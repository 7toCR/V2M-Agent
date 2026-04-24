"""
Text Understanding Agent Module

This module implements a TextAgent that uses a LangGraph-based state machine
to analyze text files and extract structured scene information (background,
background style, subject, subject mood) as JSON dictionaries via multimodal
large language model invocations.
"""

import os
import sys
import json
import re
import asyncio
from pathlib import Path
from typing import List, Literal, Dict, Any, Optional, Annotated

from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI  # type: ignore

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages, StateGraph, START, END
from operator import add
from typing_extensions import TypedDict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from promptStrategy.system_prompt_header import SystemPromptHeader
from promptStrategy.system_prompt_body import SystemPromptBody
from promptStrategy.system_prompt_profile import SystemPrompt
from promptStrategy.JSONSchema import JSONSchema  # noqa: F401  # Keep consistent dependency with pop_idx.py
from Team2.Expert.prompt import CONSTRAINTS, RESOURCES, BEST_PRACTICES, RUN_MODULE, Guide_Book_text_expert, COMMAND_text
from tools.tools import (  # Reuse the same tool parsing logic as in pop
    extract_field_from_response,
    extract_result_from_tools,
    _print_with_indent,
)

# Import agent profile
try:
    from Team2.AgentProfile.text_agent_profile import TEXT_AGENT_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import TEXT_AGENT_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    TEXT_AGENT_PROFILE = None

# Import reflection modules
try:
    from Team2.Expert.reflection_memory import ReflectionMemory, get_reflection_memory
    from Team2.Expert.reflection_agent_profile import build_reflection_prompt, parse_reflection_result
    REFLECTION_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import reflection modules: {e}")
    REFLECTION_IMPORT_SUCCESS = False
    ReflectionMemory = None
    get_reflection_memory = None
    build_reflection_prompt = None
    parse_reflection_result = None


class TextAgent:

    class Graph(TypedDict):
        # Global messages
        global_messages: Annotated[List[dict], add_messages]
        # System prompt messages
        system_prompt_messages: str
        # User messages
        user_messages: Annotated[List[dict], add_messages]
        # User task description
        tasks: str

        # Per-step messages
        think_message: Annotated[List[dict], add_messages]
        action_message: Annotated[List[dict], add_messages]
        text_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]
        reflection_message: Annotated[List[dict], add_messages]  # Reflection messages

        # Input parameters
        text_path: str  # Local path of the text to understand

        # Execution results
        text_flag: bool
        text_result: Any  # Can be Dict, List[Dict], or other formats

        # Running state
        state: Literal["start", "think", "action", "execute", "observation", "reflect", "final", "end"]
        # Execution state
        tools: Literal["None", "none", "text"]

        current_iteration: Annotated[int, add]
        max_iterations: int

        # Reflection-related fields
        reflection_flag: bool              # Reflection completion flag
        reflection_result: Dict[str, Any]  # Reflection result (analysis, strengths, weaknesses, improvement, quality, should_retry)
        reflection_count: int            # Reflection count

        complete: bool
        final_answer: Any

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        exper: Optional[ChatOpenAI] = None,
        temperature: float = 0.3,
        max_iterations: int = 10
    ) -> None:
        load_dotenv()

        config_path = os.path.join(os.path.dirname(__file__), "config_text.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.config = None

        if llm is None:
            model_config = self.config.get("model", {}) if self.config else {}
            model_name = model_config.get("name", "qwen3-max")
            base_url = model_config.get(
                "base_url",
                os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
            api_key_env = model_config.get("api_key_env", "DASHSCOPE_API_KEY")
            config_temperature = model_config.get("temperature", temperature)
            max_tokens = model_config.get("max_tokens")

            self.model = ChatOpenAI(
                model=model_name,
                api_key=os.getenv(api_key_env),
                base_url=base_url,
                temperature=config_temperature,
                max_tokens=max_tokens,
            )
        else:
            self.model = llm

        if exper is None:
            expert_config = self.config.get("expert", {}) if self.config else {}
            expert_name = expert_config.get("name", "qwen3-max")
            base_url = expert_config.get(
                "base_url",
                os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
            api_key_env = expert_config.get("api_key_env", "DASHSCOPE_API_KEY")
            config_temperature = expert_config.get("temperature", temperature)
            max_tokens = expert_config.get("max_tokens")

            self.exper = ChatOpenAI(
                model=expert_name,
                api_key=os.getenv(api_key_env),
                base_url=base_url,
                temperature=config_temperature,
                max_tokens=max_tokens,
            )
        else:
            self.exper = exper

        self.agent_profile = TEXT_AGENT_PROFILE  # Save agent profile
        self.max_iterations = max_iterations

        # Reflection memory module
        if REFLECTION_IMPORT_SUCCESS:
            self.reflection_memory = get_reflection_memory("text", CURRENT_DIR)
        else:
            self.reflection_memory = None

        self.builder = StateGraph(TextAgent.Graph)

        # Node definitions, same structure as AudioAgent
        self.builder.add_node("init", self._init_node)
        self.builder.add_node("think", self._think_node)
        self.builder.add_node("action", self._action_node)

        # Tool node: text
        self.builder.add_node("text", self._text_node)

        self.builder.add_node("observation", self._observation_node)
        self.builder.add_node("reflect", self._reflect_node)  # New: reflection node
        self.builder.add_node("final", self._final_node)

        # Edge definitions
        self.builder.add_edge(START, "init")
        self.builder.add_edge("init", "think")

        self.builder.add_conditional_edges(
            "think",
            self._route_after_think,
            {
                "action": "action",
                "final": "final",
            },
        )

        self.builder.add_conditional_edges(
            "action",
            self._route_after_action,
            {
                "text": "text",
                "observation": "observation",
            },
        )

        # Tool node returns to observation after execution
        self.builder.add_edge("text", "observation")

        # Route after observation: choose reflection or continue thinking
        self.builder.add_conditional_edges(
            "observation",
            self._route_after_observation,
            {
                "think": "think",
                "final": "final",
                "reflect": "reflect",
            },
        )

        # Route after reflection: decide next step based on reflection quality
        self.builder.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {
                "think": "think",
                "final": "final",
            },
        )

        self.builder.add_edge("final", END)

        self.graph = self.builder.compile()

    # ===================== Node Implementations =====================

    async def _init_node(self, state: "TextAgent.Graph") -> Dict[str, Any]:
        """Initialization node: build system_prompt from agent_profile."""
        task = state.get("tasks", "")
        _print_with_indent("task:", str(task), tab_count=2)

        # Build system_prompt from profile using the new method
        system_prompt = self._build_system_prompt_from_profile(task)

        #print(f"TextAgent._init_node{system_prompt}")

        return {
            "global_messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=""),
            ],
            "system_prompt_messages": system_prompt,
            "user_messages": [],
            "tasks": task,
            "think_message": [],
            "action_message": [],
            "text_message": [],
            "observation_message": [],
            "reflection_message": [],
            "text_flag": False,
            "text_result": {},
            "state": "think",
            "tools": state.get("tools", "None"),
            "current_iteration": 1,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            # Reflection-related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }

    async def _think_node(self, state: "TextAgent.Graph") -> Dict[str, Any]:

        global last_state  # Same global variable usage as AudioAgent
        global complete

        system_prompt = state.get("system_prompt_messages", "")
        messages=[SystemMessage(content=system_prompt)]
        current_iteration = state.get("current_iteration", 0)
        complete = state.get("complete", False)
        text_result = state.get("text_result", {})

        # Check if we have a valid result (non-empty dict or list)
        has_result = text_result and (
            (isinstance(text_result, dict) and text_result) or
            (isinstance(text_result, list) and len(text_result) > 0)
        )

        if complete or current_iteration > self.max_iterations or has_result:
            return {
                "current_iteration": 1,
                "complete": True,
                "state": "final",
                "text_result": text_result,  # Preserve text_result to avoid loss
            }

        think_prompt = f"""用户:根据已知条件进行思考，分析当前情况并决定下一步意图,严格遵行《回答模板》格式回答:按照以下JSON格式回答：{{"Result": ["你的思考内容"]}}只输出JSON格式，不要输出其他内容。\n"""
        messages.append(HumanMessage(content=think_prompt))
        system_prompt += think_prompt

        response = await self.model.ainvoke(messages)
        #print(f"[DEBUG]{response}\n")

        think_list = extract_field_from_response(response, "Result")
        thought = think_list[0] if isinstance(think_list, list) and len(think_list) > 0 else ""
        if isinstance(thought, list):
            thought = ''.join(thought)
        elif not isinstance(thought, str):
            thought = str(thought) if thought else ""

        if state.get("text_flag", False) and state.get("text_result"):
            thought = "None"

        _print_with_indent(f"thought{state.get('current_iteration', 0)}:", str(thought), tab_count=2)

        _complete = state.get("complete", False)
        if not _complete:
            if thought == "None":
                last_state = "final"
                _complete = True
            else:
                last_state = "action"
                _complete = False
        else:
            last_state = "final"

        content = f"thought{state.get('current_iteration', 0)}:{str(thought)}\n"
        system_prompt += content
        think_message = [{"role": "assistant", "content": content}]
        global_message = [AIMessage(content=content)]

        return {
            "global_messages": global_message,
            "system_prompt_messages": system_prompt,
            "user_messages": state.get("user_messages", []),
            "tasks": state.get("tasks", ""),
            "think_message": think_message,
            "action_message": state.get("action_message", []),
            "text_message": state.get("text_message", []),
            "observation_message": state.get("observation_message", []),
            "text_flag": state.get("text_flag", False),
            "text_result": state.get("text_result", {}),
            "state": last_state,
            "tools": state.get("tools", "None"),
            "current_iteration": 1,
            "max_iterations": state.get("max_iterations", self.max_iterations),
            "complete": _complete,
            "final_answer": state.get("final_answer", ""),
        }

    async def _action_node(self, state: "TextAgent.Graph") -> Dict[str, Any]:
        """Action node: generate Action based on latest thought and decide whether to call the text tool."""
        global last_state
        global tool

        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]

        action_prompt = f"""基于以上思考，请选择执行以下工具之一:
- text: 直接调用多模态模型，对指定本地文本进行理解并输出包含背景、背景风格、主体、主体风格的 JSON 字典
- None: 不执行任何工具
严格遵行《回答模板》格式回答,必须严格按照以下JSON格式回答：{{"Result": ["工具名字","工具参数"]}}只输出JSON格式，不要输出其他内容。\n"""

        messages.append(HumanMessage(content=action_prompt))
        system_prompt += action_prompt
        response = await self.model.ainvoke(messages)

        action_list = extract_field_from_response(response, "Result")
        action_name = action_list[0] if isinstance(action_list, list) and len(action_list) > 0 else "none"
        action_parameter = action_list[1] if isinstance(action_list, list) and len(action_list) > 1 else None

        if isinstance(action_name, list):
            action_name = ''.join(action_name)
        elif not isinstance(action_name, str):
            action_name = str(action_name) if action_name else "none"

        if isinstance(action_parameter, list):
            action_parameter = ''.join(action_parameter)
        elif not isinstance(action_parameter, str) and action_parameter is not None:
            action_parameter = str(action_parameter) if action_parameter else None

        tool = action_name.lower()
        last_state = "observation" if tool == "none" else "execute"

        action_result_content = action_name + (str(action_parameter) if action_parameter else "")
        _print_with_indent(f"action{state.get('current_iteration', 0)}:", action_result_content, tab_count=2)
        system_prompt += f"action{state.get('current_iteration', 0)}:{action_result_content}\n"

        action_result = [{"role": "assistant", "content": action_result_content}]

        return {
            "global_messages": [AIMessage(content=action_result_content)],
            "system_prompt_messages": system_prompt,
            "user_messages": state.get("user_messages", []),
            "tasks": state.get("tasks", ""),
            "think_message": state.get("think_message", []),
            "action_message": action_result,
            "text_message": state.get("text_message", []),
            "observation_message": state.get("observation_message", []),
            "text_flag": state.get("text_flag", False),
            "text_result": state.get("text_result", {}),
            "state": last_state,
            "tools": tool,
            "current_iteration": 1,
            "max_iterations": state.get("max_iterations", self.max_iterations),
            "complete": state.get("complete", False),
            "final_answer": state.get("final_answer", ""),
        }

    async def _text_node(self, state: "TextAgent.Graph") -> Dict[str, Any]:
        """
        Main function of this node: invoke the multimodal large language model
        to understand text and return JSON-formatted understanding results.
        Returns:
            A dictionary containing the following fields:
            {
                "Result": [
                    "{
                        "背景": "",
                        "背景风格": "",
                        "主体": "",
                        "主体风格": "",
                    }",
                    ...
                ]
            }
        """
        system_prompt = state.get("system_prompt_messages", "")
        text_path = state.get("text_path", "").strip()
        if not text_path:
            raise ValueError("text_path cannot be empty")

        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found: {text_path}")

        # Read the text file
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {text_path}")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

        # Define the JSONSchema for a single object
        single_item_schema = JSONSchema(
            type=JSONSchema.Type.OBJECT,
            properties={
                "背景": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="文本中描述的场景、环境或背景设定",
                    required=True,
                ),
                "背景风格": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="背景的氛围、色调、风格特征（如：温馨、冷峻、浪漫、神秘等）",
                    required=True,
                ),
                "主体": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="文本中的主要人物、角色或核心对象 以及他们正在干什么（主要人物有哪些？正在干什么）。",
                    required=True,
                ),
                "主体风格": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="主体的情感状态、情绪或心理感受（如：快乐、悲伤、焦虑、平静等）",
                    required=True,
                ),
            },
        )

        # Define the result template
        result_temple = {
            "Result": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                minItems=1,
                maxItems=10,
                items=single_item_schema,
                description="包含背景、背景风格、主体、主体风格的数组",
                required=True,
            ),
        }

        clear_system_prompt = f"""你是一个专业的文本理解助手。请仔细分析给定的文本内容，提取以下四个关键要素：
1. 背景：文本中描述的场景、环境或背景设定
2. 背景风格：背景的氛围、色调、风格特征（如：温馨、冷峻、浪漫、神秘等）
3. 主体：文本中的主要人物、角色或核心对象 以及他们正在干什么（主要人物有哪些？正在干什么）
4. 主体风格：主体的���感状态、情绪或心理感受（如：快乐、悲伤、焦虑、平静等）
请严格按照以下格式返回结果，确保所有字段都有值：
[
    "{{
        "背景": "...",
        "背景风格": "...",
        "主体": "...",
        "主体风格": "..."
    }}",
    "{{
        "背景": "...",
        "背景风格": "...",
        "主体": "...",
        "主体风格": "..."
    }}",
    ...
]
只返回上述格式，不要添加任何其他说明文字。
示例：
    文本：
        青石井栏沁着晨露，明净赤脚踩在湿润的苔痕上。他握住桶绳时，掌心传来麻绳粗糙的纹理——这是用了三年的旧绳，中间那段被井水反复浸透的位置已变成深褐色，比别处更硬实些。木桶沉入井口的瞬间，惊碎了倒映在水面的竹影，一圈圈涟漪将他的眉眼揉皱又抚平。他喜欢听桶底触水时那声短促的"咚"，像古寺晚钟的余韵被井壁收拢又送回。水将满时，绳子会发出细微的"咯吱"声，那是竹篾提梁承重时的吟唱。他匀速收绳，看着桶沿探出井口，水面上浮着两片昨夜飘入的槐叶。
        就在他转身时，厢房窗隙漏出压低的交谈声。"……那慈海法师的病，哪里是意外？"是管菜园的慧能师兄，"分明是有人在他每日打坐的蒲团里，混了毒草汁浸过的艾绒……"木桶突然倾斜，井水泼湿了僧鞋。明净听见自己手腕骨节发出僵硬的轻响。那些零碎的词句像毒蜂般钻入耳中——"住持默许的"、"为夺前年那批金箔佛像的供养"、"吐的血把殿前青砖都染透了"……
        绳子猛地勒进掌心，昨夜掌心被麻绳磨出的水泡"啵"地裂开。他盯着井中晃碎的倒影，看见自己太阳穴的青筋在跳动，像有只活物在皮下游走。满桶水突然重得提不动，不，是他整条手臂在抖，抖得桶梁撞在井壁上，砰砰砰地撞着，撞得井壁嗡嗡作响，撞得那些苔藓碎屑簌簌落进漆黑的水里。先前觉得清冽的井水腥气，此刻泛着铁锈般的味道——那是他牙关咬出的血，正顺着嘴角往下淌，滴进桶中，晕成淡红的雾，雾里浮着慈海法师去年中秋塞给他的那半块桂花糕，糕上还有师父手指的温度。
    文本分析对象格式:
    [
        "{{
            "背景": "古寺井边",
            "背景风格": "清幽古朴、静谧自然",
            "主体": "小和尚明净在打水",
            "主体风格": "专注平静"
        }}",
        "{{
            "背景": "回庙途中",
            "背景风格": "渐出静谧",
            "主体": "安静无意听到师兄讲出故事真相，小和尚明净撒到水桶",
            "主体风格": "震惊、情绪发生巨变"
        }}",
        "{{
            "背景": "已经知道真相",
            "背景风格": "打破静谧",
            "主体": "小和尚明净失态，异常愤怒",
            "主体风格": "愤怒失控、悲愤交加"
        }}"
    ]
"""

        text_prompt=f"""请你根据下面文本内容进行分析。
要求：
1. 按照文本场景个数进行划分,一段文字有多少个文本场景就划分多少个对象
2. 每个场景都必须包含"背景"、"背景风格"、"主体"、"主体风格"
3. 场景对应标签的内容必须从对应场景中总结、概括
4. 返回格式必须是JSON对象
5. 格式示例：{{"Result": [
                "{{
                    "背景": "回庙途中",
                    "背景风格": "渐出静谧",
                    "主体": "安静无意听到师兄讲出故事真相，小和尚明净撒到水桶",
                    "主体风格": "震惊、情绪发生巨变"
                }}",
                "{{
                    "背景": "已经知道真相",
                    "背景风格": "打破静谧",
                    "主体": "小和尚明净失态，青筋暴起",
                    "主体风格": "愤怒失控、悲愤交加"
                }}"
                ]
            }}
文本内容:
{text_content}
文本解析对象排布格式:
{result_temple}
请严格按照要求返回包含n个对象。
严格遵行《回答模板》格式回答,返回分析对象列表,严格按照下面格式回答：{{"Result": ["文本分析对象1", "文本分析对象2", ...]}}不要输出其他内容。\n"""
        full_system_prompt = system_prompt + clear_system_prompt
        system_prompt += clear_system_prompt
        system_prompt+=text_prompt
        messages = [
            SystemMessage(content=full_system_prompt),
            HumanMessage(content=text_prompt),
        ]

        response= await self.exper.ainvoke(messages)

        print(f"[DEBUG]{response}\n")
        try:
            result=extract_field_from_response(response,"Result")
            #print(f"\n[DEBUG]{result}\n")
        except Exception as e:
            result=''
            print(f"[ERROR]{e}")

        # Safely convert result to string for content
        if isinstance(result, (dict, list)):
            result_str = json.dumps(result, ensure_ascii=False)
        else:
            result_str = str(result) if result else ''

        content = f"text_result:{result_str}"
        text_message = [{"role": "assistant", "content": content}]
        global_message = [AIMessage(content=content)]
        system_prompt+=content

        return {
            "global_messages": global_message,
            "system_prompt_messages": system_prompt,
            "text_message": text_message,
            "text_flag": True,
            "text_result": result,
            "state": "observation",
            "tools": "None",
            "current_iteration": 1,
        }

    async def _observation_node(self, state: "TextAgent.Graph") -> Dict[str, Any]:
        """Observation node: process tool return results for the next round of thinking."""
        observation_content = ""
        current_iteration = state.get("current_iteration", 0)
        _complete = state.get("complete", False)
        if current_iteration > state.get("max_iterations", self.max_iterations):
            _complete = True

        if state.get("text_flag", False):
            text_result = state.get("text_result", {})
            if text_result:
                # Handle both dict and list formats
                if isinstance(text_result, (dict, list)):
                    result_str = json.dumps(text_result, ensure_ascii=False)
                else:
                    result_str = str(text_result)
                observation_content = f"text is: {result_str}"
        else:
            observation_content = "This action is None, and do not have observation."

        system_prompt = state.get("system_prompt_messages", "")
        _print_with_indent(f"observation{state.get('current_iteration', 0)}:", observation_content, tab_count=2)
        system_prompt += (
            f"observation{state.get('current_iteration', 0)}:{observation_content}\n"
        )

        observation_result = [{"role": "assistant", "content": observation_content}]

        return {
            "global_messages": [AIMessage(content=observation_content)],
            "system_prompt_messages": system_prompt,
            "observation_message": observation_result,
            "text_flag": False,
            "text_result": state.get("text_result", {}),  # Preserve text_result to avoid loss
            "state": "think",
            "current_iteration": 1,
            "complete": _complete,
            # Reflection-related fields
            "reflection_flag": state.get("reflection_flag", False),
            "reflection_result": state.get("reflection_result", {}),
            "reflection_count": state.get("reflection_count", 0),
        }

    async def _reflect_node(self, state: "TextAgent.Graph") -> Dict[str, Any]:
        """
        Reflection node (Reflexion mechanism):
        Generate reflection based on observation results, evaluate quality,
        and decide the next action.

        Reference to the Self-Reflection component from the Reflexion paper:
        - Analyze the match between execution results and task objectives
        - Identify success factors and shortcomings
        - Generate specific improvement suggestions
        - Evaluate reflection quality (high/medium/low)
        - Decide whether a retry is needed
        """
        _print_with_indent("", "Reflection node started...", tab_count=1)

        task_description = state.get("tasks", "")
        observation = ""

        # Build observation result string
        if state.get("text_flag", False):
            text_result = state.get("text_result", {})
            if text_result:
                if isinstance(text_result, (dict, list)):
                    result_str = json.dumps(text_result, ensure_ascii=False)
                else:
                    result_str = str(text_result)
                observation = f"Text understanding result: {result_str}"
        else:
            observation = "No valid result obtained"

        # Get historical reflection memory
        history_reflections = ""
        if self.reflection_memory:
            history_reflections = self.reflection_memory.get_summary()

        # Build reflection prompt
        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="text",
                task_description=task_description,
                observation=observation,
                history_reflections=history_reflections
            )
        else:
            reflect_prompt = f"""你是一个自我反思专家。请分析以下执行结果：

任务描述: {task_description}
观察结果: {observation}
历史反思: {history_reflections or "暂无"}

请按JSON格式输出反思：
{{
    "analysis": "分析结果是否达成目标",
    "strengths": ["成功1", "成功2"],
    "weaknesses": ["不足1"],
    "improvement": "改进建议",
    "quality": "high/medium/low",
    "should_retry": true/false
}}
"""

        system_prompt = state.get("system_prompt_messages", "")
        system_prompt += "\n## Reflection Phase\n"
        system_prompt += reflect_prompt

        messages = [SystemMessage(content=system_prompt)]

        try:
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            _print_with_indent("", f"Reflection generation failed: {e}", tab_count=2)
            response_text = '{"quality": "low", "should_retry": false}'

        # Parse reflection result
        if REFLECTION_IMPORT_SUCCESS and parse_reflection_result:
            reflection_result = parse_reflection_result(response_text)
        else:
            import re
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    reflection_result = json.loads(json_match.group())
                else:
                    reflection_result = {"quality": "low", "should_retry": False}
            except:
                reflection_result = {"quality": "low", "should_retry": False}

        reflection_content = f"Reflection analysis: {reflection_result.get('analysis', '')}\n"
        reflection_content += f"Success factors: {', '.join(reflection_result.get('strengths', []))}\n"
        reflection_content += f"Shortcomings: {', '.join(reflection_result.get('weaknesses', []))}\n"
        reflection_content += f"Improvement suggestions: {reflection_result.get('improvement', '')}\n"
        reflection_content += f"Reflection quality: {reflection_result.get('quality', 'low')}\n"
        reflection_content += f"Suggest retry: {reflection_result.get('should_retry', False)}"

        _print_with_indent("", f"Reflection result:", tab_count=2)
        _print_with_indent("", f"  Quality: {reflection_result.get('quality', 'low')}", tab_count=3)
        _print_with_indent("", f"  Suggest retry: {reflection_result.get('should_retry', False)}", tab_count=3)

        # Save reflection to memory
        if self.reflection_memory:
            self.reflection_memory.add_reflection(
                task_description=task_description,
                observation=observation,
                reflection=reflection_result.get('analysis', ''),
                improvement=reflection_result.get('improvement', ''),
                quality=reflection_result.get('quality', 'low'),
                iterations=state.get("current_iteration", 1)
            )

        system_prompt += f"\n{reflection_content}\n"

        reflection_result_dict = {
            "reflection_message": reflection_content,
            "analysis": reflection_result.get('analysis', ''),
            "strengths": reflection_result.get('strengths', []),
            "weaknesses": reflection_result.get('weaknesses', []),
            "improvement": reflection_result.get('improvement', ''),
            "quality": reflection_result.get('quality', 'low'),
            "should_retry": reflection_result.get('should_retry', False)
        }

        # Decide next step: based on reflection quality
        should_retry = reflection_result.get('should_retry', False)
        quality = reflection_result.get('quality', 'low')
        current_iteration = state.get("current_iteration", 1)
        max_iterations = state.get("max_iterations", self.max_iterations)
        reflection_count = state.get("reflection_count", 0) + 1

        # If quality is high or max iterations reached, proceed to final answer
        if quality == "high" or current_iteration >= max_iterations or reflection_count >= 3:
            next_state = "final"
            _complete = True
            _print_with_indent("", f"Reflection quality high / limit reached, proceeding to final answer node", tab_count=2)
        else:
            next_state = "think"
            _complete = False
            _print_with_indent("", f"Reflection quality low, entering retry thinking phase", tab_count=2)

        return {
            "global_messages": [AIMessage(content=reflection_content)],
            "system_prompt_messages": system_prompt,
            "reflection_message": [{"role": "assistant", "content": reflection_content}],
            "text_flag": state.get("text_flag", False),
            "text_result": state.get("text_result", {}),
            "state": next_state,
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": True,
            "reflection_result": reflection_result_dict,
            "reflection_count": reflection_count,
        }

    def save_data(self,state: "Agent.Graph") -> None:
        address=state.get("text_path", "")
        jsonl_name="text.jsonl"
        result=state.get("text_result", [])

        print(f"[DEBUG]{type(result)}")
        address_1, ext_name = os.path.split(address)
        address_2, file_name = os.path.split(address_1)
        root_name, input_name = os.path.split(address_2)

        output_root = os.path.join(root_name, "output")
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_dir= os.path.join(output_root, file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_jsonl_address= os.path.join(output_dir, jsonl_name)


        import ast
        if result:
            if isinstance(result, list):
                with open(output_jsonl_address, "w", encoding="utf-8") as f:
                    for item in result:
                        print(f"\t[DEBUG]{type(item)}")
                        if isinstance(item, str):
                            item=ast.literal_eval(item)
                        data = json.dumps(item, ensure_ascii=False)
                        f.write(data + '\n')
            else:
                with open(output_jsonl_address, "w", encoding="utf-8") as f:
                    data = json.dumps(result, ensure_ascii=False)
                    f.write(data + '\n')

    async def _final_node(self, state: "TextAgent.Graph") -> Dict[str, Any]:
        """Final answer node: output text understanding result."""
        text_result = state.get("text_result", {})
        self.save_data(state=state)
        # Return the entire dict directly
        if isinstance(text_result, dict):
            final_answer: Any = text_result
        elif text_result:
            final_answer = text_result
        else:
            final_answer = {}

        _print_with_indent(
            f"final_answer{state.get('current_iteration', 0)}:", final_answer
        )
        system_prompt = state.get("system_prompt_messages", "")
        system_prompt += (
            f"final_answer{state.get('current_iteration', 0)}:{final_answer}\n"
        )

        return {
            "global_messages": [AIMessage(content=f"Final answer: {final_answer}")],
            "system_prompt_messages": system_prompt,
            "final_answer": final_answer,
        }

    # ===================== Routing Functions =====================

    def _route_after_think(self, state: "TextAgent.Graph") -> str:
        """Route after thinking."""
        if state.get("current_iteration", 0) >= state.get(
            "max_iterations", self.max_iterations
        ):
            return "final"
        elif state.get("complete", False):
            return "final"
        else:
            return "action"

    def _route_after_action(self, state: "TextAgent.Graph") -> str:
        """Route after action."""
        tool_name: str = state.get("tools", "none")
        if tool_name in ("None", "none"):
            return "observation"
        elif tool_name == "text":
            return "text"
        else:
            return "observation"

    def _route_after_observation(self, state: "TextAgent.Graph") -> str:
        """
        Route after observation.

        Decision logic (reference to Reflexion paper):
        - If valid results exist and quality is high -> enter reflection node for evaluation
        - If iteration count has reached the limit -> enter reflection node for final evaluation
        - Otherwise -> continue to next round of thinking
        """
        text_result = state.get("text_result", {})
        has_result = text_result and (
            (isinstance(text_result, dict) and text_result) or
            (isinstance(text_result, list) and len(text_result) > 0)
        )

        current_iteration = state.get("current_iteration", 1)
        max_iterations = state.get("max_iterations", self.max_iterations)
        complete = state.get("complete", False)

        # If already complete, go directly to reflection for evaluation
        if complete:
            return "reflect"
        # If results exist, enter reflection to evaluate quality
        elif has_result:
            return "reflect"
        # If iteration count reaches the limit, enter reflection
        elif current_iteration >= max_iterations:
            return "reflect"
        else:
            return "think"

    def _route_after_reflect(self, state: "TextAgent.Graph") -> str:
        """
        Route after reflection.

        Decide next step based on reflection result (reference to Reflexion paper):
        - If reflection quality is high and result is satisfactory -> output final answer
        - If retry is suggested and max reflection count not reached -> return to think node for retry
        - Otherwise -> output final answer
        """
        reflection_result = state.get("reflection_result", {})
        quality = reflection_result.get("quality", "low")
        should_retry = reflection_result.get("should_retry", False)
        reflection_count = state.get("reflection_count", 0)

        # If too many reflections, output directly
        if reflection_count >= 3:
            return "final"

        # If quality is high, output final answer
        if quality == "high":
            return "final"

        # If retry is suggested, return to think node
        if should_retry:
            return "think"

        # Default: output final answer
        return "final"

    # ===================== Public Interface =====================

    def invoke(self, user_input: str, text_path: str) -> Dict[str, Any]:
        """Execute TextAgent and return the final result at once."""
        initial_state: TextAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "text_message": [],
            "observation_message": [],
            "reflection_message": [],
            "text_path": text_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "text_flag": False,
            "text_result": {},
            # Reflection-related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }
        return self.graph.invoke(initial_state)

    def stream(self, user_input: str, text_path: str):
        """Stream-execute TextAgent, returning intermediate states step by step."""
        initial_state: TextAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "text_message": [],
            "observation_message": [],
            "reflection_message": [],
            "text_path": text_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "text_flag": False,
            "text_result": {},
            # Reflection-related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }
        return self.graph.stream(initial_state)

    # ===================== Helper Methods =====================

    def _generate_other_system_prompt_for_task_sync(
        self, task: str
    ) -> tuple[SystemPromptHeader, SystemPromptBody]:
        """
        Synchronous wrapper for AudioAgent's generate_other_system_prompt_for_task,
        to avoid using asyncio.run directly in graph nodes.
        """
        system_prompt = SystemPrompt()
        prompt = system_prompt.build_prompt(task)

        try:
            response = self.model.invoke(prompt)
        except Exception as e:  # Keep error handling logic compatible with AudioAgent
            error_msg = str(e)
            if "Arrearage" in error_msg or "overdue-payment" in error_msg:
                raise Exception(
                    "Alibaba Cloud account has overdue payment. Please visit https://help.aliyun.com/zh/model-studio/error-code#overdue-payment for details and top up."
                )
            else:
                raise e

        return system_prompt.parse_response_content(response)

    def _build_system_prompt_from_profile(self, task: str) -> str:
        """Build the complete system_prompt from agent_profile.

        Args:
            task: Task description

        Returns:
            str: The constructed system_prompt
        """
        if not PROFILE_IMPORT_SUCCESS or self.agent_profile is None:
            # If profile loading failed, use default build method (requires async)
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If in an async environment, return a simple prompt
                return f"## My Task\n\"\"\"{task}\"\"\"\n\n## Conversation History\n"
            else:
                # Synchronous environment, can run async method
                header, body = loop.run_until_complete(self._generate_other_system_prompt_for_task_async(task))
                system_prompt = f"You are {header.agent_name} AI, {header.agent_role}.\n"
                return system_prompt

        profile = self.agent_profile
        system_prompt = ""

        # 1. Role and description
        system_prompt += f"# {profile.role.name}\n\n"
        system_prompt += f"{profile.description}\n\n"
        system_prompt += f"## Role Description\n{profile.role.description}\n\n"

        # 2. Responsibilities
        system_prompt += f"## Responsibilities\n"
        for i, resp in enumerate(profile.role.responsibilities, 1):
            system_prompt += f"{i}. {resp}\n"
        system_prompt += "\n"

        # 3. Areas of expertise
        system_prompt += f"## Areas of Expertise\n"
        for i, exp in enumerate(profile.role.expertise, 1):
            system_prompt += f"{i}. {exp}\n"
        system_prompt += "\n"

        # 4. Tools
        system_prompt += f"## Available Tools\n"
        for tool in profile.tools:
            system_prompt += f"### {tool.name}\n"
            system_prompt += f"- Description: {tool.description}\n"
            system_prompt += f"- Function signature: {tool.function_signature}\n"
            if tool.dependencies:
                system_prompt += f"- Dependencies: {', '.join(tool.dependencies)}\n"
            if tool.usage_example:
                system_prompt += f"- Usage example: {tool.usage_example}\n"
            system_prompt += "\n"

        # 5. Knowledge domains
        system_prompt += f"## Knowledge Domains\n"
        for knowledge in profile.knowledge:
            system_prompt += f"### {knowledge.domain.value}\n"
            system_prompt += f"**Core Concepts:**\n"
            for concept in knowledge.concepts:
                system_prompt += f"- {concept}\n"
            system_prompt += f"\n**Rules:**\n"
            for rule in knowledge.rules:
                system_prompt += f"- {rule}\n"
            system_prompt += "\n"

        # 6. Constraints
        system_prompt += f"## Constraints\n"
        for i, constraint in enumerate(profile.constraints, 1):
            system_prompt += f"{i}. {constraint}\n"
        system_prompt += "\n"

        # 7. Best practices
        system_prompt += f"## Best Practices\n"
        for i, practice in enumerate(profile.best_practices, 1):
            system_prompt += f"{i}. {practice}\n"
        system_prompt += "\n"

        # 8. Resources
        system_prompt += f"## Available Resources\n"
        for i, resource in enumerate(profile.resources, 1):
            system_prompt += f"{i}. {resource}\n"
        system_prompt += "\n"

        # 9. Execution methods
        system_prompt += f"## Execution Methods\n"
        for i, method in enumerate(profile.run_methods, 1):
            system_prompt += f"{i}. {method}\n"
        system_prompt += "\n"

        # 10. Commands / tool descriptions
        if profile.command:
            system_prompt += f"{profile.command}\n\n"

        # 11. Knowledge guide
        if profile.guide_book:
            system_prompt += f"{profile.guide_book}\n\n"

        # 12. My Task
        system_prompt += f'''## My Task\n"""{task}"""\n\n'''

        # 13. Conversation History
        system_prompt += f"## Conversation History\n"

        return system_prompt

    async def _generate_other_system_prompt_for_task_async(
        self, task: str
    ) -> tuple[SystemPromptHeader, SystemPromptBody]:
        """
        Async version: wrapper for AudioAgent's generate_other_system_prompt_for_task.
        """
        system_prompt = SystemPrompt()
        prompt = system_prompt.build_prompt(task)

        try:
            response = await self.model.ainvoke(prompt)
        except Exception as e:  # Keep error handling logic compatible with AudioAgent
            error_msg = str(e)
            if "Arrearage" in error_msg or "overdue-payment" in error_msg:
                raise Exception(
                    "Alibaba Cloud account has overdue payment. Please visit https://help.aliyun.com/zh/model-studio/error-code#overdue-payment for details and top up."
                )
            else:
                raise e

        return system_prompt.parse_response_content(response)

    async def ainvoke(self, user_input: str, text_path: str) -> Dict[str, Any]:
        """Asynchronously invoke TextAgent and return the final result at once."""
        initial_state: TextAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "text_message": [],
            "observation_message": [],
            "reflection_message": [],
            "text_path": text_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "text_flag": False,
            "text_result": {},
            # Reflection-related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }
        return await self.graph.ainvoke(initial_state)


if __name__ == "__main__":
    import asyncio
    load_dotenv()
    agent = TextAgent(max_iterations=5)
    demo_text_path = r"D:\Python\Project\MCP\Agent\prompt_agent\sample\input\shanghai\description1.txt"
    task = "调用工具帮我理解这个文本，并给出包含背景、背景风格、主体、主体风格的 JSON 字典"

    async def main():
        try:
            response = await agent.ainvoke(user_input=task, text_path=demo_text_path)
            text_result = response.get("text_result", [])

            # Return the entire dict directly
            if isinstance(text_result, dict):
                final_answer: Any = text_result
            elif text_result:
                final_answer = text_result
            else:
                final_answer = {}
            print(f"Final result: {final_answer}")

        except Exception as exc:
            print(f"Error: {exc}")

    asyncio.run(main())
