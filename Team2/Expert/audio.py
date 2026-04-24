"""
Audio understanding agent module.

Uses a multimodal large language model to analyze audio files,
outputting time-segmented description JSON containing subject sound content,
subject sound style, environmental sound content, and environmental sound style.
"""

import os
import sys
import json
import base64
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
from promptStrategy.JSONSchema import JSONSchema  # noqa: F401  # Keep dependency consistent with pop_idx.py
from Team2.Expert.prompt import CONSTRAINTS, RESOURCES, BEST_PRACTICES, RUN_MODULE, Guide_Book_audio_expert, COMMAND_audio
from tools.tools import (  # Reuse the same tool parsing logic as pop
    extract_field_from_response,
    extract_result_from_tools,
    _print_with_indent,
)

# Import agent profile
try:
    from Team2.AgentProfile.audio_agent_profile import AUDIO_AGENT_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import AUDIO_AGENT_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    AUDIO_AGENT_PROFILE = None

# Import reflection module
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


def _encode_audio_to_base64(audio_path: str) -> str:
    """Encode an audio file to a base64 string."""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


class AudioAgent:

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
        audio_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]
        reflection_message: Annotated[List[dict], add_messages]  # Reflection messages

        # Input parameters
        audio_path: str  # Local path of the audio file to understand

        # Execution results
        audio_flag: bool
        audio_result: Any  # Can be Dict, List[Dict], or other formats

        # Running state
        state: Literal["start", "think", "action", "execute", "observation", "reflect", "final", "end"]
        # Execution state
        tools: Literal["None", "none", "audio"]

        current_iteration: Annotated[int, add]
        max_iterations: int

        # Reflection-related fields
        reflection_flag: bool              # Reflection completion flag
        reflection_result: Dict[str, Any]  # Reflection result
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

        config_path = os.path.join(os.path.dirname(__file__), "config_audio.json")
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
            expert_name = expert_config.get("name", "qwen3-omni-flash")
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

        self.agent_profile = AUDIO_AGENT_PROFILE  # Save agent profile
        self.max_iterations = max_iterations

        # Reflection memory module
        if REFLECTION_IMPORT_SUCCESS:
            self.reflection_memory = get_reflection_memory("audio", CURRENT_DIR)
        else:
            self.reflection_memory = None

        self.builder = StateGraph(AudioAgent.Graph)

        # Node definitions, same structure as VideoAgent
        self.builder.add_node("init", self._init_node)
        self.builder.add_node("think", self._think_node)
        self.builder.add_node("action", self._action_node)

        # Tool node: audio
        self.builder.add_node("audio", self._audio_node)

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
                "audio": "audio",
                "observation": "observation",
            },
        )

        # Tool node execution returns to observation
        self.builder.add_edge("audio", "observation")

        # Post-observation routing
        self.builder.add_conditional_edges(
            "observation",
            self._route_after_observation,
            {
                "think": "think",
                "final": "final",
                "reflect": "reflect",
            },
        )

        # Post-reflection routing
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

    # ===================== Helper Methods =====================

    def _build_system_prompt_from_profile(self, task: str) -> str:
        """Build the complete system_prompt from agent_profile."""
        if not PROFILE_IMPORT_SUCCESS or self.agent_profile is None:
            # If profile import failed, return a simple fallback
            return f"## My Task\n\"\"\"{task}\"\"\"\n\n## Conversation History\n"

        profile = self.agent_profile
        system_prompt = ""

        # 1. Role and description
        system_prompt += f"# {profile.role.name}\n\n"
        system_prompt += f"{profile.role.description}\n\n"

        # 2. Core responsibilities
        if profile.role.responsibilities:
            system_prompt += "## Core Responsibilities\n\n"
            for resp in profile.role.responsibilities:
                system_prompt += f"- {resp}\n"
            system_prompt += "\n"

        # 3. Areas of expertise
        if hasattr(profile.role, 'expertise') and profile.role.expertise:
            system_prompt += "## Areas of Expertise\n\n"
            for exp in profile.role.expertise:
                system_prompt += f"- {exp}\n"
            system_prompt += "\n"

        # 4. Available tools
        if profile.tools:
            system_prompt += "## Available Tools\n\n"
            for tool in profile.tools:
                system_prompt += f"### {tool.name}\n\n"
                system_prompt += f"**Description**: {tool.description}\n\n"
                system_prompt += f"**Function Signature**: `{tool.function_signature}`\n\n"
                if tool.parameters:
                    system_prompt += "**Parameters**:\n"
                    for param in tool.parameters:
                        param_name = param.get('name', '')
                        param_type = param.get('type', '')
                        param_desc = param.get('description', '')
                        system_prompt += f"- `{param_name}` ({param_type}): {param_desc}\n"
                    system_prompt += "\n"
                if tool.returns:
                    system_prompt += f"**Returns**: {tool.returns}\n\n"
                if hasattr(tool, 'usage_example') and tool.usage_example:
                    system_prompt += f"**Example**:\n```python\n{tool.usage_example}\n```\n\n"

        # 5. Knowledge domains
        if profile.knowledge:
            system_prompt += "## Knowledge Domains\n\n"
            for knowledge in profile.knowledge:
                system_prompt += f"### {knowledge.domain}\n\n"
                if knowledge.concepts:
                    system_prompt += "**Key Concepts**:\n"
                    for concept in knowledge.concepts:
                        system_prompt += f"- {concept}\n"
                    system_prompt += "\n"
                if knowledge.rules:
                    system_prompt += "**Rules**:\n"
                    for rule in knowledge.rules:
                        system_prompt += f"- {rule}\n"
                    system_prompt += "\n"

        # 6. Constraints
        if profile.constraints:
            system_prompt += "## Constraints\n\n"
            for constraint in profile.constraints:
                system_prompt += f"- {constraint}\n"
            system_prompt += "\n"

        # 7. Best practices
        if profile.best_practices:
            system_prompt += "## Best Practices\n\n"
            for practice in profile.best_practices:
                system_prompt += f"- {practice}\n"
            system_prompt += "\n"

        # 8. Resources
        if profile.resources:
            system_prompt += "## Resources\n\n"
            for resource in profile.resources:
                system_prompt += f"- {resource}\n"
            system_prompt += "\n"

        # 9. Run methods
        if profile.run_methods:
            system_prompt += "## Run Methods\n\n"
            for method in profile.run_methods:
                system_prompt += f"{method}\n"
            system_prompt += "\n"

        # 10. Commands
        if profile.command:
            system_prompt += f"## Available Commands\n\n{profile.command}\n\n"

        # 11. Guide book
        if profile.guide_book:
            system_prompt += f"## Guide Book\n\n{profile.guide_book}\n\n"

        # 12. My task
        system_prompt += f'''## My Task\n"""{task}"""\n\n'''

        # 13. Conversation history
        system_prompt += f"## Conversation History\n"

        return system_prompt

    # ===================== Node Implementations =====================

    async def _init_node(self, state: "AudioAgent.Graph") -> Dict[str, Any]:
        """Initialization node: build system_prompt from agent_profile."""
        task = state.get("tasks", "")
        _print_with_indent("task:", str(task), tab_count=2)

        # Build system_prompt from profile using the new method
        system_prompt = self._build_system_prompt_from_profile(task)
        #print(f"AudioAgent._init_node{system_prompt}")

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
            "audio_message": [],
            "observation_message": [],
            "reflection_message": [],
            "audio_flag": False,
            "audio_result": {},
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

    async def _think_node(self, state: "AudioAgent.Graph") -> Dict[str, Any]:

        global last_state  # Same global variable usage as AudioAgent
        global complete

        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]
        current_iteration = state.get("current_iteration", 0)
        complete = state.get("complete", False)
        audio_result = state.get("audio_result", {})

        # Check if we have a valid result (non-empty dict or list)
        has_result = audio_result and (
            (isinstance(audio_result, dict) and audio_result) or
            (isinstance(audio_result, list) and len(audio_result) > 0)
        )

        if complete or current_iteration > self.max_iterations or has_result:
            return {
                "current_iteration": 1,
                "complete": True,
                "state": "final",
                "audio_result": audio_result,  # Preserve audio_result to avoid losing it
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

        if state.get("audio_flag", False) and state.get("audio_result"):
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
            "audio_message": state.get("audio_message", []),
            "observation_message": state.get("observation_message", []),
            "audio_flag": state.get("audio_flag", False),
            "audio_result": state.get("audio_result", {}),
            "state": last_state,
            "tools": state.get("tools", "None"),
            "current_iteration": 1,
            "max_iterations": state.get("max_iterations", self.max_iterations),
            "complete": _complete,
            "final_answer": state.get("final_answer", ""),
        }

    async def _action_node(self, state: "AudioAgent.Graph") -> Dict[str, Any]:
        """Action node: generate an Action based on the latest thought and decide whether to call the audio tool."""
        global last_state
        global tool

        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]

        last_thought = ""
        if state.get("think_message"):
            last_thought_msg = state["think_message"][-1]
            if isinstance(last_thought_msg, dict):
                last_thought = last_thought_msg.get("content", "")
            else:
                last_thought = str(last_thought_msg)

        action_prompt = f"""基于以上思考，请选择执行以下工具之一:
- audio: 直接调用多模态模型，对指定本地音频进行理解并输出按时间片段划分的描述 JSON
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
            "audio_message": state.get("audio_message", []),
            "observation_message": state.get("observation_message", []),
            "audio_flag": state.get("audio_flag", False),
            "audio_result": state.get("audio_result", []),
            "state": last_state,
            "tools": tool,
            "current_iteration": 1,
            "max_iterations": state.get("max_iterations", self.max_iterations),
            "complete": state.get("complete", False),
            "final_answer": state.get("final_answer", ""),
        }

    async def _audio_node(self, state: "AudioAgent.Graph") -> Dict[str, Any]:
        """
        Main function: call the multimodal large language model to understand audio,
        and return the understanding content in JSON format.
        Returns:
            A dictionary containing the following fields:
            {
                "Result": [
                    {
                        "时间段": "",
                        "主体声音内容": "...",
                        "主体声音风格": "...",
                        "环境声音内容": "...",
                        "环境声音风格": "..."
                    },
                    ...
                ]
            }
        """
        # Do not use accumulated history prompt; use a fresh, concise prompt instead.
        # This avoids overly long and confusing prompts, ensuring the model correctly understands the task.
        audio_path = state.get("audio_path", "").strip()
        if not audio_path:
            raise ValueError("audio_path cannot be empty")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_file = Path(audio_path)
        audio_format = audio_file.suffix.lstrip(".") or "wav"

        try:
            audio_base64 = _encode_audio_to_base64(str(audio_file))
        except Exception as e:
            raise Exception(f"Error encoding audio file: {str(e)}")

        audio_data_url = f"data:audio/{audio_format};base64,{audio_base64}"

        # Use a fresh, concise system_prompt, referencing the comprehension_audio structure.
        # Emphasize full coverage of the entire audio, using longer time segment examples.
        system_prompt = """你是一个专业的音频理解助手。请根据音频实际内容，按时间顺序划分若干片段，
为每个片段提取以下信息:时间段、主体声音内容、主体声音风格、环境声音内容、环境声音风格。时间可用秒数标识。

【核心要求 - 必须严格遵守】：
1. **完整覆盖**：必须完整覆盖整个音频文件，从0秒开始到音频结束，不能遗漏任何时间段。如果音频是60秒，必须从0s开始，到60s结束，中间所有时间段都要覆盖。
2. **合理划分**：根据音频内容的变化（如主体声音变化、环境声音变化、情绪或风格变化）合理划分时间段，每个时间段通常为3-15秒，但要根据实际内容灵活调整。
3. **详细描述**：每个字段的描述必须详细、具体、完整，不能过于简单或空泛。
4. **时间连续**：时间段必须连续，不能有间隔。例如：如果第一个片段是"0s-13s"，下一个片段必须是"13s-xx"，以此类推。

输出要求：仅返回 JSON 数组，每个元素都包含以下字段：
{
    "时间段": "",
    "主体声音内容": "...",
    "主体声音风格": "...",
    "环境声音内容": "...",
    "环境声音风格": "..."
}
只返回 JSON,不要附加其他文本。

示例:
[
    {
        "时间段": "0s-13s",
        "主体声音内容": "南湖红船劈开千重浪，井冈星火燃起万里光。红色血脉胸中激荡，铸就中华铁骨脊梁。",
        "主体声音风格": "男子合唱，声音铿锵有力、庄重深厚，以历史叙事感拉开序幕",
        "环境声音内容": "合唱声在教室空间内产生清晰可辨的回响，无明显杂音",
        "环境声音风格": "空间感明显，氛围肃穆，凸显集体朗诵的庄严性"
    },
    {
        "时间段": "13s-25s",
        "主体声音内容": "南昌枪声刺破长夜暗，大渡铁索擎起赤旗扬。平型关外民族怒吼，铁血丹心照耀千秋。",
        "主体声音风格": "合唱情绪逐渐上扬，节奏加快，力度增强，充满革命斗争的英勇气势",
        "环境声音内容": "仍以人声为主，可隐约听到多人同步呼吸、身体轻微移动的声音",
        "环境声音风格": "情绪推进中，环境音与朗诵情感同步升温"
    },
    {
        "时间段": "25s-38s",
        "主体声音内容": "天宫轨迹书写新和章，脱贫脚印丈量悬崖月。港珠澳大桥连沧海，北斗指路耀四方。这就是新时代的中国力量。",
        "主体声音风格": "歌声转向明亮自豪，音调昂扬，充满现代化成就的豪迈与信心",
        "环境声音内容": "朗诵声集中而响亮，环境音干净",
        "环境声音风格": "氛围开阔明亮，与歌词中"新时代"的宏大意象相呼应"
    },
    {
        "时间段": "38s-50s",
        "主体声音内容": "请听血脉里的呐喊，红旗漫卷的方向，就是吾辈征战的疆场。请党放心，强国有我！请党放心，强国有我！请党放心，强国有我！",
        "主体声音风格": "情绪达到高潮，三次"请党放心，强国有我"一次比一次高亢、坚定，充满青春誓言感",
        "环境声音内容": "合唱声极具爆发力，结尾处音量最大，空间回声显著",
        "环境声音风格": "情感充沛，声势达到顶峰，极具感染力"
    },
    {
        "时间段": "50s-52s",
        "主体声音内容": "OK,哥们儿。",
        "主体声音风格": "单个男声口语化表达，语气轻松，与之前庄严朗诵形成鲜明对比",
        "环境声音内容": "众人声音戛然而止，随后有轻微放松的动静、可能包括放手机或移动椅子的声音",
        "环境声音风格": "瞬间从仪式感切换回日常排练状态，表明活动结束"
    }
]

不要附加任何额外说明。"""

        user_prompt = "请输出多个时间片段的 JSON 数组，给出各片段的关键帧/时间点与背景、主体、情绪。必须完整覆盖整个音频，时间段必须连续。"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data_url,
                            "format": audio_format,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ]
            ),
        ]

        response = await self.exper.ainvoke(messages)

        print(f"[DEBUG]response:{response}\n")

        # Improved JSON parsing logic to handle multiple possible response formats
        response_content = response.content if hasattr(response, 'content') else str(response)

        # Clean possible markdown code block markers
        response_text = response_content.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Try to parse JSON directly
        try:
            result = json.loads(response_text)
            # If the returned value is a dict containing a Result field, extract Result
            if isinstance(result, dict) and "Result" in result:
                result = result["Result"]
            # Ensure the return value is in list format
            if isinstance(result, dict):
                result = [result]
        except json.JSONDecodeError:
            # If direct parsing fails, try using extract_field_from_response
            result = extract_field_from_response(response, "Result")
            if not result:
                # If still failing, try to extract JSON from the response
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise Exception(f"Unable to parse response as JSON: {response_text[:500]}")

        print(f"[DEBUG]result:{result}\n")

        # Safely convert result to string for content
        if isinstance(result, (dict, list)):
            result_str = json.dumps(result, ensure_ascii=False)
        else:
            result_str = str(result) if result else ''

        content = f"audio_result:{result_str}"
        audio_message = [{"role": "assistant", "content": content}]
        global_message = [AIMessage(content=content)]

        # Update system_prompt_messages in state (preserve history, but don't pollute new prompt)
        updated_system_prompt = state.get("system_prompt_messages", "") + "\n" + content

        return {
            "global_messages": global_message,
            "system_prompt_messages": updated_system_prompt,
            "audio_message": audio_message,
            "audio_flag": True,
            "audio_result": result,
            "state": "observation",
            "tools": "None",
            "current_iteration": 1,
        }

    async def _observation_node(self, state: "AudioAgent.Graph") -> Dict[str, Any]:
        """Observation node: process tool return results for use in the next round of thinking."""
        observation_content = ""
        current_iteration = state.get("current_iteration", 0)
        _complete = state.get("complete", False)
        if current_iteration > state.get("max_iterations", self.max_iterations):
            _complete = True

        if state.get("audio_flag", False):
            audio_result = state.get("audio_result", {})
            if audio_result:
                # Handle both dict and list formats
                if isinstance(audio_result, (dict, list)):
                    result_str = json.dumps(audio_result, ensure_ascii=False)
                else:
                    result_str = str(audio_result)
                observation_content = f"audio is: {result_str}"
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
            "audio_flag": False,
            "audio_result": state.get("audio_result", {}),
            "state": "think",
            "current_iteration": 1,
            "complete": _complete,
            # Reflection-related fields
            "reflection_flag": state.get("reflection_flag", False),
            "reflection_result": state.get("reflection_result", {}),
            "reflection_count": state.get("reflection_count", 0),
        }

    async def _reflect_node(self, state: "AudioAgent.Graph") -> Dict[str, Any]:
        """
        Reflection node (Reflexion mechanism): generate reflection based on observation results,
        evaluate quality.
        """
        _print_with_indent("", "Reflection node starting...", tab_count=1)

        task_description = state.get("tasks", "")
        observation = ""

        if state.get("audio_flag", False):
            audio_result = state.get("audio_result", {})
            if audio_result:
                if isinstance(audio_result, (dict, list)):
                    result_str = json.dumps(audio_result, ensure_ascii=False)
                else:
                    result_str = str(audio_result)
                observation = f"Audio understanding result: {result_str}"
        else:
            observation = "No valid result obtained"

        history_reflections = ""
        if self.reflection_memory:
            history_reflections = self.reflection_memory.get_summary()

        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="audio",
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
        system_prompt += "\n## Reflection Phase\n" + reflect_prompt

        messages = [SystemMessage(content=system_prompt)]

        try:
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            _print_with_indent("", f"Reflection generation failed: {e}", tab_count=2)
            response_text = '{"quality": "low", "should_retry": false}'

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
        reflection_content += f"Strengths: {', '.join(reflection_result.get('strengths', []))}\n"
        reflection_content += f"Weaknesses: {', '.join(reflection_result.get('weaknesses', []))}\n"
        reflection_content += f"Improvement suggestions: {reflection_result.get('improvement', '')}\n"
        reflection_content += f"Reflection quality: {reflection_result.get('quality', 'low')}\n"
        reflection_content += f"Suggest retry: {reflection_result.get('should_retry', False)}"

        _print_with_indent("", f"Reflection result:", tab_count=2)
        _print_with_indent("", f"  Quality: {reflection_result.get('quality', 'low')}", tab_count=3)
        _print_with_indent("", f"  Suggest retry: {reflection_result.get('should_retry', False)}", tab_count=3)

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

        quality = reflection_result.get('quality', 'low')
        current_iteration = state.get("current_iteration", 1)
        max_iterations = state.get("max_iterations", self.max_iterations)
        reflection_count = state.get("reflection_count", 0) + 1

        if quality == "high" or current_iteration >= max_iterations or reflection_count >= 3:
            next_state = "final"
            _complete = True
            _print_with_indent("", f"Reflection quality high / limit reached, entering final answer node", tab_count=2)
        else:
            next_state = "think"
            _complete = False
            _print_with_indent("", f"Reflection quality low, entering retry thinking phase", tab_count=2)

        return {
            "global_messages": [AIMessage(content=reflection_content)],
            "system_prompt_messages": system_prompt,
            "reflection_message": [{"role": "assistant", "content": reflection_content}],
            "audio_flag": state.get("audio_flag", False),
            "audio_result": state.get("audio_result", {}),
            "state": next_state,
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": True,
            "reflection_result": reflection_result_dict,
            "reflection_count": reflection_count,
        }

    def save_data(self,state: "Agent.Graph") -> None:
        address=state.get("audio_path", "")
        jsonl_name="audio.jsonl"
        result=state.get("audio_result", [])

        address_1, ext_name = os.path.split(address)
        address_2, file_name = os.path.split(address_1)
        root_name, input_name = os.path.split(address_2)

        #print(f"[DEBUG]Photo.save_data:{address}")
        #print(f"[DEBUG]Photo.save_data:{root_name}")
        #print(f"[DEBUG]Photo.save_data:{input_name}")
        #print(f"[DEBUG]Photo.save_data:{file_name}")
        #print(f"[DEBUG]Photo.save_data:{ext_name}")

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
                        if isinstance(item, str):
                            item=ast.literal_eval(item)
                        data = json.dumps(item, ensure_ascii=False)
                        f.write(data + '\n')
            else:
                with open(output_jsonl_address, "w", encoding="utf-8") as f:
                    data = json.dumps(result, ensure_ascii=False)
                    f.write(data + '\n')

    async def _final_node(self, state: "AudioAgent.Graph") -> Dict[str, Any]:
        """Final answer node: output the audio understanding result."""
        audio_result = state.get("audio_result", {})

        self.save_data(state=state)

        if isinstance(audio_result, list):
            # If already in list format, use directly
            final_answer: Any = audio_result
        elif isinstance(audio_result, dict):
            # If dict, check whether it contains a Result field
            if "Result" in audio_result and isinstance(audio_result["Result"], list):
                final_answer = audio_result["Result"]
            else:
                # Single dict object, convert to list
                final_answer = [audio_result]
        elif audio_result:
            # Other types, try to convert to list
            final_answer = audio_result if isinstance(audio_result, list) else [audio_result]
        else:
            # Empty result, return empty list
            final_answer = []

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

    def _route_after_think(self, state: "AudioAgent.Graph") -> str:
        """Routing after thinking."""
        if state.get("current_iteration", 0) >= state.get(
            "max_iterations", self.max_iterations
        ):
            return "final"
        elif state.get("complete", False):
            return "final"
        else:
            return "action"

    def _route_after_action(self, state: "AudioAgent.Graph") -> str:
        """Routing after action."""
        tool_name: str = state.get("tools", "none")
        if tool_name in ("None", "none"):
            return "observation"
        elif tool_name == "audio":
            return "audio"
        else:
            return "observation"

    def _route_after_observation(self, state: "AudioAgent.Graph") -> str:
        """Routing after observation."""
        audio_result = state.get("audio_result", {})
        has_result = audio_result and (
            (isinstance(audio_result, dict) and audio_result) or
            (isinstance(audio_result, list) and len(audio_result) > 0)
        )

        current_iteration = state.get("current_iteration", 1)
        max_iterations = state.get("max_iterations", self.max_iterations)
        complete = state.get("complete", False)

        if complete:
            return "reflect"
        elif has_result:
            return "reflect"
        elif current_iteration >= max_iterations:
            return "reflect"
        else:
            return "think"

    def _route_after_reflect(self, state: "AudioAgent.Graph") -> str:
        """Routing after reflection."""
        reflection_result = state.get("reflection_result", {})
        quality = reflection_result.get("quality", "low")
        should_retry = reflection_result.get("should_retry", False)
        reflection_count = state.get("reflection_count", 0)

        if reflection_count >= 3:
            return "final"
        if quality == "high":
            return "final"
        if should_retry:
            return "think"
        return "final"

    # ===================== Public Interface =====================

    def invoke(self, user_input: str, audio_path: str) -> Dict[str, Any]:
        """Execute AudioAgent, returning the final result in one call."""
        initial_state: AudioAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "audio_message": [],
            "observation_message": [],
            "reflection_message": [],
            "audio_path": audio_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "audio_flag": False,
            "audio_result": {},
            # Reflection-related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }
        return self.graph.invoke(initial_state)

    def stream(self, user_input: str, audio_path: str):
        """Stream-execute AudioAgent, returning intermediate states step by step."""
        initial_state: AudioAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "audio_message": [],
            "observation_message": [],
            "reflection_message": [],
            "audio_path": audio_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "audio_flag": False,
            "audio_result": {},
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
        Synchronous wrapper for VideoAgent's generate_other_system_prompt_for_task,
        to avoid using asyncio.run directly in graph nodes.
        """
        system_prompt = SystemPrompt()
        prompt = system_prompt.build_prompt(task)

        try:
            response = self.model.invoke(prompt)
        except Exception as e:  # Error handling logic compatible with VideoAgent
            error_msg = str(e)
            if "Arrearage" in error_msg or "overdue-payment" in error_msg:
                raise Exception(
                    "Alibaba Cloud account overdue. Please visit https://help.aliyun.com/zh/model-studio/error-code#overdue-payment for details and top up."
                )
            else:
                raise e

        return system_prompt.parse_response_content(response)

    async def _generate_other_system_prompt_for_task_async(
        self, task: str
    ) -> tuple[SystemPromptHeader, SystemPromptBody]:
        """
        Async version: wrapper for VideoAgent's generate_other_system_prompt_for_task.
        """
        system_prompt = SystemPrompt()
        prompt = system_prompt.build_prompt(task)

        try:
            response = await self.model.ainvoke(prompt)
        except Exception as e:  # Error handling logic compatible with VideoAgent
            error_msg = str(e)
            if "Arrearage" in error_msg or "overdue-payment" in error_msg:
                raise Exception(
                    "Alibaba Cloud account overdue. Please visit https://help.aliyun.com/zh/model-studio/error-code#overdue-payment for details and top up."
                )
            else:
                raise e

        return system_prompt.parse_response_content(response)

    async def ainvoke(self, user_input: str, audio_path: str) -> Dict[str, Any]:
        """Asynchronously invoke AudioAgent, returning the final result in one call."""
        initial_state: AudioAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "audio_message": [],
            "observation_message": [],
            "reflection_message": [],
            "audio_path": audio_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "audio_flag": False,
            "audio_result": [],
            # Reflection-related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }
        return await self.graph.ainvoke(initial_state)


if __name__ == "__main__":
    import asyncio
    load_dotenv()
    agent = AudioAgent(max_iterations=5)
    demo_audio_path = r"D:\Python\Project\MCP\Agent\prompt_agent\sample\input\shanghai\shanghai.MP3"  # Example path; replace with actual audio file when using
    task = "调用工具帮我理解这个音频，并给出按时间片段划分的描述 JSON 列表"

    async def main():
        try:
            response = await agent.ainvoke(user_input=task, audio_path=demo_audio_path)
            audio_result = response.get("audio_result", {})

            # Ensure the return is in standard format: a list containing dict objects
            if isinstance(audio_result, list):
                final_answer: Any = audio_result
            elif isinstance(audio_result, dict):
                # If dict, check whether it contains a Result field
                if "Result" in audio_result and isinstance(audio_result["Result"], list):
                    final_answer = audio_result["Result"]
                else:
                    # Single dict object, convert to list
                    final_answer = [audio_result]
            elif audio_result:
                # Other types, try to convert to list
                final_answer = audio_result if isinstance(audio_result, list) else [audio_result]
            else:
                # Empty result, return empty list
                final_answer = []

            print(f"Final result: {final_answer}")

        except Exception as exc:
            print(f"Error: {exc}")

    asyncio.run(main())
