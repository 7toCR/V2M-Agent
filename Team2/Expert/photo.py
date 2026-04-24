"""
PhotoAgent module — uses a multimodal large language model to understand images
and output structured JSON containing background, background style, subject,
and subject mood.  Built on LangGraph with a ReAct-style state machine
(init → think → action → tool → observation → reflect → final).
"""

import os
import sys
import json
import base64
import re
from pathlib import Path
from typing import List, Literal, Dict, Any, Optional, Annotated

import asyncio
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # Two levels up from Team2/Expert/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from promptStrategy.system_prompt_header import SystemPromptHeader
from promptStrategy.system_prompt_body import SystemPromptBody
from promptStrategy.system_prompt_profile import SystemPrompt
from promptStrategy.JSONSchema import JSONSchema  # noqa: F401  # Keep dependency consistent with pop_idx.py
from Team2.Expert.prompt import (
    CONSTRAINTS,
    RESOURCES,
    BEST_PRACTICES,
    RUN_MODULE,
    Guide_Book_photo_expert,
    COMMAND_photo,
)
from tools.tools import (  # Reuse the same tool parsing logic as in pop
    extract_field_from_response,
    extract_result_from_tools,
    _print_with_indent,
)

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

# Import agent profile
try:
    from Team2.AgentProfile.photo_agent_profile import PHOTO_AGENT_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import PHOTO_AGENT_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    PHOTO_AGENT_PROFILE = None


def _encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class PhotoAgent:

    class Graph(TypedDict):
        # Global messages
        global_messages: Annotated[List[dict], add_messages]
        # System prompt messages
        system_prompt_messages: str
        # User messages
        user_messages: Annotated[List[dict], add_messages]
        # User task description
        tasks: str

        # Messages for each step
        think_message: Annotated[List[dict], add_messages]
        action_message: Annotated[List[dict], add_messages]
        photo_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]
        reflection_message: Annotated[List[dict], add_messages]  # Reflection messages

        # Input parameters
        photo_path: str  # Local path of the image to understand

        # Execution results
        photo_flag: bool
        photo_result: Dict[str, Any]

        # Running state
        state: Literal["start", "think", "action", "execute", "observation", "reflect", "final", "end"]
        # Execution state
        tools: Literal["None", "none", "photo"]

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

        config_path = os.path.join(os.path.dirname(__file__), "config_photo.json")
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
            expert_name = expert_config.get("name", "qwen3-vl-plus")
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

        self.agent_profile = PHOTO_AGENT_PROFILE  # Save agent profile
        self.max_iterations = max_iterations

        # Reflection memory module
        if REFLECTION_IMPORT_SUCCESS:
            self.reflection_memory = get_reflection_memory("photo", CURRENT_DIR)
        else:
            self.reflection_memory = None

        self.builder = StateGraph(PhotoAgent.Graph)

        # Node definitions, same structure as AudioAgent/TextAgent
        self.builder.add_node("init", self._init_node)
        self.builder.add_node("think", self._think_node)
        self.builder.add_node("action", self._action_node)

        # Tool node: photo
        self.builder.add_node("photo", self._photo_node)

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
                "photo": "photo",
                "observation": "observation",
            },
        )

        # Tool node returns to observation after execution
        self.builder.add_edge("photo", "observation")

        # Routing after observation
        self.builder.add_conditional_edges(
            "observation",
            self._route_after_observation,
            {
                "think": "think",
                "final": "final",
                "reflect": "reflect",
            },
        )

        # Routing after reflection
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

    def _build_system_prompt_from_profile(self, task: str) -> str:
        """Build a complete system_prompt from the agent_profile."""
        if not PROFILE_IMPORT_SUCCESS or self.agent_profile is None:
            return f"## My Task\n\"\"\"{task}\"\"\"\n\n## Conversation History\n"

        profile = self.agent_profile
        system_prompt = ""

        system_prompt += f"# {profile.role.name}\n\n"
        system_prompt += f"{profile.role.description}\n\n"

        if profile.role.responsibilities:
            system_prompt += "## Core Responsibilities\n\n"
            for resp in profile.role.responsibilities:
                system_prompt += f"- {resp}\n"
            system_prompt += "\n"


        if hasattr(profile.role, 'expertise') and profile.role.expertise:
            system_prompt += "## Areas of Expertise\n\n"
            for exp in profile.role.expertise:
                system_prompt += f"- {exp}\n"
            system_prompt += "\n"

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

        # 7. Best Practices
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

        # 9. Run Methods
        if profile.run_methods:
            system_prompt += "## Run Methods\n\n"
            for method in profile.run_methods:
                system_prompt += f"{method}\n"
            system_prompt += "\n"

        # 10. Commands
        if profile.command:
            system_prompt += f"## Available Commands\n\n{profile.command}\n\n"

        # 11. Guide Book
        if profile.guide_book:
            system_prompt += f"## Guide Book\n\n{profile.guide_book}\n\n"

        # 12. My Task
        system_prompt += f'''## My Task\n"""{task}"""\n\n'''

        # 13. Conversation History
        system_prompt += f"## Conversation History\n"

        return system_prompt

    # ===================== Node Implementations =====================

    async def _init_node(self, state: "Agent.Graph") -> Dict[str, Any]:
        """Initialization node — build system_prompt from agent_profile."""
        task = state.get("tasks", "")
        _print_with_indent("task:", str(task), tab_count=2)

        # Build system_prompt from profile using the new method
        system_prompt = self._build_system_prompt_from_profile(task)
        #print(f"PhotoAgent._init_node{system_prompt}")

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
            "photo_message": [],
            "observation_message": [],
            "reflection_message": [],
            "photo_flag": False,
            "photo_result": {},
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

    async def _think_node(self, state: "Agent.Graph") -> Dict[str, Any]:

        global last_state  # Same global variable usage as AudioAgent/TextAgent
        global complete

        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]

        current_iteration = state.get("current_iteration", 0)
        complete = state.get("complete", False)
        photo_result = state.get("photo_result", {})

        # Check if we have a valid result (non-empty dict or list)
        has_result = photo_result and (
            (isinstance(photo_result, dict) and photo_result) or
            (isinstance(photo_result, list) and len(photo_result) > 0)
        )

        if complete or current_iteration > self.max_iterations or has_result:
            return {
                "current_iteration": 1,
                "complete": True,
                "state": "final",
                "photo_result": photo_result,  # Preserve photo_result to avoid losing it
            }

        think_prompt = f"""用户:根据已知条件进行思考，分析当前情况并决定下一步意图。严格遵行《回答模板》格式回答,必须严格按照以下JSON格式回答：{{"Result": ["你的思考内容"]}}只输出JSON格式，不要输出其他内容。\n"""

        messages.append(HumanMessage(content=think_prompt))
        system_prompt += think_prompt

        response = await self.model.ainvoke(messages)

        think_list = extract_field_from_response(response, "Result")
        thought = think_list[0] if isinstance(think_list, list) and len(think_list) > 0 else ""
        if isinstance(thought, list):
            thought = ''.join(thought)
        elif not isinstance(thought, str):
            thought = str(thought) if thought else ""

        if state.get("photo_flag", False) and state.get("photo_result"):
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
            "photo_message": state.get("photo_message", []),
            "observation_message": state.get("observation_message", []),
            "photo_flag": state.get("photo_flag", False),
            "photo_result": state.get("photo_result", {}),
            "state": last_state,
            "tools": state.get("tools", "None"),
            "current_iteration": 1,
            "max_iterations": state.get("max_iterations", self.max_iterations),
            "complete": _complete,
            "final_answer": state.get("final_answer", ""),
        }

    async def _action_node(self, state: "Agent.Graph") -> Dict[str, Any]:
        """Action node: generate an Action based on the latest thought and decide whether to call the photo tool."""
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
- photo: 直接调用多模态模型，对指定本地图片进行理解并输出包含背景、背景风格、主体、主体心情的 JSON 字典
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
            "photo_message": state.get("photo_message", []),
            "observation_message": state.get("observation_message", []),
            "photo_flag": state.get("photo_flag", False),
            "photo_result": state.get("photo_result", {}),
            "state": last_state,
            "tools": tool,
            "current_iteration": 1,
            "max_iterations": state.get("max_iterations", self.max_iterations),
            "complete": state.get("complete", False),
            "final_answer": state.get("final_answer", ""),
        }

    async def _photo_node(self, state: "Agent.Graph") -> Dict[str, Any]:
        """
        Main purpose of this function: call the multimodal large language model to understand an image
        and return the understanding content in JSON format.
        Returns:
            A dictionary containing the following fields:
            {
                "背景": "",
                "背景风格": "",
                "主体": "",
                "主体心情": "",
            }
        """
        # Do not use the accumulated history prompt; use a fresh, concise prompt instead.
        # This avoids overly long and confusing prompts and ensures the model correctly understands the task.
        photo_path = state.get("photo_path", "").strip()
        if not photo_path:
            raise ValueError("photo_path cannot be empty")

        if not os.path.exists(photo_path):
            raise FileNotFoundError(f"Image file not found: {photo_path}")

        image_ext = os.path.splitext(photo_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        image_mime_type = mime_types.get(image_ext, "image/jpeg")

        try:
            base64_image = _encode_image_to_base64(photo_path)
        except Exception as e:
            raise Exception(f"Error encoding image file: {str(e)}")

        image_data_url = f"data:{image_mime_type};base64,{base64_image}"

        # Use a fresh, concise system_prompt, following the structure of audio.py
        system_prompt = """你是一个专业的图片理解助手。请基于图片中的真实内容，准确、详细地描述背景、背景风格���主体与主体心情。

【核心要求 - 必须严格遵守】：
1. **完整分析**：必须全面分析图片中的所有重要元素，包括背景环境、主体对象和情绪状态。
2. **详细描述**：每个字段的描述必须详细、具体、完整，不能过于简单或空泛。
3. **准确识别**：基于图片实际内容进行描述，不要编造不存在的内容。

输出要求：仅返回 JSON 对象，包含以下字段：
{
    "背景": "...",
    "背景风格": "...",
    "主体": "...",
    "主体心情": "..."
}
只返回 JSON,不要附加其他文本。

示例:
{
    "背景": "户外，一个有白色金属栏杆的平台或桥上，远处是模糊的水面和城市建筑。",
    "背景风格": "都市休闲风",
    "主体": "一位留着棕色长发、化着精致妆容的年轻女性。她穿着一件黑白底色带彩色花卉图案的无袖连衣裙，搭配黑色高跟凉鞋，斜挎着一个黑色小包。",
    "主体心情": "开心、愉悦"
}

不要附加任何额外说明。"""

        user_prompt = "请基于这张图片提取背景、背景���格、主体与主体心情，以 JSON 格式返回。确保所有字段都有详细、具体的描述。"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
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
            # If the returned value is a list, take the first element
            if isinstance(result, list):
                if not result:
                    raise Exception("The list returned by the model is empty")
                result = result[0]
            # Ensure the returned value is in dict format
            if not isinstance(result, dict):
                result = {"主体": str(result)}
        except json.JSONDecodeError:
            # If direct parsing fails, try using extract_field_from_response
            result = extract_field_from_response(response, "Result")
            if not result:
                # If still failing, try to extract JSON from the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise Exception(f"Unable to parse response as JSON: {response_text[:500]}")

        print(f"[DEBUG]result:{result}\n")

        # Ensure all required fields are present
        required_keys = ["背景", "背景风格", "主体", "主体心情"]
        for key in required_keys:
            if key not in result or result[key] is None:
                result[key] = ""

        # Safely convert result to string for content
        if isinstance(result, dict):
            result_str = json.dumps(result, ensure_ascii=False)
        else:
            result_str = str(result) if result else ''

        content = f"photo_result:{result_str}"
        photo_message = [{"role": "assistant", "content": content}]
        global_message = [AIMessage(content=content)]

        # Update system_prompt_messages in state (preserve history but don't pollute the new prompt)
        updated_system_prompt = state.get("system_prompt_messages", "") + "\n" + content

        return {
            "global_messages": global_message,
            "system_prompt_messages": updated_system_prompt,
            "photo_message": photo_message,
            "photo_flag": True,
            "photo_result": result,
            "state": "observation",
            "tools": "None",
            "current_iteration": 1,
        }

    async def _observation_node(self, state: "Agent.Graph") -> Dict[str, Any]:
        """Observation node: process tool return results for the next round of thinking."""
        observation_content = ""
        current_iteration = state.get("current_iteration", 0)
        _complete = state.get("complete", False)
        if current_iteration > state.get("max_iterations", self.max_iterations):
            _complete = True

        if state.get("photo_flag", False):
            photo_result = state.get("photo_result", {})
            if photo_result:
                # Handle both dict and list formats
                if isinstance(photo_result, (dict, list)):
                    result_str = json.dumps(photo_result, ensure_ascii=False)
                else:
                    result_str = str(photo_result)
                observation_content = f"photo is: {result_str}"
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
            "photo_flag": False,
            "photo_result": state.get("photo_result", {}),  # Preserve photo_result to avoid losing it
            "state": "think",
            "current_iteration": 1,
            "complete": _complete,
            # Reflection-related fields
            "reflection_flag": state.get("reflection_flag", False),
            "reflection_result": state.get("reflection_result", {}),
            "reflection_count": state.get("reflection_count", 0),
        }

    async def _reflect_node(self, state: "PhotoAgent.Graph") -> Dict[str, Any]:
        """
        Reflection node (Reflexion mechanism): generate reflections based on observation results and evaluate quality.
        """
        _print_with_indent("", "Reflection node started...", tab_count=1)

        task_description = state.get("tasks", "")
        observation = ""

        if state.get("photo_flag", False):
            photo_result = state.get("photo_result", {})
            if photo_result:
                if isinstance(photo_result, (dict, list)):
                    result_str = json.dumps(photo_result, ensure_ascii=False)
                else:
                    result_str = str(photo_result)
                observation = f"Image understanding result: {result_str}"
        else:
            observation = "No valid result obtained"

        history_reflections = ""
        if self.reflection_memory:
            history_reflections = self.reflection_memory.get_summary()

        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="photo",
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
            _print_with_indent("", f"Reflection quality high / limit reached, proceeding to final answer node", tab_count=2)
        else:
            next_state = "think"
            _complete = False
            _print_with_indent("", f"Reflection quality low, entering retry thinking phase", tab_count=2)

        return {
            "global_messages": [AIMessage(content=reflection_content)],
            "system_prompt_messages": system_prompt,
            "reflection_message": [{"role": "assistant", "content": reflection_content}],
            "photo_flag": state.get("photo_flag", False),
            "photo_result": state.get("photo_result", {}),
            "state": next_state,
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": True,
            "reflection_result": reflection_result_dict,
            "reflection_count": reflection_count,
        }

    #address_x = r"D:\Python\Project\MCP\Agent\prompt_agent\sample\input\shanghai\xx.jpg"
    #output_dir = r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output"
    #address=r"D:\Python\Project\MCP\Agent\prompt_agent\understanding\girl.jpg"

    #root_name=D:\Python\Project\MCP\Agent\prompt_agent\sample
    #input_name=input
    #file_name=shanghai
    #ext_name=xx.jpg

    def save_data(self,state: "Agent.Graph") -> None:
        address=state.get("photo_path", "")
        jsonl_name="photo.jsonl"
        result=state.get("photo_result", [])

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





    async def _final_node(self, state: "Agent.Graph") -> Dict[str, Any]:
        """Final answer node: output the image understanding result."""
        photo_result = state.get("photo_result", {})

        self.save_data(state=state)

        if isinstance(photo_result, dict):
            # Already in dict format, use directly
            final_answer: Any = photo_result
        elif isinstance(photo_result, list):
            # If it's a list, check if it contains a Result field
            if len(photo_result) > 0 and isinstance(photo_result[0], dict) and "Result" in photo_result[0]:
                final_answer = photo_result[0]["Result"]
            else:
                final_answer = photo_result[0] if len(photo_result) > 0 else {}
        elif photo_result:
            # Other types, try to convert to dict
            final_answer = photo_result if isinstance(photo_result, dict) else {"主体": str(photo_result)}
        else:
            # Empty result, return empty dict
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

    def _route_after_think(self, state: "PhotoAgent.Graph") -> str:
        """Routing after thinking."""
        if state.get("current_iteration", 0) >= state.get(
            "max_iterations", self.max_iterations
        ):
            return "final"
        elif state.get("complete", False):
            return "final"
        else:
            return "action"

    def _route_after_action(self, state: "PhotoAgent.Graph") -> str:
        """Routing after action."""
        tool_name: str = state.get("tools", "none")
        if tool_name in ("None", "none"):
            return "observation"
        elif tool_name == "photo":
            return "photo"
        else:
            return "observation"

    def _route_after_observation(self, state: "PhotoAgent.Graph") -> str:
        """Routing after observation."""
        photo_result = state.get("photo_result", {})
        has_result = photo_result and (
            (isinstance(photo_result, dict) and photo_result) or
            (isinstance(photo_result, list) and len(photo_result) > 0)
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

    def _route_after_reflect(self, state: "PhotoAgent.Graph") -> str:
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

    async def ainvoke(self, user_input: str, photo_path: str) -> Dict[str, Any]:
        """Asynchronously invoke PhotoAgent and return the final result at once."""
        initial_state: PhotoAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "photo_message": [],
            "observation_message": [],
            "reflection_message": [],
            "photo_path": photo_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "photo_flag": False,
            "photo_result": {},
            # Reflection-related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }
        return await self.graph.ainvoke(initial_state)

    def invoke(self, user_input: str, photo_path: str) -> Dict[str, Any]:
        """Synchronously invoke PhotoAgent, backward compatible."""
        return asyncio.run(self.ainvoke(user_input, photo_path))

    def stream(self, user_input: str, photo_path: str):
        """Stream-execute PhotoAgent, returning intermediate states step by step."""
        initial_state: PhotoAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "photo_message": [],
            "observation_message": [],
            "reflection_message": [],
            "photo_path": photo_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "photo_flag": False,
            "photo_result": {},
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
        Synchronous wrapper for generate_other_system_prompt_for_task from AudioAgent/TextAgent,
        to avoid using asyncio.run directly inside graph nodes.
        """
        import asyncio

        system_prompt = SystemPrompt()
        prompt = system_prompt.build_prompt(task)

        try:
            # Use asyncio.run to execute async code
            response = asyncio.run(self.model.ainvoke(prompt))
        except Exception as e:  # Error handling logic compatible with AudioAgent
            error_msg = str(e)
            if "Arrearage" in error_msg or "overdue-payment" in error_msg:
                raise Exception(
                    "Alibaba Cloud account has overdue payment. Please visit https://help.aliyun.com/zh/model-studio/error-code#overdue-payment for details and top up."
                )
            else:
                raise e

        return system_prompt.parse_response_content(response)

    async def _generate_other_system_prompt_for_task_async(
        self, task: str
    ) -> tuple[SystemPromptHeader, SystemPromptBody]:
        """
        Asynchronous wrapper for generate_other_system_prompt_for_task from AudioAgent/TextAgent,
        to avoid using asyncio.run directly inside graph nodes.
        """
        system_prompt = SystemPrompt()
        prompt = system_prompt.build_prompt(task)

        try:
            response = await self.model.ainvoke(prompt)
        except Exception as e:  # Error handling logic compatible with AudioAgent
            error_msg = str(e)
            if "Arrearage" in error_msg or "overdue-payment" in error_msg:
                raise Exception(
                    "Alibaba Cloud account has overdue payment. Please visit https://help.aliyun.com/zh/model-studio/error-code#overdue-payment for details and top up."
                )
            else:
                raise e

        return system_prompt.parse_response_content(response)


if __name__ == "__main__":
    import asyncio
    load_dotenv()
    agent = PhotoAgent(max_iterations=5)
    demo_photo_path = r"D:\Python\Project\MCP\Agent\prompt_agent\sample\input\shanghai\girl.jpg"
    task = "调用工具帮我理解这个图片，并给出包含背景、背景风格、主体、主体风格的 JSON 字典"

    async def main():
        try:
            response = await agent.ainvoke(user_input=task, photo_path=demo_photo_path)
            photo_result = response.get("photo_result", {})

            # Return the entire dict directly
            if isinstance(photo_result, dict):
                final_answer: Any = photo_result
            elif photo_result:
                final_answer = photo_result
            else:
                final_answer = {}
            print(f"Final result: {final_answer}")

        except Exception as exc:
            print(f"Error: {exc}")

    asyncio.run(main())
