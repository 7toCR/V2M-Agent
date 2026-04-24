"""
Video understanding agent module.

This module implements a VideoAgent that uses a multimodal large language model
to analyze video content and extract keyframe-level scene descriptions in a
structured JSON format.
"""

import os
import sys
import json
import ast
import base64
import re
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from promptStrategy.system_prompt_header import SystemPromptHeader
from promptStrategy.system_prompt_body import SystemPromptBody
from promptStrategy.system_prompt_profile import SystemPrompt
from promptStrategy.JSONSchema import JSONSchema  # noqa: F401  # Keep consistent dependency with pop_idx.py
from Team2.Expert.prompt import CONSTRAINTS, RESOURCES, BEST_PRACTICES, RUN_MODULE, Guide_Book_video_expert, COMMAND_video
from tools.tools import (  # Reuse the same tool parsing logic as in pop
    extract_field_from_response,
    extract_result_from_tools,
    _print_with_indent,
)

# Import agent profile
try:
    from Team2.AgentProfile.video_agent_profile import VIDEO_AGENT_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import VIDEO_AGENT_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    VIDEO_AGENT_PROFILE = None

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


def _encode_video_to_base64(video_path: str) -> str:
    """Encode a video file to a base64 string."""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


class VideoAgent:

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
        video_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]
        reflection_message: Annotated[List[dict], add_messages]  # Reflection messages

        video_path: str

        # Execution results
        video_flag: bool
        video_result: List[Dict[str, Any]]

        # Running state
        state: Literal["start", "think", "action", "execute", "observation", "reflect", "final", "end"]
        # Execution state
        tools: Literal["None", "none", "video"]

        current_iteration: Annotated[int, add]
        max_iterations: int

        # Reflection related fields
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

        config_path = os.path.join(os.path.dirname(__file__), "config_video.json")
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
            api_key_env = expert_config.get("api_key_env", "MCP_API_KEY")
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

        self.agent_profile = VIDEO_AGENT_PROFILE  # Save agent profile
        self.max_iterations = max_iterations

        # Reflection memory module
        if REFLECTION_IMPORT_SUCCESS:
            self.reflection_memory = get_reflection_memory("video", CURRENT_DIR)
        else:
            self.reflection_memory = None

        self.builder = StateGraph(VideoAgent.Graph)

        # Node definitions, following the same structure as POPIdxAgent
        self.builder.add_node("init", self._init_node)
        self.builder.add_node("think", self._think_node)
        self.builder.add_node("action", self._action_node)

        # Tool node: video
        self.builder.add_node("video", self._video_node)

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
                "video": "video",
                "observation": "observation",
            },
        )

        # Tool node returns to observation after execution
        self.builder.add_edge("video", "observation")

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

    # ===================== Helper Methods =====================

    def _build_system_prompt_from_profile(self, task: str) -> str:
        """Build a complete system_prompt from the agent_profile."""
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

        # 12. My Task
        system_prompt += f'''## My Task\n"""{task}"""\n\n'''

        # 13. Conversation History
        system_prompt += f"## Conversation History\n"

        return system_prompt

    # ===================== Node Implementations =====================

    async def _init_node(self, state: "Agent.Graph") -> Dict[str, Any]:
        """Initialization node: build system_prompt from agent_profile."""
        task = state.get("tasks", "")
        _print_with_indent("task:", str(task), tab_count=2)

        # Build system_prompt from profile using the new method
        system_prompt = self._build_system_prompt_from_profile(task)
        #print(f"Video._init_node{system_prompt}")

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
            "video_message": [],
            "observation_message": [],
            "reflection_message": [],
            "video_flag": False,
            "video_result": [],
            "state": "think",
            "tools": state.get("tools", "None"),
            "current_iteration": 1,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            # Reflection related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }

    async def _think_node(self, state: "Agent.Graph") -> Dict[str, Any]:

        global last_state  # Same global variable usage as POPIdxAgent
        global complete

        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]

        current_iteration = state.get("current_iteration", 0)
        complete = state.get("complete", False)
        video_result = state.get("video_result", [])

        # Check if we have a valid result (non-empty list or dict)
        has_result = video_result and (
            (isinstance(video_result, list) and len(video_result) > 0) or
            (isinstance(video_result, dict) and video_result)
        )

        if complete or current_iteration > self.max_iterations or has_result:
            return {
                "current_iteration": 1,
                "complete": True,
                "state": "final",
                "video_result": video_result,  # Preserve video_result to avoid loss
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

        if state.get("video_flag", False) and state.get("video_result"):
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
            "video_message": state.get("video_message", []),
            "observation_message": state.get("observation_message", []),
            "video_flag": state.get("video_flag", False),
            "video_result": state.get("video_result", []),
            "state": last_state,
            "tools": state.get("tools", "None"),
            "current_iteration": 1,
            "max_iterations": state.get("max_iterations", self.max_iterations),
            "complete": _complete,
            "final_answer": state.get("final_answer", ""),
        }

    async def _action_node(self, state: "Agent.Graph") -> Dict[str, Any]:
        """Action node: generate an Action based on the latest thought and decide whether to call the video tool."""
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
- video: 直接调用多模态模型，对指定本地视频进行理解并输出关键帧 JSON
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
            "video_message": state.get("video_message", []),
            "observation_message": state.get("observation_message", []),
            "video_flag": state.get("video_flag", False),
            "video_result": state.get("video_result", []),
            "state": last_state,
            "tools": tool,
            "current_iteration": 1,
            "max_iterations": state.get("max_iterations", self.max_iterations),
            "complete": state.get("complete", False),
            "final_answer": state.get("final_answer", ""),
        }

    async def _video_node(self, state: "Agent.Graph") -> Dict[str, Any]:
        """
        Main purpose of this function: call a multimodal large language model to understand
        the video, and ultimately return the understanding content in JSON format.
        Returns:
            A list containing keyframe analysis results, where each element is a dictionary:
            [
                {
                    "关键帧": "1s",
                    "背景": "",
                    "背景风格": "",
                    "主体": "",
                    "主体心情": "",
                },
                {
                    "关键帧": "3s",
                    "背景": "",
                    "背景风格": "",
                    "主体": "",
                    "主体心情": "",
                },
                ...
            ]
        """
        # Do not use accumulated history prompt; use a completely fresh, concise prompt.
        # This avoids overly long and confusing prompts and ensures the model correctly understands the task.
        video_path = state.get("video_path", "").strip()
        if not video_path:
            raise ValueError("video_path cannot be empty")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_ext = os.path.splitext(video_path)[1].lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }
        video_mime_type = mime_types.get(video_ext, "video/mp4")
        base64_video = _encode_video_to_base64(video_path)
        video_data_url = f"data:{video_mime_type};base64,{base64_video}"

        # Use a completely fresh, concise system_prompt, referencing the structure of audio.py.
        # Emphasize full coverage of the entire video, identifying all key scene changes.
        system_prompt = """你是一个专业的视频理解助手。请仔细观看给定的视频，分析视频内容并输出按时间顺序的关键帧/场景列表。

【核心要求 - 必须严格遵守】：
1. **完整覆盖**：必须完整覆盖整个视频，识别所有重要的场景变化和关键帧，不能遗漏重要内容。
2. **合理划分**：根据视频内容的变化（如场景切换、主体动作变化、情绪或风格变化）合理划分关键帧，每个关键帧通常对应一个重要的场景或时刻。
3. **详细描述**：每个字段的描述必须详细、具体、完整，不能过于简单或空泛。
4. **时间顺序**：关键帧必须按时间顺序排列，从视频开始到结束。

对于每个关键帧，你需要详细分析并填写以下信息：
- 关键帧：该场景的时间点或场景编号（如"0s"、"10s"、"场景1"等）
- 背景：详细描述场景的背景环境
- 背景风格：描述背景的视觉风格和氛围
- 主体：描述画面中的主要对象或人物
- 主体心情：分析主体的情绪状态

输出要求：仅返回 JSON 数组，每个元素都包含以下字段：
{
    "关键帧": "",
    "背景": "...",
    "背景风格": "...",
    "主体": "...",
    "主体心情": "..."
}
只返回 JSON,不要附加其他文本。

示例:
[
    {
        "关键帧": "0s",
        "背景": "教室前方投影着大幅中国国旗，作为整个活动的核心视觉元素。",
        "背景风格": "庄重、肃穆，具有强烈的仪式感和爱国主义教育氛围。",
        "主体": "一群年轻人整齐地坐在教室内，右手掌心向内轻放在左胸心脏位置。",
        "主体心情": "专注、严肃，沉浸在活动的氛围中。"
    },
    {
        "关键帧": "10s",
        "背景": "国旗投影保持清晰明亮，是全场焦点。",
        "背景风格": "集体主义，秩序井然，凸显活动的组织性。",
        "主体": "学生们低头看着手中的手机屏幕，嘴唇同步开合，跟随内容进行朗读或歌唱。",
        "主体心情": "认真、投入，努力保持节奏一致。"
    },
    {
        "关键帧": "45s",
        "背景": "国旗背景依旧，可能伴随着屏幕上内容的切换或歌词的推进。",
        "背景风格": "情感铺垫，为即将到来的高潮营造期待感。",
        "主体": "学生们依然保持着跟唱或跟读的状态，但神情更加专注，身体微微紧绷，仿佛在准备某个重要动作。",
        "主体心情": "期待、蓄势待发，情绪逐渐升温。"
    },
    {
        "关键帧": "51s",
        "背景": "中国国旗的投影成为这一标志性动作的宏伟背景。",
        "背景风格": "戏剧性高潮，充满力量和象征意义��",
        "主体": "所有学生极其同步地将原本放在胸前的右手高高举过头顶，手臂伸直，动作坚定有力。",
        "主体心情": "激昂、坚定、自豪，情感得到充分释放。"
    },
    {
        "关键帧": "55s",
        "背景": "国旗投影下，手臂森林成为新的画面中心。",
        "背景风格": "定格瞬间，充满张力和感染力，象征着誓言的高潮或情感的顶峰。",
        "主体": "学生们保持举手姿势，表情庄严肃穆，口型可能对应着最有力的词句。",
        "主体心情": "坚定不屈、充满信念，集体荣誉感和爱国情绪达到顶峰。"
    },
    {
        "关键帧": "60s (结尾)",
        "背景": "国旗始终作为精神的象征矗立在背景中。",
        "背景风格": "仪式收尾，余韵悠长，强调活动的完整性。",
        "主体": "手势可能缓缓落下，但身体依旧挺拔，目光专注，活动接近尾声。",
        "主体心情": "满足、庄严，内心充满参与集体仪式后的共鸣与感动。"
    }
]

不要附加任何额外说明。"""

        user_prompt = "请仔细观看视频，识别所有关键场景变化，并输出 JSON 数组。每个数组元素必须包含关键帧、背景、背景风格、主体、主体心情的详细描述。必须完整覆盖整个视频，确保所有重要场景都被识别。"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "video_url", "video_url": {"url": video_data_url}},
                    {"type": "text", "text": user_prompt},
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
            # If the result is a dict containing a Result field, extract Result
            if isinstance(result, dict) and "Result" in result:
                result = result["Result"]
            # Ensure the result is in list format
            if isinstance(result, dict):
                result = [result]
            elif not isinstance(result, list):
                result = []
        except json.JSONDecodeError:
            # If direct parsing fails, try using extract_field_from_response
            result = extract_field_from_response(response, "Result")
            if not result:
                # If that also fails, try to extract JSON from the response
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Last resort: try using ast.literal_eval
                    try:
                        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                        if json_match:
                            result = ast.literal_eval(json_match.group())
                        else:
                            raise Exception(f"Unable to parse response as JSON: {response_text[:500]}")
                    except (ValueError, SyntaxError):
                        raise Exception(f"Unable to parse response as JSON: {response_text[:500]}")

        print(f"[DEBUG]result:{result}\n")

        # Normalize results to ensure all required fields are present
        normalized_results: List[Dict[str, Any]] = []
        required_keys = ["关键帧", "背景", "背景风格", "主体", "主体心情"]

        if isinstance(result, list):
            for idx, item in enumerate(result):
                if not isinstance(item, dict):
                    item = {"描述": str(item)}
                entry: Dict[str, Any] = {
                    "关键帧": str(
                        item.get("关键帧")
                        or item.get("时间")
                        or item.get("time")
                        or f"场景{idx + 1}"
                    ),
                    "背景": item.get("背景", item.get("background", "")),
                    "背景风格": item.get(
                        "背景风格", item.get("背景氛围", item.get("style", ""))
                    ),
                    "主体": item.get("主体", item.get("object", "")),
                    "主体心情": item.get("主体心情", item.get("emotion", "")),
                }
                for key in required_keys:
                    if not entry.get(key):
                        entry[key] = ""
                normalized_results.append(entry)
        elif isinstance(result, dict):
            # If it's a single dictionary, convert to a list
            entry: Dict[str, Any] = {
                "关键帧": str(
                    result.get("关键帧")
                    or result.get("时间")
                    or result.get("time")
                    or "场景1"
                ),
                "背景": result.get("背景", result.get("background", "")),
                "背景风格": result.get(
                    "背景风格", result.get("背景氛围", result.get("style", ""))
                ),
                "主体": result.get("主体", result.get("object", "")),
                "主体心情": result.get("主体心情", result.get("emotion", "")),
            }
            for key in required_keys:
                if not entry.get(key):
                    entry[key] = ""
            normalized_results.append(entry)
        else:
            raise Exception(f"Unable to parse response as valid video understanding result: {response_text[:500]}")

        # Safely convert result to string for content
        if isinstance(normalized_results, list):
            result_str = json.dumps(normalized_results, ensure_ascii=False)
        else:
            result_str = str(normalized_results) if normalized_results else ''

        content = f"video_result:{result_str}"
        video_message = [{"role": "assistant", "content": content}]
        global_message = [AIMessage(content=content)]

        # Update system_prompt_messages in state (preserve history, but do not pollute new prompts)
        updated_system_prompt = state.get("system_prompt_messages", "") + "\n" + content

        return {
            "global_messages": global_message,
            "system_prompt_messages": updated_system_prompt,
            "video_message": video_message,
            "video_flag": True,
            "video_result": normalized_results,
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

        if state.get("video_flag", False):
            video_result = state.get("video_result", [])
            if video_result:
                # Handle both dict and list formats
                if isinstance(video_result, (dict, list)):
                    result_str = json.dumps(video_result, ensure_ascii=False)
                else:
                    result_str = str(video_result)
                observation_content = f"video is: {result_str}"
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
            "video_flag": False,
            "video_result": state.get("video_result", []),
            "state": "think",
            "current_iteration": 1,
            "complete": _complete,
            # Reflection related fields
            "reflection_flag": state.get("reflection_flag", False),
            "reflection_result": state.get("reflection_result", {}),
            "reflection_count": state.get("reflection_count", 0),
        }

    async def _reflect_node(self, state: "VideoAgent.Graph") -> Dict[str, Any]:
        """
        Reflection node (Reflexion mechanism): generate reflections based on observation
        results and evaluate quality.
        """
        _print_with_indent("", "Reflection node started...", tab_count=1)

        task_description = state.get("tasks", "")
        observation = ""

        if state.get("video_flag", False):
            video_result = state.get("video_result", [])
            if video_result:
                if isinstance(video_result, (dict, list)):
                    result_str = json.dumps(video_result, ensure_ascii=False)
                else:
                    result_str = str(video_result)
                observation = f"视频理解结果: {result_str}"
        else:
            observation = "未获取到有效结果"

        history_reflections = ""
        if self.reflection_memory:
            history_reflections = self.reflection_memory.get_summary()

        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="video",
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
        reflection_content += f"Suggested retry: {reflection_result.get('should_retry', False)}"

        _print_with_indent("", f"Reflection result:", tab_count=2)
        _print_with_indent("", f"  Quality: {reflection_result.get('quality', 'low')}", tab_count=3)
        _print_with_indent("", f"  Suggested retry: {reflection_result.get('should_retry', False)}", tab_count=3)

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
            "video_flag": state.get("video_flag", False),
            "video_result": state.get("video_result", []),
            "state": next_state,
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": True,
            "reflection_result": reflection_result_dict,
            "reflection_count": reflection_count,
        }

    def save_data(self,state: "Agent.Graph") -> None:
        address=state.get("video_path", "")
        jsonl_name="video.jsonl"
        result = state.get("video_result", [])

        print(f"[DEBUG]save_data.video_result:{type(result)}")

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
                        if isinstance(item, str):
                            item=ast.literal_eval(item)
                        data = json.dumps(item, ensure_ascii=False)
                        f.write(data + '\n')
            else:
                with open(output_jsonl_address, "w", encoding="utf-8") as f:
                    data = json.dumps(result, ensure_ascii=False)
                    f.write(data + '\n')

    async def _final_node(self, state: "Agent.Graph") -> Dict[str, Any]:
        """Final answer node: output the video understanding result."""
        video_result = state.get("video_result", [])

        self.save_data(state=state)

        if isinstance(video_result, list):
            # Already in list format, use directly
            final_answer: Any = video_result
        elif isinstance(video_result, dict):
            # If it's a dict, check if it contains a Result field
            if "Result" in video_result and isinstance(video_result["Result"], list):
                final_answer = video_result["Result"]
            else:
                # Single dict object, convert to list
                final_answer = [video_result]
        elif video_result:
            # Other types, try to convert to list
            final_answer = video_result if isinstance(video_result, list) else [video_result]
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

    def _route_after_think(self, state: "VideoAgent.Graph") -> str:
        """Routing after thinking."""
        if state.get("current_iteration", 0) >= state.get(
            "max_iterations", self.max_iterations
        ):
            return "final"
        elif state.get("complete", False):
            return "final"
        else:
            return "action"

    def _route_after_action(self, state: "VideoAgent.Graph") -> str:
        """Routing after action."""
        tool_name: str = state.get("tools", "none")
        if tool_name in ("None", "none"):
            return "observation"
        elif tool_name == "video":
            return "video"
        else:
            return "observation"

    def _route_after_observation(self, state: "VideoAgent.Graph") -> str:
        """Routing after observation."""
        video_result = state.get("video_result", [])
        has_result = video_result and (
            (isinstance(video_result, list) and len(video_result) > 0) or
            (isinstance(video_result, dict) and video_result)
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

    def _route_after_reflect(self, state: "VideoAgent.Graph") -> str:
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

    async def ainvoke(self, user_input: str, video_path: str) -> Dict[str, Any]:
        """Asynchronously execute VideoAgent, returning the final result at once."""
        import asyncio

        initial_state: VideoAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "video_message": [],
            "observation_message": [],
            "reflection_message": [],
            "video_path": video_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "video_flag": False,
            "video_result": [],
            # Reflection related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }
        return await self.graph.ainvoke(initial_state)

    def invoke(self, user_input: str, video_path: str) -> Dict[str, Any]:
        """Synchronously execute VideoAgent, returning the final result at once."""
        import asyncio

        initial_state: VideoAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "video_message": [],
            "observation_message": [],
            "reflection_message": [],
            "video_path": video_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "video_flag": False,
            "video_result": [],
            # Reflection related fields
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }
        return asyncio.run(self.graph.ainvoke(initial_state))

    def stream(self, user_input: str, video_path: str):
        """Stream-execute VideoAgent, returning intermediate states step by step."""
        initial_state: VideoAgent.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "video_message": [],
            "observation_message": [],
            "reflection_message": [],
            "video_path": video_path,
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "video_flag": False,
            "video_result": [],
            # Reflection related fields
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
        Synchronous wrapper for generate_other_system_prompt_for_task from POPIdxAgent,
        to avoid using asyncio.run directly in graph nodes.
        """
        import asyncio

        system_prompt = SystemPrompt()
        prompt = system_prompt.build_prompt(task)

        try:
            # Use asyncio.run to execute the async code
            response = asyncio.run(self.model.ainvoke(prompt))
        except Exception as e:  # Error handling logic compatible with POPIdxAgent
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
        Asynchronous wrapper for generate_other_system_prompt_for_task from POPIdxAgent,
        to avoid using asyncio.run directly in graph nodes.
        """
        system_prompt = SystemPrompt()
        prompt = system_prompt.build_prompt(task)

        try:
            response = await self.model.ainvoke(prompt)
        except Exception as e:  # Error handling logic compatible with POPIdxAgent
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
    agent = VideoAgent(max_iterations=5)
    demo_video_path = r"D:\Python\Project\MCP\Agent\prompt_agent\sample\input\shanghai\shanghai.mp4"
    task = "调用工具帮我理解这个视频，并给出包含背景、背景风格、主体、主体风格的 JSON 字典"

    async def main():
        try:
            response = await agent.ainvoke(user_input=task, video_path=demo_video_path)
            video_result = response.get("video_result", {})

            if isinstance(video_result, dict):
                final_answer: Any = video_result
            elif video_result:
                final_answer = video_result
            else:
                final_answer = {}
            print(f"Final result: {final_answer}")

        except Exception as exc:
            print(f"Error: {exc}")

    asyncio.run(main())
