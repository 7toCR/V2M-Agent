"""
Stylist Agent — Adapted from pop_audio_type.py.

Selects the most suitable music audio type from 12 candidates based on
scene descriptions and Musical Blueprint. Returns 'N/A' for NotaGen mode.

Follows the Think-Act-Observe-Reflect iterative loop pattern from Team2.
"""

import os
import sys
import json
import logging
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

from Team3.Expert.prompt import (
    CONSTRAINTS, RESOURCES, BEST_PRACTICES, RUN_MODULE,
    COMMAND_stylist, Guide_Book_stylist,
)
from tools.tools import (
    extract_field_from_response,
    _print_with_indent,
)

# Import agent profile
try:
    from Team3.AgentProfile.stylist_agent_profile import STYLIST_AGENT_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import STYLIST_AGENT_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    STYLIST_AGENT_PROFILE = None

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

logger = logging.getLogger(__name__)

# The 12 valid audio type candidates
VALID_AUDIO_TYPES = [
    "Pop", "R&B", "Dance", "Jazz", "Folk", "Rock",
    "Chinese Style", "Chinese Tradition", "Metal", "Reggae",
    "Chinese Opera", "Auto",
]


def _load_songgeneration_vocab() -> Dict[str, Any]:
    """Load vocabulary from songgeneration.json."""
    for candidate in [
        os.path.join(CURRENT_DIR, "..", "pre-traing", "songgeneration.json"),
        os.path.join(CURRENT_DIR, "..", "AgentProfile", "pre-traing", "songgeneration.json"),
    ]:
        path = os.path.normpath(candidate)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


class StylistAgent:
    """Music Style Selection Expert.

    Selects auto_prompt_audio_type from 12 candidate styles:
    Pop, R&B, Dance, Jazz, Folk, Rock, Chinese Style, Chinese Tradition,
    Metal, Reggae, Chinese Opera, Auto.

    For NotaGen mode, returns 'N/A' for all pieces.
    """

    class Graph(TypedDict):
        # Global messages
        global_messages: Annotated[List[dict], add_messages]
        system_prompt_messages: str
        user_messages: Annotated[List[dict], add_messages]
        tasks: str

        # Per-step messages
        think_message: Annotated[List[dict], add_messages]
        action_message: Annotated[List[dict], add_messages]
        stylist_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]
        reflection_message: Annotated[List[dict], add_messages]

        # Input parameters
        json_scene: List[dict]
        piece: int
        blueprint: Dict[str, Any]

        # Execution results
        stylist_flag: bool
        stylist_audio_type_result: List[str]

        # Running state
        state: Literal["start", "think", "action", "execute", "observation", "reflect", "final", "end"]
        tools: Literal["None", "none", "select_audio_type"]

        current_iteration: Annotated[int, add]
        max_iterations: int

        # Reflection fields
        reflection_flag: bool
        reflection_result: Dict[str, Any]
        reflection_count: int

        complete: bool
        final_answer: Any

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        exper: Optional[ChatOpenAI] = None,
        temperature: float = 0.7,
        max_iterations: int = 15,
    ) -> None:
        load_dotenv()

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config_music_generation.json",
        )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.config = None

        # Reasoning LLM
        if llm is None:
            model_config = self.config.get("model", {}) if self.config else {}
            self.model = ChatOpenAI(
                model=model_config.get("name", "qwen3-max"),
                api_key=os.getenv(model_config.get("api_key_env", "MCP_API_KEY")),
                base_url=model_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                temperature=model_config.get("temperature", temperature),
                max_tokens=model_config.get("max_tokens"),
            )
        else:
            self.model = llm

        # Expert LLM
        if exper is None:
            expert_config = self.config.get("expert", {}) if self.config else {}
            self.exper = ChatOpenAI(
                model=expert_config.get("name", "deepseek-v3.2"),
                api_key=os.getenv(expert_config.get("api_key_env", "DASHSCOPE_API_KEY")),
                base_url=expert_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                temperature=expert_config.get("temperature", temperature),
                max_tokens=expert_config.get("max_tokens"),
            )
        else:
            self.exper = exper

        self.agent_profile = STYLIST_AGENT_PROFILE
        self.max_iterations = max_iterations

        # Load vocabulary for validation
        self.song_vocab = _load_songgeneration_vocab()
        self.valid_audio_types = self.song_vocab.get("audio_type", VALID_AUDIO_TYPES)

        # Reflection memory
        if REFLECTION_IMPORT_SUCCESS:
            self.reflection_memory = get_reflection_memory("stylist", CURRENT_DIR)
        else:
            self.reflection_memory = None

        # Build LangGraph
        self.builder = StateGraph(StylistAgent.Graph)

        self.builder.add_node("init", self._init_node)
        self.builder.add_node("think", self._think_node)
        self.builder.add_node("action", self._action_node)
        self.builder.add_node("select_audio_type", self._select_audio_type_node)
        self.builder.add_node("observation", self._observation_node)
        self.builder.add_node("reflect", self._reflect_node)
        self.builder.add_node("final", self._final_node)

        self.builder.add_edge(START, "init")
        self.builder.add_edge("init", "think")

        self.builder.add_conditional_edges(
            "think", self._route_after_think,
            {"action": "action", "final": "final"},
        )
        self.builder.add_conditional_edges(
            "action", self._route_after_action,
            {"select_audio_type": "select_audio_type", "observation": "observation"},
        )
        self.builder.add_edge("select_audio_type", "observation")
        self.builder.add_conditional_edges(
            "observation", self._route_after_observation,
            {"think": "think", "final": "final", "reflect": "reflect"},
        )
        self.builder.add_conditional_edges(
            "reflect", self._route_after_reflect,
            {"think": "think", "final": "final"},
        )
        self.builder.add_edge("final", END)

        self.graph = self.builder.compile()

    # ===================== System Prompt Builder =====================

    def _build_system_prompt_from_profile(self, task: str) -> str:
        if not PROFILE_IMPORT_SUCCESS or self.agent_profile is None:
            return f'## My Task\n"""{task}"""\n\n## Conversation History\n'

        profile = self.agent_profile
        sp = ""
        sp += f"# {profile.role.name}\n\n{profile.description}\n\n"
        sp += f"## Role Description\n{profile.role.description}\n\n"

        sp += "## Responsibilities\n"
        for i, r in enumerate(profile.role.responsibilities, 1):
            sp += f"{i}. {r}\n"
        sp += "\n"

        sp += "## Areas of Expertise\n"
        for i, e in enumerate(profile.role.expertise, 1):
            sp += f"{i}. {e}\n"
        sp += "\n"

        sp += "## Available Tools\n"
        for tool in profile.tools:
            sp += f"### {tool.name}\n- Description: {tool.description}\n"
            sp += f"- Signature: {tool.function_signature}\n"
            if tool.usage_example:
                sp += f"- Example: {tool.usage_example}\n"
            sp += "\n"

        sp += "## Knowledge Domains\n"
        for k in profile.knowledge:
            sp += f"### {k.domain.value}\n**Concepts:**\n"
            for c in k.concepts:
                sp += f"- {c}\n"
            sp += "**Rules:**\n"
            for r in k.rules:
                sp += f"- {r}\n"
            sp += "\n"

        sp += "## Constraints\n"
        for i, c in enumerate(profile.constraints, 1):
            sp += f"{i}. {c}\n"
        sp += "\n"

        sp += "## Best Practices\n"
        for i, b in enumerate(profile.best_practices, 1):
            sp += f"{i}. {b}\n"
        sp += "\n"

        if profile.command:
            sp += f"{profile.command}\n\n"
        if profile.guide_book:
            sp += f"{profile.guide_book}\n\n"

        sp += f'## My Task\n"""{task}"""\n\n## Conversation History\n'
        return sp

    # ===================== Graph Nodes =====================

    async def _init_node(self, state: "StylistAgent.Graph") -> Dict[str, Any]:
        task = state.get("tasks", "")
        _print_with_indent("task:", str(task), tab_count=2)
        system_prompt = self._build_system_prompt_from_profile(task)

        return {
            "global_messages": [SystemMessage(content=system_prompt), HumanMessage(content="")],
            "system_prompt_messages": system_prompt,
            "user_messages": [],
            "tasks": task,
            "think_message": [],
            "action_message": [],
            "stylist_message": [],
            "observation_message": [],
            "reflection_message": [],
            "stylist_flag": False,
            "stylist_audio_type_result": [],
            "state": "think",
            "tools": "None",
            "current_iteration": 1,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }

    async def _think_node(self, state: "StylistAgent.Graph") -> Dict[str, Any]:
        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]
        current_iteration = state.get("current_iteration", 0)
        _complete = state.get("complete", False)
        audio_result = state.get("stylist_audio_type_result", [])

        if _complete or current_iteration > self.max_iterations or (audio_result and len(audio_result) > 0):
            return {
                "current_iteration": 1,
                "complete": True,
                "state": "final",
                "stylist_audio_type_result": audio_result,
            }

        think_prompt = (
            'Based on known conditions, analyze the current situation and decide the next step. '
            'Reply strictly in JSON: {"Result": ["your thought"]}\n'
        )
        messages.append(HumanMessage(content=think_prompt))
        system_prompt += think_prompt

        response = await self.model.ainvoke(messages)
        think_list = extract_field_from_response(response, "Result")
        thought = think_list[0] if isinstance(think_list, list) and len(think_list) > 0 else ""
        if isinstance(thought, list):
            thought = "".join(thought)
        elif not isinstance(thought, str):
            thought = str(thought) if thought else ""

        if state.get("stylist_flag", False) and audio_result:
            thought = "None"

        _print_with_indent(f"thought{state.get('current_iteration', 0)}:", str(thought), tab_count=2)

        if not _complete:
            if thought == "None":
                next_state = "final"
                _complete = True
            else:
                next_state = "action"
        else:
            next_state = "final"

        content = f"thought{state.get('current_iteration', 0)}:{thought}\n"
        system_prompt += content

        return {
            "global_messages": [AIMessage(content=content)],
            "system_prompt_messages": system_prompt,
            "think_message": [{"role": "assistant", "content": content}],
            "state": next_state,
            "current_iteration": 1,
            "complete": _complete,
        }

    async def _action_node(self, state: "StylistAgent.Graph") -> Dict[str, Any]:
        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]

        action_prompt = (
            'Based on the above thought, select one tool:\n'
            '- select_audio_type: select music audio type\n'
            '- None: do not execute any tool\n'
            'Reply strictly in JSON: {"Result": ["tool_name", "parameters"]}\n'
        )
        messages.append(HumanMessage(content=action_prompt))
        system_prompt += action_prompt

        response = await self.model.ainvoke(messages)
        action_list = extract_field_from_response(response, "Result")
        action_name = action_list[0] if isinstance(action_list, list) and len(action_list) > 0 else "none"
        action_parameter = action_list[1] if isinstance(action_list, list) and len(action_list) > 1 else ""

        if isinstance(action_name, list):
            action_name = "".join(action_name)
        elif not isinstance(action_name, str):
            action_name = str(action_name) if action_name else "none"
        if isinstance(action_parameter, list):
            action_parameter = "".join(action_parameter)
        elif not isinstance(action_parameter, str):
            action_parameter = str(action_parameter) if action_parameter else ""

        tool = action_name.strip().lower()
        valid_tools = ["select_audio_type", "none"]
        normalized = tool
        for vt in valid_tools:
            if tool.startswith(vt):
                normalized = vt
                break
        tool = normalized

        action_content = f"{action_name} {action_parameter}".strip()
        _print_with_indent(f"action{state.get('current_iteration', 0)}:", action_content, tab_count=2)
        system_prompt += f"action{state.get('current_iteration', 0)}:{action_content}\n"

        return {
            "global_messages": [AIMessage(content=action_content)],
            "system_prompt_messages": system_prompt,
            "action_message": [{"role": "assistant", "content": action_content}],
            "state": "execute" if tool not in ("none", "") else "observation",
            "tools": tool if tool in valid_tools else "none",
            "current_iteration": 1,
        }

    async def _select_audio_type_node(self, state: "StylistAgent.Graph") -> Dict[str, Any]:
        """Core tool: select audio type from 12 candidates.

        For NotaGen mode, returns 'N/A' immediately.
        For SongGeneration mode, uses LLM to select and post-validates.
        """
        piece = state.get("piece", 2)
        json_scene = state.get("json_scene", [])
        blueprint = state.get("blueprint", {})
        system_prompt = state.get("system_prompt_messages", "")

        model_type = blueprint.get("model", "SongGeneration")

        # NotaGen mode: no audio_type
        if model_type == "NotaGen":
            audio_type_list = ["N/A"] * piece
            content = f"stylist_audio_type: {audio_type_list} (NotaGen mode)"
            _print_with_indent("select_audio_type:", content, tab_count=2)
            return {
                "global_messages": [AIMessage(content=content)],
                "system_prompt_messages": system_prompt,
                "stylist_message": [{"role": "assistant", "content": content}],
                "stylist_flag": True,
                "stylist_audio_type_result": audio_type_list,
                "state": "observation",
                "tools": "None",
                "current_iteration": 1,
            }

        # SongGeneration mode: LLM-assisted selection
        emotional_key = blueprint.get("emotional_key", "")
        scene_desc = json.dumps(json_scene, ensure_ascii=False, indent=2) if json_scene else "No scene provided."
        audio_types_str = ", ".join(self.valid_audio_types)

        audio_system = (
            "You are a professional music style selection assistant with deep knowledge of music genres. "
            f"Select {piece} music audio types from: {audio_types_str}. "
            "Choose the types that best match the scene emotion and atmosphere. "
            "Prefer diversity across selections when appropriate."
        )
        audio_prompt = (
            f"Based on the scene data, select {piece} audio types.\n"
            f"Emotional key: {emotional_key}\n"
            "Requirements:\n"
            f"1. Return exactly {piece} audio types\n"
            '2. Format: {{"Result": ["type1", "type2"]}}\n'
            f"3. Each must be one of: {audio_types_str}\n"
            "4. Only JSON, no explanations\n"
            f"Scene data:\n{scene_desc}\n"
        )

        messages = [
            SystemMessage(content=audio_system),
            HumanMessage(content=audio_prompt),
        ]
        response = await self.exper.ainvoke(messages)
        audio_list = extract_field_from_response(response, "Result")

        if not isinstance(audio_list, list):
            audio_list = [str(audio_list)]

        # Post-LLM validation: replace invalid with "Auto"
        for i, item in enumerate(audio_list):
            if item not in self.valid_audio_types:
                audio_list[i] = "Auto"

        # Pad if fewer than piece
        while len(audio_list) < piece:
            audio_list.append("Auto")

        content = f"stylist_audio_type: {json.dumps(audio_list, ensure_ascii=False)}"
        _print_with_indent("select_audio_type:", content, tab_count=2)

        return {
            "global_messages": [AIMessage(content=content)],
            "system_prompt_messages": system_prompt,
            "stylist_message": [{"role": "assistant", "content": content}],
            "stylist_flag": True,
            "stylist_audio_type_result": audio_list,
            "state": "observation",
            "tools": "None",
            "current_iteration": 1,
        }

    async def _observation_node(self, state: "StylistAgent.Graph") -> Dict[str, Any]:
        observation_content = ""
        current_iteration = state.get("current_iteration", 0)
        _complete = state.get("complete", False)
        if current_iteration > state.get("max_iterations", self.max_iterations):
            _complete = True

        if state.get("stylist_flag", False):
            audio_result = state.get("stylist_audio_type_result", [])
            observation_content = f"audio_type_list: {json.dumps(audio_result, ensure_ascii=False)}"
        else:
            observation_content = "No tool was executed."

        system_prompt = state.get("system_prompt_messages", "")
        _print_with_indent(f"observation{state.get('current_iteration', 0)}:", observation_content, tab_count=2)
        system_prompt += f"observation{state.get('current_iteration', 0)}:{observation_content}\n"

        return {
            "global_messages": [AIMessage(content=observation_content)],
            "system_prompt_messages": system_prompt,
            "observation_message": [{"role": "assistant", "content": observation_content}],
            "stylist_flag": False,
            "stylist_audio_type_result": state.get("stylist_audio_type_result", []),
            "state": "think",
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": state.get("reflection_flag", False),
            "reflection_result": state.get("reflection_result", {}),
            "reflection_count": state.get("reflection_count", 0),
        }

    async def _reflect_node(self, state: "StylistAgent.Graph") -> Dict[str, Any]:
        _print_with_indent("", "Reflection node started...", tab_count=1)

        task_description = state.get("tasks", "")
        audio_result = state.get("stylist_audio_type_result", [])
        observation = f"audio_type_list: {json.dumps(audio_result, ensure_ascii=False)}"

        history_reflections = ""
        if self.reflection_memory:
            history_reflections = self.reflection_memory.get_summary()

        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="stylist",
                task_description=task_description,
                observation=observation,
                history_reflections=history_reflections,
            )
        else:
            reflect_prompt = (
                f"Analyze results:\nTask: {task_description}\nObservation: {observation}\n"
                f"History: {history_reflections or 'None'}\n"
                'Reply JSON: {{"quality": "high/medium/low", "should_retry": true/false}}\n'
            )

        system_prompt = state.get("system_prompt_messages", "")
        system_prompt += "\n## Reflection Phase\n" + reflect_prompt
        messages = [SystemMessage(content=system_prompt)]

        try:
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            _print_with_indent("", f"Reflection failed: {e}", tab_count=2)
            response_text = '{"quality": "low", "should_retry": false}'

        if REFLECTION_IMPORT_SUCCESS and parse_reflection_result:
            reflection_result = parse_reflection_result(response_text)
        else:
            import re
            try:
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                reflection_result = json.loads(json_match.group()) if json_match else {"quality": "low", "should_retry": False}
            except Exception:
                reflection_result = {"quality": "low", "should_retry": False}

        quality = reflection_result.get("quality", "low")
        should_retry = reflection_result.get("should_retry", False)
        reflection_count = state.get("reflection_count", 0) + 1

        _print_with_indent("", f"Quality: {quality}, Retry: {should_retry}", tab_count=2)

        if self.reflection_memory:
            self.reflection_memory.add_reflection(
                task_description=task_description,
                observation=observation,
                reflection=reflection_result.get("analysis", ""),
                improvement=reflection_result.get("improvement", ""),
                quality=quality,
                iterations=state.get("current_iteration", 1),
            )

        reflection_content = f"Quality: {quality}, Retry: {should_retry}"
        system_prompt += f"\n{reflection_content}\n"

        if quality == "high" or reflection_count >= 3:
            next_state = "final"
            _complete = True
        elif should_retry:
            next_state = "think"
            _complete = False
        else:
            next_state = "final"
            _complete = True

        return {
            "global_messages": [AIMessage(content=reflection_content)],
            "system_prompt_messages": system_prompt,
            "reflection_message": [{"role": "assistant", "content": reflection_content}],
            "stylist_audio_type_result": state.get("stylist_audio_type_result", []),
            "state": next_state,
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": True,
            "reflection_result": reflection_result,
            "reflection_count": reflection_count,
        }

    async def _final_node(self, state: "StylistAgent.Graph") -> Dict[str, Any]:
        audio_result = state.get("stylist_audio_type_result", [])
        final_answer = {"audio_type_list": audio_result}

        _print_with_indent(
            f"final_answer{state.get('current_iteration', 0)}:",
            json.dumps(final_answer, ensure_ascii=False),
            tab_count=2,
        )

        system_prompt = state.get("system_prompt_messages", "")
        system_prompt += f"final_answer:{json.dumps(final_answer, ensure_ascii=False)}\n"

        return {
            "global_messages": [AIMessage(content=f"Final answer: {final_answer}")],
            "system_prompt_messages": system_prompt,
            "final_answer": final_answer,
        }

    # ===================== Routing Functions =====================

    def _route_after_think(self, state: "StylistAgent.Graph") -> str:
        if state.get("current_iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "final"
        elif state.get("complete", False):
            return "final"
        return "action"

    def _route_after_action(self, state: "StylistAgent.Graph") -> str:
        tool = state.get("tools", "none")
        if tool in ("None", "none", ""):
            return "observation"
        elif tool == "select_audio_type":
            return "select_audio_type"
        return "observation"

    def _route_after_observation(self, state: "StylistAgent.Graph") -> str:
        audio_result = state.get("stylist_audio_type_result", [])
        has_result = audio_result and len(audio_result) > 0

        if state.get("complete", False) or has_result:
            return "reflect"
        elif state.get("current_iteration", 1) >= state.get("max_iterations", self.max_iterations):
            return "reflect"
        return "think"

    def _route_after_reflect(self, state: "StylistAgent.Graph") -> str:
        reflection_result = state.get("reflection_result", {})
        quality = reflection_result.get("quality", "low")
        should_retry = reflection_result.get("should_retry", False)
        reflection_count = state.get("reflection_count", 0)

        if reflection_count >= 3 or quality == "high":
            return "final"
        if should_retry:
            return "think"
        return "final"

    # ===================== Public Interface =====================

    def invoke(self, user_input: str, _json_scene: List[dict] = None,
               piece: int = 2, blueprint: Dict[str, Any] = None) -> Dict[str, Any]:
        initial_state = self._build_initial_state(user_input, _json_scene, piece, blueprint)
        return self.graph.invoke(initial_state)

    def stream(self, user_input: str, _json_scene: List[dict] = None,
               piece: int = 2, blueprint: Dict[str, Any] = None):
        initial_state = self._build_initial_state(user_input, _json_scene, piece, blueprint)
        return self.graph.stream(initial_state)

    async def ainvoke(self, *, task: str, json_scene: List[dict] = None,
                      piece: int = 2, blueprint: Dict[str, Any] = None) -> Dict[str, Any]:
        initial_state = self._build_initial_state(task, json_scene, piece, blueprint)
        result = await self.graph.ainvoke(initial_state)
        return {
            "stylist_audio_type_result": result.get("stylist_audio_type_result", []),
        }

    def _build_initial_state(self, user_input: str, json_scene: List[dict] = None,
                             piece: int = 2, blueprint: Dict[str, Any] = None) -> Dict[str, Any]:
        return {
            "global_messages": [],
            "system_prompt_messages": "",
            "user_messages": [],
            "tasks": user_input,
            "think_message": [],
            "action_message": [],
            "stylist_message": [],
            "observation_message": [],
            "reflection_message": [],
            "json_scene": json_scene or [],
            "piece": piece,
            "blueprint": blueprint or {"model": "SongGeneration", "emotional_key": ""},
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "stylist_flag": False,
            "stylist_audio_type_result": [],
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }


if __name__ == "__main__":
    import asyncio
    load_dotenv()
    agent = StylistAgent(max_iterations=10)
    _json_scene = [
        {
            "时间段": "0s-12s",
            "主体声音内容": "教室的午后，阳光在你的睫毛上跳舞。",
            "主体声音风格": "温柔、略带迟疑",
            "环境声音内容": "笔尖划过纸张的沙沙声，窗外的蝉鸣",
            "环境声音风格": "氛围静谧、慵懒",
        },
    ]
    blueprint = {"model": "SongGeneration", "emotional_key": "romantic"}

    async def main():
        result = await agent.ainvoke(
            task="Select music audio type based on the scene data",
            json_scene=_json_scene,
            piece=2,
            blueprint=blueprint,
        )
        print(f"\naudio_type: {result.get('stylist_audio_type_result', [])}")

    asyncio.run(main())
