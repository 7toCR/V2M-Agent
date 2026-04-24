"""
Lyricist Agent — Merged from pop_gt_lyric.py + pop_idx.py.

Generates song title identifiers (idx) and lyrics (gt_lyric) based on
scene descriptions and Musical Blueprint. Supports SongGeneration vocal/BGM
modes and NotaGen classical notation mode.

Follows the Think-Act-Observe-Reflect iterative loop pattern from Team2.
"""

import os
import sys
import json
import logging
import re
from datetime import datetime
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
    COMMAND_lyricist, Guide_Book_lyricist, Gt_Lyric_system_prompt,
)
from tools.tools import (
    extract_field_from_response,
    _print_with_indent,
)

# Import agent profile
try:
    from Team3.AgentProfile.lyricist_agent_profile import LYRICIST_AGENT_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import LYRICIST_AGENT_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    LYRICIST_AGENT_PROFILE = None

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


class LyricistAgent:
    """Lyricist and Song Title Expert.

    Generates idx (datetime-prefixed song names) and gt_lyric (lyrics with
    structure tags) based on Musical Blueprint and scene data.

    Supports:
    - SongGeneration vocal mode (Chinese/English/mixed lyrics)
    - SongGeneration BGM mode (structure tags only)
    - NotaGen classical notation mode (structure tags only)
    """

    @staticmethod
    def _is_valid_generated_lyric(lyric: object, lyric_style: str = "vocal") -> bool:
        text = str(lyric or "").strip()
        if not text or "..." in text or "…" in text:
            return False

        sections = [section.strip() for section in text.split(" ; ") if section.strip()]
        if not sections:
            return False

        tags = []
        body_parts = []
        for section in sections:
            match = re.match(r"(\[[\w-]+\])\s*(.*)$", section)
            if not match:
                return False
            tags.append(match.group(1))
            body_parts.append(match.group(2).strip())

        if not tags[0].startswith("[intro") or not tags[-1].startswith("[outro"):
            return False

        if lyric_style == "vocal":
            if "[verse]" not in tags or "[chorus]" not in tags:
                return False
            body = " ".join(body_parts).replace(";", " ").strip()
            return len(body) >= 20

        return len(tags) >= 2

    class Graph(TypedDict):
        # Global messages
        global_messages: Annotated[List[dict], add_messages]
        # System prompt
        system_prompt_messages: str
        # User messages
        user_messages: Annotated[List[dict], add_messages]
        # Task description
        tasks: str

        # Per-step messages
        think_message: Annotated[List[dict], add_messages]
        action_message: Annotated[List[dict], add_messages]
        lyricist_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]
        reflection_message: Annotated[List[dict], add_messages]

        # Input parameters
        json_scene: List[dict]
        piece: int
        blueprint: Dict[str, Any]

        # Execution results
        lyricist_flag: bool
        lyricist_idx_result: List[str]
        lyricist_lyric_result: List[str]

        # Running state
        state: Literal["start", "think", "action", "execute", "observation", "reflect", "final", "end"]
        tools: Literal["None", "none", "generate_lyrics_and_title"]

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

        # Reasoning LLM (think / action / reflect)
        if llm is None:
            model_config = self.config.get("model", {}) if self.config else {}
            self.model = ChatOpenAI(
                model=model_config.get("name", "qwen3-max"),
                api_key=os.getenv(model_config.get("api_key_env", "MCP_API_KEY")),
                base_url=model_config.get(
                    "base_url",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                temperature=model_config.get("temperature", temperature),
                max_tokens=model_config.get("max_tokens"),
            )
        else:
            self.model = llm

        # Expert LLM (tool / generation nodes)
        if exper is None:
            expert_config = self.config.get("expert", {}) if self.config else {}
            self.exper = ChatOpenAI(
                model=expert_config.get("name", "deepseek-v3.2"),
                api_key=os.getenv(expert_config.get("api_key_env", "DASHSCOPE_API_KEY")),
                base_url=expert_config.get(
                    "base_url",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                temperature=expert_config.get("temperature", temperature),
                max_tokens=expert_config.get("max_tokens"),
            )
        else:
            self.exper = exper

        self.agent_profile = LYRICIST_AGENT_PROFILE
        self.max_iterations = max_iterations

        # Reflection memory
        if REFLECTION_IMPORT_SUCCESS:
            self.reflection_memory = get_reflection_memory("lyricist", CURRENT_DIR)
        else:
            self.reflection_memory = None

        # Build LangGraph
        self.builder = StateGraph(LyricistAgent.Graph)

        self.builder.add_node("init", self._init_node)
        self.builder.add_node("think", self._think_node)
        self.builder.add_node("action", self._action_node)
        self.builder.add_node("generate_lyrics_and_title", self._generate_lyrics_and_title_node)
        self.builder.add_node("observation", self._observation_node)
        self.builder.add_node("reflect", self._reflect_node)
        self.builder.add_node("final", self._final_node)

        self.builder.add_edge(START, "init")
        self.builder.add_edge("init", "think")

        self.builder.add_conditional_edges(
            "think",
            self._route_after_think,
            {"action": "action", "final": "final"},
        )
        self.builder.add_conditional_edges(
            "action",
            self._route_after_action,
            {
                "generate_lyrics_and_title": "generate_lyrics_and_title",
                "observation": "observation",
            },
        )
        self.builder.add_edge("generate_lyrics_and_title", "observation")

        self.builder.add_conditional_edges(
            "observation",
            self._route_after_observation,
            {"think": "think", "final": "final", "reflect": "reflect"},
        )
        self.builder.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {"think": "think", "final": "final"},
        )
        self.builder.add_edge("final", END)

        self.graph = self.builder.compile()

    # ===================== System Prompt Builder =====================

    def _build_system_prompt_from_profile(self, task: str) -> str:
        """Build system prompt from LYRICIST_AGENT_PROFILE."""
        if not PROFILE_IMPORT_SUCCESS or self.agent_profile is None:
            return f"## My Task\n\"\"\"{task}\"\"\"\n\n## Conversation History\n"

        profile = self.agent_profile
        sp = ""

        sp += f"# {profile.role.name}\n\n"
        sp += f"{profile.description}\n\n"
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
            sp += f"### {tool.name}\n"
            sp += f"- Description: {tool.description}\n"
            sp += f"- Signature: {tool.function_signature}\n"
            if tool.usage_example:
                sp += f"- Example: {tool.usage_example}\n"
            sp += "\n"

        sp += "## Knowledge Domains\n"
        for k in profile.knowledge:
            sp += f"### {k.domain.value}\n"
            sp += "**Concepts:**\n"
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

        sp += "## Available Resources\n"
        for i, r in enumerate(profile.resources, 1):
            sp += f"{i}. {r}\n"
        sp += "\n"

        sp += "## Execution Methods\n"
        for i, m in enumerate(profile.run_methods, 1):
            sp += f"{i}. {m}\n"
        sp += "\n"

        if profile.command:
            sp += f"{profile.command}\n\n"
        if profile.guide_book:
            sp += f"{profile.guide_book}\n\n"

        sp += f'## My Task\n"""{task}"""\n\n'
        sp += "## Conversation History\n"

        return sp

    # ===================== Graph Nodes =====================

    async def _init_node(self, state: "LyricistAgent.Graph") -> Dict[str, Any]:
        """Build system prompt from agent profile."""
        task = state.get("tasks", "")
        _print_with_indent("task:", str(task), tab_count=2)
        system_prompt = self._build_system_prompt_from_profile(task)

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
            "lyricist_message": [],
            "observation_message": [],
            "reflection_message": [],
            "lyricist_flag": False,
            "lyricist_idx_result": [],
            "lyricist_lyric_result": [],
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

    async def _think_node(self, state: "LyricistAgent.Graph") -> Dict[str, Any]:
        """Think node: reason about the current situation."""
        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]
        current_iteration = state.get("current_iteration", 0)
        _complete = state.get("complete", False)
        idx_result = state.get("lyricist_idx_result", [])
        lyric_result = state.get("lyricist_lyric_result", [])

        has_result = (idx_result and len(idx_result) > 0) and (lyric_result and len(lyric_result) > 0)

        if _complete or current_iteration > self.max_iterations or has_result:
            return {
                "current_iteration": 1,
                "complete": True,
                "state": "final",
                "lyricist_idx_result": idx_result,
                "lyricist_lyric_result": lyric_result,
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

        if state.get("lyricist_flag", False) and idx_result:
            thought = "None"

        _print_with_indent(f"thought{state.get('current_iteration', 0)}:", str(thought), tab_count=2)

        if not _complete:
            if thought == "None":
                next_state = "final"
                _complete = True
            else:
                next_state = "action"
                _complete = False
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

    async def _action_node(self, state: "LyricistAgent.Graph") -> Dict[str, Any]:
        """Action node: select tool to execute."""
        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]

        action_prompt = (
            'Based on the above thought, select one tool:\n'
            '- generate_lyrics_and_title: generate song titles and lyrics\n'
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
        # Normalize tool name
        valid_tools = ["generate_lyrics_and_title", "none"]
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

    async def _generate_lyrics_and_title_node(self, state: "LyricistAgent.Graph") -> Dict[str, Any]:
        """Core tool node: generate idx (song titles) and gt_lyric (lyrics).

        Step 1: Generate short song names via LLM, prepend datetime prefix -> idx_list
        Step 2: Generate lyrics matching scene emotion -> lyric_list
        Blueprint controls mode: vocal / BGM / NotaGen
        """
        piece = state.get("piece", 2)
        json_scene = state.get("json_scene", [])
        blueprint = state.get("blueprint", {})
        system_prompt = state.get("system_prompt_messages", "")

        model_type = blueprint.get("model", "SongGeneration")
        lyric_style = blueprint.get("lyric_style", "vocal")
        language = blueprint.get("language", "zh")
        emotional_key = blueprint.get("emotional_key", "")

        # ---- Step 1: Generate song title names ----
        now = datetime.now()

        idx_system = (
            "You are a professional music title generation assistant. "
            f"Generate {piece} concise music titles (max 10 characters each, "
            "Chinese and English only)."
        )

        scene_desc = json.dumps(json_scene, ensure_ascii=False, indent=2) if json_scene else "No scene provided."
        idx_prompt = (
            f"Based on the following scene data, generate {piece} short song titles.\n"
            f"Emotional key: {emotional_key}\n"
            f"Language preference: {language}\n"
            "Requirements:\n"
            f"1. Return exactly {piece} titles\n"
            '2. Format: {{"Result": ["title1", "title2"]}}\n'
            "3. Each title max 10 characters, Chinese or English only\n"
            "4. No explanations, only JSON\n"
            f"Scene data:\n{scene_desc}\n"
        )

        idx_messages = [
            SystemMessage(content=idx_system),
            HumanMessage(content=idx_prompt),
        ]
        idx_response = await self.exper.ainvoke(idx_messages)
        idx_names = extract_field_from_response(idx_response, "Result")
        if not isinstance(idx_names, list):
            idx_names = [str(idx_names)]

        # Prepend datetime prefix
        idx_list = [
            f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{name}"
            for name in idx_names[:piece]
        ]
        # Pad if fewer than piece
        while len(idx_list) < piece:
            idx_list.append(f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-Untitled{len(idx_list)}")

        # ---- Step 2: Generate lyrics ----
        if model_type == "NotaGen":
            # BGM / NotaGen mode: structure tags only, no lyric text
            lyric_system = (
                "You are a professional music structure designer. "
                f"Generate {piece} BGM structure tag sequences. "
                "Use only valid tags: [intro-short], [intro-medium], [intro-long], "
                "[verse], [chorus], [bridge], [outro-short], [outro-medium], [outro-long], "
                "[inst-short], [inst-medium], [inst-long]. "
                "Separate tags with ' ; '. No lyric text between tags."
            )
            lyric_prompt = (
                f"Generate {piece} BGM structure sequences.\n"
                "Requirements:\n"
                f"1. Return exactly {piece} sequences\n"
                '2. Format: {{"Result": ["[intro-short] ; [inst-medium] ; [outro-short]", ...]}}\n'
                "3. Start with [intro-*], end with [outro-*]\n"
                "4. Only JSON, no explanations\n"
                f"Emotional key: {emotional_key}\n"
                f"Scene data:\n{scene_desc}\n"
            )
        else:
            # Vocal mode: full lyrics with structure tags
            lyric_system_base = ""
            if isinstance(Gt_Lyric_system_prompt, dict):
                lyric_system_base = Gt_Lyric_system_prompt.get("system_prompt", "")
            elif isinstance(Gt_Lyric_system_prompt, str):
                lyric_system_base = Gt_Lyric_system_prompt
            else:
                lyric_system_base = str(Gt_Lyric_system_prompt) if Gt_Lyric_system_prompt else ""

            lang_instruction = {
                "zh": "Generate Chinese lyrics only (no English).",
                "en": "Generate English lyrics only (no Chinese).",
                "mixed": "Generate mixed Chinese-English lyrics.",
            }.get(language, "Generate Chinese lyrics.")

            lyric_system = (
                f"{lyric_system_base}\n\n"
                f"IMPORTANT: You must return an array of exactly {piece} lyric strings.\n"
                f"{lang_instruction}\n"
                f"Emotional key: {emotional_key}\n"
            )
            lyric_prompt = (
                f"Based on the scene data below, compose {piece} different songs.\n"
                "Requirements:\n"
                f"1. Return exactly {piece} lyrics\n"
                '2. Format: {{"Result": ["lyric1", "lyric2"]}}\n'
                "3. Each lyric must match the scene emotion, narrative, and atmosphere\n"
                "4. Use valid structure tags: [intro-*], [verse], [chorus], [bridge], [outro-*], [inst-*]\n"
                "5. Sections separated by ' ; '\n"
                "6. All punctuation must be half-width ASCII\n"
                "7. Only JSON output, no explanations\n"
                f"Scene data:\n{scene_desc}\n"
            )

        lyric_messages = [
            SystemMessage(content=lyric_system),
            HumanMessage(content=lyric_prompt),
        ]
        lyric_response = await self.exper.ainvoke(lyric_messages)
        lyric_list = extract_field_from_response(lyric_response, "Result")
        if not isinstance(lyric_list, list):
            lyric_list = [str(lyric_list)]

        lyric_style = str(blueprint.get("lyric_style", "vocal"))
        valid_lyrics = []
        for lyric in lyric_list:
            lyric_text = str(lyric or "").strip()
            if self._is_valid_generated_lyric(lyric_text, lyric_style):
                valid_lyrics.append(lyric_text)
            else:
                _print_with_indent(
                    "generate_lyrics_and_title:",
                    "discarded invalid lyric",
                    tab_count=2,
                )
        lyric_list = valid_lyrics[:piece]

        content = f"lyricist_result: idx={json.dumps(idx_list, ensure_ascii=False)}, lyric={json.dumps(lyric_list, ensure_ascii=False)}"
        _print_with_indent("generate_lyrics_and_title:", content, tab_count=2)

        return {
            "global_messages": [AIMessage(content=content)],
            "system_prompt_messages": system_prompt,
            "lyricist_message": [{"role": "assistant", "content": content}],
            "lyricist_flag": True,
            "lyricist_idx_result": idx_list,
            "lyricist_lyric_result": lyric_list,
            "state": "observation",
            "tools": "None",
            "current_iteration": 1,
        }

    async def _observation_node(self, state: "LyricistAgent.Graph") -> Dict[str, Any]:
        """Process tool results for the next round of thinking."""
        observation_content = ""
        current_iteration = state.get("current_iteration", 0)
        _complete = state.get("complete", False)
        if current_iteration > state.get("max_iterations", self.max_iterations):
            _complete = True

        if state.get("lyricist_flag", False):
            idx_result = state.get("lyricist_idx_result", [])
            lyric_result = state.get("lyricist_lyric_result", [])
            observation_content = (
                f"idx_list: {json.dumps(idx_result, ensure_ascii=False)}\n"
                f"lyric_list: {json.dumps(lyric_result, ensure_ascii=False)}"
            )
        else:
            observation_content = "No tool was executed."

        system_prompt = state.get("system_prompt_messages", "")
        _print_with_indent(f"observation{state.get('current_iteration', 0)}:", observation_content, tab_count=2)
        system_prompt += f"observation{state.get('current_iteration', 0)}:{observation_content}\n"

        return {
            "global_messages": [AIMessage(content=observation_content)],
            "system_prompt_messages": system_prompt,
            "observation_message": [{"role": "assistant", "content": observation_content}],
            "lyricist_flag": False,
            "lyricist_idx_result": state.get("lyricist_idx_result", []),
            "lyricist_lyric_result": state.get("lyricist_lyric_result", []),
            "state": "think",
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": state.get("reflection_flag", False),
            "reflection_result": state.get("reflection_result", {}),
            "reflection_count": state.get("reflection_count", 0),
        }

    async def _reflect_node(self, state: "LyricistAgent.Graph") -> Dict[str, Any]:
        """Reflection node: evaluate quality and decide whether to retry."""
        _print_with_indent("", "Reflection node started...", tab_count=1)

        task_description = state.get("tasks", "")
        idx_result = state.get("lyricist_idx_result", [])
        lyric_result = state.get("lyricist_lyric_result", [])
        observation = (
            f"idx_list: {json.dumps(idx_result, ensure_ascii=False)}\n"
            f"lyric_list: {json.dumps(lyric_result, ensure_ascii=False)}"
        )

        history_reflections = ""
        if self.reflection_memory:
            history_reflections = self.reflection_memory.get_summary()

        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="lyricist",
                task_description=task_description,
                observation=observation,
                history_reflections=history_reflections,
            )
        else:
            reflect_prompt = (
                f"Analyze the following results:\n"
                f"Task: {task_description}\n"
                f"Observation: {observation}\n"
                f"History: {history_reflections or 'None'}\n"
                'Reply JSON: {{"quality": "high/medium/low", "should_retry": true/false, '
                '"analysis": "...", "improvement": "..."}}\n'
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
            "lyricist_idx_result": state.get("lyricist_idx_result", []),
            "lyricist_lyric_result": state.get("lyricist_lyric_result", []),
            "state": next_state,
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": True,
            "reflection_result": reflection_result,
            "reflection_count": reflection_count,
        }

    async def _final_node(self, state: "LyricistAgent.Graph") -> Dict[str, Any]:
        """Output final idx_list and lyric_list."""
        idx_result = state.get("lyricist_idx_result", [])
        lyric_result = state.get("lyricist_lyric_result", [])
        final_answer = {
            "idx_list": idx_result,
            "lyric_list": lyric_result,
        }

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

    def _route_after_think(self, state: "LyricistAgent.Graph") -> str:
        if state.get("current_iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "final"
        elif state.get("complete", False):
            return "final"
        return "action"

    def _route_after_action(self, state: "LyricistAgent.Graph") -> str:
        tool = state.get("tools", "none")
        if tool in ("None", "none", ""):
            return "observation"
        elif tool == "generate_lyrics_and_title":
            return "generate_lyrics_and_title"
        return "observation"

    def _route_after_observation(self, state: "LyricistAgent.Graph") -> str:
        idx_result = state.get("lyricist_idx_result", [])
        lyric_result = state.get("lyricist_lyric_result", [])
        has_result = (idx_result and len(idx_result) > 0) and (lyric_result and len(lyric_result) > 0)

        if state.get("complete", False) or has_result:
            return "reflect"
        elif state.get("current_iteration", 1) >= state.get("max_iterations", self.max_iterations):
            return "reflect"
        return "think"

    def _route_after_reflect(self, state: "LyricistAgent.Graph") -> str:
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
        """Synchronous execution."""
        initial_state = self._build_initial_state(user_input, _json_scene, piece, blueprint)
        return self.graph.invoke(initial_state)

    def stream(self, user_input: str, _json_scene: List[dict] = None,
               piece: int = 2, blueprint: Dict[str, Any] = None):
        """Stream execution."""
        initial_state = self._build_initial_state(user_input, _json_scene, piece, blueprint)
        return self.graph.stream(initial_state)

    async def ainvoke(self, *, task: str, json_scene: List[dict] = None,
                      piece: int = 2, blueprint: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async execution (primary entry for supervisor dispatch)."""
        initial_state = self._build_initial_state(task, json_scene, piece, blueprint)
        result = await self.graph.ainvoke(initial_state)
        return {
            "lyricist_idx_result": result.get("lyricist_idx_result", []),
            "lyricist_lyric_result": result.get("lyricist_lyric_result", []),
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
            "lyricist_message": [],
            "observation_message": [],
            "reflection_message": [],
            "json_scene": json_scene or [],
            "piece": piece,
            "blueprint": blueprint or {"model": "SongGeneration", "lyric_style": "vocal", "emotional_key": "", "language": "zh"},
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "complete": False,
            "final_answer": "",
            "lyricist_flag": False,
            "lyricist_idx_result": [],
            "lyricist_lyric_result": [],
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }


if __name__ == "__main__":
    import asyncio
    load_dotenv()
    agent = LyricistAgent(max_iterations=10)
    _json_scene = [
        {
            "时间段": "0s-12s",
            "主体声音内容": "教室的午后，阳光在你的睫毛上跳舞。我假装看书，笔尖却在纸上，一遍遍临摹你的侧影。",
            "主体声音风格": "单人内心独白，音色温柔、略带迟疑，充满少年青涩的暗恋感",
            "环境声音内容": "笔尖划过纸张的沙沙声，远处隐约的课堂讲课声，窗外的蝉鸣",
            "环境声音风格": "氛围静谧、慵懒，时间仿佛被拉长，凸显私密的心事",
        },
        {
            "时间段": "12s-24s",
            "主体声音内容": "直到那天，你回头问我借半块橡皮。指尖相触的瞬间，我的心跳，盖过了整个夏天的蝉鸣。",
            "主体声音风格": "独白节奏稍快，语气中有回忆的悸动和甜蜜，声音微微发亮",
            "环境声音内容": "加入轻微的翻书声、椅子挪动声，���鸣声渐强后忽然减弱",
            "环境声音风格": "从日常环境音过渡到情感的主观聚焦",
        },
    ]
    blueprint = {
        "model": "SongGeneration",
        "lyric_style": "vocal",
        "emotional_key": "romantic",
        "language": "zh",
    }

    async def main():
        result = await agent.ainvoke(
            task="Generate Chinese song lyrics based on the scene data",
            json_scene=_json_scene,
            piece=2,
            blueprint=blueprint,
        )
        print(f"\nidx_list: {result.get('lyricist_idx_result', [])}")
        print(f"lyric_list: {result.get('lyricist_lyric_result', [])}")

    asyncio.run(main())
