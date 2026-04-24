"""
Composer Agent — Adapted from pop_descriptions.py.

Generates music description strings containing 6 key attributes:
gender, emotion, genre, timbre, instrument, bpm.

Supports both SongGeneration (6-field format) and NotaGen
(period/composer/instrument) modes.

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
    COMMAND_composer, Guide_Book_composer,
)
from tools.tools import (
    extract_field_from_response,
    _print_with_indent,
)

# Import agent profile
try:
    from Team3.AgentProfile.composer_agent_profile import COMPOSER_AGENT_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import COMPOSER_AGENT_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    COMPOSER_AGENT_PROFILE = None

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


def _load_noatgen_vocab() -> Dict[str, Any]:
    """Load vocabulary from noatgen.json."""
    for candidate in [
        os.path.join(CURRENT_DIR, "..", "pre-traing", "noatgen.json"),
        os.path.join(CURRENT_DIR, "..", "AgentProfile", "pre-traing", "noatgen.json"),
    ]:
        path = os.path.normpath(candidate)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


class ComposerAgent:
    """Music Description Composer.

    Generates descriptions strings: 'gender, emotion, genre, timbre, instrument, the bpm is N.'
    All values validated against songgeneration.json vocabulary.
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
        composer_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]
        reflection_message: Annotated[List[dict], add_messages]

        # Input parameters
        json_scene: List[dict]
        piece: int
        blueprint: Dict[str, Any]

        # Execution results
        composer_flag: bool
        composer_descriptions_result: List[str]

        # Running state
        state: Literal["start", "think", "action", "execute", "observation", "reflect", "final", "end"]
        tools: Literal["None", "none", "generate_descriptions"]

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

        self.agent_profile = COMPOSER_AGENT_PROFILE
        self.max_iterations = max_iterations

        # Load vocabularies
        self.song_vocab = _load_songgeneration_vocab()
        self.noatgen_vocab = _load_noatgen_vocab()

        # Reflection memory
        if REFLECTION_IMPORT_SUCCESS:
            self.reflection_memory = get_reflection_memory("composer", CURRENT_DIR)
        else:
            self.reflection_memory = None

        # Build LangGraph
        self.builder = StateGraph(ComposerAgent.Graph)

        self.builder.add_node("init", self._init_node)
        self.builder.add_node("think", self._think_node)
        self.builder.add_node("action", self._action_node)
        self.builder.add_node("generate_descriptions", self._generate_descriptions_node)
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
            {"generate_descriptions": "generate_descriptions", "observation": "observation"},
        )
        self.builder.add_edge("generate_descriptions", "observation")
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

    # ===================== Vocabulary Validation =====================

    def _validate_description(self, desc: str) -> str:
        """Validate and fix a single description string against songgeneration.json.

        Returns the corrected description string.
        """
        if not self.song_vocab:
            return desc

        vocab = self.song_vocab.get("descriptions", {})
        valid_genders = vocab.get("gender", [])
        valid_emotions = vocab.get("emotion", [])
        valid_genres = vocab.get("genre", [])
        valid_timbres = vocab.get("timbre", [])
        valid_instruments = vocab.get("instrument", [])
        bpm_range = vocab.get("bpm_range", [60, 200])

        # Parse: "gender, emotion, genre, timbre, instrument, the bpm is N."
        # Strategy: split by ", " carefully since instrument may contain " and "
        parts = desc.strip().rstrip(".").split(", ")

        if len(parts) < 6:
            return desc  # Cannot parse, return as-is

        # Last part should be "the bpm is N"
        bpm_part = parts[-1]
        # Instrument may span multiple parts due to " and " in names
        # Fields: gender(0), emotion(1), genre(2), timbre(3), instrument(4..n-1), bpm(n)
        gender = parts[0].strip()
        emotion = parts[1].strip() if len(parts) > 1 else ""
        genre = parts[2].strip() if len(parts) > 2 else ""
        timbre = parts[3].strip() if len(parts) > 3 else ""
        # Instrument is everything between timbre and bpm
        instrument_parts = parts[4:-1] if len(parts) > 5 else [parts[4]] if len(parts) > 4 else []
        instrument = ", ".join(instrument_parts).strip()

        # Validate and fix each field
        if gender not in valid_genders and valid_genders:
            gender = valid_genders[0]  # default: female
        if emotion not in valid_emotions and valid_emotions:
            emotion = "emotional"  # safe default
        if genre not in valid_genres and valid_genres:
            genre = "pop"  # safe default
        if timbre not in valid_timbres and valid_timbres:
            timbre = "varies"  # safe default
        if instrument not in valid_instruments and valid_instruments:
            # Try closest match
            instrument_lower = instrument.lower()
            matched = False
            for vi in valid_instruments:
                if instrument_lower == vi.lower():
                    instrument = vi
                    matched = True
                    break
            if not matched:
                instrument = "synthesizer and piano"  # safe default

        # Validate BPM
        bpm_val = 120  # default
        if "the bpm is" in bpm_part.lower():
            try:
                bpm_str = bpm_part.lower().replace("the bpm is", "").strip()
                bpm_val = int(bpm_str)
                bpm_val = max(bpm_range[0], min(bpm_range[1], bpm_val))
            except (ValueError, IndexError):
                bpm_val = 120

        return f"{gender}, {emotion}, {genre}, {timbre}, {instrument}, the bpm is {bpm_val}."

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

    async def _init_node(self, state: "ComposerAgent.Graph") -> Dict[str, Any]:
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
            "composer_message": [],
            "observation_message": [],
            "reflection_message": [],
            "composer_flag": False,
            "composer_descriptions_result": [],
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

    async def _think_node(self, state: "ComposerAgent.Graph") -> Dict[str, Any]:
        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]
        current_iteration = state.get("current_iteration", 0)
        _complete = state.get("complete", False)
        desc_result = state.get("composer_descriptions_result", [])

        if _complete or current_iteration > self.max_iterations or (desc_result and len(desc_result) > 0):
            return {
                "current_iteration": 1,
                "complete": True,
                "state": "final",
                "composer_descriptions_result": desc_result,
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

        if state.get("composer_flag", False) and desc_result:
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

    async def _action_node(self, state: "ComposerAgent.Graph") -> Dict[str, Any]:
        system_prompt = state.get("system_prompt_messages", "")
        messages = [SystemMessage(content=system_prompt)]

        action_prompt = (
            'Based on the above thought, select one tool:\n'
            '- generate_descriptions: generate music description strings\n'
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
        valid_tools = ["generate_descriptions", "none"]
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

    async def _generate_descriptions_node(self, state: "ComposerAgent.Graph") -> Dict[str, Any]:
        """Core tool: generate descriptions strings with vocabulary validation.

        For SongGeneration: 'gender, emotion, genre, timbre, instrument, the bpm is N.'
        For NotaGen: period/composer/instrument description from noatgen.json.
        """
        piece = state.get("piece", 2)
        json_scene = state.get("json_scene", [])
        blueprint = state.get("blueprint", {})
        system_prompt = state.get("system_prompt_messages", "")

        model_type = blueprint.get("model", "SongGeneration")
        emotional_key = blueprint.get("emotional_key", "")

        scene_desc = json.dumps(json_scene, ensure_ascii=False, indent=2) if json_scene else "No scene provided."

        if model_type == "NotaGen":
            # NotaGen mode: period/composer/instrument from noatgen.json
            noatgen_info = json.dumps(self.noatgen_vocab, ensure_ascii=False, indent=2) if self.noatgen_vocab else "{}"
            desc_system = (
                "You are a classical music notation expert. "
                f"Generate {piece} NotaGen descriptions using period, composer, and instrument category "
                "from the noatgen.json vocabulary provided."
            )
            desc_prompt = (
                f"Generate {piece} NotaGen descriptions.\n"
                f"NotaGen vocabulary:\n{noatgen_info}\n"
                f"Emotional key: {emotional_key}\n"
                f"Scene data:\n{scene_desc}\n"
                "Requirements:\n"
                f"1. Return exactly {piece} descriptions\n"
                '2. Format: {{"Result": ["description1", "description2"]}}\n'
                "3. Each description should follow NotaGen format\n"
                "4. Only JSON, no explanations\n"
            )
        else:
            # SongGeneration mode: build prompt with valid vocabulary
            vocab = self.song_vocab.get("descriptions", {}) if self.song_vocab else {}
            genders = ", ".join(vocab.get("gender", ["female", "male"]))
            emotions = ", ".join(vocab.get("emotion", []))
            genres = ", ".join(vocab.get("genre", []))
            timbres = ", ".join(vocab.get("timbre", []))
            instruments = ", ".join(vocab.get("instrument", []))
            bpm_range = vocab.get("bpm_range", [60, 200])

            desc_system = (
                "You are a professional music analysis assistant with deep knowledge of music attributes. "
                f"Generate {piece} music description strings, each selecting one value from each category:\n"
                f"  [gender]: {genders}\n"
                f"  [emotion]: {emotions}\n"
                f"  [genre]: {genres}\n"
                f"  [timbre]: {timbres}\n"
                f"  [instrument]: {instruments}\n"
                f"  [bpm]: integer in [{bpm_range[0]}, {bpm_range[1]}]\n"
                "Output format per description: 'gender, emotion, genre, timbre, instrument, the bpm is N.'\n"
                "Select values that match the scene emotion and blueprint."
            )
            desc_prompt = (
                f"Based on the scene data, generate {piece} music descriptions.\n"
                f"Emotional key: {emotional_key}\n"
                "Requirements:\n"
                f"1. Return exactly {piece} descriptions\n"
                '2. Format: {{"Result": ["desc1", "desc2"]}}\n'
                "3. Each description: 'gender, emotion, genre, timbre, instrument, the bpm is N.'\n"
                "4. All values must come from the valid vocabulary listed above\n"
                "5. Only JSON, no explanations\n"
                f"Scene data:\n{scene_desc}\n"
            )

        messages = [
            SystemMessage(content=desc_system),
            HumanMessage(content=desc_prompt),
        ]
        response = await self.exper.ainvoke(messages)
        desc_list = extract_field_from_response(response, "Result")
        if not isinstance(desc_list, list):
            desc_list = [str(desc_list)]

        # Post-LLM vocabulary validation (SongGeneration mode only)
        if model_type != "NotaGen":
            desc_list = [self._validate_description(d) for d in desc_list]

        # Pad if fewer than piece
        while len(desc_list) < piece:
            desc_list.append("female, emotional, pop, varies, synthesizer and piano, the bpm is 120.")

        content = f"composer_descriptions: {json.dumps(desc_list, ensure_ascii=False)}"
        _print_with_indent("generate_descriptions:", content, tab_count=2)

        return {
            "global_messages": [AIMessage(content=content)],
            "system_prompt_messages": system_prompt,
            "composer_message": [{"role": "assistant", "content": content}],
            "composer_flag": True,
            "composer_descriptions_result": desc_list,
            "state": "observation",
            "tools": "None",
            "current_iteration": 1,
        }

    async def _observation_node(self, state: "ComposerAgent.Graph") -> Dict[str, Any]:
        observation_content = ""
        current_iteration = state.get("current_iteration", 0)
        _complete = state.get("complete", False)
        if current_iteration > state.get("max_iterations", self.max_iterations):
            _complete = True

        if state.get("composer_flag", False):
            desc_result = state.get("composer_descriptions_result", [])
            observation_content = f"descriptions_list: {json.dumps(desc_result, ensure_ascii=False)}"
        else:
            observation_content = "No tool was executed."

        system_prompt = state.get("system_prompt_messages", "")
        _print_with_indent(f"observation{state.get('current_iteration', 0)}:", observation_content, tab_count=2)
        system_prompt += f"observation{state.get('current_iteration', 0)}:{observation_content}\n"

        return {
            "global_messages": [AIMessage(content=observation_content)],
            "system_prompt_messages": system_prompt,
            "observation_message": [{"role": "assistant", "content": observation_content}],
            "composer_flag": False,
            "composer_descriptions_result": state.get("composer_descriptions_result", []),
            "state": "think",
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": state.get("reflection_flag", False),
            "reflection_result": state.get("reflection_result", {}),
            "reflection_count": state.get("reflection_count", 0),
        }

    async def _reflect_node(self, state: "ComposerAgent.Graph") -> Dict[str, Any]:
        _print_with_indent("", "Reflection node started...", tab_count=1)

        task_description = state.get("tasks", "")
        desc_result = state.get("composer_descriptions_result", [])
        observation = f"descriptions_list: {json.dumps(desc_result, ensure_ascii=False)}"

        history_reflections = ""
        if self.reflection_memory:
            history_reflections = self.reflection_memory.get_summary()

        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="composer",
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
            "composer_descriptions_result": state.get("composer_descriptions_result", []),
            "state": next_state,
            "current_iteration": 1,
            "complete": _complete,
            "reflection_flag": True,
            "reflection_result": reflection_result,
            "reflection_count": reflection_count,
        }

    async def _final_node(self, state: "ComposerAgent.Graph") -> Dict[str, Any]:
        desc_result = state.get("composer_descriptions_result", [])
        final_answer = {"descriptions_list": desc_result}

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

    def _route_after_think(self, state: "ComposerAgent.Graph") -> str:
        if state.get("current_iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "final"
        elif state.get("complete", False):
            return "final"
        return "action"

    def _route_after_action(self, state: "ComposerAgent.Graph") -> str:
        tool = state.get("tools", "none")
        if tool in ("None", "none", ""):
            return "observation"
        elif tool == "generate_descriptions":
            return "generate_descriptions"
        return "observation"

    def _route_after_observation(self, state: "ComposerAgent.Graph") -> str:
        desc_result = state.get("composer_descriptions_result", [])
        has_result = desc_result and len(desc_result) > 0

        if state.get("complete", False) or has_result:
            return "reflect"
        elif state.get("current_iteration", 1) >= state.get("max_iterations", self.max_iterations):
            return "reflect"
        return "think"

    def _route_after_reflect(self, state: "ComposerAgent.Graph") -> str:
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
            "composer_descriptions_result": result.get("composer_descriptions_result", []),
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
            "composer_message": [],
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
            "composer_flag": False,
            "composer_descriptions_result": [],
            "reflection_flag": False,
            "reflection_result": {},
            "reflection_count": 0,
        }


if __name__ == "__main__":
    import asyncio
    load_dotenv()
    agent = ComposerAgent(max_iterations=10)
    _json_scene = [
        {
            "时间段": "0s-12s",
            "主体声音内容": "教室的午后，阳光在你的睫毛上跳舞。我假装看书，笔尖却在纸上，一遍遍临摹你的侧影。",
            "主体声音风格": "单人内心独白，音色温柔、略带迟疑，充满少年青涩的暗恋感",
            "环境声音内容": "笔尖划过纸张的沙沙声，远处隐约的课堂讲课声，窗外的蝉鸣",
            "环境声音风格": "氛围静谧、慵懒",
        },
    ]
    blueprint = {"model": "SongGeneration", "emotional_key": "romantic"}

    async def main():
        result = await agent.ainvoke(
            task="Generate music descriptions based on the scene data",
            json_scene=_json_scene,
            piece=2,
            blueprint=blueprint,
        )
        print(f"\ndescriptions: {result.get('composer_descriptions_result', [])}")

    asyncio.run(main())
