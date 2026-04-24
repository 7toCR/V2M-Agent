"""
Music Generation Supervisor (Team 3)

Coordinator agent that executes the three-step music generation workflow:
1. Strategic Planning — generate Musical Blueprint B from scene data
2. Creative Execution — dispatch parallel experts (Lyricist, Composer, Stylist)
3. Cross-Expert Alignment — invoke independent MusicVerifier, apply
   Collaborative Modification Operator M when verification fails

Follows Think-Act-Observe-Reflect iterative loop via LangGraph.

Paper reference (Section 3.3.3):
    B  = StrategicPlanning(S)
    E  = ParallelExperts(B, S)
    B* = M(E_lyric, E_comp, E_style | S)  when verifier rejects
"""

import os
import sys
import re
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages, StateGraph, START, END
from typing_extensions import TypedDict
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM3_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TEAM3_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tools.tools import _print_with_indent as _base_print

logger = logging.getLogger(__name__)


def _log(msg: str, indent: int = 1):
    _base_print("", msg, tab_count=indent)


# ---------------------------------------------------------------------------
# Import supervisor profile
# ---------------------------------------------------------------------------
try:
    from Team3.AgentProfile.music_generation_supervisor_profile import (
        MUSIC_GENERATION_SUPERVISOR_PROFILE,
    )
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import MUSIC_GENERATION_SUPERVISOR_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    MUSIC_GENERATION_SUPERVISOR_PROFILE = None

# ---------------------------------------------------------------------------
# Import reflection modules
# ---------------------------------------------------------------------------
try:
    from Team2.Expert.reflection_memory import get_reflection_memory
    from Team2.Expert.reflection_agent_profile import (
        build_reflection_prompt,
        parse_reflection_result,
    )
    REFLECTION_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import reflection modules: {e}")
    REFLECTION_IMPORT_SUCCESS = False
    get_reflection_memory = None
    build_reflection_prompt = None
    parse_reflection_result = None

# ---------------------------------------------------------------------------
# Import expert agents
# ---------------------------------------------------------------------------
try:
    from Team3.Expert.lyricist import LyricistAgent
    from Team3.Expert.composer import ComposerAgent
    from Team3.Expert.stylist import StylistAgent
    EXPERTS_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import expert agents: {e}")
    EXPERTS_IMPORT_SUCCESS = False
    LyricistAgent = None
    ComposerAgent = None
    StylistAgent = None

# ---------------------------------------------------------------------------
# Import verifier
# ---------------------------------------------------------------------------
try:
    from Team3.verifier.music_verifier import MusicVerifier
    VERIFIER_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import MusicVerifier: {e}")
    VERIFIER_IMPORT_SUCCESS = False
    MusicVerifier = None

# ---------------------------------------------------------------------------
# Import task creator (Algorithm 1)
# ---------------------------------------------------------------------------
try:
    from task.task_create import AsyncTaskCreator
    TASK_CREATOR_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import AsyncTaskCreator: {e}")
    TASK_CREATOR_IMPORT_SUCCESS = False
    AsyncTaskCreator = None

# Import expert AgentProfiles for Algorithm 1 task generation
try:
    from Team3.AgentProfile.lyricist_agent_profile import LYRICIST_AGENT_PROFILE
    from Team3.AgentProfile.composer_agent_profile import COMPOSER_AGENT_PROFILE
    from Team3.AgentProfile.stylist_agent_profile import STYLIST_AGENT_PROFILE
    EXPERT_PROFILES_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import expert AgentProfiles: {e}")
    EXPERT_PROFILES_IMPORT_SUCCESS = False
    LYRICIST_AGENT_PROFILE = None
    COMPOSER_AGENT_PROFILE = None
    STYLIST_AGENT_PROFILE = None

# ---------------------------------------------------------------------------
# Vocabulary loading (for blueprint validation)
# ---------------------------------------------------------------------------

def _load_songgeneration_vocab() -> Dict[str, Any]:
    for candidate in [
        os.path.join(TEAM3_DIR, "pre-traing", "songgeneration.json"),
        os.path.join(TEAM3_DIR, "AgentProfile", "pre-traing", "songgeneration.json"),
    ]:
        path = os.path.normpath(candidate)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _load_noatgen_vocab() -> Dict[str, Any]:
    for candidate in [
        os.path.join(TEAM3_DIR, "pre-traing", "noatgen.json"),
        os.path.join(TEAM3_DIR, "AgentProfile", "pre-traing", "noatgen.json"),
    ]:
        path = os.path.normpath(candidate)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# MusicGenerationSupervisor
# ═════════════════════════════════════════════════════════════���═════════════

class MusicGenerationSupervisor:
    """
    Team 3 Supervisor: coordinates music prompt generation.

    Responsibilities:
    1. Strategic Planning — analyse scene S → Musical Blueprint B
    2. Parallel expert dispatch (Lyricist, Composer, Stylist)
    3. Assemble expert results into 4-field prompt dicts
    4. Invoke independent MusicVerifier
    5. Collaborative Modification Operator M on failure (max 2 rounds)
    6. Output pop_prompt_result and save to lyric.jsonl
    """

    class Graph(TypedDict):
        # Messages
        global_messages: Annotated[List[dict], add_messages]
        system_prompt_messages: str
        think_message: Annotated[List[dict], add_messages]
        action_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]

        # Input from Team 1 / Team 2
        task_packet: Dict[str, Any]
        json_scene: List[Dict[str, Any]]
        user_requirement: str

        # Musical Blueprint
        blueprint: Dict[str, Any]
        blueprint_generated: bool

        # Expert results
        lyricist_idx_result: List[str]
        lyricist_lyric_result: List[str]
        composer_descriptions_result: List[str]
        stylist_audio_type_result: List[str]
        experts_complete: bool

        # Generated expert tasks (Algorithm 1)
        generated_lyricist_task: str
        generated_composer_task: str
        generated_stylist_task: str
        expert_tasks_generated: bool

        # Assembled prompts
        pop_prompt_result: List[Dict[str, str]]
        prompts_assembled: bool

        # Verifier
        verification_result: Dict[str, Any]
        verification_passed: bool
        verifier_invoked: bool

        # Control flow
        state: str
        tools: str
        current_iteration: int
        complete: bool
        final_answer: str
        piece: int

        # Reflection
        reflection_result: Dict[str, Any]
        reflection_count: int
        max_reflect_retries: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, llm=None):
        self.profile = MUSIC_GENERATION_SUPERVISOR_PROFILE

        # Load configuration
        config_path = os.path.join(TEAM3_DIR, "config_music_generation.json")
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        # Reasoning LLM (supervisor decisions)
        if llm is not None:
            self.model = llm
        else:
            model_cfg = self.config.get("model", {})
            api_key = os.environ.get(model_cfg.get("api_key_env", "MCP_API_KEY"), "")
            self.model = ChatOpenAI(
                model=model_cfg.get("name", "qwen3-max"),
                openai_api_base=model_cfg.get("base_url",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                openai_api_key=api_key,
                temperature=model_cfg.get("temperature", 0.7),
                max_tokens=model_cfg.get("max_tokens", 8192),
            )

        # Expert-generation LLM (passed to experts)
        expert_cfg = self.config.get("expert", {})
        expert_key = os.environ.get(expert_cfg.get("api_key_env", "DASHSCOPE_API_KEY"), "")
        if expert_key:
            self.expert_llm = ChatOpenAI(
                model=expert_cfg.get("name", "deepseek-v3.2"),
                openai_api_base=expert_cfg.get("base_url",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                openai_api_key=expert_key,
                temperature=expert_cfg.get("temperature", 0.7),
                max_tokens=expert_cfg.get("max_tokens", 8192),
            )
        else:
            self.expert_llm = None

        sup_cfg = self.config.get("supervisor", {})
        self.max_reflect_retries = sup_cfg.get("max_reflect_retries", 2)
        self.default_piece = sup_cfg.get("default_piece", 2)
        self.default_model = sup_cfg.get("default_model", "SongGeneration")

        # Load vocabulary for blueprint validation
        self.song_vocab = _load_songgeneration_vocab()
        self.noatgen_vocab = _load_noatgen_vocab()

        # Initialise expert agents
        self.agents_available = False
        if EXPERTS_IMPORT_SUCCESS:
            try:
                self.lyricist = LyricistAgent(llm=self.expert_llm)
                self.composer = ComposerAgent(llm=self.expert_llm)
                self.stylist = StylistAgent(llm=self.expert_llm)
                self.agents_available = True
                _log("[MusicSupervisor] Expert agents initialised")
            except Exception as e:
                _log(f"[MusicSupervisor] Failed to initialise expert agents: {e}")

        # Initialise verifier
        self.verifier = None
        if VERIFIER_IMPORT_SUCCESS and MusicVerifier is not None:
            try:
                self.verifier = MusicVerifier(llm=self.model)
                _log("[MusicSupervisor] MusicVerifier initialised")
            except Exception as e:
                _log(f"[MusicSupervisor] Failed to initialise MusicVerifier: {e}")

        # Reflection memory
        if REFLECTION_IMPORT_SUCCESS and get_reflection_memory:
            self.reflection_memory = get_reflection_memory("music_supervisor", CURRENT_DIR)
        else:
            self.reflection_memory = None

        self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(MusicGenerationSupervisor.Graph)

        builder.add_node("init", self._init_node)
        builder.add_node("think", self._think_node)
        builder.add_node("act", self._act_node)
        builder.add_node("strategic_planning", self._strategic_planning_node)
        builder.add_node("generate_expert_tasks", self._generate_expert_tasks_node)
        builder.add_node("dispatch_experts", self._dispatch_experts_node)
        builder.add_node("assemble_prompts", self._assemble_prompts_node)
        builder.add_node("invoke_verifier", self._invoke_verifier_node)
        builder.add_node("observation", self._observation_node)
        builder.add_node("reflect", self._reflect_node)
        builder.add_node("final", self._final_node)

        builder.add_edge(START, "init")
        builder.add_edge("init", "think")

        builder.add_conditional_edges(
            "think", self._route_after_think, {"act": "act", "final": "final"}
        )
        builder.add_conditional_edges(
            "act", self._route_after_act, {
                "strategic_planning": "strategic_planning",
                "generate_expert_tasks": "generate_expert_tasks",
                "dispatch_experts": "dispatch_experts",
                "assemble_prompts": "assemble_prompts",
                "invoke_verifier": "invoke_verifier",
                "observation": "observation",
            }
        )

        for tool_node in ("strategic_planning", "generate_expert_tasks",
                          "dispatch_experts",
                          "assemble_prompts", "invoke_verifier"):
            builder.add_edge(tool_node, "observation")

        builder.add_conditional_edges(
            "observation", self._route_after_observation,
            {"think": "think", "reflect": "reflect", "final": "final"}
        )
        builder.add_conditional_edges(
            "reflect", self._route_after_reflect,
            {"think": "think", "final": "final"}
        )
        builder.add_edge("final", END)

        self.graph = builder.compile()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_think(self, state: "MusicGenerationSupervisor.Graph") -> str:
        return "final" if state.get("state") == "final" else "act"

    def _route_after_act(self, state: "MusicGenerationSupervisor.Graph") -> str:
        tool = state.get("tools", "none")
        mapping = {
            "strategic_planning": "strategic_planning",
            "generate_expert_tasks": "generate_expert_tasks",
            "dispatch_parallel_experts": "dispatch_experts",
            "assemble_prompts": "assemble_prompts",
            "invoke_verifier": "invoke_verifier",
        }
        return mapping.get(tool, "observation")

    def _route_after_observation(self, state: "MusicGenerationSupervisor.Graph") -> str:
        s = state.get("state", "think")
        if s in ("reflect", "final"):
            return s
        return "think"

    def _route_after_reflect(self, state: "MusicGenerationSupervisor.Graph") -> str:
        return "final" if state.get("state") == "final" else "think"

    # ------------------------------------------------------------------
    # Init node
    # ------------------------------------------------------------------

    def _init_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        """Parse task_packet and initialise supervisor state."""
        _log("[MusicSupervisor] Initialising", indent=1)

        packet = state.get("task_packet", {})
        json_scene = packet.get("json_scene", state.get("json_scene", []))
        user_req = packet.get("user_requirement", state.get("user_requirement", ""))
        piece = packet.get("piece", state.get("piece", self.default_piece))

        scene_summary = ""
        if json_scene:
            try:
                scene_summary = json.dumps(json_scene[:2], ensure_ascii=False)[:300]
            except Exception:
                scene_summary = str(json_scene[:2])[:300]

        _log(f"[MusicSupervisor] piece={piece}, scenes={len(json_scene)}", indent=2)

        sys_prompt = (
            "You are the Music Generation Supervisor for Team 3.\n"
            "You coordinate the three-step music generation workflow:\n"
            "1. Strategic Planning → Musical Blueprint\n"
            "2. Creative Execution → Parallel experts (Lyricist, Composer, Stylist)\n"
            "3. Cross-Expert Alignment → Verification and collaborative modification\n"
            f"Scene count: {len(json_scene)}\n"
            f"Pieces to generate: {piece}\n"
            f"User requirement: {user_req}\n"
        )

        return {
            "system_prompt_messages": sys_prompt,
            "json_scene": json_scene,
            "user_requirement": user_req,
            "piece": piece,
            "state": "think",
            "current_iteration": 0,
        }

    # ------------------------------------------------------------------
    # Think node (deterministic sequential dispatch)
    # ------------------------------------------------------------------

    def _think_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        """Deterministic think: sequential stage dispatch."""
        iteration = state.get("current_iteration", 0) + 1
        _log(f"[MusicSupervisor] Think (iteration {iteration})", indent=1)

        if not state.get("blueprint_generated"):
            _log("[MusicSupervisor] → Strategic Planning", indent=2)
            return {"state": "act", "tools": "strategic_planning",
                    "current_iteration": iteration}

        if not state.get("expert_tasks_generated"):
            _log("[MusicSupervisor] → Generate Expert Tasks (Algorithm 1)", indent=2)
            return {"state": "act", "tools": "generate_expert_tasks",
                    "current_iteration": iteration}

        if not state.get("experts_complete"):
            _log("[MusicSupervisor] → Dispatch parallel experts", indent=2)
            return {"state": "act", "tools": "dispatch_parallel_experts",
                    "current_iteration": iteration}

        if not state.get("prompts_assembled"):
            _log("[MusicSupervisor] → Assemble prompts", indent=2)
            return {"state": "act", "tools": "assemble_prompts",
                    "current_iteration": iteration}

        if not state.get("verifier_invoked"):
            _log("[MusicSupervisor] → Invoke verifier", indent=2)
            return {"state": "act", "tools": "invoke_verifier",
                    "current_iteration": iteration}

        _log("[MusicSupervisor] → All steps complete, proceeding to final", indent=2)
        return {"state": "final", "tools": "none", "current_iteration": iteration}

    # ------------------------------------------------------------------
    # Act node
    # ------------------------------------------------------------------

    def _act_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        tool = state.get("tools", "none")
        _log(f"[MusicSupervisor] Act → dispatching: {tool}", indent=2)
        return {"state": "execute"}

    # ------------------------------------------------------------------
    # Observation node
    # ------------------------------------------------------------------

    def _observation_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        _log("[MusicSupervisor] Observation", indent=1)

        if state.get("verifier_invoked") and not state.get("verification_passed"):
            issues = state.get("verification_result", {}).get("issues", [])
            count = state.get("reflection_count", 0)
            if issues and count < state.get("max_reflect_retries", 2):
                _log(f"[MusicSupervisor] Verification failed ({len(issues)} issues) → reflect", indent=2)
                return {"state": "reflect"}
            elif issues:
                _log("[MusicSupervisor] Verification failed, max retries reached → final", indent=2)
                return {"state": "final"}

        return {"state": "think"}

    # ══════════════════════════════════════════════════════════════════
    # Tool: Strategic Planning
    # ══════════════════════════════════════════════════════════════════

    async def _strategic_planning_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        """
        Step 1: Generate Musical Blueprint B from scene data S.

        Blueprint fields:
            model          — "SongGeneration" or "NotaGen"
            lyric_style    — "vocal" or "bgm"
            emotional_key  — dominant emotion (e.g. "romantic", "sad")
            variant        — "Ours-Vocal" or "Ours-BGM"
            language       — "zh" or "en" or "mixed"
            (NotaGen only) period, composer, instrument_category
        """
        _log("[MusicSupervisor] Strategic Planning — generating Musical Blueprint", indent=1)

        json_scene = state.get("json_scene", [])
        user_req = state.get("user_requirement", "")

        # Prepare scene summary for the LLM
        scene_text = json.dumps(json_scene, ensure_ascii=False, indent=2)[:4000]

        # Valid emotion keys from songgeneration.json
        valid_emotions = self.song_vocab.get("descriptions", {}).get("emotion", [])
        valid_audio_types = self.song_vocab.get("audio_type", [])

        # NotaGen vocabulary summary
        noatgen_periods = [p["name"] for p in self.noatgen_vocab.get("periods", [])]

        prompt = f"""You are a Musical Blueprint planner. Based on the scene data and user requirement,
generate a Musical Blueprint that will guide three expert agents (Lyricist, Composer, Stylist).

## Scene Data
{scene_text}

## User Requirement
{user_req or "(no explicit requirement)"}

## Instructions
Analyze the scene data and decide:
1. **model**: Choose "SongGeneration" for modern pop/vocal music, or "NotaGen" for classical instrumental notation.
   Default to "SongGeneration" unless the scene explicitly calls for classical/orchestral music.
2. **lyric_style**: "vocal" if lyrics with singing are needed, "bgm" if instrumental background music only.
3. **emotional_key**: The dominant emotion that should unify all components.
   Valid emotions: {json.dumps(valid_emotions)}
4. **language**: "zh" for Chinese lyrics, "en" for English, "mixed" for bilingual.
   Detect from scene content language.
5. **variant**: "Ours-Vocal" if model=SongGeneration + lyric_style=vocal, else "Ours-BGM".

If model is "NotaGen", also provide:
- **period**: one of {json.dumps(noatgen_periods)}
- **composer**: a composer from that period (check noatgen.json)
- **instrument_category**: from the composer's available instruments

Return ONLY a JSON object with these fields, no extra text.

Example (SongGeneration):
{{
    "model": "SongGeneration",
    "lyric_style": "vocal",
    "emotional_key": "romantic",
    "variant": "Ours-Vocal",
    "language": "zh"
}}

Example (NotaGen):
{{
    "model": "NotaGen",
    "lyric_style": "bgm",
    "emotional_key": "melancholic",
    "variant": "Ours-BGM",
    "language": "en",
    "period": "Romantic",
    "composer": "Chopin, Frederic",
    "instrument_category": "Keyboard"
}}
"""

        try:
            messages = [SystemMessage(content=prompt)]
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, "content") else str(response)

            # Strip markdown fences
            text = response_text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            # Parse JSON
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                blueprint = json.loads(json_match.group())
            else:
                blueprint = json.loads(text)

            # Validate and set defaults
            if blueprint.get("model") not in ("SongGeneration", "NotaGen"):
                blueprint["model"] = self.default_model
            if blueprint.get("lyric_style") not in ("vocal", "bgm"):
                blueprint["lyric_style"] = "vocal" if blueprint["model"] == "SongGeneration" else "bgm"
            # SongGeneration always needs full lyrics — enforce vocal style
            if blueprint.get("model") == "SongGeneration" and blueprint.get("lyric_style") != "vocal":
                blueprint["lyric_style"] = "vocal"
            if not blueprint.get("emotional_key"):
                blueprint["emotional_key"] = "emotional"
            scene_mood_text = json.dumps(json_scene, ensure_ascii=False).lower()
            sad_markers = (
                "sad", "melancholic", "lonely", "sorrow", "grief",
                "悲伤", "忧伤", "孤寂", "哀愁", "凄",
            )
            if blueprint.get("emotional_key") == "melancholic" and not any(
                marker in scene_mood_text for marker in sad_markers
            ):
                blueprint["emotional_key"] = "emotional"
            if not blueprint.get("language"):
                blueprint["language"] = "zh"
            if not blueprint.get("variant"):
                blueprint["variant"] = (
                    "Ours-Vocal" if blueprint["model"] == "SongGeneration"
                    and blueprint["lyric_style"] == "vocal"
                    else "Ours-BGM"
                )

            # NotaGen-specific validation
            if blueprint["model"] == "NotaGen":
                if blueprint.get("period") not in noatgen_periods:
                    blueprint["period"] = "Romantic"
                # Validate composer exists in the selected period
                period_data = next(
                    (p for p in self.noatgen_vocab.get("periods", [])
                     if p["name"] == blueprint["period"]),
                    None,
                )
                if period_data:
                    composer_names = [c["name"] for c in period_data.get("composers", [])]
                    if blueprint.get("composer") not in composer_names:
                        blueprint["composer"] = composer_names[0] if composer_names else "Unknown"
                    # Validate instrument
                    composer_data = next(
                        (c for c in period_data.get("composers", [])
                         if c["name"] == blueprint["composer"]),
                        None,
                    )
                    if composer_data:
                        valid_inst = composer_data.get("instruments", [])
                        if blueprint.get("instrument_category") not in valid_inst:
                            blueprint["instrument_category"] = valid_inst[0] if valid_inst else "Keyboard"

            _log(f"[MusicSupervisor] Blueprint: {json.dumps(blueprint, ensure_ascii=False)}", indent=2)

        except Exception as e:
            _log(f"[MusicSupervisor] Strategic Planning LLM failed: {e}, using defaults", indent=2)
            blueprint = {
                "model": self.default_model,
                "lyric_style": "vocal",
                "emotional_key": "emotional",
                "variant": "Ours-Vocal",
                "language": "zh",
            }

        return {
            "blueprint": blueprint,
            "blueprint_generated": True,
        }

    # ══════════════════════════════════════════════════════════════════
    # Tool: Generate Expert Tasks (Algorithm 1)
    # ══════════════════════════════════════════════════════════════════

    async def _generate_expert_tasks_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        """
        Algorithm 1: DetermineNeed → ReadProfile → Match → ExtractInstruction → GenerateConstraints.
        Generates per-expert task prompts from AgentProfiles via AsyncTaskCreator.
        """
        _log("[MusicSupervisor] Generating expert tasks (Algorithm 1)", indent=1)

        blueprint = state.get("blueprint", {})
        user_req = state.get("user_requirement", "")

        updates: Dict[str, Any] = {
            "expert_tasks_generated": True,
            "generated_lyricist_task": "",
            "generated_composer_task": "",
            "generated_stylist_task": "",
        }

        # DetermineNeed: all 3 experts are always needed for music generation
        # (unlike Team2 where modality presence gates expert selection)
        try:
            profiles_to_use = []
            if EXPERT_PROFILES_IMPORT_SUCCESS:
                if LYRICIST_AGENT_PROFILE is not None:
                    profiles_to_use.append(("lyricist", LYRICIST_AGENT_PROFILE))
                if COMPOSER_AGENT_PROFILE is not None:
                    profiles_to_use.append(("composer", COMPOSER_AGENT_PROFILE))
                if STYLIST_AGENT_PROFILE is not None:
                    profiles_to_use.append(("stylist", STYLIST_AGENT_PROFILE))

            if profiles_to_use and TASK_CREATOR_IMPORT_SUCCESS and AsyncTaskCreator is not None:
                json_scene_data = state.get("json_scene", [])[:2] or [
                    {"关键帧": "0s", "背景": "generic background", "主体": "generic subject", "心情": "neutral"}
                ]

                task_creator = AsyncTaskCreator(llm=self.model)
                profiles_list = [p for _, p in profiles_to_use]

                task_strings = await task_creator.create_tasks_for_all_agents(
                    profiles=profiles_list,
                    user_requirement=f"{user_req}\nMusical Blueprint: {json.dumps(blueprint, ensure_ascii=False)}",
                    json_scene_data=json_scene_data,
                    num_tasks_per_agent=1,
                    save_to_file=False,
                )

                idx = 0
                for name, _ in profiles_to_use:
                    if idx < len(task_strings):
                        updates[f"generated_{name}_task"] = task_strings[idx]
                        _log(f"[MusicSupervisor] Generated task for {name}: {task_strings[idx][:80]}...", indent=2)
                        idx += 1

                _log(f"[MusicSupervisor] Algorithm 1 task generation complete ({len(profiles_to_use)} experts)", indent=2)
            else:
                # Fallback: simple default tasks
                _log("[MusicSupervisor] AsyncTaskCreator unavailable, using default tasks", indent=2)
                piece = state.get("piece", self.default_piece)
                updates["generated_lyricist_task"] = (
                    f"Generate song titles and lyrics for {piece} piece(s). "
                    f"User requirement: {user_req}. "
                    f"Blueprint: {json.dumps(blueprint, ensure_ascii=False)}"
                )
                updates["generated_composer_task"] = (
                    f"Generate music descriptions (6-field format) for {piece} piece(s). "
                    f"User requirement: {user_req}. "
                    f"Blueprint: {json.dumps(blueprint, ensure_ascii=False)}"
                )
                updates["generated_stylist_task"] = (
                    f"Select audio type for {piece} piece(s). "
                    f"User requirement: {user_req}. "
                    f"Blueprint: {json.dumps(blueprint, ensure_ascii=False)}"
                )

        except Exception as e:
            _log(f"[MusicSupervisor] Algorithm 1 task generation failed: {e}, using defaults", indent=2)
            piece = state.get("piece", self.default_piece)
            bp_str = json.dumps(blueprint, ensure_ascii=False)
            updates["generated_lyricist_task"] = f"Generate titles and lyrics for {piece} pieces. Blueprint: {bp_str}"
            updates["generated_composer_task"] = f"Generate descriptions for {piece} pieces. Blueprint: {bp_str}"
            updates["generated_stylist_task"] = f"Select audio types for {piece} pieces. Blueprint: {bp_str}"

        return updates

    # ══════════════════════════════════════════════════════════════════
    # Tool: Dispatch Parallel Experts
    # ══════════════════════════════════════════════════════════════════

    async def _dispatch_experts_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        """
        Step 2: Creative Execution — dispatch Lyricist, Composer, Stylist in parallel.
        Each expert runs its own Think-Act-Observe-Reflect loop internally.
        """
        _log("[MusicSupervisor] Dispatching parallel experts", indent=1)

        json_scene = state.get("json_scene", [])
        blueprint = state.get("blueprint", {})
        piece = state.get("piece", self.default_piece)
        user_req = state.get("user_requirement", "")

        updates: Dict[str, Any] = {
            "experts_complete": True,
            "lyricist_idx_result": [],
            "lyricist_lyric_result": [],
            "composer_descriptions_result": [],
            "stylist_audio_type_result": [],
        }

        if not self.agents_available:
            _log("[MusicSupervisor] Expert agents not available", indent=2)
            return updates

        # Use Algorithm 1 generated tasks if available, otherwise fallback
        default_task = (
            f"Generate music components for {piece} piece(s).\n"
            f"User requirement: {user_req}\n"
            f"Musical Blueprint: {json.dumps(blueprint, ensure_ascii=False)}"
        )
        lyricist_task = state.get("generated_lyricist_task") or default_task
        composer_task = state.get("generated_composer_task") or default_task
        stylist_task = state.get("generated_stylist_task") or default_task

        tasks = [
            self.lyricist.ainvoke(
                task=lyricist_task, json_scene=json_scene,
                piece=piece, blueprint=blueprint,
            ),
            self.composer.ainvoke(
                task=composer_task, json_scene=json_scene,
                piece=piece, blueprint=blueprint,
            ),
            self.stylist.ainvoke(
                task=stylist_task, json_scene=json_scene,
                piece=piece, blueprint=blueprint,
            ),
        ]

        _log("[MusicSupervisor] Launching 3 experts in parallel: Lyricist, Composer, Stylist", indent=2)

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300.0,
            )

            labels = ["Lyricist", "Composer", "Stylist"]
            for label, result in zip(labels, results):
                if isinstance(result, Exception):
                    _log(f"[MusicSupervisor] Expert '{label}' failed: {result}", indent=2)
                else:
                    if label == "Lyricist":
                        updates["lyricist_idx_result"] = result.get("lyricist_idx_result", [])
                        updates["lyricist_lyric_result"] = result.get("lyricist_lyric_result", [])
                        _log(
                            f"[MusicSupervisor] Lyricist complete: "
                            f"{len(updates['lyricist_idx_result'])} idx, "
                            f"{len(updates['lyricist_lyric_result'])} lyrics",
                            indent=2,
                        )
                    elif label == "Composer":
                        updates["composer_descriptions_result"] = result.get(
                            "composer_descriptions_result", []
                        )
                        _log(
                            f"[MusicSupervisor] Composer complete: "
                            f"{len(updates['composer_descriptions_result'])} descriptions",
                            indent=2,
                        )
                    elif label == "Stylist":
                        updates["stylist_audio_type_result"] = result.get(
                            "stylist_audio_type_result", []
                        )
                        _log(
                            f"[MusicSupervisor] Stylist complete: "
                            f"{len(updates['stylist_audio_type_result'])} audio types",
                            indent=2,
                        )

        except asyncio.TimeoutError:
            _log("[MusicSupervisor] Expert execution timed out (>300s)", indent=2)
        except Exception as e:
            _log(f"[MusicSupervisor] Expert execution failed: {e}", indent=2)

        return updates

    # ══════════════════════════════════════════════════════════════════
    # Tool: Assemble Prompts
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_valid_prompt_lyric(lyric: object, lyric_style: str = "vocal") -> bool:
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

    @staticmethod
    def _genre_to_audio_type(descriptions: object) -> Optional[str]:
        parts = [part.strip() for part in str(descriptions or "").split(",")]
        if len(parts) < 3:
            return None

        text = " ".join(parts[2:]).lower()
        if "r&b" in text or "soul" in text:
            return "R&B"
        if "jazz" in text:
            return "Jazz"
        if "folk" in text or "country" in text:
            return "Folk"
        if "reggae" in text:
            return "Reggae"
        if "dance" in text or "electronic" in text or "k-pop" in text:
            return "Dance"
        if "rock" in text or "metal" in text:
            return "Rock"
        if "pop" in text:
            return "Pop"
        return None

    @classmethod
    def _align_audio_type(cls, descriptions: object, audio_type: object) -> str:
        expected = cls._genre_to_audio_type(descriptions)
        current = str(audio_type or "").strip() or "Auto"
        if expected and current != expected:
            return expected
        return current

    def _assemble_prompts_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        """
        Zip expert results into 4-field prompt dicts.

        Each prompt: {idx, gt_lyric, descriptions, auto_prompt_audio_type}
        """
        _log("[MusicSupervisor] Assembling prompts from expert results", indent=1)

        idx_list = state.get("lyricist_idx_result", [])
        lyric_list = state.get("lyricist_lyric_result", [])
        desc_list = state.get("composer_descriptions_result", [])
        audio_type_list = state.get("stylist_audio_type_result", [])
        piece = state.get("piece", self.default_piece)
        lyric_style = str(state.get("blueprint", {}).get("lyric_style", "vocal"))

        # Determine count — use the minimum length across all results,
        # but at most `piece`
        counts = [
            len(lst) for lst in (idx_list, lyric_list, desc_list, audio_type_list)
            if lst
        ]
        n = min(counts) if counts else 0
        n = min(n, piece)

        pop_prompt_result: List[Dict[str, str]] = []
        for i in range(n):
            lyric = lyric_list[i] if i < len(lyric_list) else ""
            desc = desc_list[i] if i < len(desc_list) else ""
            if not self._is_valid_prompt_lyric(lyric, lyric_style):
                _log(f"[MusicSupervisor] Dropped invalid lyric[{i}]", indent=2)
                continue
            if not str(desc or "").strip():
                _log(f"[MusicSupervisor] Dropped prompt[{i}] with empty descriptions", indent=2)
                continue

            prompt_dict = {
                "idx": idx_list[i] if i < len(idx_list) else "",
                "gt_lyric": lyric,
                "descriptions": desc,
                "auto_prompt_audio_type": self._align_audio_type(
                    desc,
                    audio_type_list[i] if i < len(audio_type_list) else "Auto",
                ),
            }
            pop_prompt_result.append(prompt_dict)

        _log(f"[MusicSupervisor] Assembled {len(pop_prompt_result)} prompt(s)", indent=2)

        if pop_prompt_result:
            _log(f"[MusicSupervisor] Sample prompt[0]: {json.dumps(pop_prompt_result[0], ensure_ascii=False)[:200]}", indent=2)

        return {
            "pop_prompt_result": pop_prompt_result,
            "prompts_assembled": True,
        }

    # ══════════════════════════════════════════════════════════════════
    # Tool: Invoke Verifier
    # ══════════════════════════════════════════════════════════════════

    async def _invoke_verifier_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        """
        Step 3: Cross-Expert Alignment — invoke independent MusicVerifier.
        """
        _log("[MusicSupervisor] Invoking MusicVerifier", indent=1)

        pop_prompt_result = state.get("pop_prompt_result", [])
        blueprint = state.get("blueprint", {})
        verification_result = state.get("verification_result", {})
        if verification_result and verification_result.get("passed") is False:
            _log("[MusicSupervisor] Verification failed; refusing invalid prompts", indent=2)
            pop_prompt_result = []
        json_scene = state.get("json_scene", [])

        if self.verifier is None:
            _log("[MusicSupervisor] MusicVerifier not available, skipping", indent=2)
            return {
                "verifier_invoked": True,
                "verification_passed": True,
                "verification_result": {
                    "passed": True,
                    "issues": [],
                    "warnings": ["Verifier not available"],
                },
            }

        if not pop_prompt_result:
            _log("[MusicSupervisor] No prompts to verify", indent=2)
            return {
                "verifier_invoked": True,
                "verification_passed": False,
                "verification_result": {
                    "passed": False,
                    "issues": [{"type": "ERROR", "severity": "ERROR", "stage": 2,
                                "field": "pop_prompt_result",
                                "message": "No prompts were assembled"}],
                    "warnings": [],
                },
            }

        try:
            result = await self.verifier.ainvoke(
                pop_prompt_result=pop_prompt_result,
                blueprint=blueprint,
                json_scene=json_scene,
            )

            passed = result.get("passed", False)
            issues = result.get("issues", [])
            warnings = result.get("warnings", [])

            _log(
                f"[MusicSupervisor] Verifier result: {'PASSED' if passed else 'FAILED'} "
                f"({len(issues)} issues, {len(warnings)} warnings)",
                indent=2,
            )
            if not passed:
                for issue in issues[:5]:
                    _log(f"[MusicSupervisor]   Issue: {issue}", indent=3)

            return {
                "verifier_invoked": True,
                "verification_passed": passed,
                "verification_result": result,
            }

        except Exception as e:
            _log(f"[MusicSupervisor] Verifier execution failed: {e}", indent=2)
            return {
                "verifier_invoked": True,
                "verification_passed": False,
                "verification_result": {
                    "passed": False,
                    "issues": [{"type": "ERROR", "message": f"Verifier exception: {e}"}],
                    "warnings": [],
                },
            }

    # ══════════════════════════════════════════════════════════════════
    # Reflect node — Collaborative Modification Operator M
    # ══════════════════════════════════════════════════════════════════

    async def _reflect_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        """
        Collaborative Modification Operator:
        B* = M(E_lyric, E_comp, E_style | S)

        1. Vocabulary issues (Stage 1/2): auto-correct deterministically
        2. Semantic issues (Stage 3): LLM-assisted re-planning
        3. Reset verifier flag and selectively re-dispatch
        """
        _log("[MusicSupervisor] Reflect — Collaborative Modification", indent=1)

        verification = state.get("verification_result", {})
        issues = verification.get("issues", [])
        warnings = verification.get("warnings", [])
        reflection_count = state.get("reflection_count", 0) + 1
        blueprint = state.get("blueprint", {})
        pop_prompt_result = list(state.get("pop_prompt_result", []))

        # Categorise issues
        vocab_issues = [i for i in issues if i.get("severity") == "ERROR"
                        and i.get("stage") in ("vocabulary_validation", "structure_validation", 1, 2)]
        semantic_issues = [i for i in issues if i.get("severity") == "ERROR"
                          and i.get("stage") in ("consistency_check", 3)]

        _log(
            f"[MusicSupervisor] Reflection #{reflection_count}: "
            f"{len(vocab_issues)} vocab issues, {len(semantic_issues)} semantic issues",
            indent=2,
        )

        # -- Deterministic auto-correction for vocabulary issues --
        needs_re_dispatch = set()

        valid_audio_types = self.song_vocab.get("audio_type", [])
        valid_descriptions = self.song_vocab.get("descriptions", {})

        for issue in vocab_issues:
            idx = issue.get("prompt_index", 0)
            field = issue.get("field", "")
            if field == "pop_prompt_result":
                needs_re_dispatch.update({"lyricist", "composer", "stylist"})
                continue
            if idx >= len(pop_prompt_result):
                continue

            prompt = pop_prompt_result[idx]

            if field == "auto_prompt_audio_type" or "audio_type" in str(issue.get("message", "")):
                if prompt.get("auto_prompt_audio_type") not in valid_audio_types:
                    prompt["auto_prompt_audio_type"] = "Auto"
                    _log(f"[MusicSupervisor]   Auto-corrected audio_type[{idx}] → Auto", indent=3)

            elif field == "descriptions" or "description" in str(issue.get("message", "")):
                # Re-dispatch composer for this piece
                needs_re_dispatch.add("composer")

            elif field == "gt_lyric" or "lyric" in str(issue.get("message", "")):
                needs_re_dispatch.add("lyricist")

        # -- LLM-assisted reflection for semantic issues --
        if semantic_issues:
            observation = (
                f"Verification failed with {len(semantic_issues)} semantic issues.\n"
                f"Issues: {json.dumps(semantic_issues, ensure_ascii=False, default=str)}\n"
                f"Current Blueprint: {json.dumps(blueprint, ensure_ascii=False)}"
            )

            history_reflections = ""
            if self.reflection_memory:
                history_reflections = self.reflection_memory.get_summary()

            if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
                reflect_prompt = build_reflection_prompt(
                    task_type="music_generation",
                    task_description="Music generation coordination and cross-expert consistency",
                    observation=observation,
                    history_reflections=history_reflections,
                )
            else:
                reflect_prompt = f"""You are a music generation self-reflection expert.
Analyze the verification results and suggest corrections.

Task: Music generation coordination
Observation: {observation}
History: {history_reflections or "None"}

Return JSON:
{{
    "analysis": "Root cause analysis of the semantic conflicts",
    "blueprint_adjustments": {{}},
    "re_dispatch": ["lyricist", "composer", "stylist"],
    "quality": "high/medium/low",
    "should_retry": true
}}

Only include experts in re_dispatch that produced conflicting output.
blueprint_adjustments should contain only fields to change (e.g. {{"emotional_key": "sad"}}).
"""

            try:
                messages = [SystemMessage(content=reflect_prompt)]
                response = await self.model.ainvoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)

                text = response_text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)

                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    reflection_result = json.loads(json_match.group())
                else:
                    reflection_result = {"quality": "low", "should_retry": False}

                # Apply blueprint adjustments
                adjustments = reflection_result.get("blueprint_adjustments", {})
                if adjustments and isinstance(adjustments, dict):
                    for k, v in adjustments.items():
                        if k in blueprint and v:
                            blueprint[k] = v
                            _log(f"[MusicSupervisor]   Blueprint adjustment: {k} → {v}", indent=3)

                # Merge re-dispatch recommendations
                for expert in reflection_result.get("re_dispatch", []):
                    if expert in ("lyricist", "composer", "stylist"):
                        needs_re_dispatch.add(expert)

            except Exception as e:
                _log(f"[MusicSupervisor] Reflection LLM failed: {e}", indent=2)
                reflection_result = {"quality": "low", "should_retry": False}
        else:
            reflection_result = {"quality": "medium", "should_retry": bool(vocab_issues)}

        # Save to memory
        if self.reflection_memory:
            self.reflection_memory.add_reflection(
                task_description="Music generation supervision",
                observation=json.dumps(issues[:3], ensure_ascii=False, default=str)[:200],
                reflection=reflection_result.get("analysis", ""),
                improvement=str(reflection_result.get("blueprint_adjustments", "")),
                quality=reflection_result.get("quality", "low"),
                iterations=reflection_count,
            )

        _log(
            f"[MusicSupervisor] Reflection result: quality={reflection_result.get('quality')}, "
            f"re-dispatch={needs_re_dispatch or 'none'}",
            indent=2,
        )

        # If re-dispatch is needed, reset expert and verifier flags
        if needs_re_dispatch and reflection_count < self.max_reflect_retries:
            return {
                "reflection_result": reflection_result,
                "reflection_count": reflection_count,
                "blueprint": blueprint,
                "pop_prompt_result": pop_prompt_result,
                "experts_complete": False,
                "prompts_assembled": False,
                "verifier_invoked": False,
                "state": "think",
            }

        # If only vocab auto-corrections were applied, re-verify
        if vocab_issues and not needs_re_dispatch and reflection_count < self.max_reflect_retries:
            return {
                "reflection_result": reflection_result,
                "reflection_count": reflection_count,
                "pop_prompt_result": pop_prompt_result,
                "verifier_invoked": False,
                "state": "think",
            }

        # Accept current result
        return {
            "reflection_result": reflection_result,
            "reflection_count": reflection_count,
            "pop_prompt_result": pop_prompt_result,
            "state": "final",
        }

    # ══════════════════════════════════════════════════════════════════
    # Final node
    # ══════════════════════════════════════════════════════════════════

    def _final_node(self, state: "MusicGenerationSupervisor.Graph") -> dict:
        """Output final results and save to lyric.jsonl."""
        _log("[MusicSupervisor] Final node", indent=1)

        pop_prompt_result = state.get("pop_prompt_result", [])
        blueprint = state.get("blueprint", {})
        verification_result = state.get("verification_result", {})

        # Save to lyric.jsonl (same location as pop/pop.py)
        output_path = os.path.join(TEAM3_DIR, "lyric.jsonl")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for prompt in pop_prompt_result:
                    f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
            _log(f"[MusicSupervisor] Results saved to: {output_path}", indent=2)
        except Exception as e:
            _log(f"[MusicSupervisor] Failed to save results: {e}", indent=2)

        # Also try to write to const_pop_prompt (inference pipeline path)
        try:
            from inference.address import const_pop_prompt
            with open(const_pop_prompt, "w", encoding="utf-8") as f:
                for prompt in pop_prompt_result:
                    f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
            _log(f"[MusicSupervisor] Results saved to inference path: {const_pop_prompt}", indent=2)
        except (ImportError, Exception):
            pass

        final_result = {
            "pop_prompt_result": pop_prompt_result,
            "blueprint": blueprint,
            "verification_result": verification_result,
        }
        final_answer = json.dumps(final_result, ensure_ascii=False, separators=(",", ":"))

        _log(f"[MusicSupervisor] Generated {len(pop_prompt_result)} prompt(s)", indent=2)

        return {
            "final_answer": final_answer,
            "pop_prompt_result": pop_prompt_result,
            "blueprint": blueprint,
            "verification_result": verification_result,
            "complete": True,
        }

    # ══════════════════════════════════════════════════════════════════
    # Public entry points
    # ══════════════════════════════════════════════════════════════════

    async def ainvoke_from_packet(self, task_packet: Dict[str, Any]) -> dict:
        """
        Primary entry: receive Team 1 task_packet.

        Args:
            task_packet: {
                "team_name": "Team3",
                "instruction": str,
                "constraints": List[str],
                "json_scene": List[Dict],      # From Team 2
                "user_requirement": str,
                "piece": int,                   # Number of songs (default 2)
            }
        """
        piece = task_packet.get("piece", self.default_piece)

        initial_state: MusicGenerationSupervisor.Graph = {
            # Messages
            "global_messages": [],
            "system_prompt_messages": "",
            "think_message": [],
            "action_message": [],
            "observation_message": [],
            # Input
            "task_packet": task_packet,
            "json_scene": task_packet.get("json_scene", []),
            "user_requirement": task_packet.get("user_requirement", ""),
            # Blueprint
            "blueprint": {},
            "blueprint_generated": False,
            # Expert results
            "lyricist_idx_result": [],
            "lyricist_lyric_result": [],
            "composer_descriptions_result": [],
            "stylist_audio_type_result": [],
            "experts_complete": False,
            # Generated expert tasks (Algorithm 1)
            "generated_lyricist_task": "",
            "generated_composer_task": "",
            "generated_stylist_task": "",
            "expert_tasks_generated": False,
            # Assembled prompts
            "pop_prompt_result": [],
            "prompts_assembled": False,
            # Verifier
            "verification_result": {},
            "verification_passed": False,
            "verifier_invoked": False,
            # Control
            "state": "start",
            "tools": "none",
            "current_iteration": 0,
            "complete": False,
            "final_answer": "",
            "piece": piece,
            # Reflection
            "reflection_result": {},
            "reflection_count": 0,
            "max_reflect_retries": self.max_reflect_retries,
        }

        _log("=" * 60, indent=0)
        _log("[MusicGenerationSupervisor] Starting pipeline", indent=0)
        _log("=" * 60, indent=0)

        response = await self.graph.ainvoke(initial_state, config={"recursion_limit": 50})

        return {
            "final_answer": response.get("final_answer", ""),
            "pop_prompt_result": response.get("pop_prompt_result", []),
            "blueprint": response.get("blueprint", {}),
            "verification_result": response.get("verification_result", {}),
            "complete": response.get("complete", False),
            # Individual expert results
            "lyricist_idx_result": response.get("lyricist_idx_result", []),
            "lyricist_lyric_result": response.get("lyricist_lyric_result", []),
            "composer_descriptions_result": response.get("composer_descriptions_result", []),
            "stylist_audio_type_result": response.get("stylist_audio_type_result", []),
        }

    async def ainvoke(
        self,
        user_input: str,
        json_scene: Optional[List[dict]] = None,
        piece: int = 2,
    ) -> dict:
        """
        Backward-compatible entry point (same signature as pop/pop.py).

        Constructs a task_packet internally and delegates to ainvoke_from_packet.
        """
        if not json_scene:
            return {
                "final_answer": "",
                "pop_prompt_result": [],
                "complete": False,
                "error": "No json_scene provided",
            }

        packet = {
            "team_name": "Team3",
            "instruction": user_input,
            "constraints": [],
            "json_scene": json_scene,
            "user_requirement": user_input,
            "piece": piece,
        }
        return await self.ainvoke_from_packet(packet)

    def invoke(
        self,
        user_input: str,
        json_scene: Optional[List[dict]] = None,
        piece: int = 2,
    ) -> dict:
        """Synchronous wrapper for ainvoke."""
        return asyncio.run(self.ainvoke(user_input, json_scene, piece))


# ═══════════════════════════════════════════════════════════════════════════
# Standalone execution for testing
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEAM3_DIR = os.path.dirname(CURRENT_DIR)

    def _bar(title: str, char: str = "=", width: int = 76):
        print(f"\n{char * width}")
        print(f"  {title}")
        print(f"{char * width}")

    async def main():
        load_dotenv()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        _header("Music Generation Supervisor — End-to-End Test")

        supervisor = MusicGenerationSupervisor()

        # Test scene data (from Team2 beauty_01 output)
        test_json_scene = [
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
                "环境声音内容": "加入轻微的翻书声、椅子挪动声，蝉鸣声渐强后忽然减弱",
                "环境声音风格": "从日常环境音过渡到情感的主观聚焦",
            },
        ]

        user_input = "Generate romantic Chinese pop music prompts based on the scene data"

        # Test 1: Backward-compatible ainvoke
        _header("Test 1: ainvoke (backward-compatible)", char="-", width=60)
        result = await supervisor.ainvoke(
            user_input=user_input,
            json_scene=test_json_scene,
            piece=2,
        )

        _header("Musical Blueprint", char="-", width=60)
        blueprint = result.get("blueprint", {})
        print(json.dumps(blueprint, ensure_ascii=False, indent=2))

        _header("Individual Expert Results", char="-", width=60)
        print(f"  Lyricist IDX: {result.get('lyricist_idx_result', [])}")
        print(f"  Lyricist Lyrics: {len(result.get('lyricist_lyric_result', []))} item(s)")
        for i, lyric in enumerate(result.get("lyricist_lyric_result", [])):
            print(f"    [{i}] {lyric[:120]}...")
        print(f"  Composer Descriptions: {result.get('composer_descriptions_result', [])}")
        print(f"  Stylist Audio Types: {result.get('stylist_audio_type_result', [])}")

        _header("Assembled Prompts (pop_prompt_result)", char="-", width=60)
        for i, prompt in enumerate(result.get("pop_prompt_result", [])):
            print(f"\n  Prompt [{i}]:")
            print(json.dumps(prompt, ensure_ascii=False, indent=4))

        _header("Verification Result", char="-", width=60)
        verification = result.get("verification_result", {})
        print(json.dumps(verification, ensure_ascii=False, indent=2))

        _header("Summary", char="-", width=60)
        print(f"  Complete: {result.get('complete', False)}")
        print(f"  Prompts generated: {len(result.get('pop_prompt_result', []))}")
        print(f"  Verification passed: {verification.get('passed', 'N/A')}")

        # Verify output format matches lyric.jsonl schema
        _header("Output Format Validation", char="-", width=60)
        required_fields = {"idx", "gt_lyric", "descriptions", "auto_prompt_audio_type"}
        all_valid = True
        for i, prompt in enumerate(result.get("pop_prompt_result", [])):
            missing = required_fields - set(prompt.keys())
            if missing:
                print(f"  Prompt [{i}] INVALID — missing fields: {missing}")
                all_valid = False
            else:
                print(f"  Prompt [{i}] OK — all 4 fields present")
        if all_valid and result.get("pop_prompt_result"):
            print("  All prompts have valid 4-field format")
        elif not result.get("pop_prompt_result"):
            print("  WARNING: No prompts were generated")

        # Test 2: ainvoke_from_packet
        _header("Test 2: ainvoke_from_packet", char="-", width=60)
        packet = {
            "team_name": "Team3",
            "instruction": user_input,
            "constraints": ["Must use Chinese lyrics", "Romantic mood"],
            "json_scene": test_json_scene,
            "user_requirement": user_input,
            "piece": 2,
        }
        result2 = await supervisor.ainvoke_from_packet(packet)
        print(f"  Complete: {result2.get('complete', False)}")
        print(f"  Prompts: {len(result2.get('pop_prompt_result', []))}")

    asyncio.run(main())
