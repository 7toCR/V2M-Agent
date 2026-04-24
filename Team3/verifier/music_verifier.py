"""
Music Verifier — 3-stage validation pipeline for Team 3.

Stage 1: Vocabulary Validation (pure Python, STRICT)
  - Validate descriptions fields against songgeneration.json
  - Validate audio_type against 12 candidates
  - Validate BPM range

Stage 2: Lyric Structure Validation (pure Python, FLEXIBLE)
  - Validate structure tags
  - Check vocal mode requirements
  - Half-width punctuation check

Stage 3: Cross-Component Consistency (deterministic rules + LLM)
  - Apply cross_field_rules from songgeneration.json
  - LLM semantic check for lyrics-emotion alignment

Follows the Think-Act-Observe-Reflect iterative loop pattern from Team2.
"""

import os
import sys
import re
import json
import asyncio
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
TEAM3_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TEAM3_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tools.tools import _print_with_indent

# Import agent profile
try:
    from Team3.AgentProfile.music_verifier_profile import MUSIC_VERIFIER_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import MUSIC_VERIFIER_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    MUSIC_VERIFIER_PROFILE = None

# Import reflection modules
try:
    from Team2.Expert.reflection_memory import ReflectionMemory, get_reflection_memory
    from Team2.Expert.reflection_agent_profile import build_reflection_prompt, parse_reflection_result
    REFLECTION_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import reflection modules: {e}")
    REFLECTION_IMPORT_SUCCESS = False

logger = logging.getLogger(__name__)


# ===================== Vocabulary Loading =====================

def _load_songgeneration_vocab() -> Dict[str, Any]:
    """Load vocabulary from songgeneration.json."""
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
    """Load vocabulary from noatgen.json."""
    for candidate in [
        os.path.join(TEAM3_DIR, "pre-traing", "noatgen.json"),
        os.path.join(TEAM3_DIR, "AgentProfile", "pre-traing", "noatgen.json"),
    ]:
        path = os.path.normpath(candidate)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


class MusicVerifier:
    """Independent Music Generation Verifier.

    Validates generated music prompts through a 3-stage pipeline:
    Stage 1: Vocabulary Validation (pure Python, STRICT — errors block)
    Stage 2: Lyric Structure Validation (pure Python, FLEXIBLE — warnings)
    Stage 3: Cross-Component Consistency (rules + LLM — errors block)
    """

    class Graph(TypedDict):
        # Messages
        global_messages: Annotated[List[dict], add_messages]
        system_prompt_messages: str
        think_message: Annotated[List[dict], add_messages]
        action_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]

        # Inputs
        pop_prompt_result: List[Dict[str, str]]
        blueprint: Dict[str, Any]
        json_scene: List[Dict[str, Any]]

        # Stage completion
        stage1_complete: bool
        stage2_complete: bool
        stage3_complete: bool

        # Validation results
        validation_passed: bool
        validation_issues: List[Dict[str, Any]]
        validation_warnings: List[Dict[str, Any]]

        # Control flow
        state: Literal["start", "think", "act", "execute", "observation", "reflect", "final", "end"]
        tools: str
        current_iteration: Annotated[int, add]
        max_iterations: int
        complete: bool
        final_answer: Any

        # Reflection
        reflection_result: Dict[str, Any]
        reflection_count: int
        max_reflect_retries: int

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        max_reflect_retries: int = 2,
    ) -> None:
        load_dotenv()

        config_path = os.path.join(TEAM3_DIR, "config_music_generation.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.config = None

        if llm is None:
            model_config = self.config.get("model", {}) if self.config else {}
            self.model = ChatOpenAI(
                model=model_config.get("name", "qwen3-max"),
                api_key=os.getenv(model_config.get("api_key_env", "MCP_API_KEY")),
                base_url=model_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                temperature=model_config.get("temperature", 0.3),
                max_tokens=model_config.get("max_tokens"),
            )
        else:
            self.model = llm

        self.max_reflect_retries = max_reflect_retries
        self.agent_profile = MUSIC_VERIFIER_PROFILE

        # Load vocabularies
        self.song_vocab = _load_songgeneration_vocab()
        self.noatgen_vocab = _load_noatgen_vocab()

        # Reflection memory
        if REFLECTION_IMPORT_SUCCESS:
            self.reflection_memory = get_reflection_memory("music_verifier", CURRENT_DIR)
        else:
            self.reflection_memory = None

        self._build_graph()

    def _build_graph(self) -> None:
        builder = StateGraph(MusicVerifier.Graph)

        builder.add_node("init", self._init_node)
        builder.add_node("think", self._think_node)
        builder.add_node("act", self._act_node)
        builder.add_node("vocabulary_validation", self._vocabulary_validation_node)
        builder.add_node("structure_validation", self._structure_validation_node)
        builder.add_node("consistency_check", self._consistency_check_node)
        builder.add_node("observation", self._observation_node)
        builder.add_node("reflect", self._reflect_node)
        builder.add_node("final", self._final_node)

        builder.add_edge(START, "init")
        builder.add_edge("init", "think")

        builder.add_conditional_edges(
            "think", self._route_after_think,
            {"act": "act", "final": "final"},
        )
        builder.add_conditional_edges(
            "act", self._route_after_act,
            {
                "vocabulary_validation": "vocabulary_validation",
                "structure_validation": "structure_validation",
                "consistency_check": "consistency_check",
                "observation": "observation",
            },
        )

        builder.add_edge("vocabulary_validation", "observation")
        builder.add_edge("structure_validation", "observation")
        builder.add_edge("consistency_check", "observation")

        builder.add_conditional_edges(
            "observation", self._route_after_observation,
            {"think": "think", "reflect": "reflect", "final": "final"},
        )
        builder.add_conditional_edges(
            "reflect", self._route_after_reflect,
            {"think": "think", "final": "final"},
        )
        builder.add_edge("final", END)

        self.graph = builder.compile()

    # ===================== Routing =====================

    def _route_after_think(self, state: "MusicVerifier.Graph") -> str:
        if state.get("state") == "final":
            return "final"
        return "act"

    def _route_after_act(self, state: "MusicVerifier.Graph") -> str:
        tool = state.get("tools", "observation")
        valid = {"vocabulary_validation", "structure_validation", "consistency_check"}
        return tool if tool in valid else "observation"

    def _route_after_observation(self, state: "MusicVerifier.Graph") -> str:
        # All 3 stages complete -> check if issues exist
        if state.get("stage1_complete") and state.get("stage2_complete") and state.get("stage3_complete"):
            issues = state.get("validation_issues", [])
            if issues and state.get("reflection_count", 0) < state.get("max_reflect_retries", self.max_reflect_retries):
                return "reflect"
            return "final"
        return "think"

    def _route_after_reflect(self, state: "MusicVerifier.Graph") -> str:
        reflection = state.get("reflection_result", {})
        should_retry = reflection.get("should_retry", False)
        count = state.get("reflection_count", 0)
        if should_retry and count < state.get("max_reflect_retries", self.max_reflect_retries):
            return "think"
        return "final"

    # ===================== Nodes =====================

    async def _init_node(self, state: "MusicVerifier.Graph") -> Dict[str, Any]:
        _print_with_indent("MusicVerifier:", "Starting 3-stage validation pipeline", tab_count=1)

        sp = ""
        if self.agent_profile:
            sp += f"# {self.agent_profile.role.name}\n\n"
            sp += f"{self.agent_profile.description}\n\n"
            if self.agent_profile.guide_book:
                sp += f"{self.agent_profile.guide_book}\n\n"

        return {
            "global_messages": [SystemMessage(content=sp)],
            "system_prompt_messages": sp,
            "think_message": [],
            "action_message": [],
            "observation_message": [],
            "stage1_complete": False,
            "stage2_complete": False,
            "stage3_complete": False,
            "validation_passed": True,
            "validation_issues": [],
            "validation_warnings": [],
            "state": "think",
            "tools": "None",
            "current_iteration": 1,
            "max_iterations": 20,
            "complete": False,
            "final_answer": None,
            "reflection_result": {},
            "reflection_count": 0,
            "max_reflect_retries": self.max_reflect_retries,
        }

    async def _think_node(self, state: "MusicVerifier.Graph") -> Dict[str, Any]:
        """Deterministic stage sequencer: Stage1 -> Stage2 -> Stage3 -> final."""
        if not state.get("stage1_complete"):
            tool = "vocabulary_validation"
            _print_with_indent("think:", "Stage 1 — Vocabulary Validation", tab_count=2)
        elif not state.get("stage2_complete"):
            tool = "structure_validation"
            _print_with_indent("think:", "Stage 2 — Lyric Structure Validation", tab_count=2)
        elif not state.get("stage3_complete"):
            tool = "consistency_check"
            _print_with_indent("think:", "Stage 3 — Cross-Component Consistency", tab_count=2)
        else:
            _print_with_indent("think:", "All stages complete", tab_count=2)
            return {
                "state": "final",
                "tools": "None",
                "current_iteration": 1,
            }

        return {
            "state": "act",
            "tools": tool,
            "current_iteration": 1,
        }

    async def _act_node(self, state: "MusicVerifier.Graph") -> Dict[str, Any]:
        tool = state.get("tools", "None")
        _print_with_indent("act:", f"Dispatching -> {tool}", tab_count=2)
        return {"state": "execute", "current_iteration": 1}

    # ===================== Stage 1: Vocabulary Validation =====================

    async def _vocabulary_validation_node(self, state: "MusicVerifier.Graph") -> Dict[str, Any]:
        """Stage 1: Pure Python vocabulary validation. Severity: ERROR."""
        _print_with_indent("Stage 1:", "Vocabulary Validation (pure Python)", tab_count=2)

        prompts = state.get("pop_prompt_result", [])
        blueprint = state.get("blueprint", {})
        model_type = blueprint.get("model", "SongGeneration")
        issues = list(state.get("validation_issues", []))
        warnings = list(state.get("validation_warnings", []))

        if model_type == "NotaGen":
            _print_with_indent("Stage 1:", "NotaGen mode — skipping SongGeneration vocabulary check", tab_count=3)
            return {
                "stage1_complete": True,
                "validation_issues": issues,
                "validation_warnings": warnings,
                "state": "observation",
                "current_iteration": 1,
            }

        vocab = self.song_vocab.get("descriptions", {})
        valid_genders = set(vocab.get("gender", []))
        valid_emotions = set(vocab.get("emotion", []))
        valid_genres = set(vocab.get("genre", []))
        valid_timbres = set(vocab.get("timbre", []))
        valid_instruments = set(vocab.get("instrument", []))
        bpm_range = vocab.get("bpm_range", [60, 200])
        valid_audio_types = set(self.song_vocab.get("audio_type", []))

        for idx, prompt in enumerate(prompts):
            desc = prompt.get("descriptions", "")
            audio_type = prompt.get("auto_prompt_audio_type", "")

            # Validate descriptions fields
            parts = desc.strip().rstrip(".").split(", ")
            if len(parts) < 6:
                issues.append({
                    "prompt_index": idx, "stage": 1, "severity": "ERROR",
                    "field": "descriptions", "type": "parse_error",
                    "message": f"Cannot parse into 6 fields: '{desc}'",
                })
                continue

            # Parse fields: gender, emotion, genre, timbre, instrument..., the bpm is N
            gender = parts[0].strip()
            emotion = parts[1].strip()
            genre = parts[2].strip()
            timbre = parts[3].strip()
            bpm_part = parts[-1].strip()
            instrument = ", ".join(p.strip() for p in parts[4:-1])

            if gender not in valid_genders:
                issues.append({
                    "prompt_index": idx, "stage": 1, "severity": "ERROR",
                    "field": "gender", "value": gender,
                    "allowed": list(valid_genders),
                })
            if emotion not in valid_emotions:
                issues.append({
                    "prompt_index": idx, "stage": 1, "severity": "ERROR",
                    "field": "emotion", "value": emotion,
                    "allowed": list(valid_emotions),
                })
            if genre not in valid_genres:
                issues.append({
                    "prompt_index": idx, "stage": 1, "severity": "ERROR",
                    "field": "genre", "value": genre,
                    "allowed": list(valid_genres),
                })
            if timbre not in valid_timbres:
                issues.append({
                    "prompt_index": idx, "stage": 1, "severity": "ERROR",
                    "field": "timbre", "value": timbre,
                    "allowed": list(valid_timbres),
                })
            if instrument not in valid_instruments:
                issues.append({
                    "prompt_index": idx, "stage": 1, "severity": "ERROR",
                    "field": "instrument", "value": instrument,
                    "allowed": "see songgeneration.json",
                })

            # Validate BPM
            bpm_match = re.search(r"the bpm is (\d+)", bpm_part, re.IGNORECASE)
            if bpm_match:
                bpm_val = int(bpm_match.group(1))
                if bpm_val < bpm_range[0] or bpm_val > bpm_range[1]:
                    issues.append({
                        "prompt_index": idx, "stage": 1, "severity": "ERROR",
                        "field": "bpm", "value": bpm_val,
                        "allowed": f"[{bpm_range[0]}, {bpm_range[1]}]",
                    })
            else:
                issues.append({
                    "prompt_index": idx, "stage": 1, "severity": "ERROR",
                    "field": "bpm", "type": "parse_error",
                    "message": f"Cannot parse BPM from: '{bpm_part}'",
                })

            # Validate audio_type
            if audio_type and audio_type not in valid_audio_types:
                issues.append({
                    "prompt_index": idx, "stage": 1, "severity": "ERROR",
                    "field": "audio_type", "value": audio_type,
                    "allowed": list(valid_audio_types),
                })

        stage1_issues = [i for i in issues if i.get("stage") == 1]
        _print_with_indent("Stage 1:", f"Found {len(stage1_issues)} issues", tab_count=3)

        return {
            "stage1_complete": True,
            "validation_issues": issues,
            "validation_warnings": warnings,
            "state": "observation",
            "current_iteration": 1,
        }

    # ===================== Stage 2: Lyric Structure Validation =====================

    async def _structure_validation_node(self, state: "MusicVerifier.Graph") -> Dict[str, Any]:
        """Stage 2: Pure Python lyric structure validation. Severity: WARNING."""
        _print_with_indent("Stage 2:", "Lyric Structure Validation (pure Python)", tab_count=2)

        prompts = state.get("pop_prompt_result", [])
        blueprint = state.get("blueprint", {})
        lyric_style = blueprint.get("lyric_style", "vocal")
        issues = list(state.get("validation_issues", []))
        warnings = list(state.get("validation_warnings", []))

        valid_tags = set(self.song_vocab.get("lyric_structure_tags", []))

        for idx, prompt in enumerate(prompts):
            gt_lyric = prompt.get("gt_lyric", "")
            if not gt_lyric:
                issues.append({
                    "prompt_index": idx, "stage": 2, "severity": "ERROR",
                    "field": "gt_lyric", "message": "Empty gt_lyric",
                })
                continue
            if "..." in gt_lyric or "…" in gt_lyric:
                issues.append({
                    "prompt_index": idx, "stage": 2, "severity": "ERROR",
                    "field": "gt_lyric", "message": "Placeholder lyric content",
                })

            # Split sections by ' ; '
            sections = [s.strip() for s in gt_lyric.split(" ; ") if s.strip()]

            # Check each section starts with a valid tag
            found_tags = []
            for section in sections:
                tag_match = re.match(r"(\[[\w-]+\])", section)
                if tag_match:
                    tag = tag_match.group(1)
                    found_tags.append(tag)
                    if valid_tags and tag not in valid_tags:
                        warnings.append({
                            "prompt_index": idx, "stage": 2, "severity": "WARNING",
                            "field": "gt_lyric", "tag": tag,
                            "message": f"Invalid structure tag: {tag}",
                        })
                else:
                    warnings.append({
                        "prompt_index": idx, "stage": 2, "severity": "WARNING",
                        "field": "gt_lyric",
                        "message": f"Section without tag: '{section[:50]}...'",
                    })

            # Vocal mode: must have [verse] + [chorus]
            if lyric_style == "vocal":
                if "[verse]" not in found_tags:
                    issues.append({
                        "prompt_index": idx, "stage": 2, "severity": "ERROR",
                        "field": "gt_lyric",
                        "message": "Vocal mode requires at least one [verse]",
                    })
                if "[chorus]" not in found_tags:
                    issues.append({
                        "prompt_index": idx, "stage": 2, "severity": "ERROR",
                        "field": "gt_lyric",
                        "message": "Vocal mode requires at least one [chorus]",
                    })
                body = re.sub(r"\[[^\]]+\]", "", gt_lyric).replace(";", " ").strip()
                if len(body) < 20:
                    issues.append({
                        "prompt_index": idx, "stage": 2, "severity": "ERROR",
                        "field": "gt_lyric",
                        "message": "Vocal lyric body is too short",
                    })

            # Should start with [intro-*] and end with [outro-*]
            if found_tags:
                if not found_tags[0].startswith("[intro"):
                    warnings.append({
                        "prompt_index": idx, "stage": 2, "severity": "WARNING",
                        "field": "gt_lyric",
                        "message": f"Song should start with [intro-*], found: {found_tags[0]}",
                    })
                if not found_tags[-1].startswith("[outro"):
                    warnings.append({
                        "prompt_index": idx, "stage": 2, "severity": "WARNING",
                        "field": "gt_lyric",
                        "message": f"Song should end with [outro-*], found: {found_tags[-1]}",
                    })

            # Check half-width punctuation
            full_width_chars = re.findall(r'[，。；：！？]', gt_lyric)
            if full_width_chars:
                warnings.append({
                    "prompt_index": idx, "stage": 2, "severity": "WARNING",
                    "field": "gt_lyric",
                    "message": f"Full-width punctuation found: {full_width_chars[:5]}",
                })

        stage2_warnings = [w for w in warnings if w.get("stage") == 2]
        _print_with_indent("Stage 2:", f"Found {len(stage2_warnings)} warnings", tab_count=3)

        return {
            "stage2_complete": True,
            "validation_issues": issues,
            "validation_warnings": warnings,
            "state": "observation",
            "current_iteration": 1,
        }

    # ===================== Stage 3: Cross-Component Consistency =====================

    async def _consistency_check_node(self, state: "MusicVerifier.Graph") -> Dict[str, Any]:
        """Stage 3: Deterministic rules + LLM semantic check. Severity: ERROR."""
        _print_with_indent("Stage 3:", "Cross-Component Consistency", tab_count=2)

        prompts = state.get("pop_prompt_result", [])
        blueprint = state.get("blueprint", {})
        json_scene = state.get("json_scene", [])
        issues = list(state.get("validation_issues", []))
        warnings = list(state.get("validation_warnings", []))

        cross_field_rules = self.song_vocab.get("cross_field_rules", [])

        # --- Deterministic rules ---
        for idx, prompt in enumerate(prompts):
            desc = prompt.get("descriptions", "")
            audio_type = prompt.get("auto_prompt_audio_type", "")

            parts = desc.strip().rstrip(".").split(", ")
            if len(parts) < 6:
                continue

            emotion = parts[1].strip()
            genre = parts[2].strip()
            timbre = parts[3].strip()

            for rule in cross_field_rules:
                rule_name = rule.get("rule", "")

                if rule_name == "emotion_timbre_alignment":
                    for conflict in rule.get("conflicts", []):
                        if emotion in conflict.get("if_emotion", []):
                            disallowed = conflict.get("disallow_timbre", [])
                            if timbre in disallowed:
                                issues.append({
                                    "prompt_index": idx, "stage": 3, "severity": "ERROR",
                                    "rule": rule_name,
                                    "message": f"emotion '{emotion}' conflicts with timbre '{timbre}'",
                                })

                elif rule_name == "emotion_genre_alignment":
                    for conflict in rule.get("conflicts", []):
                        if emotion in conflict.get("if_emotion", []):
                            disallowed = conflict.get("disallow_genre", [])
                            if genre in disallowed:
                                issues.append({
                                    "prompt_index": idx, "stage": 3, "severity": "ERROR",
                                    "rule": rule_name,
                                    "message": f"emotion '{emotion}' conflicts with genre '{genre}'",
                                })

                elif rule_name == "audio_type_genre_coherence":
                    for conflict in rule.get("conflicts", []):
                        if audio_type == conflict.get("if_audio_type"):
                            preferred = conflict.get("prefer_genre", [])
                            if preferred and genre not in preferred:
                                warnings.append({
                                    "prompt_index": idx, "stage": 3, "severity": "WARNING",
                                    "rule": rule_name,
                                    "message": f"audio_type '{audio_type}' prefers genre {preferred}, got '{genre}'",
                                })

        # --- LLM semantic check (only if scene data available) ---
        if json_scene and prompts:
            try:
                scene_summary = json.dumps(json_scene[:3], ensure_ascii=False, indent=2)
                prompts_summary = json.dumps(prompts[:3], ensure_ascii=False, indent=2)
                emotional_key = blueprint.get("emotional_key", "")

                semantic_prompt = (
                    "You are a music consistency checker. Analyze the following:\n"
                    f"Scene data (first 3):\n{scene_summary}\n\n"
                    f"Music prompts (first 3):\n{prompts_summary}\n\n"
                    f"Blueprint emotional_key: {emotional_key}\n\n"
                    "Check for contradictions:\n"
                    "1. Do lyrics emotion match scene mood?\n"
                    "2. Does BPM match scene intensity?\n"
                    "3. Does audio_type align with overall mood?\n"
                    "4. Does title language match lyric language?\n\n"
                    "Reply JSON: {\"contradictions\": [{\"prompt_index\": 0, \"issue\": \"...\"}], "
                    "\"coherent\": true/false}\n"
                    "If no contradictions, return {\"contradictions\": [], \"coherent\": true}\n"
                )

                messages = [HumanMessage(content=semantic_prompt)]
                response = await self.model.ainvoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)

                # Parse response
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    semantic_result = json.loads(json_match.group())
                    for contradiction in semantic_result.get("contradictions", []):
                        issues.append({
                            "prompt_index": contradiction.get("prompt_index", -1),
                            "stage": 3, "severity": "ERROR",
                            "rule": "llm_semantic_check",
                            "message": contradiction.get("issue", "Semantic contradiction"),
                        })
            except Exception as e:
                _print_with_indent("Stage 3:", f"LLM semantic check failed: {e}", tab_count=3)

        stage3_issues = [i for i in issues if i.get("stage") == 3]
        stage3_warnings = [w for w in warnings if w.get("stage") == 3]
        _print_with_indent("Stage 3:", f"Found {len(stage3_issues)} issues, {len(stage3_warnings)} warnings", tab_count=3)

        return {
            "stage3_complete": True,
            "validation_issues": issues,
            "validation_warnings": warnings,
            "state": "observation",
            "current_iteration": 1,
        }

    # ===================== Observation & Reflection =====================

    async def _observation_node(self, state: "MusicVerifier.Graph") -> Dict[str, Any]:
        issues = state.get("validation_issues", [])
        warnings = state.get("validation_warnings", [])
        s1 = state.get("stage1_complete", False)
        s2 = state.get("stage2_complete", False)
        s3 = state.get("stage3_complete", False)

        obs = f"Stages: S1={s1}, S2={s2}, S3={s3}. Issues: {len(issues)}, Warnings: {len(warnings)}"
        _print_with_indent("observation:", obs, tab_count=2)

        sp = state.get("system_prompt_messages", "")
        sp += f"observation:{obs}\n"

        return {
            "global_messages": [AIMessage(content=obs)],
            "system_prompt_messages": sp,
            "observation_message": [{"role": "assistant", "content": obs}],
            "state": "think",
            "current_iteration": 1,
        }

    async def _reflect_node(self, state: "MusicVerifier.Graph") -> Dict[str, Any]:
        """Reflection: analyze validation failures and decide whether to retry."""
        _print_with_indent("reflect:", "Analyzing validation failures...", tab_count=2)

        issues = state.get("validation_issues", [])
        warnings = state.get("validation_warnings", [])
        reflection_count = state.get("reflection_count", 0) + 1

        issue_summary = json.dumps(issues[:10], ensure_ascii=False, indent=2)

        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="music_verifier",
                task_description="Validate music generation prompts",
                observation=f"Issues: {issue_summary}",
                history_reflections="",
            )
        else:
            reflect_prompt = (
                f"Validation found {len(issues)} errors and {len(warnings)} warnings.\n"
                f"Issues: {issue_summary}\n"
                'Reply JSON: {{"quality": "high/medium/low", "should_retry": true/false, '
                '"analysis": "...", "improvement": "..."}}\n'
            )

        try:
            messages = [HumanMessage(content=reflect_prompt)]
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, "content") else str(response)
        except Exception:
            response_text = '{"quality": "low", "should_retry": false}'

        if REFLECTION_IMPORT_SUCCESS and parse_reflection_result:
            reflection_result = parse_reflection_result(response_text)
        else:
            try:
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                reflection_result = json.loads(json_match.group()) if json_match else {"quality": "low", "should_retry": False}
            except Exception:
                reflection_result = {"quality": "low", "should_retry": False}

        _print_with_indent("reflect:", f"Quality: {reflection_result.get('quality')}, Count: {reflection_count}", tab_count=2)

        return {
            "reflection_result": reflection_result,
            "reflection_count": reflection_count,
            "state": "final" if not reflection_result.get("should_retry", False) else "think",
            "current_iteration": 1,
        }

    # ===================== Final =====================

    async def _final_node(self, state: "MusicVerifier.Graph") -> Dict[str, Any]:
        issues = state.get("validation_issues", [])
        warnings = state.get("validation_warnings", [])
        errors = [i for i in issues if i.get("severity") == "ERROR"]
        passed = len(errors) == 0

        final = {
            "passed": passed,
            "issues": issues,
            "warnings": warnings,
        }

        _print_with_indent("MusicVerifier:", f"Passed={passed}, Errors={len(errors)}, Warnings={len(warnings)}", tab_count=1)

        return {
            "validation_passed": passed,
            "validation_issues": issues,
            "validation_warnings": warnings,
            "complete": True,
            "final_answer": final,
        }

    # ===================== Public Interface =====================

    async def ainvoke(
        self,
        pop_prompt_result: List[Dict[str, str]],
        blueprint: Dict[str, Any],
        json_scene: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async entry point for music verification.

        Args:
            pop_prompt_result: Assembled prompts [{idx, gt_lyric, descriptions, auto_prompt_audio_type}]
            blueprint: Musical Blueprint from supervisor
            json_scene: Scene data from Team 2 (for semantic consistency check)

        Returns:
            {passed: bool, issues: List[Dict], warnings: List[Dict]}
        """
        initial_state: MusicVerifier.Graph = {
            "global_messages": [],
            "system_prompt_messages": "",
            "think_message": [],
            "action_message": [],
            "observation_message": [],
            "pop_prompt_result": pop_prompt_result,
            "blueprint": blueprint,
            "json_scene": json_scene or [],
            "stage1_complete": False,
            "stage2_complete": False,
            "stage3_complete": False,
            "validation_passed": True,
            "validation_issues": [],
            "validation_warnings": [],
            "state": "start",
            "tools": "None",
            "current_iteration": 0,
            "max_iterations": 20,
            "complete": False,
            "final_answer": None,
            "reflection_result": {},
            "reflection_count": 0,
            "max_reflect_retries": self.max_reflect_retries,
        }

        result = await self.graph.ainvoke(initial_state, config={"recursion_limit": 50})

        return {
            "passed": result.get("validation_passed", False),
            "issues": result.get("validation_issues", []),
            "warnings": result.get("validation_warnings", []),
        }


if __name__ == "__main__":
    import asyncio
    load_dotenv()

    verifier = MusicVerifier(max_reflect_retries=2)

    # Test with valid prompts
    test_prompts = [
        {
            "idx": "2026-4-11-14-30-Sunset Dreams",
            "gt_lyric": "[intro-short] ; [verse] A golden sunset line ; [chorus] We dance in the light ; [outro-short]",
            "descriptions": "female, romantic, pop, bright, synthesizer and piano, the bpm is 125.",
            "auto_prompt_audio_type": "Pop",
        },
        {
            "idx": "2026-4-11-14-30-Summer Rain",
            "gt_lyric": "[intro-medium] ; [verse] Rain falls gently ; [chorus] Washing away tears ; [outro-short]",
            "descriptions": "male, melancholic, folk, soft, acoustic guitar and piano, the bpm is 80.",
            "auto_prompt_audio_type": "Folk",
        },
    ]

    test_blueprint = {
        "model": "SongGeneration",
        "lyric_style": "vocal",
        "emotional_key": "romantic",
        "language": "en",
    }

    async def main():
        result = await verifier.ainvoke(
            pop_prompt_result=test_prompts,
            blueprint=test_blueprint,
            json_scene=[],
        )
        print(f"\nVerification result:")
        print(f"  Passed: {result['passed']}")
        print(f"  Issues: {len(result['issues'])}")
        print(f"  Warnings: {len(result['warnings'])}")
        for issue in result["issues"]:
            print(f"    ERROR: {issue}")
        for warning in result["warnings"]:
            print(f"    WARNING: {warning}")

    asyncio.run(main())
