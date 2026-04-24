"""
Scene Understanding Verifier (Team 2)

Implements the four-stage multimodal fusion pipeline and consistency validation
as a full LangGraph agent following Think-Act-Observe-Reflect iterative logic.

Independent from the Supervisor - receives expert results + context directly.

Paper reference (Section 3.3.2):
    Stage 1: Temporal Anchoring        - K_V = {(v_i, t_i)}
    Stage 2: Inter-modal Interpolation - K_VI = K_V ∪ {(img_j, t'_j)}
    Stage 3: Audio-Visual Alignment    - K_VIA = Φ(K_VI, A)
    Stage 4: Semantic Refinement       - S = R(K_VIA | T_bg, F_later→F_early)
    + Consistency Validation
"""

import os
import sys
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages, StateGraph, START, END
from typing_extensions import TypedDict
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM2_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TEAM2_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tools.tools import _print_with_indent as _base_print


def _log(msg: str, indent: int = 1):
    """Convenience wrapper for _log(prefix, content, tab_count)."""
    _base_print("", msg, tab_count=indent)

# Import Verifier profile
try:
    from Team2.AgentProfile.scene_verifier_profile import SCENE_VERIFIER_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import SCENE_VERIFIER_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    SCENE_VERIFIER_PROFILE = None

# Import reflection modules
try:
    from Team2.Expert.reflection_memory import ReflectionMemory, get_reflection_memory
    from Team2.Expert.reflection_agent_profile import build_reflection_prompt, parse_reflection_result
    REFLECTION_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import reflection modules: {e}")
    REFLECTION_IMPORT_SUCCESS = False
    get_reflection_memory = None
    build_reflection_prompt = None
    parse_reflection_result = None

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# 9-field scene representation keys (Chinese by design — paper standard)
# ──────────────────────────────────────────────────────────────────────────────
NINE_FIELDS = [
    "关键帧", "主体", "主体心情",
    "主体声音内容", "主体声音风格",
    "背景", "背景风格",
    "背景声音内容", "背景声音风格",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _parse_timestamp(ts_str: str) -> float:
    """Parse a timestamp string (e.g. '0s', '3.5s', '00:01:30') into seconds."""
    if not ts_str or not isinstance(ts_str, str):
        return 0.0
    ts_str = ts_str.strip()
    # Format: '3.5s' or '3s'
    match = re.match(r'^([\d.]+)\s*s?$', ts_str)
    if match:
        return float(match.group(1))
    # Format: 'MM:SS' or 'HH:MM:SS'
    parts = ts_str.replace('s', '').split(':')
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    # Last resort: extract first number
    nums = re.findall(r'[\d.]+', ts_str)
    return float(nums[0]) if nums else 0.0


def _extract_start_time(time_range: str) -> float:
    """Extract start time from a time range string like '0s-3s' or '00:00-00:05'."""
    if not time_range or not isinstance(time_range, str):
        return 0.0
    parts = time_range.split('-')
    return _parse_timestamp(parts[0].strip())


def _clean_markdown_json(text: str) -> str:
    """Strip markdown code fences from LLM response text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _safe_parse_json_array(text: str) -> Optional[list]:
    """Try to parse a JSON array from text, tolerating markdown wrapping."""
    text = _clean_markdown_json(text)
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "result" in result:
            result = result["result"]
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Fallback: extract JSON array with regex
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Validation keywords for mood-voice consistency checking
# ──────────────────────────────────────────────────────────────────────────────
POSITIVE_MOODS = ["开心", "快乐", "高兴", "愉悦", "兴奋", "欢快", "轻松"]
NEGATIVE_MOODS = ["悲伤", "难过", "沮丧", "忧郁", "痛苦", "低沉", "沉重"]
CONFLICTING_VOICE_KW = ["烦躁", "不满", "抱怨", "愤怒", "悲伤"]
POSITIVE_VOICE_KW = ["温柔", "欢快", "愉悦", "轻快", "明亮"]


class SceneVerifier:
    """
    Team 2 Verifier: performs four-stage multimodal fusion and validates results.

    Follows Think-Act-Observe-Reflect iterative loop via LangGraph.
    Independent from the Supervisor — receives expert results + context directly.
    """

    class Graph(TypedDict):
        # Messages
        global_messages: Annotated[List[dict], add_messages]
        system_prompt_messages: str
        think_message: Annotated[List[dict], add_messages]
        action_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]

        # Inputs (immutable during execution)
        user_requirement: str
        team1_instruction: str
        team1_constraints: List[str]
        expert_results: Dict[str, Any]

        # Four-stage intermediate outputs
        temporal_skeleton: List[Dict[str, Any]]
        interpolated_timeline: List[Dict[str, Any]]
        av_aligned_timeline: List[Dict[str, Any]]
        refined_scenes: List[Dict[str, Any]]

        # Stage completion flags
        stage1_complete: bool
        stage2_complete: bool
        stage3_complete: bool
        stage4_complete: bool

        # Validation
        json_scene_result: List[Dict[str, Any]]
        validation_passed: bool
        validation_issues: List[str]
        validation_warnings: List[str]
        validation_run: bool

        # Control flow
        state: str
        tools: str
        current_iteration: int
        complete: bool
        final_answer: str

        # Reflection
        reflection_result: Dict[str, Any]
        reflection_count: int
        max_reflect_retries: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, llm=None):
        self.profile = SCENE_VERIFIER_PROFILE

        # Load configuration
        config_path = os.path.join(TEAM2_DIR, "config_scene_understanding.json")
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        # Initialize LLM
        if llm is not None:
            self.model = llm
        else:
            model_cfg = self.config.get("model", {})
            api_key = os.environ.get(model_cfg.get("api_key_env", "MCP_API_KEY"), "")
            self.model = ChatOpenAI(
                model=model_cfg.get("name", "qwen3-max"),
                openai_api_base=model_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                openai_api_key=api_key,
                temperature=model_cfg.get("temperature", 0.7),
                max_tokens=model_cfg.get("max_tokens", 8192),
            )

        self.max_reflect_retries = self.config.get("verifier", {}).get("max_reflect_retries", 2)

        # Reflection memory
        if REFLECTION_IMPORT_SUCCESS and get_reflection_memory:
            self.reflection_memory = get_reflection_memory("scene_verifier", CURRENT_DIR)
        else:
            self.reflection_memory = None

        # Build LangGraph
        self._build_graph()

    def _build_graph(self):
        builder = StateGraph(SceneVerifier.Graph)

        # Nodes
        builder.add_node("init", self._init_node)
        builder.add_node("think", self._think_node)
        builder.add_node("act", self._act_node)
        builder.add_node("temporal_anchoring", self._temporal_anchoring_node)
        builder.add_node("interpolation", self._interpolation_node)
        builder.add_node("av_alignment", self._av_alignment_node)
        builder.add_node("refinement", self._semantic_refinement_node)
        builder.add_node("validate", self._validate_node)
        builder.add_node("observation", self._observation_node)
        builder.add_node("reflect", self._reflect_node)
        builder.add_node("final", self._final_node)

        # Edges
        builder.add_edge(START, "init")
        builder.add_edge("init", "think")

        builder.add_conditional_edges("think", self._route_after_think,
                                      {"act": "act", "final": "final"})

        builder.add_conditional_edges("act", self._route_after_act, {
            "temporal_anchoring": "temporal_anchoring",
            "interpolation": "interpolation",
            "av_alignment": "av_alignment",
            "refinement": "refinement",
            "validate": "validate",
            "observation": "observation",
        })

        for tool_node in ["temporal_anchoring", "interpolation", "av_alignment",
                          "refinement", "validate"]:
            builder.add_edge(tool_node, "observation")

        builder.add_conditional_edges("observation", self._route_after_observation,
                                      {"think": "think", "reflect": "reflect", "final": "final"})

        builder.add_conditional_edges("reflect", self._route_after_reflect,
                                      {"think": "think", "final": "final"})

        builder.add_edge("final", END)

        self.graph = builder.compile()

    # ------------------------------------------------------------------
    # Routing functions
    # ------------------------------------------------------------------

    def _route_after_think(self, state: Graph) -> str:
        return "final" if state.get("state") == "final" else "act"

    def _route_after_act(self, state: Graph) -> str:
        tool = state.get("tools", "none")
        mapping = {
            "temporal_anchoring": "temporal_anchoring",
            "inter_modal_interpolation": "interpolation",
            "audio_visual_alignment": "av_alignment",
            "semantic_refinement": "refinement",
            "validate_scenes": "validate",
        }
        return mapping.get(tool, "observation")

    def _route_after_observation(self, state: Graph) -> str:
        s = state.get("state", "think")
        if s in ("reflect", "final"):
            return s
        return "think"

    def _route_after_reflect(self, state: Graph) -> str:
        return "final" if state.get("state") == "final" else "think"

    # ------------------------------------------------------------------
    # Init node
    # ------------------------------------------------------------------

    def _init_node(self, state: Graph) -> dict:
        """Initialize verifier state and build system prompt."""
        _log("[Verifier] Initializing four-stage fusion pipeline", indent=1)

        expert_results = state.get("expert_results", {})
        modalities = []
        if expert_results.get("video_result"):
            modalities.append("video")
        if expert_results.get("audio_result"):
            modalities.append("audio")
        if expert_results.get("photo_result"):
            modalities.append("photo")
        if expert_results.get("text_result"):
            modalities.append("text")

        sys_prompt = (
            "You are the Scene Understanding Verifier for Team 2.\n"
            "You perform four-stage multimodal fusion and validate scene consistency.\n"
            f"Available modalities: {', '.join(modalities) if modalities else 'none'}\n"
            f"User requirement: {state.get('user_requirement', '')}\n"
        )

        return {
            "system_prompt_messages": sys_prompt,
            "state": "think",
            "current_iteration": 0,
        }

    # ------------------------------------------------------------------
    # Think / Act / Observation nodes
    # ------------------------------------------------------------------

    def _think_node(self, state: Graph) -> dict:
        """
        Think: decide which stage tool to execute next.
        Sequential order: Stage1 → Stage2 → Stage3 → Stage4 → Validate → Final
        """
        iteration = state.get("current_iteration", 0) + 1
        _log(f"[Verifier] Think (iteration {iteration})", indent=1)

        if not state.get("stage1_complete"):
            _log("[Verifier] → Stage 1: Temporal Anchoring", indent=2)
            return {"state": "act", "tools": "temporal_anchoring", "current_iteration": iteration}

        if not state.get("stage2_complete"):
            _log("[Verifier] → Stage 2: Inter-modal Interpolation", indent=2)
            return {"state": "act", "tools": "inter_modal_interpolation", "current_iteration": iteration}

        if not state.get("stage3_complete"):
            _log("[Verifier] → Stage 3: Audio-Visual Alignment", indent=2)
            return {"state": "act", "tools": "audio_visual_alignment", "current_iteration": iteration}

        if not state.get("stage4_complete"):
            _log("[Verifier] → Stage 4: Semantic Refinement", indent=2)
            return {"state": "act", "tools": "semantic_refinement", "current_iteration": iteration}

        if state.get("json_scene_result") and not state.get("validation_run"):
            _log("[Verifier] → Validation", indent=2)
            return {"state": "act", "tools": "validate_scenes", "current_iteration": iteration}

        _log("[Verifier] → All stages complete, proceeding to final", indent=2)
        return {"state": "final", "tools": "none", "current_iteration": iteration}

    def _act_node(self, state: Graph) -> dict:
        """Act: dispatch to selected tool (routing handles actual dispatch)."""
        tool = state.get("tools", "none")
        _log(f"[Verifier] Act → dispatching tool: {tool}", indent=2)
        return {"state": "execute"}

    def _observation_node(self, state: Graph) -> dict:
        """Observe: check tool execution results and decide next step."""
        _log("[Verifier] Observation", indent=1)

        # If validation just ran and failed, consider reflection
        if state.get("validation_run") and not state.get("validation_passed"):
            issues = state.get("validation_issues", [])
            if issues and state.get("reflection_count", 0) < state.get("max_reflect_retries", 2):
                _log(f"[Verifier] Validation failed with {len(issues)} issues → reflect", indent=2)
                return {"state": "reflect"}
            else:
                _log("[Verifier] Validation failed but max retries reached → final", indent=2)
                return {"state": "final"}

        return {"state": "think"}

    # ------------------------------------------------------------------
    # Stage 1: Temporal Anchoring (Pure Python — no LLM)
    # ------------------------------------------------------------------

    def _temporal_anchoring_node(self, state: Graph) -> dict:
        """
        Stage 1: K_V = {(v_i, t_i) | i=1,...,n}
        Build keyframe skeleton from video_result.
        Fallback hierarchy: video → audio → photo/text → single frame.
        """
        _log("[Verifier] Stage 1: Temporal Anchoring", indent=1)

        expert_results = state.get("expert_results", {})
        video_result = expert_results.get("video_result", [])
        audio_result = expert_results.get("audio_result", [])
        photo_result = expert_results.get("photo_result", {})
        text_result = expert_results.get("text_result", {})

        skeleton = []

        if video_result and isinstance(video_result, list) and len(video_result) > 0:
            # Primary: video keyframes as temporal anchors
            _log(f"[Verifier] Using video modality ({len(video_result)} keyframes)", indent=2)
            for frame in video_result:
                ts_str = frame.get("关键帧", "0s")
                skeleton.append({
                    "关键帧": ts_str,
                    "_ts": _parse_timestamp(ts_str),
                    "主体": frame.get("主体", ""),
                    "主体心情": frame.get("主体心情", ""),
                    "主体声音内容": "",
                    "主体声音风格": "",
                    "背景": frame.get("背景", ""),
                    "背景风格": frame.get("背景风格", ""),
                    "背景声音内容": "",
                    "背景声音风格": "",
                    "_source": "video",
                })
            skeleton.sort(key=lambda x: x["_ts"])

        elif audio_result and isinstance(audio_result, list) and len(audio_result) > 0:
            # Fallback 1: audio time segments
            _log(f"[Verifier] Fallback to audio modality ({len(audio_result)} segments)", indent=2)
            for seg in audio_result:
                start = _extract_start_time(seg.get("时间段", seg.get("关键帧", "0s")))
                skeleton.append({
                    "关键帧": f"{start}s",
                    "_ts": start,
                    "主体": "",
                    "主体心情": "",
                    "主体声音内容": seg.get("主体声音内容", ""),
                    "主体声音风格": seg.get("主体声音风格", ""),
                    "背景": "",
                    "背景风格": "",
                    "背景声音内容": seg.get("环境声音内容", seg.get("背景声音内容", "")),
                    "背景声音风格": seg.get("环境声音风格", seg.get("背景声音风格", "")),
                    "_source": "audio",
                })
            skeleton.sort(key=lambda x: x["_ts"])

        elif isinstance(text_result, list) and len(text_result) > 0:
            # Fallback 2: text analysis produced multiple scenes — use each as an anchor
            _log(f"[Verifier] Fallback to text modality ({len(text_result)} scenes)", indent=2)
            for idx, scene in enumerate(text_result):
                # Handle both dict and JSON-string items
                if isinstance(scene, str):
                    try:
                        scene = json.loads(scene)
                    except (json.JSONDecodeError, ValueError):
                        continue
                if not isinstance(scene, dict):
                    continue
                skeleton.append({
                    "关键帧": scene.get("关键帧", f"{idx * 10}s"),
                    "_ts": _parse_timestamp(scene.get("关键帧", f"{idx * 10}s")),
                    "主体": scene.get("主体", ""),
                    "主体心情": scene.get("主体心情", scene.get("心情", "")),
                    "主体声音内容": scene.get("主体声音内容", ""),
                    "主体声音风格": scene.get("主体声音风格", ""),
                    "背景": scene.get("背景", ""),
                    "背景风格": scene.get("背景风格", ""),
                    "背景声音内容": scene.get("背景声音内容", ""),
                    "背景声音风格": scene.get("背景声音风格", ""),
                    "_source": "text",
                })
            if skeleton:
                skeleton.sort(key=lambda x: x["_ts"])

        else:
            # Fallback 3: single frame from photo or text dict
            _log("[Verifier] Fallback to single frame", indent=2)
            src = {}
            if isinstance(photo_result, dict) and photo_result:
                src = photo_result
            elif isinstance(text_result, dict) and text_result:
                src = text_result
            skeleton.append({
                "关键帧": "0s",
                "_ts": 0.0,
                "主体": src.get("主体", ""),
                "主体心情": src.get("主体心情", ""),
                "主体声音内容": "",
                "主体声音风格": "",
                "背景": src.get("背景", ""),
                "背景风格": src.get("背景风格", ""),
                "背景声音内容": "",
                "背景声音风格": "",
                "_source": "fallback",
            })

        _log(f"[Verifier] Stage 1 complete: {len(skeleton)} anchor points", indent=2)
        return {
            "temporal_skeleton": skeleton,
            "stage1_complete": True,
            "observation_message": [{"role": "assistant", "content": f"Stage 1 complete: {len(skeleton)} temporal anchors"}],
        }

    # ------------------------------------------------------------------
    # Stage 2: Inter-modal Interpolation (LLM-assisted)
    # ------------------------------------------------------------------

    async def _interpolation_node(self, state: Graph) -> dict:
        """
        Stage 2: K_VI = K_V ∪ {(img_j, t'_j)}

        Insert image keyframes into the temporal skeleton.
        If img_j falls between t_i and t_{i+1}, it receives midpoint timestamp
        t'_j = mid(t_i, t_{i+1}).

        LLM decides: merge into existing frame OR insert as new keyframe.
        """
        _log("[Verifier] Stage 2: Inter-modal Interpolation", indent=1)

        skeleton = state.get("temporal_skeleton", [])
        photo_result = state.get("expert_results", {}).get("photo_result", {})

        if not skeleton:
            _log("[Verifier] No skeleton available, passing through", indent=2)
            return {"interpolated_timeline": [], "stage2_complete": True}

        if not photo_result or not isinstance(photo_result, dict) or not any(
            photo_result.get(k) for k in NINE_FIELDS if k != "关键帧"
        ):
            _log("[Verifier] No photo data to interpolate, passing through", indent=2)
            return {"interpolated_timeline": [dict(f) for f in skeleton], "stage2_complete": True}

        timeline = [dict(f) for f in skeleton]

        # Build LLM prompt for merge/insert decision
        skeleton_summary = json.dumps(
            [{"index": i, "关键帧": f["关键帧"], "主体": f.get("主体", ""), "背景": f.get("背景", "")}
             for i, f in enumerate(timeline)],
            ensure_ascii=False, indent=2)

        photo_summary = json.dumps(
            {k: photo_result.get(k, "") for k in NINE_FIELDS},
            ensure_ascii=False, indent=2)

        prompt = f"""You are performing Stage 2 (Inter-modal Interpolation) of multimodal fusion.

Current temporal skeleton (from video/audio):
{skeleton_summary}

Image analysis result to integrate:
{photo_summary}

Decision required:
1. If the image content closely matches an existing keyframe, MERGE it (supplement empty fields only).
2. If the image represents a distinct moment not in the skeleton, INSERT it as a new keyframe with midpoint timestamp.

Return a JSON object:
{{
    "action": "merge" or "insert",
    "target_index": <index of keyframe to merge into (if merge)>,
    "reasoning": "<brief explanation>"
}}

Only return the JSON object, nothing else."""

        try:
            messages = [HumanMessage(content=prompt)]
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            response_text = _clean_markdown_json(response_text)

            # Extract JSON from potentially verbose response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                decision = {"action": "merge", "target_index": 0}

            action = decision.get("action", "merge")
            target_idx = decision.get("target_index", 0)

            if action == "insert":
                # Calculate midpoint timestamp
                if len(timeline) >= 2:
                    idx = min(target_idx, len(timeline) - 1)
                    if idx < len(timeline) - 1:
                        mid_ts = (timeline[idx]["_ts"] + timeline[idx + 1]["_ts"]) / 2.0
                    else:
                        mid_ts = timeline[-1]["_ts"] + 1.0
                else:
                    mid_ts = timeline[0]["_ts"] + 1.0

                new_frame = {
                    "关键帧": f"{mid_ts}s",
                    "_ts": mid_ts,
                    "_source": "photo_insert",
                }
                for field in NINE_FIELDS:
                    if field != "关键帧":
                        new_frame[field] = photo_result.get(field, "")
                timeline.append(new_frame)
                timeline.sort(key=lambda x: x["_ts"])
                _log(f"[Verifier] Stage 2: Inserted image as new keyframe at {mid_ts}s", indent=2)
            else:
                # Merge into existing frame (fill empty fields only)
                idx = min(target_idx, len(timeline) - 1) if timeline else 0
                if idx < len(timeline):
                    for field in NINE_FIELDS:
                        if field != "关键帧" and not timeline[idx].get(field):
                            timeline[idx][field] = photo_result.get(field, "")
                    _log(f"[Verifier] Stage 2: Merged image into keyframe {idx}", indent=2)

        except Exception as e:
            _log(f"[Verifier] Stage 2 LLM failed ({e}), falling back to merge-into-first", indent=2)
            # Fallback: merge into first frame
            if timeline:
                for field in NINE_FIELDS:
                    if field != "关键帧" and not timeline[0].get(field):
                        timeline[0][field] = photo_result.get(field, "")

        _log(f"[Verifier] Stage 2 complete: {len(timeline)} keyframes", indent=2)
        return {
            "interpolated_timeline": timeline,
            "stage2_complete": True,
            "observation_message": [{"role": "assistant", "content": f"Stage 2 complete: {len(timeline)} keyframes"}],
        }

    # ------------------------------------------------------------------
    # Stage 3: Audio-Visual Alignment (LLM-assisted)
    # ------------------------------------------------------------------

    async def _av_alignment_node(self, state: Graph) -> dict:
        """
        Stage 3: K_VIA = Φ(K_VI, A)

        Map audio features into overlapping keyframes.
        Audio attaches to the visual skeleton — never overrides it.
        Keyframe count must not decrease.
        """
        _log("[Verifier] Stage 3: Audio-Visual Alignment", indent=1)

        timeline = state.get("interpolated_timeline", [])
        audio_result = state.get("expert_results", {}).get("audio_result", [])

        if not timeline:
            _log("[Verifier] No timeline to align, passing through", indent=2)
            return {"av_aligned_timeline": [], "stage3_complete": True}

        if not audio_result or not isinstance(audio_result, list) or len(audio_result) == 0:
            _log("[Verifier] No audio data to align, passing through", indent=2)
            return {"av_aligned_timeline": [dict(f) for f in timeline], "stage3_complete": True}

        aligned = [dict(f) for f in timeline]

        # Build LLM prompt for audio-visual alignment
        timeline_summary = json.dumps(
            [{"index": i, "关键帧": f["关键帧"], "_ts": f.get("_ts", 0)}
             for i, f in enumerate(aligned)],
            ensure_ascii=False, indent=2)

        audio_summary = json.dumps(audio_result, ensure_ascii=False, indent=2)

        prompt = f"""You are performing Stage 3 (Audio-Visual Alignment) of multimodal fusion.

Visual timeline keyframes:
{timeline_summary}

Audio analysis segments:
{audio_summary}

Task: Map each audio segment to the nearest visual keyframe by temporal proximity.
Rules:
- Audio ATTACHES to visual keyframes, never replaces visual content
- Only fill audio-related fields that are currently empty (主体声音内容, 主体声音风格, 背景声音内容, 背景声音风格)
- Do NOT reduce the keyframe count

Return a JSON array of mappings:
[
    {{"audio_index": 0, "target_keyframe_index": 0, "fields": {{"主体声音内容": "...", "主体声音风格": "...", "背景声音内容": "...", "背景声音风格": "..."}}}}
]

Only return the JSON array, nothing else."""

        try:
            messages = [HumanMessage(content=prompt)]
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            mappings = _safe_parse_json_array(response_text)

            if mappings and isinstance(mappings, list):
                for mapping in mappings:
                    target = mapping.get("target_keyframe_index", 0)
                    fields = mapping.get("fields", {})
                    if 0 <= target < len(aligned):
                        audio_fields = ["主体声音内容", "主体声音风格", "背景声音内容", "背景声音风格"]
                        for af in audio_fields:
                            if af in fields and fields[af] and not aligned[target].get(af):
                                aligned[target][af] = fields[af]
                _log(f"[Verifier] Stage 3: Applied {len(mappings)} audio mappings", indent=2)
            else:
                raise ValueError("Could not parse LLM mapping response")

        except Exception as e:
            _log(f"[Verifier] Stage 3 LLM failed ({e}), using time-distance fallback", indent=2)
            aligned = self._simple_audio_alignment(aligned, audio_result)

        _log(f"[Verifier] Stage 3 complete: {len(aligned)} keyframes", indent=2)
        return {
            "av_aligned_timeline": aligned,
            "stage3_complete": True,
            "observation_message": [{"role": "assistant", "content": f"Stage 3 complete: {len(aligned)} keyframes"}],
        }

    def _simple_audio_alignment(self, timeline: list, audio_result: list) -> list:
        """Fallback: simple time-distance audio alignment (no LLM)."""
        for seg in audio_result:
            seg_start = _extract_start_time(seg.get("时间段", seg.get("关键帧", "0s")))
            # Find nearest keyframe
            best_idx = 0
            best_dist = float('inf')
            for i, frame in enumerate(timeline):
                dist = abs(frame.get("_ts", 0) - seg_start)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            # Fill audio fields if empty
            for af, sf in [("主体声音内容", "主体声音内容"), ("主体声音风格", "主体声音风格"),
                           ("背景声音内容", "环境声音内容"), ("背景声音风格", "环境声音风格")]:
                if not timeline[best_idx].get(af):
                    val = seg.get(sf, seg.get(af, ""))
                    if val:
                        timeline[best_idx][af] = val
        return timeline

    # ------------------------------------------------------------------
    # Stage 4: Semantic Refinement (LLM-assisted)
    # ------------------------------------------------------------------

    async def _semantic_refinement_node(self, state: Graph) -> dict:
        """
        Stage 4: S = R(K_VIA | T_bg, F_later → F_early)

        Apply posterior correction using global constraints T_bg from text analysis.
        Later high-confidence features correct earlier ambiguous descriptions
        (backward propagation).
        """
        _log("[Verifier] Stage 4: Semantic Refinement", indent=1)

        timeline = state.get("av_aligned_timeline", [])
        text_result = state.get("expert_results", {}).get("text_result", {})
        constraints = state.get("team1_constraints", [])
        user_req = state.get("user_requirement", "")

        if not timeline:
            _log("[Verifier] No timeline to refine", indent=2)
            return {"refined_scenes": [], "json_scene_result": [], "stage4_complete": True}

        # Extract T_bg (global background constraint from text)
        t_bg = ""
        if isinstance(text_result, dict):
            bg_parts = [text_result.get("背景", ""), text_result.get("背景风格", "")]
            t_bg = " ".join(p for p in bg_parts if p).strip()

        # Include reflection feedback if available (for retry)
        reflection_feedback = ""
        reflection = state.get("reflection_result", {})
        if reflection and reflection.get("feedback"):
            reflection_feedback = f"\n\nPrevious validation feedback to address:\n{reflection['feedback']}"

        # Build refinement prompt
        scenes_json = json.dumps(timeline, ensure_ascii=False, indent=2)
        constraint_text = "\n".join(f"- {c}" for c in constraints) if constraints else "None"

        system_prompt = f"""You are a semantic refinement expert performing posterior correction on multimodal scene data.

## Posterior Correction Mechanism
S = R(K_VIA | T_bg, F_later→F_early)
- T_bg (global background constraint): {t_bg or 'Not available'}
- Backward propagation: later high-confidence features correct earlier ambiguous descriptions

## Refinement Rules
1. Temporal consistency: check from last scene to first, ensure logical timeline
2. Mood-voice consistency: subject mood must match voice style (no contradictions)
3. Background consistency: background style must be coherent with background sound style
4. Global constraint: apply T_bg to ensure overall atmosphere coherence
5. Preserve keyframe count — do NOT add or remove scenes
6. Only modify field values to resolve inconsistencies, do not change 关键帧 timestamps
7. Ensure ALL 9 fields are non-empty for every scene

## User Requirement
{user_req}

## Team 1 Constraints
{constraint_text}
{reflection_feedback}

## Required 9 Fields per Scene
关键帧, 主体, 主体心情, 主体声音内容, 主体声音风格, 背景, 背景风格, 背景声音内容, 背景声音风格

## Output Format
Return a JSON array of scenes, each with exactly the 9 fields listed above.
Remove any temporary fields (_ts, _source).
Only return the JSON array, nothing else."""

        user_prompt = f"""Please refine the following scene timeline:

{scenes_json}

Apply posterior correction (backward propagation) and ensure all consistency requirements are met.
Return the refined scene list as a JSON array."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            _log("[Verifier] Calling LLM for semantic refinement...", indent=2)
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            refined = _safe_parse_json_array(response_text)

            if refined is None or not isinstance(refined, list) or len(refined) == 0:
                _log("[Verifier] LLM refinement parse failed, using fallback", indent=2)
                refined = self._fallback_refinement(timeline, t_bg)

        except Exception as e:
            _log(f"[Verifier] Stage 4 LLM failed ({e}), using fallback", indent=2)
            refined = self._fallback_refinement(timeline, t_bg)

        # Clean temporary fields
        for scene in refined:
            scene.pop("_ts", None)
            scene.pop("_source", None)
            # Ensure all 9 fields exist
            for field in NINE_FIELDS:
                if field not in scene:
                    scene[field] = ""

        _log(f"[Verifier] Stage 4 complete: {len(refined)} refined scenes", indent=2)
        return {
            "refined_scenes": refined,
            "json_scene_result": refined,
            "stage4_complete": True,
            "observation_message": [{"role": "assistant", "content": f"Stage 4 complete: {len(refined)} scenes"}],
        }

    def _fallback_refinement(self, timeline: list, t_bg: str) -> list:
        """Fallback refinement: clean fields, apply T_bg, ensure 9-field completeness."""
        refined = []
        for frame in timeline:
            scene = {}
            for field in NINE_FIELDS:
                scene[field] = frame.get(field, "")
            # Apply T_bg to empty background fields
            if t_bg:
                if not scene["背景"]:
                    scene["背景"] = t_bg.split()[0] if t_bg.split() else t_bg
                if not scene["背景风格"]:
                    scene["背景风格"] = t_bg
            refined.append(scene)
        return refined

    # ------------------------------------------------------------------
    # Validation (Pure Python — no LLM, deterministic)
    # ------------------------------------------------------------------

    def _validate_node(self, state: Graph) -> dict:
        """
        Validate fused scenes for consistency and completeness.
        Pure Python — deterministic, no LLM.

        Checks:
        V1: All 9 fields present and non-empty          (ERROR)
        V2: Keyframe timestamps ascending order          (ERROR)
        V3: Keyframe count >= video_result count          (ERROR)
        V4: Subject mood vs voice style consistency       (ERROR)
        V5: Background style vs sound style consistency   (WARNING)
        V6: Timestamp format parseable                    (WARNING)
        V7: Natural emotional transitions                 (WARNING)
        """
        _log("[Verifier] Validation", indent=1)

        scenes = state.get("json_scene_result", [])
        video_result = state.get("expert_results", {}).get("video_result", [])
        video_count = len(video_result) if isinstance(video_result, list) else 0

        issues = []
        warnings = []

        if not scenes:
            issues.append("No scenes to validate")
            return {
                "validation_passed": False,
                "validation_issues": issues,
                "validation_warnings": warnings,
                "validation_run": True,
            }

        # V1: 9-field completeness
        for i, scene in enumerate(scenes):
            missing = [f for f in NINE_FIELDS if not scene.get(f)]
            if missing:
                issues.append(f"Scene {i}: missing fields: {', '.join(missing)}")

        # V2: Keyframe timestamp ordering
        timestamps = []
        for i, scene in enumerate(scenes):
            ts = _parse_timestamp(scene.get("关键帧", ""))
            timestamps.append(ts)
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                issues.append(
                    f"Scene {i}: timestamp {timestamps[i]}s < previous {timestamps[i-1]}s (disorder)")

        # V3: Keyframe count vs video count
        if video_count > 0 and len(scenes) < video_count:
            issues.append(
                f"Keyframe count ({len(scenes)}) < video keyframe count ({video_count})")

        # V4: Mood-voice consistency
        for i, scene in enumerate(scenes):
            mood = scene.get("主体心情", "")
            voice_style = scene.get("主体声音风格", "")
            is_positive = any(m in mood for m in POSITIVE_MOODS)
            is_negative = any(m in mood for m in NEGATIVE_MOODS)

            if is_positive and any(kw in voice_style for kw in CONFLICTING_VOICE_KW):
                issues.append(
                    f"Scene {i}: mood-voice conflict: mood='{mood}' vs voice='{voice_style}'")
            if is_negative and any(kw in voice_style for kw in POSITIVE_VOICE_KW):
                issues.append(
                    f"Scene {i}: mood-voice conflict: mood='{mood}' vs voice='{voice_style}'")

        # V5: Background style vs sound style coherence (WARNING only)
        for i, scene in enumerate(scenes):
            bg_style = scene.get("背景风格", "")
            bg_sound = scene.get("背景声音风格", "")
            if bg_style and bg_sound:
                # Simple heuristic: both should share some semantic similarity
                # For now just flag empty mismatches
                pass

        # V6: Timestamp format parseable (WARNING)
        for i, scene in enumerate(scenes):
            ts_str = scene.get("关键帧", "")
            if ts_str and _parse_timestamp(ts_str) == 0.0 and ts_str != "0s" and ts_str != "0":
                warnings.append(f"Scene {i}: timestamp '{ts_str}' may not be parseable")

        # V7: Natural emotional transitions (WARNING)
        for i in range(1, len(scenes)):
            prev_mood = scenes[i - 1].get("主体心情", "")
            curr_mood = scenes[i].get("主体心情", "")
            prev_positive = any(m in prev_mood for m in POSITIVE_MOODS)
            curr_negative = any(m in curr_mood for m in NEGATIVE_MOODS)
            if prev_positive and curr_negative:
                warnings.append(
                    f"Scene {i}: abrupt mood shift from positive ('{prev_mood}') to negative ('{curr_mood}')")

        passed = len(issues) == 0
        _log(
            f"[Verifier] Validation {'PASSED' if passed else 'FAILED'}: "
            f"{len(issues)} issues, {len(warnings)} warnings", indent=2)

        return {
            "validation_passed": passed,
            "validation_issues": issues,
            "validation_warnings": warnings,
            "validation_run": True,
        }

    # ------------------------------------------------------------------
    # Reflect node
    # ------------------------------------------------------------------

    async def _reflect_node(self, state: Graph) -> dict:
        """
        Reflect: analyze validation failures, decide whether to retry Stage 4.
        Based on the Reflexion framework.
        """
        _log("[Verifier] Reflect", indent=1)

        issues = state.get("validation_issues", [])
        warnings_list = state.get("validation_warnings", [])
        reflection_count = state.get("reflection_count", 0) + 1

        # Build observation summary
        observation = (
            f"Validation failed with {len(issues)} issues and {len(warnings_list)} warnings.\n"
            f"Issues: {json.dumps(issues, ensure_ascii=False)}\n"
            f"Warnings: {json.dumps(warnings_list, ensure_ascii=False)}"
        )

        # Get historical reflections
        history_reflections = ""
        if self.reflection_memory:
            history_reflections = self.reflection_memory.get_summary()

        # Build reflection prompt
        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="scene_verification",
                task_description="Four-stage multimodal fusion and scene validation",
                observation=observation,
                history_reflections=history_reflections,
            )
        else:
            reflect_prompt = f"""You are a self-reflection expert. Analyze the following validation results:

Task: Four-stage multimodal fusion and scene validation
Observation: {observation}
History: {history_reflections or "None"}

Return JSON:
{{
    "analysis": "What went wrong in the fusion/refinement",
    "strengths": ["..."],
    "weaknesses": ["..."],
    "improvement": "Specific guidance for the next refinement attempt",
    "quality": "high/medium/low",
    "should_retry": true/false,
    "feedback": "Specific instructions to fix the issues"
}}"""

        try:
            messages = [SystemMessage(content=reflect_prompt)]
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            if REFLECTION_IMPORT_SUCCESS and parse_reflection_result:
                reflection_result = parse_reflection_result(response_text)
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        reflection_result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        reflection_result = {"quality": "low", "should_retry": False}
                else:
                    reflection_result = {"quality": "low", "should_retry": False}

            # Ensure feedback field exists
            if "feedback" not in reflection_result:
                reflection_result["feedback"] = reflection_result.get("improvement", "")

        except Exception as e:
            _log(f"[Verifier] Reflection LLM failed: {e}", indent=2)
            reflection_result = {
                "quality": "low",
                "should_retry": False,
                "feedback": f"Reflection failed: {e}",
            }

        # Save to memory
        if self.reflection_memory:
            self.reflection_memory.add_reflection(
                task_description="Scene verification",
                observation=observation[:200],
                reflection=reflection_result.get("analysis", ""),
                improvement=reflection_result.get("improvement", ""),
                quality=reflection_result.get("quality", "low"),
                iterations=reflection_count,
            )

        should_retry = reflection_result.get("should_retry", False)
        quality = reflection_result.get("quality", "low")

        _log(
            f"[Verifier] Reflection #{reflection_count}: quality={quality}, retry={should_retry}",
            indent=2)

        if should_retry and reflection_count < self.max_reflect_retries:
            # Reset Stage 4 to trigger re-run with feedback
            return {
                "reflection_result": reflection_result,
                "reflection_count": reflection_count,
                "stage4_complete": False,
                "validation_run": False,
                "state": "think",
            }

        # Accept current result
        return {
            "reflection_result": reflection_result,
            "reflection_count": reflection_count,
            "state": "final",
        }

    # ------------------------------------------------------------------
    # Final node
    # ------------------------------------------------------------------

    def _final_node(self, state: Graph) -> dict:
        """Output final results."""
        _log("[Verifier] Final node", indent=1)

        scenes = state.get("json_scene_result", [])
        passed = state.get("validation_passed", False)

        final_result = {
            "json_scene_result": scenes,
            "verification": {
                "passed": passed,
                "issues": state.get("validation_issues", []),
                "warnings": state.get("validation_warnings", []),
            },
        }

        return {
            "final_answer": json.dumps(final_result, ensure_ascii=False),
            "complete": True,
        }

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def ainvoke(
        self,
        expert_results: Dict[str, Any],
        user_requirement: str = "",
        team1_instruction: str = "",
        team1_constraints: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry: perform four-stage fusion and validation.

        Args:
            expert_results: {"text_result": Dict, "audio_result": List,
                             "photo_result": Dict, "video_result": List}
            user_requirement: R_user — original user input
            team1_instruction: Team 1's instruction for Team 2
            team1_constraints: Constraints from Team 1 task packet
        """
        initial_state = {
            "expert_results": expert_results,
            "user_requirement": user_requirement,
            "team1_instruction": team1_instruction,
            "team1_constraints": team1_constraints or [],
            # Four-stage outputs
            "temporal_skeleton": [],
            "interpolated_timeline": [],
            "av_aligned_timeline": [],
            "refined_scenes": [],
            # Stage flags
            "stage1_complete": False,
            "stage2_complete": False,
            "stage3_complete": False,
            "stage4_complete": False,
            # Validation
            "json_scene_result": [],
            "validation_passed": False,
            "validation_issues": [],
            "validation_warnings": [],
            "validation_run": False,
            # Control
            "state": "start",
            "tools": "none",
            "current_iteration": 0,
            "complete": False,
            "final_answer": "",
            # Reflection
            "reflection_result": {},
            "reflection_count": 0,
            "max_reflect_retries": self.max_reflect_retries,
            # Messages
            "global_messages": [],
            "system_prompt_messages": "",
            "think_message": [],
            "action_message": [],
            "observation_message": [],
        }

        _log("=" * 60, indent=0)
        _log("[SceneVerifier] Starting four-stage fusion pipeline", indent=0)
        _log("=" * 60, indent=0)

        response = await self.graph.ainvoke(initial_state, config={"recursion_limit": 50})

        return {
            "json_scene_result": response.get("json_scene_result", []),
            "verification": {
                "passed": response.get("validation_passed", False),
                "issues": response.get("validation_issues", []),
                "warnings": response.get("validation_warnings", []),
            },
            "stage_outputs": {
                "temporal_skeleton": response.get("temporal_skeleton", []),
                "interpolated_timeline": response.get("interpolated_timeline", []),
                "av_aligned_timeline": response.get("av_aligned_timeline", []),
                "refined_scenes": response.get("refined_scenes", []),
            },
            "reflection_history": response.get("reflection_result", {}),
            "complete": response.get("complete", False),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Standalone execution for testing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def _test():
        verifier = SceneVerifier()

        # Test with mock expert results
        expert_results = {
            "video_result": [
                {"关键帧": "0s", "主体": "warrior", "主体心情": "determined", "背景": "battlefield", "背景风格": "epic and intense"},
                {"关键帧": "5s", "主体": "warrior", "主体心情": "fierce", "背景": "burning village", "背景风格": "dark and smoky"},
                {"关键帧": "12s", "主体": "warrior", "主体心情": "sorrowful", "背景": "ruins", "背景风格": "desolate"},
            ],
            "audio_result": [
                {"时间段": "0s-5s", "主体声音内容": "battle cries", "主体声音风格": "fierce and powerful", "环境声音内容": "clashing swords", "环境声音风格": "metallic and intense"},
                {"时间段": "5s-12s", "主体声音内容": "heavy breathing", "主体声音风格": "exhausted", "环境声音内容": "crackling fire", "环境声音风格": "ominous"},
            ],
            "photo_result": {
                "主体": "armored warrior with sword",
                "主体心情": "resolute",
                "背景": "ancient fortress",
                "背景风格": "imposing stone walls",
            },
            "text_result": {
                "背景": "A tale of war and sacrifice in ancient China",
                "背景风格": "epic historical narrative",
                "主体": "a legendary general",
                "主体心情": "conflicted between duty and compassion",
            },
        }

        result = await verifier.ainvoke(
            expert_results=expert_results,
            user_requirement="Generate background music for a historical war drama",
            team1_instruction="Analyze all modalities and produce unified scene descriptions",
            team1_constraints=["Maintain historical accuracy", "Ensure emotional depth"],
        )

        print("\n" + "=" * 60)
        print("VERIFIER RESULT")
        print("=" * 60)
        print(f"Complete: {result['complete']}")
        print(f"Verification passed: {result['verification']['passed']}")
        print(f"Issues: {result['verification']['issues']}")
        print(f"Warnings: {result['verification']['warnings']}")
        print(f"\nScenes ({len(result['json_scene_result'])}):")
        for i, scene in enumerate(result['json_scene_result']):
            print(f"\n  Scene {i}:")
            for k, v in scene.items():
                print(f"    {k}: {v}")

    asyncio.run(_test())
