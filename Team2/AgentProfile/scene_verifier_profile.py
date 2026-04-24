"""
Scene Verifier Agent Profile

Defines the Team 2 Scene Understanding Verifier that performs four-stage
multimodal fusion and validates the resulting Unified Scene Representation.

Independent from the Supervisor — receives expert outputs, user requirement,
and Team 1 constraints directly.

Paper reference (Section 3.3.2):
    Stage 1: Temporal Anchoring — K_V = {(v_i, t_i)}
    Stage 2: Inter-modal Interpolation — K_VI = K_V ∪ {(img_j, t'_j)}
    Stage 3: Audio-Visual Alignment — K_VIA = Φ(K_VI, A)
    Stage 4: Semantic Refinement — S = R(K_VIA | T_bg, F_later→F_early)
    + Consistency Validation
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Use local dataclass definitions to avoid Pydantic enum validation issues
# with task.task_profile's ToolCategory (which lacks FUSION values).

class ToolCategory(str, Enum):
    ANALYSIS = "analysis"
    FUSION = "fusion"
    VALIDATION = "validation"


class KnowledgeDomain(str, Enum):
    MULTIMODAL = "multimodal_understanding"
    FUSION = "multimodal_fusion"
    VALIDATION = "validation"


@dataclass
class AgentRole:
    name: str
    description: str
    responsibilities: List[str]
    expertise: List[str]


@dataclass
class AgentTool:
    name: str
    function_signature: str
    description: str
    parameters: List[Dict[str, str]]
    returns: str
    category: ToolCategory
    usage_example: Optional[str] = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AgentKnowledge:
    domain: KnowledgeDomain
    concepts: List[str]
    rules: List[str]


@dataclass
class AgentProfile:
    agent_id: str
    description: str = ""
    role: AgentRole = None
    tools: List[AgentTool] = None
    knowledge: List[AgentKnowledge] = None
    constraints: List[str] = None
    best_practices: List[str] = None
    resources: List[str] = None
    run_methods: List[str] = None
    command: Optional[str] = None
    guide_book: Optional[str] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.knowledge is None:
            self.knowledge = []
        if self.constraints is None:
            self.constraints = []
        if self.best_practices is None:
            self.best_practices = []
        if self.resources is None:
            self.resources = []
        if self.run_methods is None:
            self.run_methods = []


SCENE_VERIFIER_ROLE = AgentRole(
    name="Scene Understanding Verifier",
    description=(
        "Independent verifier for Team 2 that performs four-stage multimodal fusion "
        "and validates the resulting Unified Scene Representation. Receives expert "
        "outputs, user requirement, and Team 1 constraints independently from the Supervisor."
    ),
    responsibilities=[
        "Perform Stage 1: Temporal Anchoring — build keyframe skeleton from video",
        "Perform Stage 2: Inter-modal Interpolation — insert image keyframes into timeline",
        "Perform Stage 3: Audio-Visual Alignment — map audio segments to visual timeline",
        "Perform Stage 4: Semantic Refinement — posterior correction R(K_VIA | T_bg, F_later→F_early)",
        "Validate scene consistency (mood-voice, background-sound, temporal ordering)",
        "Report pass/fail with detailed issue list and warnings",
    ],
    expertise=[
        "Four-stage multimodal fusion pipeline",
        "Temporal anchoring and keyframe extraction",
        "Cross-modal interpolation and alignment",
        "Posterior correction mechanisms",
        "Scene consistency validation",
        "9-field unified scene representation",
    ],
)


SCENE_VERIFIER_TOOLS = [
    AgentTool(
        name="temporal_anchoring",
        function_signature="temporal_anchoring(video_result: List[Dict], audio_result: List[Dict]) -> List[Dict]",
        description=(
            "Stage 1: Build temporal keyframe skeleton K_V = {(v_i, t_i)} from video keyframes. "
            "Fallback hierarchy: video → audio → photo/text → single frame."
        ),
        parameters=[
            {"name": "video_result", "type": "List[Dict]", "description": "Video analysis result with keyframe timestamps"},
            {"name": "audio_result", "type": "List[Dict]", "description": "Audio analysis result with time segments (fallback)"},
        ],
        returns="List of temporal anchor points forming the keyframe skeleton",
        category=ToolCategory.FUSION,
    ),
    AgentTool(
        name="inter_modal_interpolation",
        function_signature="inter_modal_interpolation(skeleton: List[Dict], photo_result: Dict) -> List[Dict]",
        description=(
            "Stage 2: K_VI = K_V ∪ {(img_j, t'_j)}. Insert image keyframes into the temporal "
            "skeleton. If img_j falls between t_i and t_{i+1}, it receives midpoint timestamp "
            "t'_j = mid(t_i, t_{i+1}). LLM decides: merge into existing frame or insert as new."
        ),
        parameters=[
            {"name": "skeleton", "type": "List[Dict]", "description": "Temporal skeleton from Stage 1"},
            {"name": "photo_result", "type": "Dict", "description": "Photo analysis result"},
        ],
        returns="Interpolated timeline with image keyframes inserted",
        category=ToolCategory.FUSION,
    ),
    AgentTool(
        name="audio_visual_alignment",
        function_signature="audio_visual_alignment(timeline: List[Dict], audio_result: List[Dict]) -> List[Dict]",
        description=(
            "Stage 3: K_VIA = Φ(K_VI, A). Map audio features into overlapping keyframes. "
            "Audio attaches to the visual skeleton — never overrides it. "
            "Keyframe count must not decrease."
        ),
        parameters=[
            {"name": "timeline", "type": "List[Dict]", "description": "Interpolated timeline from Stage 2"},
            {"name": "audio_result", "type": "List[Dict]", "description": "Audio analysis result with time segments"},
        ],
        returns="Audio-visual aligned timeline",
        category=ToolCategory.FUSION,
    ),
    AgentTool(
        name="semantic_refinement",
        function_signature="semantic_refinement(timeline: List[Dict], text_result: Dict, constraints: List[str]) -> List[Dict]",
        description=(
            "Stage 4: S = R(K_VIA | T_bg, F_later→F_early). Apply posterior correction using "
            "global constraints T_bg from text analysis. Later high-confidence features "
            "correct earlier ambiguous descriptions (backward propagation)."
        ),
        parameters=[
            {"name": "timeline", "type": "List[Dict]", "description": "AV-aligned timeline from Stage 3"},
            {"name": "text_result", "type": "Dict", "description": "Text analysis providing global constraint T_bg"},
            {"name": "constraints", "type": "List[str]", "description": "Additional constraints from Team 1"},
        ],
        returns="Refined scene list with posterior corrections applied",
        category=ToolCategory.FUSION,
    ),
    AgentTool(
        name="validate_scene_consistency",
        function_signature="validate_scene_consistency(scenes: List[Dict], video_count: int) -> Dict[str, Any]",
        description=(
            "Validate fused scenes for consistency and completeness. Checks: "
            "9-field presence, timestamp ordering, keyframe count, mood-voice consistency, "
            "background-sound consistency. Returns pass/fail with issues and warnings."
        ),
        parameters=[
            {"name": "scenes", "type": "List[Dict]", "description": "Final scene list to validate"},
            {"name": "video_count", "type": "int", "description": "Original video keyframe count for comparison"},
        ],
        returns="Dict with passed (bool), issues (List[str]), warnings (List[str])",
        category=ToolCategory.VALIDATION,
    ),
]


SCENE_VERIFIER_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.MULTIMODAL,
        concepts=[
            "K_V = {(v_i, t_i)} — temporal anchoring from video keyframes",
            "K_VI = K_V ∪ {(img_j, t'_j)} — inter-modal interpolation with midpoint timestamps",
            "K_VIA = Φ(K_VI, A) — audio-visual alignment mapping function",
            "S = R(K_VIA | T_bg, F_later→F_early) — semantic refinement with posterior correction",
            "9-field unified scene representation: keyframe, subject, subject_mood, subject_voice_content, subject_voice_style, background, background_style, background_sound_content, background_sound_style",
        ],
        rules=[
            "Video modality provides the temporal anchoring baseline",
            "Image keyframes receive midpoint timestamps t'_j = mid(t_i, t_{i+1})",
            "Audio attaches to the visual skeleton, never overrides it",
            "Text provides global constraint T_bg for semantic refinement",
            "Later high-confidence features correct earlier ambiguous descriptions (backward propagation)",
            "Keyframe count must not decrease through any fusion stage",
            "All output scenes must have exactly 9 fields",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.VALIDATION,
        concepts=[
            "Scene consistency validation",
            "Mood-voice alignment checking",
            "Background-sound coherence verification",
            "Temporal ordering enforcement",
            "9-field completeness validation",
        ],
        rules=[
            "All 9 fields must be present and non-empty",
            "Keyframe timestamps must be in ascending order",
            "Subject mood must be consistent with voice style",
            "Background style must be coherent with sound style",
            "Keyframe count after fusion must be >= original video keyframe count",
        ],
    ),
]


SCENE_VERIFIER_CONSTRAINTS = [
    "Must execute four fusion stages sequentially: Temporal Anchoring → Interpolation → AV Alignment → Refinement",
    "Must preserve keyframe count — count may increase but never decrease through stages",
    "All output scenes must have exactly 9 fields with non-empty values",
    "Audio content must attach to the visual skeleton, not override it",
    "Semantic refinement must apply backward propagation (later→earlier correction)",
    "Validation must be deterministic (pure Python, no LLM) for reproducibility",
    "Must report all issues as errors (blocking) or warnings (non-blocking)",
    "Maximum 2 reflect-retry cycles before accepting current result",
]


SCENE_VERIFIER_BEST_PRACTICES = [
    "Start with Temporal Anchoring using video as primary timeline",
    "Use fallback hierarchy (video → audio → photo/text) when video is unavailable",
    "Preserve existing field values when merging — only fill empty fields",
    "Extract global constraint T_bg from text_result for Stage 4",
    "Apply backward propagation in Stage 4: iterate from last scene to first",
    "Clean temporary fields (_ts, _source) before outputting final scenes",
    "Run validation immediately after Stage 4 to catch issues early",
]


SCENE_VERIFIER_PROFILE = AgentProfile(
    agent_id="scene_verifier_v1",
    description=(
        "Team 2 Scene Understanding Verifier implementing four-stage multimodal fusion "
        "(Temporal Anchoring, Inter-modal Interpolation, Audio-Visual Alignment, Semantic Refinement) "
        "and consistency validation. Follows Think-Act-Observe-Reflect iterative loop. "
        "Independent from the Supervisor — receives expert results and context directly."
    ),
    role=SCENE_VERIFIER_ROLE,
    tools=SCENE_VERIFIER_TOOLS,
    knowledge=SCENE_VERIFIER_KNOWLEDGE,
    constraints=SCENE_VERIFIER_CONSTRAINTS,
    best_practices=SCENE_VERIFIER_BEST_PRACTICES,
    resources=[
        "Expert results: text_result (Dict), audio_result (List), photo_result (Dict), video_result (List)",
        "User requirement (R_user) from original input",
        "Team 1 instruction and constraints for Team 2",
        "DashScope multimodal LLM for Stages 2, 3, 4",
        "Reflection memory module for experience-based improvement",
    ],
    run_methods=[
        "Receive expert results + context from Supervisor",
        "Stage 1: Temporal Anchoring — build keyframe skeleton from video/audio",
        "Stage 2: Inter-modal Interpolation — insert photo keyframes (LLM-assisted)",
        "Stage 3: Audio-Visual Alignment — map audio to visual timeline (LLM-assisted)",
        "Stage 4: Semantic Refinement — posterior correction with T_bg (LLM-assisted)",
        "Validate scene consistency (pure Python, deterministic)",
        "If validation fails: Reflect → Retry Stage 4 (max 2 retries)",
        "Output json_scene_result with 9-field unified scenes",
    ],
    command="scene_verification",
    guide_book=(
        "Four-Stage Multimodal Fusion Pipeline:\n"
        "Stage 1: Temporal Anchoring — K_V = {(v_i, t_i)} from video keyframes\n"
        "Stage 2: Inter-modal Interpolation — K_VI = K_V ∪ {(img_j, t'_j)} with midpoint timestamps\n"
        "Stage 3: Audio-Visual Alignment — K_VIA = Φ(K_VI, A) mapping audio to visual skeleton\n"
        "Stage 4: Semantic Refinement — S = R(K_VIA | T_bg, F_later→F_early) posterior correction\n"
        "\n"
        "Validation Checks:\n"
        "V1: All 9 fields present and non-empty (ERROR)\n"
        "V2: Keyframe timestamps in ascending order (ERROR)\n"
        "V3: Keyframe count >= original video count (ERROR)\n"
        "V4: Subject mood vs voice style consistency (ERROR)\n"
        "V5: Background style vs sound style coherence (WARNING)\n"
        "V6: Timestamp format parseable (WARNING)\n"
        "V7: Natural emotional transitions between adjacent scenes (WARNING)\n"
    ),
)


if __name__ == "__main__":
    print("Scene Understanding Verifier Agent Profile")
    print("=" * 80)
    print(f"Agent ID: {SCENE_VERIFIER_PROFILE.agent_id}")
    print(f"Role: {SCENE_VERIFIER_PROFILE.role.name}")
    print(f"Description: {SCENE_VERIFIER_PROFILE.description}")
    print(f"\nTools ({len(SCENE_VERIFIER_PROFILE.tools)}):")
    for tool in SCENE_VERIFIER_PROFILE.tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")
    print(f"\nConstraints ({len(SCENE_VERIFIER_PROFILE.constraints)}):")
    for i, c in enumerate(SCENE_VERIFIER_PROFILE.constraints, 1):
        print(f"  {i}. {c}")
