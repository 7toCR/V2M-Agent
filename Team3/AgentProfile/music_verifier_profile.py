"""
Music Verifier Profile — Pattern from Team2 scene_verifier_profile.
Role: Music Generation Verifier (3-stage validation pipeline).
"""
import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class ToolCategory(str, Enum):
    VALIDATION = "validation"
    ANALYSIS = "analysis"


class KnowledgeDomain(str, Enum):
    VOCABULARY = "vocabulary_validation"
    STRUCTURE = "structure_validation"
    CONSISTENCY = "consistency_validation"


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
        for attr in ("tools", "knowledge", "constraints", "best_practices", "resources", "run_methods"):
            if getattr(self, attr) is None:
                setattr(self, attr, [])


MUSIC_VERIFIER_ROLE = AgentRole(
    name="Music Generation Verifier",
    description=(
        "Independent verifier for Team 3 that validates generated music prompts through "
        "a 3-stage pipeline: vocabulary validation, lyric structure validation, and "
        "cross-component consistency checking. Receives assembled prompts, blueprint, "
        "and scene data independently from the Supervisor."
    ),
    responsibilities=[
        "Stage 1: Vocabulary Validation — check descriptions and audio_type against songgeneration.json",
        "Stage 2: Lyric Structure Validation — verify structure tags and song form",
        "Stage 3: Cross-Component Consistency — check emotion/timbre/genre/audio_type alignment",
        "Report pass/fail with detailed issue and warning lists",
        "Distinguish ERROR (blocking) from WARNING (non-blocking) issues",
        "Support both SongGeneration and NotaGen model validation",
    ],
    expertise=[
        "songgeneration.json vocabulary mastery (descriptions + audio_type)",
        "noatgen.json vocabulary mastery (periods, composers, instruments)",
        "Lyric structure tag validation",
        "Cross-field consistency rule checking",
        "LLM-assisted semantic consistency analysis",
    ],
)

MUSIC_VERIFIER_TOOLS = [
    AgentTool(
        name="vocabulary_validation",
        function_signature="vocabulary_validation(pop_prompt_result: List[Dict], blueprint: Dict) -> Dict",
        description=(
            "Stage 1: Parse each descriptions string into 6 fields and validate each "
            "against songgeneration.json vocabulary. Check audio_type against the 12 candidates. "
            "Pure Python — no LLM needed. Any mismatch is an ERROR."
        ),
        parameters=[
            {"name": "pop_prompt_result", "type": "List[Dict[str, str]]", "description": "Assembled prompts"},
            {"name": "blueprint", "type": "Dict", "description": "Musical Blueprint"},
        ],
        returns="Dict with issues list (field, value, allowed)",
        category=ToolCategory.VALIDATION,
    ),
    AgentTool(
        name="structure_validation",
        function_signature="structure_validation(pop_prompt_result: List[Dict], blueprint: Dict) -> Dict",
        description=(
            "Stage 2: Validate lyric structure tags. Check each gt_lyric section starts with "
            "a valid tag from songgeneration.json. For vocal mode: require [verse] + [chorus]. "
            "Check [intro-*] start and [outro-*] end. Pure Python. Violations are WARNINGs."
        ),
        parameters=[
            {"name": "pop_prompt_result", "type": "List[Dict[str, str]]", "description": "Assembled prompts"},
            {"name": "blueprint", "type": "Dict", "description": "Musical Blueprint"},
        ],
        returns="Dict with warnings list",
        category=ToolCategory.VALIDATION,
    ),
    AgentTool(
        name="consistency_check",
        function_signature="consistency_check(pop_prompt_result: List[Dict], blueprint: Dict, json_scene: List[Dict]) -> Dict",
        description=(
            "Stage 3: Cross-component consistency. Apply deterministic rules from "
            "songgeneration.json cross_field_rules (emotion-timbre, emotion-genre, "
            "audio_type-genre). Then LLM semantic check for lyrics-emotion alignment, "
            "BPM-intensity coherence. Contradictions are ERRORs."
        ),
        parameters=[
            {"name": "pop_prompt_result", "type": "List[Dict[str, str]]", "description": "Assembled prompts"},
            {"name": "blueprint", "type": "Dict", "description": "Musical Blueprint"},
            {"name": "json_scene", "type": "List[Dict]", "description": "Scene data for semantic check"},
        ],
        returns="Dict with issues and warnings lists",
        category=ToolCategory.VALIDATION,
    ),
]

MUSIC_VERIFIER_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.VOCABULARY,
        concepts=[
            "descriptions format: 'gender, emotion, genre, timbre, instrument, the bpm is N.'",
            "gender: [female, male]",
            "emotion: [sad, emotional, angry, happy, uplifting, intense, romantic, melancholic]",
            "audio_type: 12 candidates (Pop, R&B, Dance, Jazz, Folk, Rock, Chinese Style, "
            "Chinese Tradition, Metal, Reggae, Chinese Opera, Auto)",
            "BPM range: [60, 200]",
        ],
        rules=[
            "Every descriptions field must exist in songgeneration.json vocabulary",
            "audio_type must be one of the 12 exact candidates",
            "BPM must parse as integer in [60, 200]",
            "For NotaGen mode: descriptions follow different format (period/composer/instrument)",
            "Vocabulary checks are STRICT — zero tolerance for invalid values",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.STRUCTURE,
        concepts=[
            "Valid structure tags: [intro-short/medium/long], [verse], [chorus], [bridge], "
            "[outro-short/medium/long], [inst-short/medium/long]",
            "Vocal mode: must have >= 1 [verse] and >= 1 [chorus]",
            "BGM mode: structure tags only, no lyric text",
            "Song should start with [intro-*] and end with [outro-*]",
            "Sections separated by ' ; ' (space-semicolon-space)",
        ],
        rules=[
            "All tags must match songgeneration.json lyric_structure_tags exactly",
            "Vocal songs must contain [verse] and [chorus]",
            "BGM songs must not have text between tags",
            "Structure validation issues are WARNINGs (flexible, not blocking)",
            "Punctuation must be half-width ASCII only",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.CONSISTENCY,
        concepts=[
            "Emotion-timbre alignment (sad/melancholic + bright/rock = conflict)",
            "Emotion-genre alignment (angry + folk/jazz = conflict)",
            "Audio_type-genre coherence (Rock -> rock/hard rock/classic rock/pop rock/rock and roll)",
            "Language-title-lyric match (Chinese title -> Chinese lyrics)",
            "Blueprint emotional_key -> descriptions emotion alignment",
        ],
        rules=[
            "Apply cross_field_rules from songgeneration.json deterministically",
            "LLM semantic check for lyrics-emotion match, BPM-intensity, audio_type-mood",
            "Deterministic rule violations are ERRORs",
            "LLM-detected contradictions are ERRORs",
            "Maximum 2 reflect-retry cycles before accepting",
        ],
    ),
]

MUSIC_VERIFIER_CONSTRAINTS = [
    "Must execute three validation stages sequentially: vocabulary -> structure -> consistency",
    "Vocabulary validation (Stage 1) is pure Python — no LLM, zero tolerance",
    "Structure validation (Stage 2) is pure Python — violations are WARNINGs",
    "Consistency check (Stage 3) uses deterministic rules first, then LLM semantic check",
    "Must report all issues with severity: ERROR (blocking) or WARNING (non-blocking)",
    "Overall passed = True only when zero ERRORs",
    "Maximum 2 reflect-retry cycles",
    "Must support both SongGeneration and NotaGen validation modes",
]

MUSIC_VERIFIER_BEST_PRACTICES = [
    "Run Stage 1 first — vocabulary errors indicate fundamental generation failures",
    "Parse descriptions carefully: fields may contain commas (e.g., 'dance, electronic')",
    "Use the full vocabulary lists from songgeneration.json, not hardcoded subsets",
    "For Stage 3 deterministic rules, apply all cross_field_rules entries",
    "In LLM semantic check, provide scene context for better judgment",
    "Report specific field/value pairs in issues for targeted correction",
    "Aggregate all issues before returning — do not short-circuit on first error",
]

MUSIC_VERIFIER_PROFILE = AgentProfile(
    agent_id="music_verifier_v1",
    description=(
        "Team 3 Music Generation Verifier implementing 3-stage validation: "
        "Stage 1 Vocabulary Validation (pure Python, strict), "
        "Stage 2 Lyric Structure Validation (pure Python, warnings), "
        "Stage 3 Cross-Component Consistency (deterministic rules + LLM semantic check). "
        "Independent from the Supervisor. Follows Think-Act-Observe-Reflect iterative loop."
    ),
    role=MUSIC_VERIFIER_ROLE,
    tools=MUSIC_VERIFIER_TOOLS,
    knowledge=MUSIC_VERIFIER_KNOWLEDGE,
    constraints=MUSIC_VERIFIER_CONSTRAINTS,
    best_practices=MUSIC_VERIFIER_BEST_PRACTICES,
    resources=[
        "songgeneration.json (SongGeneration model vocabulary and rules)",
        "noatgen.json (NotaGen classical notation vocabulary)",
        "Assembled pop_prompt_result from supervisor",
        "Musical Blueprint from supervisor",
        "json_scene from Team 2 (for semantic consistency check)",
        "DashScope LLM (for Stage 3 semantic check only)",
    ],
    run_methods=[
        "Receive pop_prompt_result, blueprint, json_scene",
        "Stage 1: Vocabulary Validation (pure Python)",
        "Stage 2: Lyric Structure Validation (pure Python)",
        "Stage 3: Cross-Component Consistency (rules + LLM)",
        "Aggregate issues and warnings",
        "If ERRORs > 0: Reflect-Retry (max 2)",
        "Output {passed, issues, warnings}",
    ],
    command="music_verification",
    guide_book=(
        "Three-Stage Music Verification Pipeline:\n\n"
        "Stage 1 — Vocabulary Validation (Pure Python, STRICT):\n"
        "  V1: Parse descriptions into 6 fields, validate each against songgeneration.json\n"
        "  V2: Check audio_type against 12 candidates\n"
        "  V3: Validate BPM as integer in [60, 200]\n"
        "  Severity: ERROR (blocking)\n\n"
        "Stage 2 — Lyric Structure Validation (Pure Python, FLEXIBLE):\n"
        "  V4: All tags must be valid structure tags\n"
        "  V5: Vocal mode requires [verse] + [chorus]\n"
        "  V6: Song should start with [intro-*], end with [outro-*]\n"
        "  V7: Half-width punctuation only\n"
        "  Severity: WARNING (non-blocking)\n\n"
        "Stage 3 — Cross-Component Consistency (Rules + LLM):\n"
        "  V8: Emotion-timbre alignment (cross_field_rules)\n"
        "  V9: Emotion-genre alignment (cross_field_rules)\n"
        "  V10: Audio_type-genre coherence (cross_field_rules)\n"
        "  V11: Language-title-lyric match\n"
        "  V12: LLM semantic check (lyrics-emotion, BPM-intensity, audio_type-mood)\n"
        "  Severity: ERROR (blocking)\n"
    ),
)


if __name__ == "__main__":
    print(f"Agent ID: {MUSIC_VERIFIER_PROFILE.agent_id}")
    print(f"Role: {MUSIC_VERIFIER_PROFILE.role.name}")
    print(f"Tools: {[t.name for t in MUSIC_VERIFIER_PROFILE.tools]}")
    print(f"Constraints: {len(MUSIC_VERIFIER_PROFILE.constraints)}")
