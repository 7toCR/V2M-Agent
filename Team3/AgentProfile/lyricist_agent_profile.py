"""
Lyricist Agent Profile — Merged from pop_gt_lyric + pop_idx profiles.
Role: Lyricist and Song Title Expert (generates idx + gt_lyric).
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
    LYRIC_CREATION = "lyric_creation"
    COMPOSITION = "composition"
    ID_GENERATION = "id_generation"


class KnowledgeDomain(str, Enum):
    LYRIC_WRITING = "lyric_writing"
    MUSIC_COMPOSITION = "music_composition"
    STRUCTURE_ANALYSIS = "structure_analysis"
    NAMING = "naming"
    MULTILINGUAL = "multilingual"


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


from Team3.Expert.prompt import COMMAND_lyricist, Guide_Book_lyricist  # noqa: E402

LYRICIST_AGENT_ROLE = AgentRole(
    name="Lyricist and Song Title Expert",
    description=(
        "Responsible for generating song title identifiers (idx) and lyrics (gt_lyric) "
        "based on scene descriptions. Supports vocal songs (Chinese/English/mixed), "
        "BGM-only structure tags, and NotaGen classical notation mode."
    ),
    responsibilities=[
        "Generate time-based song title identifiers (idx) with meaningful names",
        "Create vocal lyrics matching scene emotion, narrative, and atmosphere",
        "Design standard music structure tag sequences for BGM mode",
        "Support NotaGen classical mode with structure-only output",
        "Ensure title language matches lyric language",
        "Output standardized format compatible with lyric.jsonl schema",
    ],
    expertise=[
        "Lyric creation art (Chinese, English, mixed)",
        "Modern music composition techniques",
        "Song ID generation with datetime prefix",
        "Blueprint-aware mode switching (SongGeneration vs NotaGen)",
        "Batch lyric and title generation",
    ],
)

LYRICIST_AGENT_TOOLS = [
    AgentTool(
        name="generate_lyrics_and_title",
        function_signature=(
            "async def _generate_lyrics_and_title_node(self, state: 'LyricistAgent.Graph') "
            "-> Dict[str, Any]"
        ),
        description=(
            "Generate song titles (idx) and lyrics (gt_lyric) based on json_scene and blueprint. "
            "In SongGeneration vocal mode, creates full lyrics with structure tags. "
            "In SongGeneration BGM mode, creates structure-tag-only sequences. "
            "In NotaGen mode, creates classical notation structure tags."
        ),
        parameters=[
            {"name": "json_scene", "type": "List[Dict]", "description": "Scene data array from Team 2"},
            {"name": "piece", "type": "int", "description": "Number of songs to generate"},
            {"name": "blueprint", "type": "Dict", "description": "Musical Blueprint from supervisor"},
        ],
        returns='Dict with "idx_list" (List[str]) and "lyric_list" (List[str])',
        category=ToolCategory.LYRIC_CREATION,
        usage_example=(
            '{"idx_list": ["2026-04-11-14-30-Sunset Dreams"], '
            '"lyric_list": ["[intro-short] ; [verse] A line of gold... ; [chorus] We dance... ; [outro-short]"]}'
        ),
    ),
]

LYRICIST_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.LYRIC_WRITING,
        concepts=[
            "Verse: narrative foundation, emotion buildup (4-8 sentences)",
            "Chorus: emotional climax, core memory point (4-6 sentences)",
            "Bridge: emotional twist, perspective elevation (2-4 sentences)",
            "Instrumental sections: [intro-*], [outro-*], [inst-*] — no lyric content",
            "Lyric form patterns: A+C/B+C, A1+A2+C/B1+B2+C, A1+B2+C1+C2",
        ],
        rules=[
            "All punctuation must be half-width ASCII: comma ',', period '.', semicolon ';'",
            "Structure tags separated by ' ; ' (space-semicolon-space)",
            "Only valid tags: [intro-short/medium/long], [verse], [chorus], [bridge], [outro-short/medium/long], [inst-short/medium/long]",
            "Vocal mode: must contain at least one [verse] and one [chorus]",
            "BGM mode: only structure tags, no lyric text between tags",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.NAMING,
        concepts=[
            "Song ID format: YYYY-MM-DD-HH-MM-SongName",
            "Song name should reflect scene emotion and theme",
            "Chinese titles pair with Chinese lyrics, English titles with English lyrics",
            "Names should be concise, memorable, and musically evocative",
            "Batch IDs share timestamp prefix but have unique names",
        ],
        rules=[
            "idx must start with datetime prefix in format YYYY-MM-DD-HH-MM",
            "Song name follows datetime prefix, separated by hyphen",
            "Title language must match lyrics language (Chinese title = Chinese lyrics)",
            "No duplicate song names within the same batch",
            "Names should be 1-4 words, modern music style",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.MUSIC_COMPOSITION,
        concepts=[
            "Melody-lyrics synchronization: upward melody for rising emotion, downward for release",
            "Syllabic vs melismatic singing: one-note-per-syllable for narrative, multi-note for emotion",
            "Breath point design: align with lyric punctuation, phrase length 4-8 bars",
            "Song structure: intro -> verse -> chorus -> bridge -> outro",
            "Emotional arc: setup-development-climax-resolution",
        ],
        rules=[
            "Extract core emotion from json_scene for lyric creation",
            "Maintain consistent emotional tone across all verses and choruses",
            "Each chorus should have a memorable hook line",
            "Bridge provides contrast or perspective shift",
            "Lyrics should be poetic, literary, and emotionally deep",
        ],
    ),
]

LYRICIST_AGENT_CONSTRAINTS = [
    "Must create lyrics based on json_scene emotion and narrative",
    "Output in standardized JSON format with idx_list and lyric_list keys",
    "Only use valid structure tags — do NOT invent new tags",
    "All punctuation must be half-width (ASCII): comma, period, semicolon",
    "No empty lyrics for vocal mode — every [verse], [chorus], [bridge] must contain text",
    "Title language must match lyric language",
    "Support batch generation for multiple pieces",
    "In NotaGen mode, output structure tags only (no lyrics)",
]

LYRICIST_AGENT_BEST_PRACTICES = [
    "Analyze scene emotion deeply before starting creation",
    "Draw inspiration from scene imagery but reconstruct artistically",
    "Use varied sentence lengths and poetic language",
    "Design memorable chorus hooks that capture scene essence",
    "For BGM mode, create musically logical structure progression",
    "Generate diverse styles across batch — avoid repetition",
    "Verify half-width punctuation before finalizing output",
]

LYRICIST_AGENT_PROFILE = AgentProfile(
    agent_id="lyricist_agent_v1",
    description=(
        "Team 3 Lyricist and Song Title Expert. Generates idx (song title identifiers) "
        "and gt_lyric (lyrics with structure tags) based on Musical Blueprint and scene data. "
        "Supports vocal (Chinese/English/mixed), BGM, and NotaGen modes. "
        "Follows Think-Act-Observe-Reflect iterative loop."
    ),
    role=LYRICIST_AGENT_ROLE,
    tools=LYRICIST_AGENT_TOOLS,
    knowledge=LYRICIST_AGENT_KNOWLEDGE,
    constraints=LYRICIST_AGENT_CONSTRAINTS,
    best_practices=LYRICIST_AGENT_BEST_PRACTICES,
    resources=[
        "json_scene from Team 2 (9-field unified scene representation)",
        "Musical Blueprint from supervisor (model, lyric_style, emotional_key, language)",
        "songgeneration.json (lyric_structure_tags for validation)",
        "noatgen.json (classical notation structure reference)",
        "DashScope LLM (qwen3-max for reasoning, deepseek-v3.2 for generation)",
    ],
    run_methods=[
        "Receive task, json_scene, piece count, and blueprint",
        "Generate idx list (datetime prefix + LLM-generated song names)",
        "Generate lyric list based on blueprint mode (vocal/BGM/NotaGen)",
        "Validate structure tags and punctuation",
        "Reflect if quality insufficient (max 2 retries)",
        "Output {idx_list, lyric_list}",
    ],
    command=COMMAND_lyricist,
    guide_book=Guide_Book_lyricist,
)


if __name__ == "__main__":
    print(f"Agent ID: {LYRICIST_AGENT_PROFILE.agent_id}")
    print(f"Role: {LYRICIST_AGENT_PROFILE.role.name}")
    print(f"Tools: {[t.name for t in LYRICIST_AGENT_PROFILE.tools]}")
    print(f"Constraints: {len(LYRICIST_AGENT_PROFILE.constraints)}")
