"""
Stylist Agent Profile — Adapted from pop_audio_type_agent_profile.
Role: Music Style Selection Expert (selects audio_type).
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
    MUSIC_STYLE = "music_style"
    GENRE_ANALYSIS = "genre_analysis"


class KnowledgeDomain(str, Enum):
    MUSIC_GENRE = "music_genre"
    STYLE_ANALYSIS = "style_analysis"
    EMOTION_MATCHING = "emotion_matching"


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


from Team3.Expert.prompt import COMMAND_stylist, Guide_Book_stylist  # noqa: E402

STYLIST_AGENT_ROLE = AgentRole(
    name="Music Style Selection Expert",
    description=(
        "Responsible for selecting the most suitable music audio type based on scene "
        "descriptions. Chooses from 12 candidate styles defined in songgeneration.json. "
        "Returns 'N/A' for NotaGen mode (classical notation does not use audio_type)."
    ),
    responsibilities=[
        "Analyze scene emotion and atmosphere for style selection",
        "Select from 12 candidate music styles per piece",
        "Provide diverse style recommendations across batch",
        "Ensure style-scene emotional coherence",
        "Validate selections against songgeneration.json audio_type list",
        "Output standardized list format",
    ],
    expertise=[
        "Music genre analysis and classification",
        "Emotion-to-style mapping",
        "Scene atmosphere understanding",
        "Audio type vocabulary (12 candidates)",
        "Batch style recommendation with diversity",
    ],
)

STYLIST_AGENT_TOOLS = [
    AgentTool(
        name="select_audio_type",
        function_signature=(
            "async def _select_audio_type_node(self, state: 'StylistAgent.Graph') "
            "-> Dict[str, Any]"
        ),
        description=(
            "Select music audio type based on json_scene and blueprint. "
            "Must select from 12 candidates: Pop, R&B, Dance, Jazz, Folk, Rock, "
            "Chinese Style, Chinese Tradition, Metal, Reggae, Chinese Opera, Auto. "
            "For NotaGen mode, returns 'N/A'."
        ),
        parameters=[
            {"name": "json_scene", "type": "List[Dict]", "description": "Scene data array from Team 2"},
            {"name": "piece", "type": "int", "description": "Number of audio types to select"},
            {"name": "blueprint", "type": "Dict", "description": "Musical Blueprint from supervisor"},
        ],
        returns='Dict with "audio_type_list" (List[str])',
        category=ToolCategory.MUSIC_STYLE,
        usage_example='["Pop", "Folk"]',
    ),
]

STYLIST_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.MUSIC_GENRE,
        concepts=[
            "12 candidate audio types: Pop, R&B, Dance, Jazz, Folk, Rock, Chinese Style, "
            "Chinese Tradition, Metal, Reggae, Chinese Opera, Auto",
            "Pop: catchy melodies, electronic production, bright and positive",
            "R&B: soulful vocals, smooth production, emotional and sensual",
            "Rock: electric guitar/bass/drums core, high energy, direct expression",
            "Chinese Style: Chinese classical elements with modern production",
        ],
        rules=[
            "Each selection must be one of the 12 candidates exactly",
            "Default to 'Auto' when no clear style match exists",
            "Chinese scene imagery -> prefer Chinese Style or Chinese Tradition",
            "High energy scenes -> prefer Rock, Dance, or Metal",
            "Intimate/romantic scenes -> prefer Pop, R&B, or Jazz",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.EMOTION_MATCHING,
        concepts=[
            "Emotion-style mapping: joyful -> Pop/Dance, solemn -> Chinese Tradition",
            "Energy level matching: high energy -> Rock/Metal/Dance, low energy -> Jazz/Folk",
            "Scene formality: formal -> Classical/Chinese Tradition, casual -> Pop/Rock",
            "Social context: party -> Dance/Pop, intimate -> R&B/Jazz",
            "Cultural context: traditional Chinese -> Chinese Style/Tradition/Opera",
        ],
        rules=[
            "Match emotional tone first, then energy level",
            "Consider scene formality and social context",
            "Avoid contradictory pairings (e.g., Metal for romantic dinner)",
            "Use 'Auto' as safe fallback, not primary choice",
            "For NotaGen mode, return 'N/A' — classical notation has no audio_type",
        ],
    ),
]

STYLIST_AGENT_CONSTRAINTS = [
    "Must select from exactly 12 candidate audio types (strict vocabulary)",
    "Default return 1 audio type per piece",
    "Selection must be based on scene analysis and blueprint emotional_key",
    "Invalid selections must be replaced with 'Auto'",
    "For NotaGen mode, return 'N/A' for all pieces",
    "No fabricated or hallucinated audio type names",
    "Output as standardized list",
]

STYLIST_AGENT_BEST_PRACTICES = [
    "Analyze scene mood before selecting style",
    "Use blueprint.emotional_key as primary guidance",
    "Cross-reference with descriptions genre for coherence",
    "Vary styles across batch pieces when scenes differ",
    "Post-validate all selections against songgeneration.json audio_type list",
    "Default to 'Auto' when uncertain rather than guessing",
    "Consider audio_type-genre alignment rules from songgeneration.json",
]

STYLIST_AGENT_PROFILE = AgentProfile(
    agent_id="stylist_agent_v1",
    description=(
        "Team 3 Music Style Selection Expert. Selects auto_prompt_audio_type from "
        "12 candidate styles based on Musical Blueprint and scene data. "
        "Strict vocabulary validation against songgeneration.json. "
        "Follows Think-Act-Observe-Reflect iterative loop."
    ),
    role=STYLIST_AGENT_ROLE,
    tools=STYLIST_AGENT_TOOLS,
    knowledge=STYLIST_AGENT_KNOWLEDGE,
    constraints=STYLIST_AGENT_CONSTRAINTS,
    best_practices=STYLIST_AGENT_BEST_PRACTICES,
    resources=[
        "json_scene from Team 2 (9-field unified scene representation)",
        "Musical Blueprint from supervisor",
        "songgeneration.json (12 valid audio_type candidates)",
        "DashScope LLM for style selection",
    ],
    run_methods=[
        "Receive task, json_scene, piece count, and blueprint",
        "Check blueprint model — if NotaGen, return 'N/A' immediately",
        "Analyze scene emotion and atmosphere",
        "Select audio_type per piece (LLM-assisted)",
        "Post-validate against songgeneration.json audio_type list",
        "Replace invalid selections with 'Auto'",
        "Reflect if quality insufficient (max 2 retries)",
        "Output {audio_type_list}",
    ],
    command=COMMAND_stylist,
    guide_book=Guide_Book_stylist,
)


if __name__ == "__main__":
    print(f"Agent ID: {STYLIST_AGENT_PROFILE.agent_id}")
    print(f"Role: {STYLIST_AGENT_PROFILE.role.name}")
    print(f"Tools: {[t.name for t in STYLIST_AGENT_PROFILE.tools]}")
    print(f"Constraints: {len(STYLIST_AGENT_PROFILE.constraints)}")
