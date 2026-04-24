"""
Composer Agent Profile — Adapted from pop_description_agent_profile.
Role: Music Description Composer (generates descriptions string).
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
    MUSIC_DESCRIPTION = "music_description"
    ATTRIBUTE_SELECTION = "attribute_selection"


class KnowledgeDomain(str, Enum):
    MUSIC_ATTRIBUTES = "music_attributes"
    EMOTION_ANALYSIS = "emotion_analysis"
    STYLE_COORDINATION = "style_coordination"
    INSTRUMENT_KNOWLEDGE = "instrument_knowledge"
    TEMPO_ANALYSIS = "tempo_analysis"


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


from Team3.Expert.prompt import COMMAND_composer, Guide_Book_composer  # noqa: E402

COMPOSER_AGENT_ROLE = AgentRole(
    name="Music Description Composer",
    description=(
        "Responsible for generating complete music description strings containing "
        "gender, emotion, genre, timbre, instrument, and bpm — six key attributes. "
        "Supports both SongGeneration (6-field format) and NotaGen (period/composer/instrument) modes."
    ),
    responsibilities=[
        "Analyze scene emotion and atmosphere for attribute selection",
        "Select one value from each of the 6 description categories",
        "Generate coordinated description string matching scene mood",
        "Validate all selected values against songgeneration.json vocabulary",
        "Support NotaGen mode with period/composer/instrument from noatgen.json",
        "Output standardized format: 'gender, emotion, genre, timbre, instrument, the bpm is N.'",
    ],
    expertise=[
        "Music attribute analysis and selection",
        "Emotion-to-style mapping",
        "Instrument pairing coordination",
        "Rhythm and tempo analysis",
        "Batch description generation with variety",
    ],
)

COMPOSER_AGENT_TOOLS = [
    AgentTool(
        name="generate_descriptions",
        function_signature=(
            "async def _generate_descriptions_node(self, state: 'ComposerAgent.Graph') "
            "-> Dict[str, Any]"
        ),
        description=(
            "Generate music description strings based on json_scene and blueprint. "
            "For SongGeneration: select from 6 categories (gender, emotion, genre, timbre, instrument, bpm). "
            "For NotaGen: generate period/composer/instrument description from noatgen.json."
        ),
        parameters=[
            {"name": "json_scene", "type": "List[Dict]", "description": "Scene data array from Team 2"},
            {"name": "piece", "type": "int", "description": "Number of descriptions to generate"},
            {"name": "blueprint", "type": "Dict", "description": "Musical Blueprint from supervisor"},
        ],
        returns='Dict with "descriptions_list" (List[str])',
        category=ToolCategory.MUSIC_DESCRIPTION,
        usage_example='"female, romantic, pop, bright, synthesizer and piano, the bpm is 125."',
    ),
]

COMPOSER_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.MUSIC_ATTRIBUTES,
        concepts=[
            "6 description categories: gender, emotion, genre, timbre, instrument, bpm",
            "Gender: female, male",
            "Emotion: sad, emotional, angry, happy, uplifting, intense, romantic, melancholic",
            "Genre: 27 valid values from pop to pop rock (see songgeneration.json)",
            "Timbre: dark, bright, warm, rock, varies, soft, vocal",
        ],
        rules=[
            "Must select exactly one value from each of the 6 categories",
            "All values must come from songgeneration.json vocabulary — no hallucination",
            "BPM must be integer in range [60, 200]",
            "Output format: 'gender, emotion, genre, timbre, instrument, the bpm is N.'",
            "For NotaGen mode: use period/composer/instrument from noatgen.json instead",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.EMOTION_ANALYSIS,
        concepts=[
            "Scene mood to music emotion mapping",
            "Emotion-timbre alignment (e.g., sad -> dark/soft, happy -> bright)",
            "Emotion-genre coherence (e.g., angry -> rock, romantic -> pop/R&B)",
            "Multi-scene emotional arc analysis",
            "Blueprint emotional_key as primary guidance",
        ],
        rules=[
            "Emotion must align with scene mood — do not contradict",
            "sad/melancholic emotions should NOT pair with bright/rock timbre",
            "angry emotion should NOT pair with folk/jazz genre",
            "Use blueprint.emotional_key as primary emotion reference",
            "Maintain consistent emotional tone across batch pieces",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.INSTRUMENT_KNOWLEDGE,
        concepts=[
            "39 valid instrument combinations from songgeneration.json",
            "Instrument-genre affinity (e.g., guitar and drums -> rock)",
            "Instrument-emotion compatibility (e.g., piano and strings -> romantic/emotional)",
            "Single vs dual instrument configurations",
            "NotaGen instrument categories: Chamber, Choral, Keyboard, Orchestral, etc.",
        ],
        rules=[
            "Instrument must be exactly one entry from songgeneration.json instrument list",
            "Instrument should complement the selected genre and emotion",
            "Avoid contradictory pairings (e.g., beats for classical genre)",
            "For NotaGen: select instrument category that the chosen composer supports",
            "Consider scene atmosphere when selecting instrument",
        ],
    ),
]

COMPOSER_AGENT_CONSTRAINTS = [
    "Must select one value from each of the 6 categories",
    "All values must exist in songgeneration.json vocabulary (strict validation)",
    "Output must follow exact format: 'gender, emotion, genre, timbre, instrument, the bpm is N.'",
    "BPM must be integer in [60, 200]",
    "Description must align with scene emotion and blueprint emotional_key",
    "No fabricated or hallucinated attribute values",
    "Support batch generation for multiple pieces with variety",
    "For NotaGen mode: generate classical notation descriptions from noatgen.json",
]

COMPOSER_AGENT_BEST_PRACTICES = [
    "Analyze scene mood patterns before selecting attributes",
    "Use blueprint.emotional_key as anchor for emotion selection",
    "Cross-validate emotion-timbre and emotion-genre pairs against conflict rules",
    "Vary BPM and genre across batch pieces for diversity",
    "Post-validate all generated descriptions against songgeneration.json",
    "Replace invalid tokens with closest valid match rather than failing",
    "For NotaGen: validate composer + instrument_category pairing against noatgen.json",
]

COMPOSER_AGENT_PROFILE = AgentProfile(
    agent_id="composer_agent_v1",
    description=(
        "Team 3 Music Description Composer. Generates descriptions strings containing "
        "6 key attributes (gender, emotion, genre, timbre, instrument, bpm) based on "
        "Musical Blueprint and scene data. Strict vocabulary validation against "
        "songgeneration.json. Follows Think-Act-Observe-Reflect iterative loop."
    ),
    role=COMPOSER_AGENT_ROLE,
    tools=COMPOSER_AGENT_TOOLS,
    knowledge=COMPOSER_AGENT_KNOWLEDGE,
    constraints=COMPOSER_AGENT_CONSTRAINTS,
    best_practices=COMPOSER_AGENT_BEST_PRACTICES,
    resources=[
        "json_scene from Team 2 (9-field unified scene representation)",
        "Musical Blueprint from supervisor",
        "songgeneration.json (valid vocabulary for all 6 categories)",
        "noatgen.json (classical notation: periods, composers, instruments)",
        "DashScope LLM for attribute selection",
    ],
    run_methods=[
        "Receive task, json_scene, piece count, and blueprint",
        "Analyze scene mood patterns across all scenes",
        "Generate description string per piece (LLM-assisted selection)",
        "Post-validate each field against songgeneration.json vocabulary",
        "Replace invalid tokens with closest valid match",
        "Reflect if quality insufficient (max 2 retries)",
        "Output {descriptions_list}",
    ],
    command=COMMAND_composer,
    guide_book=Guide_Book_composer,
)


if __name__ == "__main__":
    print(f"Agent ID: {COMPOSER_AGENT_PROFILE.agent_id}")
    print(f"Role: {COMPOSER_AGENT_PROFILE.role.name}")
    print(f"Tools: {[t.name for t in COMPOSER_AGENT_PROFILE.tools]}")
    print(f"Constraints: {len(COMPOSER_AGENT_PROFILE.constraints)}")
