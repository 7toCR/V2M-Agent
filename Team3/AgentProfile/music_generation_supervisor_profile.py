"""
Music Generation Supervisor Agent Profile

This profile defines the Team 3 Music Generation Supervisor that coordinates
the music generation workflow with strategic planning, parallel experts,
verification, and collaborative modification.
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


class ToolCategory(str, Enum):
    PLANNING = "planning"
    ORCHESTRATION = "orchestration"
    GENERATION = "generation"
    VALIDATION = "validation"
    MODIFICATION = "modification"
    ANALYSIS = "analysis"


class KnowledgeDomain(str, Enum):
    MUSIC = "music"
    WORKFLOW = "workflow"
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


MUSIC_GENERATION_SUPERVISOR_ROLE = AgentRole(
    name="Music Generation Supervisor",
    description=(
        "Team 3 supervisor responsible for coordinating music prompt generation "
        "through strategic planning, parallel expert execution, cross-expert verification, "
        "and collaborative modification."
    ),
    responsibilities=[
        "Execute strategic planning to generate Musical Blueprint",
        "Coordinate parallel execution of music generation experts",
        "Generate idx, gt_lyric, descriptions, auto_prompt_audio_type",
        "Verify consistency across all generated components",
        "Trigger collaborative modification when conflicts detected",
        "Output structured prompt list",
    ],
    expertise=[
        "Musical blueprint planning",
        "Multi-expert coordination",
        "Cross-expert verification",
        "Collaborative modification operator",
    ],
)


MUSIC_GENERATION_SUPERVISOR_TOOLS = [
    AgentTool(
        name="strategic_planning",
        function_signature="strategic_planning(scenes: List[Dict]) -> Dict[str, Any]",
        description="Generate Musical Blueprint defining model, lyric_style, and emotional_key",
        parameters=[
            {"name": "scenes", "type": "List[Dict]", "description": "Scene data from Team 2"},
        ],
        returns="Musical Blueprint with model, lyric_style, emotional_key",
        category=ToolCategory.PLANNING,
    ),
    AgentTool(
        name="coordinate_parallel_experts",
        function_signature="coordinate_parallel_experts(blueprint: Dict, scenes: List[Dict], piece: int) -> Dict[str, Any]",
        description="Coordinate parallel execution of Lyricist, Composer, Stylist, Index experts",
        parameters=[
            {"name": "blueprint", "type": "Dict", "description": "Musical Blueprint"},
            {"name": "scenes", "type": "List[Dict]", "description": "Scene data"},
            {"name": "piece", "type": "int", "description": "Number of pieces to generate"},
        ],
        returns="Dict containing results from all music experts",
        category=ToolCategory.ORCHESTRATION,
    ),
    AgentTool(
        name="verify_consistency",
        function_signature="verify_consistency(prompt: Dict[str, str], blueprint: Dict) -> Dict[str, Any]",
        description="Verify consistency across all generated components",
        parameters=[
            {"name": "prompt", "type": "Dict[str, str]", "description": "Generated prompt"},
            {"name": "blueprint", "type": "Dict", "description": "Musical Blueprint"},
        ],
        returns="Verification result with passed flag and conflict list",
        category=ToolCategory.VALIDATION,
    ),
    AgentTool(
        name="collaborative_modification",
        function_signature="collaborative_modification(prompt: Dict, conflicts: List[Dict], blueprint: Dict) -> Dict[str, str]",
        description="Apply collaborative modification operator to resolve conflicts",
        parameters=[
            {"name": "prompt", "type": "Dict", "description": "Original prompt with conflicts"},
            {"name": "conflicts", "type": "List[Dict]", "description": "Detected conflicts"},
            {"name": "blueprint", "type": "Dict", "description": "Musical Blueprint"},
        ],
        returns="Modified prompt with conflicts resolved",
        category=ToolCategory.MODIFICATION,
    ),
]


MUSIC_GENERATION_SUPERVISOR_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.MUSIC,
        concepts=[
            "Musical Blueprint: model, lyric_style, emotional_key",
            "Strategic Planning: analyze scenes to determine musical direction",
            "Four-field prompt structure: idx, gt_lyric, descriptions, auto_prompt_audio_type",
        ],
        rules=[
            "Blueprint must be generated before expert execution",
            "Title language must match lyric language",
            "Emotional tone must align across all components",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.WORKFLOW,
        concepts=[
            "Three-step workflow: Strategic Planning → Creative Execution → Cross-Expert Alignment",
            "Parallel expert execution pattern",
        ],
        rules=[
            "Strategic Planning must complete before Creative Execution",
            "Creative Execution experts run in parallel",
            "Verification runs after all experts complete",
        ],
    ),
]


MUSIC_GENERATION_SUPERVISOR_PROFILE = AgentProfile(
    agent_id="music_generation_supervisor_v1",
    description=(
        "Team 3 Music Generation Supervisor implementing the three-step music generation "
        "workflow with strategic planning, parallel expert execution, and cross-expert verification."
    ),
    role=MUSIC_GENERATION_SUPERVISOR_ROLE,
    tools=MUSIC_GENERATION_SUPERVISOR_TOOLS,
    knowledge=MUSIC_GENERATION_SUPERVISOR_KNOWLEDGE,
    constraints=[
        "Must execute three steps sequentially",
        "Cannot skip strategic planning",
        "Must verify all prompts before output",
        "All output prompts must have exactly four fields",
    ],
    best_practices=[
        "Extract emotional key from scene mood patterns",
        "Use asyncio.gather() for parallel expert execution",
        "Apply collaborative modification only to conflicting components",
    ],
    resources=[
        "Team3/Expert/lyricist.py - Lyricist expert (idx + gt_lyric)",
        "Team3/Expert/composer.py - Composer expert (descriptions)",
        "Team3/Expert/stylist.py - Stylist expert (audio_type)",
        "Team3/verifier/music_verifier.py - Independent Music Verifier",
        "Team3/pre-traing/songgeneration.json - SongGeneration vocabulary",
        "Team3/pre-traing/noatgen.json - NotaGen vocabulary",
        "DashScope LLM (qwen3-max for reasoning)",
    ],
    run_methods=[
        "Receive task_packet from Team 1 (contains json_scene from Team 2)",
        "Step 1: Strategic Planning - generate Musical Blueprint B",
        "Step 2: Creative Execution - parallel expert generation via asyncio.gather",
        "Assemble expert results into 4-field prompt dicts",
        "Step 3: Cross-Expert Alignment - invoke independent MusicVerifier",
        "If verification fails: Collaborative Modification Operator M (max 2 rounds)",
        "Output pop_prompt_result and save to lyric.jsonl",
    ],
    command="music_generation",
    guide_book=(
        "Three-Step Music Generation Workflow:\n"
        "Step 1: Strategic Planning - Generate Musical Blueprint\n"
        "Step 2: Creative Execution - Parallel experts generate components\n"
        "Step 3: Cross-Expert Alignment - Verify and resolve conflicts\n"
    ),
)


if __name__ == "__main__":
    print("Music Generation Supervisor Agent Profile")
    print("=" * 80)
    print(f"Agent ID: {MUSIC_GENERATION_SUPERVISOR_PROFILE.agent_id}")
    print(f"Role: {MUSIC_GENERATION_SUPERVISOR_PROFILE.role.name}")
