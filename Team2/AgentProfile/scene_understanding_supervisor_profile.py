"""
Scene Understanding Supervisor Agent Profile

This profile defines the Team 2 Scene Understanding Supervisor that coordinates
the four-stage multimodal scene understanding workflow from the paper.
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
    ANALYSIS = "analysis"
    ORCHESTRATION = "orchestration"
    VALIDATION = "validation"


class KnowledgeDomain(str, Enum):
    MULTIMODAL = "multimodal_understanding"
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


SCENE_UNDERSTANDING_SUPERVISOR_ROLE = AgentRole(
    name="Scene Understanding Supervisor",
    description=(
        "Team 2 supervisor responsible for coordinating multimodal scene understanding "
        "through a four-stage workflow: Temporal Anchoring, Inter-modal Interpolation, "
        "Audio-Visual Alignment, and Semantic Refinement."
    ),
    responsibilities=[
        "Coordinate parallel execution of modality-specific experts",
        "Execute Stage 1: Temporal Anchoring",
        "Execute Stage 2: Inter-modal Interpolation",
        "Execute Stage 3: Audio-Visual Alignment",
        "Execute Stage 4: Semantic Refinement",
        "Verify scene consistency",
        "Output unified 9-field scene representations",
    ],
    expertise=[
        "Multimodal fusion and alignment",
        "Four-stage scene understanding workflow",
        "Posterior correction mechanisms",
        "Scene consistency verification",
    ],
)


SCENE_UNDERSTANDING_SUPERVISOR_TOOLS = [
    AgentTool(
        name="generate_expert_tasks",
        function_signature="generate_expert_tasks(user_requirement: str, modality_addresses: Dict) -> Dict[str, str]",
        description=(
            "Read expert AgentProfiles (Text, Audio, Photo, Video) and generate per-expert "
            "structured tasks using Algorithm 1 via AsyncTaskCreator. Implements DetermineNeed → "
            "ReadProfile → Match → ExtractInstruction → GenerateConstraints."
        ),
        parameters=[
            {"name": "user_requirement", "type": "str", "description": "Task instruction from Team 1"},
            {"name": "modality_addresses", "type": "Dict", "description": "Available modality file paths"},
        ],
        returns="Dict mapping modality name to generated task string",
        category=ToolCategory.ORCHESTRATION,
    ),
    AgentTool(
        name="dispatch_parallel_experts",
        function_signature="dispatch_parallel_experts(expert_tasks: Dict[str, str], modality_addresses: Dict) -> Dict[str, Any]",
        description=(
            "Execute Text, Audio, Photo, Video experts in parallel using asyncio.gather "
            "with return_exceptions=True. Each expert runs its own Think-Act-Observe-Reflect loop."
        ),
        parameters=[
            {"name": "expert_tasks", "type": "Dict[str, str]", "description": "Per-expert task strings from generate_expert_tasks"},
            {"name": "modality_addresses", "type": "Dict", "description": "File paths per modality"},
        ],
        returns="Dict with text_result, audio_result, photo_result, video_result",
        category=ToolCategory.ORCHESTRATION,
    ),
    AgentTool(
        name="invoke_verifier",
        function_signature="invoke_verifier(expert_results: Dict, user_requirement: str, team1_instruction: str, constraints: List[str]) -> Dict",
        description=(
            "Forward all expert results plus context to the SceneVerifier for four-stage "
            "fusion and validation. The Verifier is independent — it receives all inputs directly."
        ),
        parameters=[
            {"name": "expert_results", "type": "Dict", "description": "Results from all 4 experts"},
            {"name": "user_requirement", "type": "str", "description": "Original user requirement R_user"},
            {"name": "team1_instruction", "type": "str", "description": "Instruction from Team 1 task packet"},
            {"name": "constraints", "type": "List[str]", "description": "Constraints from Team 1"},
        ],
        returns="Dict with json_scene_result, verification {passed, issues, warnings, stage_outputs}",
        category=ToolCategory.ORCHESTRATION,
    ),
]


SCENE_UNDERSTANDING_SUPERVISOR_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.MULTIMODAL,
        concepts=[
            "Temporal Anchoring: video keyframes establish timeline baseline",
            "Semantic Refinement: posterior correction using global constraints",
            "9-field unified scene representation",
        ],
        rules=[
            "Video modality provides temporal anchoring baseline",
            "Text modality provides global constraints T_bg for refinement",
            "All scenes must have 9 fields",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.WORKFLOW,
        concepts=[
            "Four-stage sequential workflow",
            "Parallel expert execution before Stage 1",
        ],
        rules=[
            "Stages must execute in order: 1 → 2 → 3 → 4",
            "Modality experts execute in parallel before Stage 1",
        ],
    ),
]


SCENE_UNDERSTANDING_SUPERVISOR_PROFILE = AgentProfile(
    agent_id="scene_understanding_supervisor_v1",
    description=(
        "Team 2 Scene Understanding Supervisor that reads expert AgentProfiles (Text, Audio, "
        "Photo, Video), generates per-expert tasks via Algorithm 1, dispatches parallel expert "
        "execution, and forwards results to the SceneVerifier for four-stage fusion and validation. "
        "Follows Think-Act-Observe-Reflect iterative loop via LangGraph."
    ),
    role=SCENE_UNDERSTANDING_SUPERVISOR_ROLE,
    tools=SCENE_UNDERSTANDING_SUPERVISOR_TOOLS,
    knowledge=SCENE_UNDERSTANDING_SUPERVISOR_KNOWLEDGE,
    constraints=[
        "Must read expert AgentProfiles before generating tasks (Algorithm 1)",
        "Must dispatch experts in parallel using asyncio.gather with return_exceptions=True",
        "Must forward ALL expert results to the Verifier — never skip a modality",
        "Must handle Verifier failure via Reflect → Retry loop (max 2 retries)",
        "All output scenes must have exactly 9 fields",
        "Follows Think-Act-Observe-Reflect iterative loop",
    ],
    best_practices=[
        "Parse Team 1 task_packet to extract instruction, constraints, and modality addresses",
        "Only generate tasks for modalities that have input files (DetermineNeed)",
        "Use AsyncTaskCreator with expert profiles for task generation",
        "Start expert dispatch only after all tasks are generated",
        "Forward complete context (user_requirement, team1_instruction, constraints) to Verifier",
    ],
    resources=[
        "Team2/Expert/text.py - Text modality expert agent",
        "Team2/Expert/audio.py - Audio modality expert agent",
        "Team2/Expert/photo.py - Photo modality expert agent",
        "Team2/Expert/video.py - Video modality expert agent",
        "Team2/AgentProfile/text_agent_profile.py - Text expert profile",
        "Team2/AgentProfile/audio_agent_profile.py - Audio expert profile",
        "Team2/AgentProfile/photo_agent_profile.py - Photo expert profile",
        "Team2/AgentProfile/video_agent_profile.py - Video expert profile",
        "Team2/verifier/scene_verifier.py - Scene Understanding Verifier",
        "task/task_create.py - AsyncTaskCreator for task generation",
    ],
    run_methods=[
        "Parse Team 1 task_packet (instruction, constraints, modality_addresses)",
        "Read expert AgentProfiles (Algorithm 1: DetermineNeed → ReadProfile → Match)",
        "Generate per-expert tasks via AsyncTaskCreator",
        "Dispatch parallel expert execution via asyncio.gather",
        "Collect expert results (text_result, audio_result, photo_result, video_result)",
        "Forward results + context to SceneVerifier",
        "Handle Verifier feedback: Reflect → Retry if verification fails",
        "Output final json_scene_result",
    ],
    command="scene_understanding",
    guide_book=(
        "Supervisor Workflow:\n"
        "1. Parse team2_task_packet from Team 1\n"
        "2. Read expert AgentProfiles and run Algorithm 1\n"
        "3. Generate per-expert tasks via AsyncTaskCreator\n"
        "4. Dispatch experts in parallel (asyncio.gather)\n"
        "5. Forward results to SceneVerifier\n"
        "6. Handle verification: pass → output, fail → reflect → retry\n"
    ),
)


if __name__ == "__main__":
    print("Scene Understanding Supervisor Agent Profile")
    print("=" * 80)
    print(f"Agent ID: {SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.agent_id}")
    print(f"Role: {SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.role.name}")
