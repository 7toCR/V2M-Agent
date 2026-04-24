"""
Requirement Supervisor Agent Profile

This profile defines the Team 1 Requirement Analysis Supervisor that implements
Algorithm 1 from the paper: Task Prompt Generation based on Agent Profile.
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
    ANALYSIS = "分析"
    GENERATION = "生成"
    RETRIEVAL = "检索"
    ORCHESTRATION = "协调"
    PLANNING = "规划"
    VALIDATION = "验证"
    MODIFICATION = "修改"


class KnowledgeDomain(str, Enum):
    MULTIMODAL = "多模态理解"
    MUSIC = "音乐"
    WORKFLOW = "工作流"


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


REQUIREMENT_SUPERVISOR_ROLE = AgentRole(
    name="Requirement Analysis Supervisor",
    description=(
        "Team 1 supervisor responsible for parsing user requirements, determining "
        "which teams and experts are needed, matching requirements to agent profiles, "
        "and generating structured task packets with constraints for Team 2 and Team 3."
    ),
    responsibilities=[
        "Parse user requirements and multimodal input specifications",
        "Determine which modalities (text, audio, image, video) are present and relevant",
        "Decide which teams (Team 2 Scene Understanding, Team 3 Music Generation) need to participate",
        "Read and match agent profiles to user requirements",
        "Extract specific instructions for each participating team and expert",
        "Generate constraints based on agent profiles and user requirements",
        "Output structured task packets for downstream teams",
        "Coordinate requirement analysis workflow following Algorithm 1",
    ],
    expertise=[
        "Requirement analysis and decomposition",
        "Multimodal input classification",
        "Agent capability matching",
        "Task packet generation",
        "Constraint extraction from profiles",
        "Team coordination and orchestration",
        "Algorithm 1: Parse, DetermineNeed, ReadProfile, Match, ExtractInstruction, GenerateConstraints",
    ],
)


REQUIREMENT_SUPERVISOR_TOOLS = [
    AgentTool(
        name="parse_requirement",
        function_signature="parse_requirement(user_requirement: str) -> Dict[str, Any]",
        description="Parse user requirement text and extract key objectives, constraints, and modality specifications",
        parameters=[
            {"name": "user_requirement", "type": "str", "description": "Raw user requirement string"},
        ],
        returns="Dict containing parsed objectives, constraints, and modality hints",
        category=ToolCategory.ANALYSIS,
    ),
    AgentTool(
        name="determine_need",
        function_signature="determine_need(parsed_req: Dict, modalities: List[str]) -> Dict[str, List[str]]",
        description="Determine which teams and experts are needed based on parsed requirements and available modalities",
        parameters=[
            {"name": "parsed_req", "type": "Dict", "description": "Parsed requirement dictionary"},
            {"name": "modalities", "type": "List[str]", "description": "List of available modalities"},
        ],
        returns="Dict mapping team names to lists of required expert IDs",
        category=ToolCategory.ANALYSIS,
    ),
    AgentTool(
        name="read_profile",
        function_signature="read_profile(agent_ids: List[str]) -> List[AgentProfile]",
        description="Read agent profiles for specified teams and experts",
        parameters=[
            {"name": "agent_ids", "type": "List[str]", "description": "List of agent IDs to retrieve profiles for"},
        ],
        returns="List of AgentProfile objects",
        category=ToolCategory.RETRIEVAL,
    ),
    AgentTool(
        name="generate_task_packet",
        function_signature="generate_task_packet(team_name: str, instructions: Dict, constraints: Dict) -> Dict[str, Any]",
        description="Generate structured task packet for a team containing instructions and constraints",
        parameters=[
            {"name": "team_name", "type": "str", "description": "Name of the target team"},
            {"name": "instructions", "type": "Dict", "description": "Task instructions per agent"},
            {"name": "constraints", "type": "Dict", "description": "Constraints per agent"},
        ],
        returns="Structured task packet with instructions, constraints, and metadata",
        category=ToolCategory.GENERATION,
    ),
]


REQUIREMENT_SUPERVISOR_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.MULTIMODAL,
        concepts=[
            "Multimodal input types: text, audio, image, video",
            "Modality characteristics and processing requirements",
            "Cross-modal alignment and fusion strategies",
        ],
        rules=[
            "Text modality provides global background constraints (T_bg)",
            "Video modality establishes temporal anchoring baseline",
            "At least one modality must be present for processing",
        ],
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.WORKFLOW,
        concepts=[
            "Algorithm 1: Task Prompt Generation workflow",
            "Team coordination patterns",
            "Task packet structure and format",
        ],
        rules=[
            "Algorithm 1 steps must execute in order: Parse → DetermineNeed → ReadProfile → Match → ExtractInstruction → GenerateConstraints",
            "Team 2 (Scene Understanding) must execute before Team 3 (Music Generation)",
            "Task packets must include both instructions (I) and constraints (C)",
        ],
    ),
]


REQUIREMENT_SUPERVISOR_PROFILE = AgentProfile(
    agent_id="requirement_supervisor_v1",
    description=(
        "Team 1 Requirement Analysis Supervisor implementing Algorithm 1 for task prompt generation. "
        "Parses user requirements, determines team participation, matches requirements to agent capabilities, "
        "and generates structured task packets with instructions and constraints for Team 2 and Team 3."
    ),
    role=REQUIREMENT_SUPERVISOR_ROLE,
    tools=REQUIREMENT_SUPERVISOR_TOOLS,
    knowledge=REQUIREMENT_SUPERVISOR_KNOWLEDGE,
    constraints=[
        "Must follow Algorithm 1 workflow strictly",
        "Cannot skip DetermineNeed step",
        "Cannot invent constraints - must extract from agent profiles",
        "Must validate that at least one modality is present",
    ],
    best_practices=[
        "Start with modality detection to understand available inputs",
        "Use DetermineNeed to avoid unnecessary expert activation",
        "Extract constraints directly from profile.constraints and profile.knowledge.rules",
        "Generate task packets with clear success criteria",
    ],
    resources=[
        "task/task_profile.py - AgentProfile schema definitions",
        "understanding/*_agent_profile.py - Team 2 expert profiles",
        "pop/*_agent_profile.py - Team 3 expert profiles",
    ],
    run_methods=[
        "Initialize with user requirement and multimodal input paths",
        "Execute Algorithm 1 workflow sequentially",
        "Parse user requirement to extract objectives and hints",
        "Determine which modalities are present",
        "Decide which teams and experts need to participate",
        "Read agent profiles for selected teams and experts",
        "Match parsed requirements to agent capabilities",
        "Extract specific task instructions for each agent",
        "Generate constraints from agent profiles",
        "Assemble Team 2 and Team 3 task packets",
        "Output requirement_analysis_result",
    ],
    command="requirement_analysis",
    guide_book=(
        "Algorithm 1 Implementation Guide:\n"
        "1. Parse(R_user): Extract objectives, modalities, constraints\n"
        "2. DetermineNeed(objectives, modalities): Decide which teams/experts needed\n"
        "3. ReadProfile(P_supervisor, P_experts): Load relevant agent profiles\n"
        "4. Match(objectives, profiles): Map requirements to capabilities\n"
        "5. ExtractInstruction(matches): Generate task instructions (I_i)\n"
        "6. GenerateConstraints(profiles, instructions): Extract constraints (C_i)\n"
        "7. Output: T_experts = [(I_1, C_1), ..., (I_n, C_n)] as task packets\n"
    ),
)


if __name__ == "__main__":
    import json
    print("Requirement Supervisor Agent Profile")
    print("=" * 80)
    print(f"Agent ID: {REQUIREMENT_SUPERVISOR_PROFILE.agent_id}")
    print(f"Role: {REQUIREMENT_SUPERVISOR_PROFILE.role.name}")
    print(f"Tools: {len(REQUIREMENT_SUPERVISOR_PROFILE.tools)}")
    print(f"Knowledge domains: {len(REQUIREMENT_SUPERVISOR_PROFILE.knowledge)}")
