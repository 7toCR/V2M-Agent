"""
VideoAgent Profile File

Defines the role, tools, knowledge, constraints, and best practices for the VideoAgent.
This file is located under the Team2/AgentProfile directory for use by the VideoAgent.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Import base types
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from task.task_profile import AgentProfile, AgentRole, AgentTool, AgentKnowledge, ToolCategory, KnowledgeDomain
except ImportError:
    # If the task module is not available, use local definitions
    from dataclasses import dataclass
    from enum import Enum

    class ToolCategory(str, Enum):
        VIDEO_ANALYSIS = "video_analysis"
        CONTENT_GENERATION = "content_generation"
        ANALYSIS = "analysis"
        OTHER = "other"

    class KnowledgeDomain(str, Enum):
        VIDEO = "video_analysis"
        SCENE = "scene_understanding"
        EMOTION = "emotion_analysis"
        TEMPORAL = "temporal_analysis"
        OTHER = "other"

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

# VideoAgent Role Definition
VIDEO_AGENT_ROLE = AgentRole(
    name="Video Scene Analysis Expert",
    description="Responsible for analyzing video content, extracting keyframes, actions, emotions, and scene information to generate video scene descriptions",
    responsibilities=[
        "Analyze the visual content of videos",
        "Identify keyframes and actions in videos",
        "Extract the emotional tone of videos",
        "Analyze scene composition and spatial relationships",
        "Generate standardized video scene descriptions"
    ],
    expertise=[
        "Video content analysis",
        "Keyframe identification",
        "Action analysis",
        "Visual emotion recognition",
        "Spatiotemporal relationship understanding"
    ]
)

from Team2.Expert.prompt import COMMAND_video, Guide_Book_video_expert

# VideoAgent Tool Definitions
VIDEO_AGENT_TOOLS = [
    AgentTool(
        name="video_analysis",
        function_signature="async def _video_node(self, state: 'VideoAgent.Graph') -> dict",
        description="Analyze video content and extract scene information",
        parameters=[
            {"name": "state", "type": "VideoAgent.Graph", "description": "Current state graph state"}
        ],
        returns="Updated state dictionary containing video analysis results",
        category=ToolCategory.VIDEO_ANALYSIS,
        usage_example="Analyze a video file and extract scene descriptions",
        dependencies=[]
    )
]

# VideoAgent Knowledge Definitions
VIDEO_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.VIDEO,
        concepts=[
            "Video content analysis",
            "Keyframe extraction",
            "Action recognition",
            "Shot transition analysis",
            "Temporal relationship understanding"
        ],
        rules=[
            "Analyze the chronological development of the video",
            "Identify key actions and events",
            "Understand the significance of shot transitions",
            "Extract important visual elements",
            "Maintain coherence in the analysis"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.SCENE,
        concepts=[
            "Scene composition analysis",
            "Spatial relationship understanding",
            "Background and environment analysis",
            "Subject and object identification",
            "Visual focal point analysis"
        ],
        rules=[
            "Identify the main elements in a scene",
            "Analyze spatial relationships between elements",
            "Understand the influence of the background environment",
            "Determine the visual focal point",
            "Maintain completeness in scene descriptions"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.EMOTION,
        concepts=[
            "Visual emotion recognition",
            "Action-based emotion analysis",
            "Color-emotion association",
            "Rhythm-emotion influence",
            "Facial expression analysis"
        ],
        rules=[
            "Analyze the emotional tone of the video",
            "Identify emotional turning points",
            "Understand the impact of color on emotions",
            "Analyze the relationship between rhythm and emotion",
            "Maintain consistency in emotion analysis"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.TEMPORAL,
        concepts=[
            "Time series analysis",
            "Event timeline",
            "Rhythm and pace",
            "Duration analysis",
            "Temporal relationship understanding"
        ],
        rules=[
            "Analyze the chronological order of events",
            "Understand the significance of time intervals",
            "Evaluate the pacing of the video",
            "Identify key moments",
            "Maintain accuracy in temporal analysis"
        ]
    )
]

# VideoAgent Constraints
VIDEO_AGENT_CONSTRAINTS = [
    "Must analyze the complete video content",
    "Must identify keyframes and actions",
    "Must extract the emotional tone",
    "Must analyze scene composition",
    "Must generate descriptions in a standardized format",
    "Must not omit important visual information",
    "Must maintain objectivity in the analysis"
]

# VideoAgent Best Practices
VIDEO_AGENT_BEST_PRACTICES = [
    "Watch the video as a whole first, then analyze the details",
    "Identify keyframes as analysis benchmarks",
    "Pay attention to the timing of emotional changes",
    "Analyze the spatial relationships within scenes",
    "Verify the reasonableness of analysis results",
    "Provide detailed analysis logs",
    "Handle potential analysis errors"
]

# VideoAgent Profile
VIDEO_AGENT_PROFILE = AgentProfile(
    agent_id="video_agent",
    description="A specialized agent responsible for analyzing video content, extracting keyframes, actions, emotions, and scene information to generate video scene descriptions.",
    role=VIDEO_AGENT_ROLE,
    tools=VIDEO_AGENT_TOOLS,
    knowledge=VIDEO_AGENT_KNOWLEDGE,
    constraints=VIDEO_AGENT_CONSTRAINTS,
    best_practices=VIDEO_AGENT_BEST_PRACTICES,
    resources=[
        "DashScope multimodal large language model",
        "Video base64 encoding tool",
        "Video keyframe extraction knowledge base",
        "Video emotion recognition model",
        "JSON format output specification"
    ],
    run_methods=[
        "Follow the ReAct (Reasoning + Acting) pattern workflow",
        "Process through the Thought -> Action -> Observation -> Final loop",
        "Directly invoke the video tool node, passing in the state parameter",
        "The tool internally handles video base64 encoding and model invocation",
        "Return a JSON list of keyframes",
        "Note: Execute the video analysis tool even if no video file is present (video files are not transmitted via text; they are provided directly in base64 format when the tool is invoked)",
        "Note: Do not extract keyframes at fixed time intervals; instead, identify significant visual changes to define keyframes",
    ],
    command=COMMAND_video,
    guide_book=Guide_Book_video_expert
)

# Export functions
def get_video_agent_profile() -> AgentProfile:
    """Get the VideoAgent configuration information"""
    return VIDEO_AGENT_PROFILE

def print_video_agent_profile():
    """Print VideoAgent configuration information"""
    profile = VIDEO_AGENT_PROFILE
    print("=" * 80)
    print("VideoAgent Profile")
    print("=" * 80)

    print(f"\nAgent ID: {profile.agent_id}")
    print(f"Description: {profile.description}")
    print(f"\nRole: {profile.role.name}")
    print(f"Role Description: {profile.role.description}")
    print(f"\nResponsibilities ({len(profile.role.responsibilities)} items):")
    for i, resp in enumerate(profile.role.responsibilities, 1):
        print(f"  {i}. {resp}")
    print(f"\nExpertise ({len(profile.role.expertise)} items):")
    for i, exp in enumerate(profile.role.expertise, 1):
        print(f"  {i}. {exp}")

    print(f"\nTools ({len(profile.tools)} total):")
    for tool in profile.tools:
        print(f"  - {tool.name}: {tool.description}")
        print(f"    Function Signature: {tool.function_signature}")

    print(f"\nKnowledge Domains ({len(profile.knowledge)} total):")
    for knowledge in profile.knowledge:
        print(f"  - {knowledge.domain.value}:")
        print(f"    Concepts: {', '.join(knowledge.concepts[:3])}...")

    print(f"\nConstraints ({len(profile.constraints)} total):")
    for i, constraint in enumerate(profile.constraints, 1):
        print(f"  {i}. {constraint}")

    print(f"\nBest Practices ({len(profile.best_practices)} total):")
    for i, practice in enumerate(profile.best_practices, 1):
        print(f"  {i}. {practice}")

    print(f"\nResources ({len(profile.resources)} items):")
    for i, resource in enumerate(profile.resources, 1):
        print(f"  {i}. {resource}")

    print(f"\nRun Methods ({len(profile.run_methods)} items):")
    for i, method in enumerate(profile.run_methods, 1):
        print(f"  {i}. {method}")

    print(f"\nCOMMAND: {profile.command}")
    print(f"Guide Book: {profile.guide_book}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_video_agent_profile()
