"""
TextAgent Profile File

Defines the role, tools, knowledge, constraints, and best practices for TextAgent.
This file is located under Team2/AgentProfile/ and is used by TextAgent.
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
        TEXT_ANALYSIS = "text_analysis"
        MULTIMODAL = "multimodal_understanding"
        DATA_PROCESSING = "data_processing"
        OTHER = "other"

    class KnowledgeDomain(str, Enum):
        TEXT = "text_analysis"
        SEMANTIC = "semantic_understanding"
        EMOTION = "emotion_analysis"
        SCENE = "scene_analysis"
        STYLE = "style_recognition"
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

# TextAgent Role Definition
TEXT_AGENT_ROLE = AgentRole(
    name="Text Understanding and Analysis Expert",
    description="Responsible for analyzing text content, extracting key information including background, background style, subject, and subject mood, providing text analysis support for multimodal scene understanding",
    responsibilities=[
        "Analyze text file content to identify key scenes and plot points",
        "Extract background information and style characteristics from text",
        "Identify the text subject and its emotional state",
        "Output standardized JSON-formatted analysis results",
        "Support text analysis needs in multimodal scene understanding"
    ],
    expertise=[
        "Text semantic analysis",
        "Emotion recognition and classification",
        "Scene element extraction",
        "JSON format output",
        "Multimodal text processing"
    ]
)

from Team2.Expert.prompt import COMMAND_text, Guide_Book_text_expert

# TextAgent Tool Definition
TEXT_AGENT_TOOLS = [
    AgentTool(
        name="text",
        function_signature="async def _text_node(self, state: 'TextAgent.Graph') -> Dict[str, Any]",
        description="Calls a multimodal large language model to understand text file content and returns a JSON dictionary containing background, background style, subject, and subject style",
        parameters=[
            {"name": "state", "type": "TextAgent.Graph", "description": "Current state graph state"},
            {"name": "text_path", "type": "str", "description": "Path to the text file to be analyzed"}
        ],
        returns="Updated state dictionary containing text analysis results",
        category=ToolCategory.TEXT_ANALYSIS,
        usage_example="Analyze a text file to extract background, style, subject, and emotion information",
        dependencies=[]
    )
]

# TextAgent Knowledge Definition
TEXT_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.TEXT,
        concepts=[
            "Text scene segmentation and identification",
            "Background information extraction techniques",
            "Subject identification and description methods",
            "Emotional state analysis methods",
            "Text style feature recognition"
        ],
        rules=[
            "Must extract all four elements: background, background style, subject, and subject style",
            "Analysis results must be in standardized JSON format",
            "Preserve the original semantics of the text content",
            "Correctly handle multi-scene text content",
            "Ensure accuracy of emotion analysis"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.SEMANTIC,
        concepts=[
            "Semantic understanding and parsing",
            "Contextual relationship analysis",
            "Keyword extraction techniques",
            "Plot development identification",
            "Character relationship analysis"
        ],
        rules=[
            "Accurately understand the semantic content of the text",
            "Identify key plot turning points in the text",
            "Extract important keywords and phrases",
            "Analyze dynamic relationships between characters",
            "Understand the deeper meaning of the text"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.EMOTION,
        concepts=[
            "Emotion recognition and classification",
            "Emotional intensity assessment",
            "Emotional change tracking",
            "Emotional expression pattern analysis",
            "Emotion-scene correlation"
        ],
        rules=[
            "Accurately identify the emotional tone in the text",
            "Assess the intensity and developmental changes of emotions",
            "Analyze the specific ways emotions are expressed",
            "Establish correlations between emotions and scenes",
            "Maintain consistency in emotion analysis"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.SCENE,
        concepts=[
            "Scene identification and segmentation",
            "Scene element extraction",
            "Scene atmosphere analysis",
            "Scene transition identification",
            "Scene chronological order"
        ],
        rules=[
            "Correctly segment different scenes in the text",
            "Extract key elements and details of each scene",
            "Analyze the overall atmosphere and style of scenes",
            "Identify transition relationships between scenes",
            "Maintain chronological order in scene descriptions"
        ]
    )
]

# TextAgent Constraints
TEXT_AGENT_CONSTRAINTS = [
    "Must analyze text file content; do not fabricate information",
    "Must include all four elements: background, background style, subject, and subject style",
    "Output must be in standardized JSON format",
    "Must not omit key scene information from the text",
    "Must preserve the original semantics of the text",
    "Must not add content that does not appear in the text",
    "Must correctly handle segmentation of multi-scene text"
]

# TextAgent Best Practices
TEXT_AGENT_BEST_PRACTICES = [
    "Read through the text content first to understand the overall structure and theme",
    "Identify key scene turning points in the text",
    "Accurately extract background information and style characteristics",
    "Carefully analyze the emotional state of the subject",
    "Use standardized JSON format for output",
    "Verify the completeness and accuracy of analysis results",
    "Provide detailed execution logs and analysis process"
]

# TextAgent Profile
TEXT_AGENT_PROFILE = AgentProfile(
    agent_id="text_agent",
    description="A specialized agent responsible for analyzing text content, extracting key information including background, background style, subject, and subject mood, providing text analysis support for multimodal scene understanding.",
    role=TEXT_AGENT_ROLE,
    tools=TEXT_AGENT_TOOLS,
    knowledge=TEXT_AGENT_KNOWLEDGE,
    constraints=TEXT_AGENT_CONSTRAINTS,
    best_practices=TEXT_AGENT_BEST_PRACTICES,
    resources=[
        "DashScope multimodal large language model",
        "Text file reading tool",
        "Text semantic analysis knowledge base",
        "Emotion recognition and classification model",
        "JSON format output specification"
    ],
    run_methods=[
        "Follow the ReAct (Reasoning + Acting) workflow pattern",
        "Process through the Thought -> Action -> Observation -> Final loop",
        "Directly call the text tool node, passing in the state parameter",
        "The tool internally handles text file reading and model invocation",
        "Return a JSON dictionary containing background, background style, subject, and subject mood",
        "Note: Execute the text analysis tool even without text files (text files are not transmitted via text; they are provided directly in base64 format when the tool is called)",
    ],
    command=COMMAND_text,
    guide_book=Guide_Book_text_expert
)

# Export functions
def get_text_agent_profile() -> AgentProfile:
    """Get the TextAgent profile configuration"""
    return TEXT_AGENT_PROFILE

def print_text_agent_profile():
    """Print TextAgent profile information"""
    profile = TEXT_AGENT_PROFILE
    print("=" * 80)
    print("TextAgent Profile")
    print("=" * 80)

    print(f"\nAgent ID: {profile.agent_id}")
    print(f"Overall Description: {profile.description}")
    print(f"\nRole: {profile.role.name}")
    print(f"Role Description: {profile.role.description}")
    print(f"\nResponsibilities ({len(profile.role.responsibilities)} items):")
    for i, resp in enumerate(profile.role.responsibilities, 1):
        print(f"  {i}. {resp}")
    print(f"\nExpertise Areas ({len(profile.role.expertise)} items):")
    for i, exp in enumerate(profile.role.expertise, 1):
        print(f"  {i}. {exp}")

    print(f"\nTools ({len(profile.tools)}):")
    for tool in profile.tools:
        print(f"  - {tool.name}: {tool.description}")
        print(f"    Function Signature: {tool.function_signature}")
        if tool.dependencies:
            print(f"    Dependencies: {', '.join(tool.dependencies)}")

    print(f"\nKnowledge Domains ({len(profile.knowledge)}):")
    for knowledge in profile.knowledge:
        print(f"  - {knowledge.domain.value}:")
        print(f"    Concepts: {', '.join(knowledge.concepts[:3])}...")
        print(f"    Rules: {', '.join(knowledge.rules[:2])}...")

    print(f"\nConstraints ({len(profile.constraints)}):")
    for i, constraint in enumerate(profile.constraints, 1):
        print(f"  {i}. {constraint}")

    print(f"\nBest Practices ({len(profile.best_practices)}):")
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
    print_text_agent_profile()
