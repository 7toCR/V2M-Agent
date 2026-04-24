"""
PhotoAgent Profile File

Defines the role, tools, knowledge, constraints, and best practices for the PhotoAgent.
This file is located in the Team2/AgentProfile directory for use by the PhotoAgent.
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
        IMAGE_ANALYSIS = "image_analysis"
        VISUAL_UNDERSTANDING = "visual_understanding"
        MULTIMODAL = "multimodal_understanding"
        OTHER = "other"

    class KnowledgeDomain(str, Enum):
        IMAGE = "image_analysis"
        VISUAL = "visual_understanding"
        COMPOSITION = "composition_analysis"
        COLOR = "color_analysis"
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

# PhotoAgent Role Definition
PHOTO_AGENT_ROLE = AgentRole(
    name="Image Understanding and Analysis Expert",
    description="Responsible for analyzing image content, extracting key information including background, background style, subject, and subject mood, providing image analysis support for multimodal scene understanding",
    responsibilities=[
        "Analyze image file content and identify visual elements",
        "Extract background information and style features from images",
        "Identify the image subject and its emotional state",
        "Output standardized JSON-formatted analysis results",
        "Support image analysis needs within multimodal scene understanding"
    ],
    expertise=[
        "Image content recognition",
        "Visual emotion analysis",
        "Scene composition analysis",
        "Color and lighting analysis",
        "JSON format output",
        "Multimodal image processing"
    ]
)

from Team2.Expert.prompt import COMMAND_photo, Guide_Book_photo_expert

# PhotoAgent Tool Definitions
PHOTO_AGENT_TOOLS = [
    AgentTool(
        name="photo",
        function_signature="async def _photo_node(self, state: 'PhotoAgent.Graph') -> Dict[str, Any]",
        description="Invoke a multimodal large language model to understand image file content and return a JSON dictionary containing background, background style, subject, and subject mood",
        parameters=[
            {"name": "state", "type": "PhotoAgent.Graph", "description": "Current state graph state"},
            {"name": "photo_path", "type": "str", "description": "Path to the image file to be analyzed"}
        ],
        returns="Updated state dictionary containing image analysis results",
        category=ToolCategory.IMAGE_ANALYSIS,
        usage_example="Analyze an image file to extract background, style, subject, and emotion information",
        dependencies=[]
    )
]

# PhotoAgent Knowledge Definitions
PHOTO_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.IMAGE,
        concepts=[
            "Image content recognition techniques",
            "Visual element extraction methods",
            "Composition analysis techniques",
            "Image quality assessment",
            "Visual feature classification"
        ],
        rules=[
            "Must extract all four elements: background, background style, subject, and subject mood",
            "Analysis results must be in standardized JSON format",
            "Preserve the original visual information of the image content",
            "Accurately identify the main subject in the image",
            "Ensure the accuracy of visual analysis"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.VISUAL,
        concepts=[
            "Visual understanding and parsing",
            "Spatial relationship analysis",
            "Viewpoint and angle recognition",
            "Depth and layering analysis",
            "Visual focal point identification"
        ],
        rules=[
            "Accurately understand the visual content of the image",
            "Analyze spatial relationships and relative positions",
            "Identify shooting viewpoints and angles",
            "Understand the depth and layers in the image",
            "Identify visual focal points and attention areas"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.COLOR,
        concepts=[
            "Color analysis and recognition",
            "Tone and saturation analysis",
            "Color-emotion association",
            "Lighting and shadow analysis",
            "Color contrast assessment"
        ],
        rules=[
            "Analyze the color composition of the image",
            "Identify tone and saturation characteristics",
            "Establish associations between colors and emotions",
            "Analyze lighting and shadow effects",
            "Assess color contrast and harmony"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.EMOTION,
        concepts=[
            "Visual emotion recognition",
            "Emotional expression analysis",
            "Facial expression interpretation",
            "Body language analysis",
            "Atmosphere and mood association"
        ],
        rules=[
            "Identify the emotional tone in the image",
            "Analyze methods of emotional expression",
            "Interpret facial expression characteristics",
            "Analyze body language cues",
            "Establish associations between atmosphere and mood"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.COMPOSITION,
        concepts=[
            "Composition principle analysis",
            "Visual balance assessment",
            "Focal point and leading line identification",
            "Proportion and symmetry analysis",
            "Visual rhythm identification"
        ],
        rules=[
            "Analyze the composition principles of the image",
            "Assess visual balance and stability",
            "Identify visual focal points and leading lines",
            "Analyze proportion and symmetry relationships",
            "Identify visual rhythm and patterns"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.STYLE,
        concepts=[
            "Visual style recognition",
            "Artistic style classification",
            "Cultural style analysis",
            "Period style identification",
            "Personal style characteristics"
        ],
        rules=[
            "Identify the visual style of the image",
            "Classify artistic styles and genres",
            "Analyze cultural style features",
            "Determine period style characteristics",
            "Identify personal style traits"
        ]
    )
]

# PhotoAgent Constraints
PHOTO_AGENT_CONSTRAINTS = [
    "Must analyze actual image file content; do not fabricate information",
    "Must include all four elements: background, background style, subject, and subject mood",
    "Output must be in standardized JSON format",
    "Must not omit key visual information from the image",
    "Must preserve the original visual content of the image",
    "Must not add content that does not appear in the image",
    "Must correctly handle image files in various formats"
]

# PhotoAgent Best Practices
PHOTO_AGENT_BEST_PRACTICES = [
    "First observe the image as a whole to understand its general content and composition",
    "Identify the main visual elements and subject in the image",
    "Accurately extract background information and style features",
    "Carefully analyze the subject's emotional state and expressions",
    "Use standardized JSON format for output",
    "Verify the completeness and accuracy of analysis results",
    "Provide detailed execution logs and analysis processes",
]

# PhotoAgent Profile
PHOTO_AGENT_PROFILE = AgentProfile(
    agent_id="photo_agent",
    description="A specialized agent responsible for analyzing image content, extracting key information including background, background style, subject, and subject mood, providing image analysis support for multimodal scene understanding.",
    role=PHOTO_AGENT_ROLE,
    tools=PHOTO_AGENT_TOOLS,
    knowledge=PHOTO_AGENT_KNOWLEDGE,
    constraints=PHOTO_AGENT_CONSTRAINTS,
    best_practices=PHOTO_AGENT_BEST_PRACTICES,
    resources=[
        "DashScope multimodal large language model",
        "Image base64 encoding tool",
        "Image understanding and analysis knowledge base",
        "Visual emotion recognition model",
        "JSON format output specification"
    ],
    run_methods=[
        "Follow the ReAct (Reasoning + Acting) workflow pattern",
        "Process through the Thought -> Action -> Observation -> Final loop",
        "Directly invoke the photo tool node, passing the state parameter",
        "The tool internally handles image base64 encoding and model invocation",
        "Return a JSON dictionary containing background, background style, subject, and subject mood",
        "Note: Execute the image analysis tool even when no image file is present (image files are not transmitted via text; they are provided directly in base64 format when the tool is invoked)",
    ],
    command=COMMAND_photo,
    guide_book=Guide_Book_photo_expert
)

# Export functions
def get_photo_agent_profile() -> AgentProfile:
    """Get the PhotoAgent configuration profile"""
    return PHOTO_AGENT_PROFILE

def print_photo_agent_profile():
    """Print the PhotoAgent configuration profile"""
    profile = PHOTO_AGENT_PROFILE
    print("=" * 80)
    print("PhotoAgent Profile")
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
        if tool.dependencies:
            print(f"    Dependencies: {', '.join(tool.dependencies)}")

    print(f"\nKnowledge Domains ({len(profile.knowledge)} total):")
    for knowledge in profile.knowledge:
        print(f"  - {knowledge.domain.value}:")
        print(f"    Concepts: {', '.join(knowledge.concepts[:3])}...")
        print(f"    Rules: {', '.join(knowledge.rules[:2])}...")

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
    print_photo_agent_profile()
