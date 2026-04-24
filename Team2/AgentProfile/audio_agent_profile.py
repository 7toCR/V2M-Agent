"""
AudioAgent Profile File

Defines the role, tools, knowledge, constraints, and best practices for AudioAgent.
This file is located in the Team2/AgentProfile directory for use by AudioAgent.
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
        AUDIO_ANALYSIS = "audio_analysis"
        MULTIMODAL = "multimodal_understanding"
        TIME_SEGMENTATION = "time_segmentation"
        OTHER = "other"

    class KnowledgeDomain(str, Enum):
        AUDIO = "audio_analysis"
        SOUND = "sound_recognition"
        EMOTION = "emotion_analysis"
        RHYTHM = "rhythm_analysis"
        SCENE = "scene_analysis"
        TIME = "time_segmentation"
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

AUDIO_AGENT_ROLE = AgentRole(
    name="Audio Understanding and Analysis Expert",
    description="Responsible for analyzing audio content, extracting key information by time segments including main sound content, main sound style, ambient sound content, and ambient sound style, providing audio analysis support for multimodal scene understanding",
    responsibilities=[
        "Analyze audio file content with time-based segmentation",
        "Extract main sound content and style for each time segment",
        "Identify ambient sound content and style for each time segment",
        "Output analysis results as a list ordered by time sequence",
        "Support audio analysis needs in multimodal scene understanding"
    ],
    expertise=[
        "Audio time-based segmentation analysis",
        "Sound content recognition",
        "Audio emotion analysis",
        "Ambient sound recognition",
        "JSON format output",
        "Multimodal audio processing"
    ]
)

from Team2.Expert.prompt import COMMAND_audio, Guide_Book_audio_expert

AUDIO_AGENT_TOOLS = [
    AgentTool(
        name="audio",
        function_signature="async def _audio_node(self, state: 'AudioAgent.Graph') -> Dict[str, Any]",
        description="Invoke a multimodal large language model to understand audio file content and return a list of descriptions divided by time segments",
        parameters=[
            {"name": "state", "type": "AudioAgent.Graph", "description": "Current state graph state"},
            {"name": "audio_path", "type": "str", "description": "Path to the audio file to be analyzed"}
        ],
        returns="Updated state dictionary containing time-segmented audio analysis results",
        category=ToolCategory.AUDIO_ANALYSIS,
        usage_example="Analyze an audio file, extracting main sound and ambient sound information by time segments",
        dependencies=[]
    )
]

# AudioAgent Knowledge Definitions
AUDIO_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.AUDIO,
        concepts=[
            "Audio time segmentation techniques",
            "Main sound recognition methods",
            "Ambient sound analysis techniques",
            "Audio feature extraction methods",
            "Sound style classification standards"
        ],
        rules=[
            "Must perform reasonable segmentation by time",
            "Each segment must include main sound content and style",
            "Each segment must include ambient sound content and style",
            "Output must be a list ordered by time sequence",
            "Ensure temporal accuracy of audio analysis"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.SOUND,
        concepts=[
            "Sound content recognition techniques",
            "Human voice vs. non-human voice distinction",
            "Timbre and pitch analysis",
            "Volume variation recognition",
            "Sound quality assessment"
        ],
        rules=[
            "Accurately identify the specific content of main sounds",
            "Distinguish between human voice, music, and ambient sounds",
            "Analyze timbre and pitch characteristics of sounds",
            "Identify volume changes and intensity",
            "Assess sound quality and clarity"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.EMOTION,
        concepts=[
            "Audio emotion recognition",
            "Emotional intensity analysis",
            "Emotional change tracking",
            "Tone and intonation analysis",
            "Emotional expression pattern recognition"
        ],
        rules=[
            "Identify the emotional undertone in audio",
            "Analyze the intensity and development of emotions",
            "Track the process of emotional changes",
            "Analyze tone and intonation characteristics",
            "Identify specific ways of emotional expression"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.TIME,
        concepts=[
            "Time segmentation principles",
            "Key time point identification",
            "Time order preservation",
            "Time label formatting",
            "Segmentation reasonableness assessment"
        ],
        rules=[
            "Perform reasonable segmentation based on content changes",
            "Identify key turning points in the audio",
            "Maintain strict time order",
            "Use standard time label formats",
            "Ensure reasonableness and continuity of segmentation"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.RHYTHM,
        concepts=[
            "Rhythm and beat analysis",
            "Prosodic pattern recognition",
            "Tempo variation detection",
            "Musical structure analysis",
            "Dynamic change recognition"
        ],
        rules=[
            "Analyze rhythm and beat characteristics of audio",
            "Identify prosodic patterns and repetitive structures",
            "Detect changes in tempo and intensity",
            "Analyze the structural composition of music",
            "Identify the process of dynamic changes"
        ]
    )
]

# AudioAgent Constraints
AUDIO_AGENT_CONSTRAINTS = [
    "Must perform segmented analysis by time; holistic processing is not allowed",
    "Each time segment must contain all four complete elements",
    "Output must be a list ordered by time sequence",
    "Must not omit key sound changes in the audio",
    "Must maintain temporal accuracy and continuity",
    "Must not add content not present in the audio",
    "Must properly handle long-duration audio files"
]

# AudioAgent Best Practices
AUDIO_AGENT_BEST_PRACTICES = [
    "Listen to the audio as a whole first to understand the general content",
    "Perform reasonable time segmentation based on content changes",
    "Accurately extract main sound information for each segment",
    "Carefully analyze the characteristics of ambient sounds",
    "Use standard time label formats",
    "Verify the completeness and accuracy of analysis results",
    "Provide detailed execution logs and analysis process"
]

# AudioAgent Profile
AUDIO_AGENT_PROFILE = AgentProfile(
    agent_id="audio_agent",
    description="A specialized agent responsible for analyzing audio content, extracting key information by time segments including main sound content, main sound style, ambient sound content, and ambient sound style, providing audio analysis support for multimodal scene understanding.",
    role=AUDIO_AGENT_ROLE,
    tools=AUDIO_AGENT_TOOLS,
    knowledge=AUDIO_AGENT_KNOWLEDGE,
    constraints=AUDIO_AGENT_CONSTRAINTS,
    best_practices=AUDIO_AGENT_BEST_PRACTICES,
    resources=[
        "DashScope multimodal large language model",
        "Audio base64 encoding utility",
        "Audio time segmentation analysis knowledge base",
        "Sound recognition and classification model",
        "JSON format output specification"
    ],
    run_methods=[
        "Follow the ReAct (Reasoning + Acting) pattern workflow",
        "Process through Thought -> Action -> Observation -> Final loop",
        "Directly invoke the audio tool node, passing in the state parameter",
        "The tool internally handles audio base64 encoding and model invocation",
        "Return a JSON list of descriptions divided by time segments",
        "Note: Execute the audio analysis tool even when no audio file is present (audio files are not transmitted via text; they are provided directly in base64 format when the tool is invoked)",
    ],
    command=COMMAND_audio,
    guide_book=Guide_Book_audio_expert
)

def get_audio_agent_profile() -> AgentProfile:
    """Get the AudioAgent configuration information"""
    return AUDIO_AGENT_PROFILE

def print_audio_agent_profile():
    """Print AudioAgent configuration information"""
    profile = AUDIO_AGENT_PROFILE
    print("=" * 80)
    print("AudioAgent Profile")
    print("=" * 80)

    print(f"\nAgent ID: {profile.agent_id}")
    print(f"Overall Description: {profile.description}")
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
    print_audio_agent_profile()
