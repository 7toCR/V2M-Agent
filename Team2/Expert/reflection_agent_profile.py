"""
Reflection Agent Profile

Defines the role, responsibilities, and reflection mechanism for the reflection node.
Based on the Reflexion paper's Self-Reflection model design.

Reflexion core components:
- Actor: generates text and actions
- Evaluator: evaluates output quality
- Self-Reflection: generates verbal feedback signals
- Memory: stores reflection experiences
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
    # Check if our needed enum values exist; if not, fall through to local defs
    _check = ToolCategory.REFLECTION  # noqa: F841
    _check2 = KnowledgeDomain.REFLECTION  # noqa: F841
except (ImportError, AttributeError):
    from dataclasses import dataclass
    from enum import Enum

    class ToolCategory(str, Enum):
        TEXT_ANALYSIS = "text_analysis"
        MULTIMODAL = "multimodal_understanding"
        DATA_PROCESSING = "data_processing"
        REFLECTION = "reflection"
        OTHER = "other"

    class KnowledgeDomain(str, Enum):
        TEXT = "text_analysis"
        SEMANTIC = "semantic_understanding"
        EMOTION = "emotion_analysis"
        SCENE = "scene_analysis"
        STYLE = "style_recognition"
        REFLECTION = "self_reflection"
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


# Reflection Agent Role Definition
REFLECTION_AGENT_ROLE = AgentRole(
    name="Self-Reflection and Improvement Expert",
    description="Responsible for analyzing execution results, identifying issues, generating improvement suggestions, and storing experiences in the memory system for future use",
    responsibilities=[
        "Analyze the match between execution results and task objectives",
        "Identify deficiencies and errors in the execution process",
        "Generate specific, actionable improvement suggestions",
        "Evaluate reflection quality and decide whether retry is needed",
        "Maintain reflection memory to support experience accumulation"
    ],
    expertise=[
        "Result quality assessment",
        "Problem diagnosis and analysis",
        "Improvement suggestion generation",
        "Self-assessment and calibration",
        "Experience learning and summarization"
    ]
)


# Reflection Agent Tool Definition
REFLECTION_AGENT_TOOLS = [
    AgentTool(
        name="reflect",
        function_signature="async def _reflect_node(self, state: 'Graph') -> Dict[str, Any]",
        description="Reflection node: generates reflection based on observation results, evaluates quality, and decides next action",
        parameters=[
            {"name": "state", "type": "Graph", "description": "Current state graph state, containing task, execution results, observations, etc."}
        ],
        returns="Updated state containing reflection results and improvement suggestions",
        category=ToolCategory.REFLECTION,
        usage_example="Analyze execution results, generate reflection, and decide whether retry is needed",
        dependencies=["observation_result"]
    )
]


# Reflection Agent Knowledge Definition
REFLECTION_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.REFLECTION,
        concepts=[
            "Reflexion self-reflection mechanism",
            "Language reinforcement learning",
            "Experience memory storage",
            "Sparse feedback signal amplification",
            "Quality assessment criteria"
        ],
        rules=[
            "Reflection must be based on actual execution results",
            "Improvement suggestions must be specific and actionable",
            "Assessment results are used to decide whether to retry",
            "High-quality reflections output directly; low-quality reflections trigger retry",
            "Memory uses a sliding window strategy"
        ]
    )
]


# Reflection Agent Constraints
REFLECTION_AGENT_CONSTRAINTS = [
    "Reflection must be based on actual observed results",
    "Improvement suggestions must be specific and executable",
    "Quality assessment must be objective and fair",
    "Memory storage follows the sliding window strategy",
    "Reflection content is used to guide the next action"
]


# Reflection Agent Best Practices
REFLECTION_AGENT_BEST_PRACTICES = [
    "First evaluate whether results meet task objectives",
    "Identify specific success and failure factors",
    "Attribute failures to specific steps",
    "Generate verifiable improvement suggestions",
    "Reference historical reflections to avoid repeating mistakes"
]


# Reflection System Prompt Template
REFLECTION_SYSTEM_PROMPT_TEMPLATE = """You are a professional self-reflection expert.

Your responsibility is to analyze execution results, identify issues, and generate improvement suggestions.

## Reflection Mechanism (Based on Reflexion Paper)

Reflexion is a framework for self-reinforcement through language feedback:
1. **Evaluate**: Analyze whether current execution results meet objectives
2. **Diagnose**: Identify success factors and shortcomings
3. **Suggest**: Generate specific improvement directions
4. **Decide**: Evaluate reflection quality and decide whether retry is needed

## Historical Reflection Memory

{history_reflections}

## Current Task

Task type: {task_type}
Task description: {task_description}

## Execution Results

Observation: {observation}

## Reflection Requirements

Please output your reflection in the following JSON format:

{{
    "analysis": "Analysis of execution results, determining whether objectives were achieved",
    "strengths": ["successful aspect 1", "successful aspect 2"],
    "weaknesses": ["shortcoming 1", "shortcoming 2"],
    "improvement": "Specific actionable improvement suggestions",
    "quality": "high/medium/low - reflection quality assessment",
    "should_retry": true/false - whether retry is recommended
}}

Requirements:
- Analysis must be based on actual observation results
- Improvement suggestions must be specific and actionable
- Quality assessment must be objective and truthful
- Only set should_retry=false when quality=high
"""

REFLECTION_COMMAND = """## Reflection Node Execution Commands

When performing reflection analysis, follow these steps:

1. **Load Historical Memory**
   - Retrieve recent reflection records from the reflection memory module
   - Analyze improvement suggestions from historical reflections

2. **Analyze Observation Results**
   - Evaluate the quality of current execution results
   - Identify gaps between results and task objectives

3. **Generate Reflection**
   - Analyze success factors and shortcomings
   - Generate specific improvement suggestions
   - Evaluate reflection quality

4. **Decide Next Step**
   - If quality is high and results are satisfactory -> output final answer
   - If quality is low or results are unsatisfactory -> return to think node for retry
   - If maximum iterations reached -> output current best result

5. **Update Memory**
   - Store reflection results in the memory module
   - Maintain memory using the sliding window strategy
"""


# Reflection Agent Profile
REFLECTION_AGENT_PROFILE = AgentProfile(
    agent_id="reflection_agent",
    description="Professional agent responsible for self-reflection and improvement, implementing language feedback reinforcement learning based on the Reflexion framework",
    role=REFLECTION_AGENT_ROLE,
    tools=REFLECTION_AGENT_TOOLS,
    knowledge=REFLECTION_AGENT_KNOWLEDGE,
    constraints=REFLECTION_AGENT_CONSTRAINTS,
    best_practices=REFLECTION_AGENT_BEST_PRACTICES,
    resources=[
        "Reflection memory module",
        "Historical reflection data",
        "Reflexion framework"
    ],
    run_methods=[
        "Receive observation results as input",
        "Analyze match between results and objectives",
        "Generate improvement suggestions",
        "Evaluate reflection quality",
        "Decide whether retry is needed",
        "Update reflection memory"
    ],
    command=REFLECTION_COMMAND
)


def build_reflection_prompt(
    task_type: str,
    task_description: str,
    observation: str,
    history_reflections: str = ""
) -> str:
    """
    Build reflection prompt.

    Args:
        task_type: Task type
        task_description: Task description
        observation: Observation result
        history_reflections: Historical reflection summary

    Returns:
        Formatted reflection prompt
    """
    return REFLECTION_SYSTEM_PROMPT_TEMPLATE.format(
        history_reflections=history_reflections or "No historical reflection records.",
        task_type=task_type,
        task_description=task_description,
        observation=observation
    )


def parse_reflection_result(response_text: str) -> Dict[str, Any]:
    """
    Parse reflection result.

    Args:
        response_text: Reflection text returned by LLM

    Returns:
        Parsed reflection result dictionary
    """
    import re
    import json

    # Try to extract JSON content
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response_text, re.DOTALL)

    for match in matches:
        try:
            result = json.loads(match)
            if all(k in result for k in ['analysis', 'strengths', 'weaknesses', 'improvement', 'quality']):
                return result
        except json.JSONDecodeError:
            continue

    # If parsing fails, return defaults
    return {
        "analysis": response_text[:200] if response_text else "Unable to parse reflection result",
        "strengths": [],
        "weaknesses": ["Reflection result parsing failed"],
        "improvement": "Please check reflection output format",
        "quality": "low",
        "should_retry": True
    }


def get_reflection_agent_profile() -> AgentProfile:
    """Get the reflection agent's configuration."""
    return REFLECTION_AGENT_PROFILE


if __name__ == "__main__":
    # Test code
    prompt = build_reflection_prompt(
        task_type="text",
        task_description="Analyze an ancient martial arts text, extract background, style, subject, and emotion",
        observation="Extracted 3 scenes including background and style info, but subject emotion analysis was insufficiently accurate",
        history_reflections="Reflection 1: First analysis succeeded but missed details"
    )

    print("Reflection prompt:")
    print(prompt)
    print("\n" + "=" * 50)
    print("Reflection Agent Profile:")
    print(f"Agent ID: {REFLECTION_AGENT_PROFILE.agent_id}")
    print(f"Description: {REFLECTION_AGENT_PROFILE.description}")
    print(f"Role: {REFLECTION_AGENT_PROFILE.role.name}")
