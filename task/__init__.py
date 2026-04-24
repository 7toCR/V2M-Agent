"""
Task Generation Module

This module provides functionality to generate tasks for sub-agents based on:
- Agent roles
- Available tools
- Agent knowledge base
"""

from .task_profile import AgentProfile, AgentRole, AgentTool, AgentKnowledge
from .task_generator import TaskGenerator, TaskList

__all__ = [
    "AgentProfile",
    "AgentRole",
    "AgentTool",
    "AgentKnowledge",
    "TaskGenerator",
    "TaskList",
]
