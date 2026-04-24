"""
Requirement Supervisor Agent

Team 1 supervisor implementing Algorithm 1 for requirement analysis and task distribution.
Follows Think-Act-Observe-Reflect pattern with LangGraph state machine.
"""

import os
import sys
import json
import asyncio
from typing import TypedDict, Annotated, Literal, Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tools.live_status import LiveStatus

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

try:
    from Team1.AgentProfile.requirement_supervisor_profile import REQUIREMENT_SUPERVISOR_PROFILE
except ImportError:
    from AgentProfile.requirement_supervisor_profile import REQUIREMENT_SUPERVISOR_PROFILE

try:
    from Team1.verifier.requirement_verifier import RequirementVerifier
except ImportError:
    from verifier.requirement_verifier import RequirementVerifier

try:
    from Team2.AgentProfile.scene_understanding_supervisor_profile import SCENE_UNDERSTANDING_SUPERVISOR_PROFILE
except ImportError:
    SCENE_UNDERSTANDING_SUPERVISOR_PROFILE = None

try:
    from Team3.AgentProfile.music_generation_supervisor_profile import MUSIC_GENERATION_SUPERVISOR_PROFILE
except ImportError:
    MUSIC_GENERATION_SUPERVISOR_PROFILE = None


load_dotenv()


class RequirementSupervisor:
    """
    Team 1 Requirement Analysis Supervisor

    Implements Algorithm 1: Task Prompt Generation based on Agent Profile
    - Parse: Extract objectives from user requirement
    - DetermineNeed: Decide which teams/experts are needed
    - ReadProfile: Load Team 2/3 supervisor profiles
    - Match: Map requirements to team capabilities
    - ExtractInstruction: Generate task instructions
    - GenerateConstraints: Extract constraints from profiles
    """

    class Graph(TypedDict):
        """State schema for RequirementSupervisor"""
        # Input
        user_requirement: str
        text_address: List[str]
        audio_address: List[str]
        photo_address: List[str]
        video_address: List[str]

        # Workflow state
        state: Literal["init", "think", "act", "observe", "reflect", "final"]
        iteration: int
        max_iterations: int

        # Messages
        global_messages: List[Any]
        system_prompt_messages: List[Any]

        # Algorithm 1 intermediate results
        parsed_requirement: Optional[Dict[str, Any]]
        available_modalities: List[str]
        needed_teams: Dict[str, List[str]]  # team_name -> list of expert_ids
        team_profiles: Dict[str, Any]  # team_name -> profile
        requirement_matches: Dict[str, Any]
        task_instructions: Dict[str, str]  # team_name -> instruction
        task_constraints: Dict[str, List[str]]  # team_name -> constraints

        # Output
        team2_task_packet: Optional[Dict[str, Any]]
        team3_task_packet: Optional[Dict[str, Any]]
        requirement_analysis_result: Optional[Dict[str, Any]]
        final_answer: str
        complete: bool

        # Verification
        verification_result: Optional[Dict[str, Any]]
        verify_retry_count: int

        # Reflection
        reflection_history: List[Dict[str, Any]]
        quality_score: float

    def __init__(self, config_path: Optional[str] = None):
        """Initialize RequirementSupervisor"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(CURRENT_DIR), "config_requirement_supervisor.json")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        model_config = self.config["model"]
        self.llm = ChatOpenAI(
            model=model_config["name"],
            api_key=os.getenv(model_config["api_key_env"]),
            base_url=model_config["base_url"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
        )

        self.max_iterations = self.config["agents"]["max_iterations"]
        self.profile = REQUIREMENT_SUPERVISOR_PROFILE
        self.verifier = RequirementVerifier()
        self.max_verify_retries = 3

        # Build state machine
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine"""
        workflow = StateGraph(self.Graph)

        # Add nodes
        workflow.add_node("init", self._init_node)
        workflow.add_node("think", self._think_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("final", self._final_node)

        # Add edges
        workflow.set_entry_point("init")
        workflow.add_edge("init", "think")
        workflow.add_conditional_edges("think", self._route_after_think)
        workflow.add_edge("act", "observe")
        workflow.add_conditional_edges("observe", self._route_after_observe)
        workflow.add_conditional_edges("verify", self._route_after_verify)
        workflow.add_conditional_edges("reflect", self._route_after_reflect)
        workflow.add_edge("final", END)

        return workflow.compile()

    async def _init_node(self, state: Graph) -> Dict[str, Any]:
        """Initialize state and system prompt"""
        system_prompt = self._build_system_prompt()

        return {
            "state": "think",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "system_prompt_messages": [SystemMessage(content=system_prompt)],
            "global_messages": [],
            "parsed_requirement": None,
            "available_modalities": [],
            "needed_teams": {},
            "team_profiles": {},
            "requirement_matches": {},
            "task_instructions": {},
            "task_constraints": {},
            "team2_task_packet": None,
            "team3_task_packet": None,
            "requirement_analysis_result": None,
            "final_answer": "",
            "complete": False,
            "verification_result": None,
            "verify_retry_count": 0,
            "reflection_history": [],
            "quality_score": 0.0,
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt from profile"""
        prompt_parts = [
            f"# Role: {self.profile.role.name}",
            f"\n{self.profile.role.description}\n",
            "\n## Responsibilities:",
        ]
        for resp in self.profile.role.responsibilities:
            prompt_parts.append(f"- {resp}")

        prompt_parts.append("\n## Expertise:")
        for exp in self.profile.role.expertise:
            prompt_parts.append(f"- {exp}")

        prompt_parts.append("\n## Available Tools:")
        for tool in self.profile.tools:
            prompt_parts.append(f"- {tool.name}: {tool.description}")

        prompt_parts.append("\n## Constraints:")
        for constraint in self.profile.constraints:
            prompt_parts.append(f"- {constraint}")

        prompt_parts.append("\n## Workflow:")
        for method in self.profile.run_methods:
            prompt_parts.append(f"- {method}")

        prompt_parts.append(f"\n## Guide:\n{self.profile.guide_book}")

        return "\n".join(prompt_parts)

    async def _think_node(self, state: Graph) -> Dict[str, Any]:
        """Think: Decide next action based on Algorithm 1 workflow"""
        iteration = state["iteration"] + 1
        needed_teams = state.get("needed_teams", {})

        # Helper: check if all needed packets are generated
        def _all_packets_ready():
            if "Team2" in needed_teams and not state.get("team2_task_packet"):
                return False
            if "Team3" in needed_teams and not state.get("team3_task_packet"):
                return False
            # Need at least one packet to be "ready"
            return bool(state.get("team2_task_packet") or state.get("team3_task_packet"))

        # Check if we've completed all Algorithm 1 steps -> go to verify
        if needed_teams and _all_packets_ready():
            LiveStatus.update("用户理解", f"[Think] Iteration {iteration}: All needed packets ready, moving to verify")
            return {
                "state": "verify",
                "iteration": iteration,
            }

        # Check max iterations
        if iteration >= state["max_iterations"]:
            LiveStatus.update("用户理解", f"[Think] Iteration {iteration}: Max iterations reached, moving to final")
            return {
                "state": "final",
                "iteration": iteration,
            }

        # Determine next step in Algorithm 1
        if not state.get("parsed_requirement"):
            next_action = "parse_requirement"
        elif not state.get("available_modalities"):
            next_action = "determine_modalities"
        elif not needed_teams:
            next_action = "determine_need"
        elif not state.get("team_profiles"):
            next_action = "read_profiles"
        elif not state.get("requirement_matches"):
            next_action = "match_requirements"
        elif not state.get("task_instructions"):
            next_action = "extract_instructions"
        elif not state.get("task_constraints"):
            next_action = "generate_constraints"
        elif "Team2" in needed_teams and not state.get("team2_task_packet"):
            next_action = "generate_team2_packet"
        elif "Team3" in needed_teams and not state.get("team3_task_packet"):
            next_action = "generate_team3_packet"
        else:
            next_action = "finalize"

        LiveStatus.update("用户理解", f"[Think] Iteration {iteration}: Next action = {next_action}")

        return {
            "state": "act",
            "iteration": iteration,
        }

    async def _act_node(self, state: Graph) -> Dict[str, Any]:
        """Act: Execute the determined action"""
        updates = {"state": "observe"}
        needed_teams = state.get("needed_teams", {})

        # Execute based on current workflow state
        if not state.get("parsed_requirement"):
            LiveStatus.update("用户理解", "[Act] Parsing requirement...")
            updates["parsed_requirement"] = await self._parse_requirement(state)

        elif not state.get("available_modalities"):
            LiveStatus.update("用户理解", "[Act] Determining modalities...")
            updates["available_modalities"] = self._determine_modalities(state)

        elif not needed_teams:
            LiveStatus.update("用户理解", "[Act] Determining needed teams...")
            updates["needed_teams"] = await self._determine_need(state)

        elif not state.get("team_profiles"):
            LiveStatus.update("用户理解", "[Act] Reading team profiles...")
            updates["team_profiles"] = self._read_profiles(state)

        elif not state.get("requirement_matches"):
            LiveStatus.update("用户理解", "[Act] Matching requirements...")
            updates["requirement_matches"] = await self._match_requirements(state)

        elif not state.get("task_instructions"):
            LiveStatus.update("用户理解", "[Act] Extracting instructions...")
            updates["task_instructions"] = await self._extract_instructions(state)

        elif not state.get("task_constraints"):
            LiveStatus.update("用户理解", "[Act] Generating constraints...")
            updates["task_constraints"] = self._generate_constraints(state)

        elif "Team2" in needed_teams and not state.get("team2_task_packet"):
            LiveStatus.update("用户理解", "[Act] Generating Team 2 packet...")
            updates["team2_task_packet"] = self._generate_task_packet("Team2", state)

        elif "Team3" in needed_teams and not state.get("team3_task_packet"):
            LiveStatus.update("用户理解", "[Act] Generating Team 3 packet...")
            updates["team3_task_packet"] = self._generate_task_packet("Team3", state)

        return updates

    async def _parse_requirement(self, state: Graph) -> Dict[str, Any]:
        """Algorithm 1 Step 1: Parse user requirement"""
        prompt = f"""
Parse the following user requirement and extract:
1. Main objectives
2. Constraints mentioned
3. Modality hints (text/audio/image/video references)

User requirement: {state['user_requirement']}

Return as JSON with keys: objectives, constraints, modality_hints
"""
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        # Extract JSON from response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def _determine_modalities(self, state: Graph) -> List[str]:
        """Algorithm 1 Step 2a: Determine available modalities"""
        modalities = []
        if state.get("text_address"):
            modalities.append("text")
        if state.get("audio_address"):
            modalities.append("audio")
        if state.get("photo_address"):
            modalities.append("photo")
        if state.get("video_address"):
            modalities.append("video")
        return modalities

    async def _determine_need(self, state: Graph) -> Dict[str, List[str]]:
        """Algorithm 1 Step 2b: Determine which teams are needed"""
        prompt = f"""
Based on the parsed requirement and available modalities, determine which teams are needed:

Parsed requirement: {json.dumps(state['parsed_requirement'], ensure_ascii=False)}
Available modalities: {state['available_modalities']}

Available teams:
- Team2: Scene Understanding (handles multimodal scene analysis)
- Team3: Music Generation (generates music prompts from scenes)

Rules:
- Team2 is needed if any modality is present
- Team3 is needed if music generation is requested
- Team3 requires Team2 output

Return as JSON: {{"Team2": ["scene_understanding_supervisor"], "Team3": ["music_generation_supervisor"]}}
Or omit teams that are not needed.
"""
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def _read_profiles(self, state: Graph) -> Dict[str, Any]:
        """Algorithm 1 Step 3: Read team profiles"""
        profiles = {}

        if "Team2" in state["needed_teams"] and SCENE_UNDERSTANDING_SUPERVISOR_PROFILE:
            profiles["Team2"] = {
                "agent_id": SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.agent_id,
                "description": SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.description,
                "role": {
                    "name": SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.role.name,
                    "description": SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.role.description,
                    "responsibilities": SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.role.responsibilities,
                    "expertise": SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.role.expertise,
                },
                "constraints": SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.constraints,
                "knowledge": [
                    {
                        "domain": k.domain.value,
                        "concepts": k.concepts,
                        "rules": k.rules,
                    }
                    for k in SCENE_UNDERSTANDING_SUPERVISOR_PROFILE.knowledge
                ],
            }

        if "Team3" in state["needed_teams"] and MUSIC_GENERATION_SUPERVISOR_PROFILE:
            profiles["Team3"] = {
                "agent_id": MUSIC_GENERATION_SUPERVISOR_PROFILE.agent_id,
                "description": MUSIC_GENERATION_SUPERVISOR_PROFILE.description,
                "role": {
                    "name": MUSIC_GENERATION_SUPERVISOR_PROFILE.role.name,
                    "description": MUSIC_GENERATION_SUPERVISOR_PROFILE.role.description,
                    "responsibilities": MUSIC_GENERATION_SUPERVISOR_PROFILE.role.responsibilities,
                    "expertise": MUSIC_GENERATION_SUPERVISOR_PROFILE.role.expertise,
                },
                "constraints": MUSIC_GENERATION_SUPERVISOR_PROFILE.constraints,
                "knowledge": [
                    {
                        "domain": k.domain.value,
                        "concepts": k.concepts,
                        "rules": k.rules,
                    }
                    for k in MUSIC_GENERATION_SUPERVISOR_PROFILE.knowledge
                ],
            }

        return profiles

    async def _match_requirements(self, state: Graph) -> Dict[str, Any]:
        """Algorithm 1 Step 4: Match requirements to team capabilities"""
        prompt = f"""
Match the parsed requirements to team capabilities:

Requirements: {json.dumps(state['parsed_requirement'], ensure_ascii=False)}
Available modalities: {state['available_modalities']}

Team profiles:
{json.dumps(state['team_profiles'], ensure_ascii=False, indent=2)}

For each team, identify:
1. Which objectives they should handle
2. Which modalities they should process
3. Confidence score (0-1)

Return as JSON with team names as keys.
"""
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    async def _extract_instructions(self, state: Graph) -> Dict[str, str]:
        """Algorithm 1 Step 5: Extract task instructions"""
        prompt = f"""
Generate specific task instructions for each team based on the requirement matches:

Matches: {json.dumps(state['requirement_matches'], ensure_ascii=False)}
User requirement: {state['user_requirement']}

For each team, generate a clear, actionable instruction (I_i).

Return as JSON: {{"Team2": "instruction text", "Team3": "instruction text"}}
"""
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def _generate_constraints(self, state: Graph) -> Dict[str, List[str]]:
        """Algorithm 1 Step 6: Generate constraints from profiles"""
        constraints = {}

        for team_name, profile_dict in state["team_profiles"].items():
            team_constraints = []

            # Extract from profile constraints
            if "constraints" in profile_dict:
                team_constraints.extend(profile_dict["constraints"])

            # Extract from knowledge rules
            if "knowledge" in profile_dict:
                for knowledge in profile_dict["knowledge"]:
                    if "rules" in knowledge:
                        team_constraints.extend(knowledge["rules"])

            constraints[team_name] = team_constraints

        return constraints

    def _generate_task_packet(self, team_name: str, state: Graph) -> Dict[str, Any]:
        """Generate structured task packet for a team"""
        return {
            "team_name": team_name,
            "instruction": state["task_instructions"].get(team_name, ""),
            "constraints": state["task_constraints"].get(team_name, []),
            "modalities": state["available_modalities"],
            "modality_addresses": {
                "text_address": state.get("text_address", []),
                "audio_address": state.get("audio_address", []),
                "photo_address": state.get("photo_address", []),
                "video_address": state.get("video_address", []),
            },
            "user_requirement": state["user_requirement"],
            "parsed_requirement": state["parsed_requirement"],
            "match_info": state["requirement_matches"].get(team_name, {}),
        }

    async def _observe_node(self, state: Graph) -> Dict[str, Any]:
        """Observe: Check if action was successful and determine next state"""
        # Check if we've completed all Algorithm 1 steps -> go to verify
        if state.get("team2_task_packet") and state.get("team3_task_packet"):
            return {"state": "verify"}

        # Check if we've exceeded max iterations
        if state["iteration"] >= state["max_iterations"]:
            return {"state": "verify"}

        # Otherwise continue to next step
        return {"state": "think"}

    def _route_after_think(self, state: Graph) -> str:
        """Route after think node"""
        if state["state"] == "final":
            return "final"
        if state["state"] == "verify":
            return "verify"
        return "act"

    def _route_after_observe(self, state: Graph) -> str:
        """Route after observe node"""
        if state["state"] == "verify":
            return "verify"
        if state["state"] == "think":
            return "think"
        return "verify"

    async def _verify_node(self, state: Graph) -> Dict[str, Any]:
        """Verify: Validate task packets using RequirementVerifier"""
        retry_count = state.get("verify_retry_count", 0)

        # Build the analysis result for verification
        analysis_result = {
            "user_requirement": state["user_requirement"],
            "parsed_requirement": state.get("parsed_requirement"),
            "available_modalities": state.get("available_modalities", []),
            "needed_teams": state.get("needed_teams", {}),
            "team2_task_packet": state.get("team2_task_packet"),
            "team3_task_packet": state.get("team3_task_packet"),
        }

        LiveStatus.update(
            "用户理解",
            f"[Verify] Running verification (attempt {retry_count + 1}/{self.max_verify_retries})...",
        )
        verification_result = self.verifier.verify(analysis_result)

        if verification_result["passed"]:
            LiveStatus.update("用户理解", "[Verify] PASSED - task packets are valid")
            return {
                "state": "final",
                "verification_result": verification_result,
                "quality_score": 1.0,
            }

        # Verification failed
        if retry_count >= self.max_verify_retries:
            LiveStatus.update(
                "用户理解",
                f"[Verify] FAILED after {retry_count + 1} attempts, proceeding with best effort",
            )
            return {
                "state": "final",
                "verification_result": verification_result,
                "quality_score": 0.5,
            }

        issue_preview = "; ".join(str(issue) for issue in verification_result["issues"][:2])
        LiveStatus.update(
            "用户理解",
            f"[Verify] FAILED - {len(verification_result['issues'])} issues found: {issue_preview}",
        )

        return {
            "state": "reflect",
            "verification_result": verification_result,
            "verify_retry_count": retry_count + 1,
        }

    def _route_after_verify(self, state: Graph) -> str:
        """Route after verify node"""
        if state["state"] == "final":
            return "final"
        return "reflect"

    async def _reflect_node(self, state: Graph) -> Dict[str, Any]:
        """Reflect: Use LLM to analyze verification issues and decide what to fix"""
        verification_result = state.get("verification_result", {})
        issues = verification_result.get("issues", [])
        warnings = verification_result.get("warnings", [])
        recommendations = verification_result.get("recommendations", [])

        # Use LLM to analyze issues and decide what to redo
        prompt = f"""
You are the Requirement Analysis Supervisor reflecting on verification failures.

The verifier found these issues with the task packets:

Issues (must fix):
{json.dumps(issues, ensure_ascii=False, indent=2)}

Warnings (optional):
{json.dumps(warnings, ensure_ascii=False, indent=2)}

Recommendations:
{json.dumps(recommendations, ensure_ascii=False, indent=2)}

Current state:
- User requirement: {state['user_requirement']}
- Available modalities: {state.get('available_modalities', [])}
- Team 2 instruction length: {len(state.get('task_instructions', {}).get('Team2', ''))}
- Team 3 instruction length: {len(state.get('task_instructions', {}).get('Team3', ''))}
- Team 2 constraints count: {len(state.get('task_constraints', {}).get('Team2', []))}
- Team 3 constraints count: {len(state.get('task_constraints', {}).get('Team3', []))}

Analyze the issues and decide which steps need to be re-executed.
Return a JSON object with:
- "analysis": brief explanation of what went wrong
- "fix_strategy": which steps to redo, one of:
  - "redo_instructions" - regenerate task instructions (for instruction quality issues)
  - "redo_constraints" - regenerate constraints (for constraint issues)
  - "redo_packets" - regenerate task packets (for structural/modality issues)
  - "redo_all" - redo instructions, constraints, and packets
"""
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            reflection = json.loads(content)
        except json.JSONDecodeError:
            reflection = {"analysis": content, "fix_strategy": "redo_all"}

        fix_strategy = reflection.get("fix_strategy", "redo_all")
        LiveStatus.update(
            "用户理解",
            f"[Reflect] Strategy: {fix_strategy}; Analysis: {reflection.get('analysis', 'N/A')}",
        )

        # Record reflection
        history_entry = {
            "iteration": state["iteration"],
            "issues": issues,
            "strategy": fix_strategy,
            "analysis": reflection.get("analysis", ""),
        }
        updated_history = list(state.get("reflection_history", []))
        updated_history.append(history_entry)

        # Clear the appropriate state fields to trigger re-execution
        updates: Dict[str, Any] = {
            "state": "think",
            "reflection_history": updated_history,
        }

        if fix_strategy in ("redo_instructions", "redo_all"):
            updates["task_instructions"] = {}
            updates["task_constraints"] = {}
            updates["team2_task_packet"] = None
            updates["team3_task_packet"] = None
        elif fix_strategy == "redo_constraints":
            updates["task_constraints"] = {}
            updates["team2_task_packet"] = None
            updates["team3_task_packet"] = None
        elif fix_strategy == "redo_packets":
            updates["team2_task_packet"] = None
            updates["team3_task_packet"] = None

        return updates

    def _route_after_reflect(self, state: Graph) -> str:
        """Route after reflect node"""
        if state["state"] == "final":
            return "final"
        return "think"

    async def _final_node(self, state: Graph) -> Dict[str, Any]:
        """Final: Assemble final output"""
        requirement_analysis_result = {
            "user_requirement": state["user_requirement"],
            "parsed_requirement": state["parsed_requirement"],
            "available_modalities": state["available_modalities"],
            "needed_teams": state["needed_teams"],
            "team2_task_packet": state["team2_task_packet"],
            "team3_task_packet": state["team3_task_packet"],
            "quality_score": state["quality_score"],
            "verification_passed": state.get("verification_result", {}).get("passed", False),
            "verify_attempts": state.get("verify_retry_count", 0),
        }

        final_answer = json.dumps(requirement_analysis_result, ensure_ascii=False, indent=2)

        LiveStatus.update(
            "用户理解",
            (
                f"[Final] Quality={state['quality_score']}, "
                f"passed={requirement_analysis_result['verification_passed']}, "
                f"attempts={requirement_analysis_result['verify_attempts']}"
            ),
        )

        return {
            "state": "final",
            "requirement_analysis_result": requirement_analysis_result,
            "final_answer": final_answer,
            "complete": True,
        }

    async def ainvoke(
        self,
        user_requirement: str,
        multimodal_input: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Async invoke the agent"""
        if multimodal_input is None:
            multimodal_input = {}

        initial_state: RequirementSupervisor.Graph = {
            "user_requirement": user_requirement,
            "text_address": multimodal_input.get("text_address", []),
            "audio_address": multimodal_input.get("audio_address", []),
            "photo_address": multimodal_input.get("photo_address", []),
            "video_address": multimodal_input.get("video_address", []),
            "state": "init",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "global_messages": [],
            "system_prompt_messages": [],
            "parsed_requirement": None,
            "available_modalities": [],
            "needed_teams": {},
            "team_profiles": {},
            "requirement_matches": {},
            "task_instructions": {},
            "task_constraints": {},
            "team2_task_packet": None,
            "team3_task_packet": None,
            "requirement_analysis_result": None,
            "final_answer": "",
            "complete": False,
            "verification_result": None,
            "verify_retry_count": 0,
            "reflection_history": [],
            "quality_score": 0.0,
        }

        result = await self.graph.ainvoke(
            initial_state,
            {"recursion_limit": 80}
        )
        return result

    def invoke(
        self,
        user_requirement: str,
        multimodal_input: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Sync invoke wrapper"""
        return asyncio.run(self.ainvoke(user_requirement, multimodal_input))


async def main():
    """Test RequirementSupervisor"""
    print("=" * 80)
    print("Testing RequirementSupervisor")
    print("=" * 80)

    supervisor = RequirementSupervisor()

    # Test case 1: Full multimodal
    print("\n[Test 1] Full multimodal input")
    result = await supervisor.ainvoke(
        user_requirement="请对提供的文本、音频、图片和视频进行多模态理解,并基于理解结果生成现代音乐的prompt。",
        multimodal_input={
            "text_address": ["sample/input/test/description.txt"],
            "audio_address": ["sample/input/test/audio.mp3"],
            "photo_address": ["sample/input/test/image.jpg"],
            "video_address": ["sample/input/test/video.mp4"],
        }
    )
    print(f"\nResult:\n{result['final_answer']}")

    # Test case 2: Text only
    print("\n" + "=" * 80)
    print("[Test 2] Text only input")
    result = await supervisor.ainvoke(
        user_requirement="分析这段文本并生成音乐",
        multimodal_input={
            "text_address": ["sample/input/test/description.txt"],
        }
    )
    print(f"\nResult:\n{result['final_answer']}")

    # Test case 3: Video only
    print("\n" + "=" * 80)
    print("[Test 3] Video only input")
    result = await supervisor.ainvoke(
        user_requirement="理解视频内容",
        multimodal_input={
            "video_address": ["sample/input/test/video.mp4"],
        }
    )
    print(f"\nResult:\n{result['final_answer']}")


if __name__ == "__main__":
    asyncio.run(main())
