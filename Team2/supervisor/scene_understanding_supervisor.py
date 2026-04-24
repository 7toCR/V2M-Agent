"""
Scene Understanding Supervisor (Team 2)

Coordinator agent that reads expert AgentProfiles, generates per-expert tasks
via Algorithm 1 (AsyncTaskCreator), dispatches parallel expert execution,
and forwards results to the SceneVerifier for four-stage fusion and validation.

Follows Think-Act-Observe-Reflect iterative loop via LangGraph.

Paper reference (Section 3.1, 3.3):
    - Supervisor reads expert profiles to determine domains and boundaries
    - Decomposes requirements into subtasks assigned to best-matched experts
    - Experts execute in parallel, each following Think-Act-Observe-Reflect
    - Verifier performs independent four-stage fusion and validation
"""

import os
import sys
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages, StateGraph, START, END
from typing_extensions import TypedDict
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM2_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TEAM2_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tools.tools import _print_with_indent as _base_print


def _log(msg: str, indent: int = 1):
    """Convenience wrapper for _log(prefix, content, tab_count)."""
    _base_print("", msg, tab_count=indent)

# Import Supervisor profile
try:
    from Team2.AgentProfile.scene_understanding_supervisor_profile import SCENE_UNDERSTANDING_SUPERVISOR_PROFILE
    PROFILE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import SCENE_UNDERSTANDING_SUPERVISOR_PROFILE: {e}")
    PROFILE_IMPORT_SUCCESS = False
    SCENE_UNDERSTANDING_SUPERVISOR_PROFILE = None

# Import reflection modules
try:
    from Team2.Expert.reflection_memory import get_reflection_memory
    from Team2.Expert.reflection_agent_profile import build_reflection_prompt, parse_reflection_result
    REFLECTION_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import reflection modules: {e}")
    REFLECTION_IMPORT_SUCCESS = False
    get_reflection_memory = None
    build_reflection_prompt = None
    parse_reflection_result = None

# Import expert agents
try:
    from Team2.Expert.text import TextAgent
    from Team2.Expert.audio import AudioAgent
    from Team2.Expert.photo import PhotoAgent
    from Team2.Expert.video import VideoAgent
    EXPERTS_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import expert agents: {e}")
    EXPERTS_IMPORT_SUCCESS = False
    TextAgent = None
    AudioAgent = None
    PhotoAgent = None
    VideoAgent = None

# Import verifier
try:
    from Team2.verifier.scene_verifier import SceneVerifier
    VERIFIER_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import SceneVerifier: {e}")
    VERIFIER_IMPORT_SUCCESS = False
    SceneVerifier = None

# Import task creator
try:
    from task.task_create import AsyncTaskCreator
    TASK_CREATOR_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import AsyncTaskCreator: {e}")
    TASK_CREATOR_IMPORT_SUCCESS = False
    AsyncTaskCreator = None

load_dotenv()


class SceneUnderstandingSupervisor:
    """
    Team 2 Supervisor: coordinates multimodal scene understanding.

    Responsibilities:
    1. Parse team2_task_packet from Team 1
    2. Read expert AgentProfiles and run Algorithm 1
    3. Generate per-expert tasks via AsyncTaskCreator
    4. Dispatch experts in parallel (asyncio.gather)
    5. Forward results to SceneVerifier
    6. Handle verification feedback (reflect/retry if needed)

    Follows Think-Act-Observe-Reflect iterative loop via LangGraph.
    """

    class Graph(TypedDict):
        # Messages
        global_messages: Annotated[List[dict], add_messages]
        system_prompt_messages: str
        think_message: Annotated[List[dict], add_messages]
        action_message: Annotated[List[dict], add_messages]
        observation_message: Annotated[List[dict], add_messages]

        # Team 1 input
        task_packet: Dict[str, Any]
        user_requirement: str
        team1_instruction: str
        team1_constraints: List[str]

        # Modality addresses
        text_address: List[str]
        audio_address: List[str]
        photo_address: List[str]
        video_address: List[str]

        # Per-expert generated tasks
        generated_text_task: str
        generated_audio_task: str
        generated_photo_task: str
        generated_video_task: str
        expert_tasks_generated: bool

        # Expert results
        text_result: Dict[str, Any]
        audio_result: List[Dict[str, Any]]
        photo_result: Dict[str, Any]
        video_result: List[Dict[str, Any]]
        experts_complete: bool

        # Verifier output
        json_scene_result: List[Dict[str, Any]]
        verification_result: Dict[str, Any]
        verification_passed: bool
        verifier_invoked: bool

        # Control flow
        state: str
        tools: str
        current_iteration: int
        complete: bool
        final_answer: str

        # Reflection
        reflection_result: Dict[str, Any]
        reflection_count: int
        max_reflect_retries: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, llm=None):
        self.profile = SCENE_UNDERSTANDING_SUPERVISOR_PROFILE

        # Load configuration
        config_path = os.path.join(TEAM2_DIR, "config_scene_understanding.json")
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        # Initialize LLM
        if llm is not None:
            self.model = llm
        else:
            model_cfg = self.config.get("model", {})
            api_key = os.environ.get(model_cfg.get("api_key_env", "MCP_API_KEY"), "")
            self.model = ChatOpenAI(
                model=model_cfg.get("name", "qwen3-max"),
                openai_api_base=model_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                openai_api_key=api_key,
                temperature=model_cfg.get("temperature", 0.7),
                max_tokens=model_cfg.get("max_tokens", 8192),
            )

        self.max_reflect_retries = self.config.get("supervisor", {}).get("max_reflect_retries", 2)

        # Initialize expert agents
        self.agents_available = False
        if EXPERTS_IMPORT_SUCCESS:
            try:
                self.text_agent = TextAgent(llm=self.model)
                self.audio_agent = AudioAgent(llm=self.model)
                self.photo_agent = PhotoAgent(llm=self.model)
                self.video_agent = VideoAgent(llm=self.model)
                self.agents_available = True
                _log("[Supervisor] Expert agents initialized", indent=1)
            except Exception as e:
                _log(f"[Supervisor] Failed to initialize expert agents: {e}", indent=1)

        # Initialize verifier
        self.verifier = None
        if VERIFIER_IMPORT_SUCCESS and SceneVerifier is not None:
            try:
                self.verifier = SceneVerifier(llm=self.model)
                _log("[Supervisor] SceneVerifier initialized", indent=1)
            except Exception as e:
                _log(f"[Supervisor] Failed to initialize SceneVerifier: {e}", indent=1)

        # Reflection memory
        if REFLECTION_IMPORT_SUCCESS and get_reflection_memory:
            self.reflection_memory = get_reflection_memory("scene_supervisor", CURRENT_DIR)
        else:
            self.reflection_memory = None

        # Build LangGraph
        self._build_graph()

    def _build_graph(self):
        builder = StateGraph(SceneUnderstandingSupervisor.Graph)

        # Nodes
        builder.add_node("init", self._init_node)
        builder.add_node("think", self._think_node)
        builder.add_node("act", self._act_node)
        builder.add_node("generate_expert_tasks", self._generate_expert_tasks_node)
        builder.add_node("dispatch_experts", self._dispatch_experts_node)
        builder.add_node("invoke_verifier", self._invoke_verifier_node)
        builder.add_node("observation", self._observation_node)
        builder.add_node("reflect", self._reflect_node)
        builder.add_node("final", self._final_node)

        # Edges
        builder.add_edge(START, "init")
        builder.add_edge("init", "think")

        builder.add_conditional_edges("think", self._route_after_think,
                                      {"act": "act", "final": "final"})

        builder.add_conditional_edges("act", self._route_after_act, {
            "generate_expert_tasks": "generate_expert_tasks",
            "dispatch_experts": "dispatch_experts",
            "invoke_verifier": "invoke_verifier",
            "observation": "observation",
        })

        for tool_node in ["generate_expert_tasks", "dispatch_experts", "invoke_verifier"]:
            builder.add_edge(tool_node, "observation")

        builder.add_conditional_edges("observation", self._route_after_observation,
                                      {"think": "think", "reflect": "reflect", "final": "final"})

        builder.add_conditional_edges("reflect", self._route_after_reflect,
                                      {"think": "think", "final": "final"})

        builder.add_edge("final", END)

        self.graph = builder.compile()

    # ------------------------------------------------------------------
    # Routing functions
    # ------------------------------------------------------------------

    def _route_after_think(self, state: Graph) -> str:
        return "final" if state.get("state") == "final" else "act"

    def _route_after_act(self, state: Graph) -> str:
        tool = state.get("tools", "none")
        mapping = {
            "generate_expert_tasks": "generate_expert_tasks",
            "dispatch_parallel_experts": "dispatch_experts",
            "invoke_verifier": "invoke_verifier",
        }
        return mapping.get(tool, "observation")

    def _route_after_observation(self, state: Graph) -> str:
        s = state.get("state", "think")
        if s in ("reflect", "final"):
            return s
        return "think"

    def _route_after_reflect(self, state: Graph) -> str:
        return "final" if state.get("state") == "final" else "think"

    # ------------------------------------------------------------------
    # Init node
    # ------------------------------------------------------------------

    def _init_node(self, state: Graph) -> dict:
        """Parse task packet and initialize supervisor state."""
        _log("[Supervisor] Initializing", indent=1)

        packet = state.get("task_packet", {})

        # Parse modality addresses from packet
        addrs = packet.get("modality_addresses", {})
        text_addr = addrs.get("text_address", state.get("text_address", []))
        audio_addr = addrs.get("audio_address", state.get("audio_address", []))
        photo_addr = addrs.get("photo_address", state.get("photo_address", []))
        video_addr = addrs.get("video_address", state.get("video_address", []))

        user_req = packet.get("user_requirement", state.get("user_requirement", ""))
        instruction = packet.get("instruction", state.get("team1_instruction", user_req))
        constraints = packet.get("constraints", state.get("team1_constraints", []))

        modalities = []
        if text_addr:
            modalities.append("text")
        if audio_addr:
            modalities.append("audio")
        if photo_addr:
            modalities.append("photo")
        if video_addr:
            modalities.append("video")

        _log(f"[Supervisor] Available modalities: {', '.join(modalities) if modalities else 'none'}", indent=2)
        _log(f"[Supervisor] User requirement: {user_req[:100]}...", indent=2)

        sys_prompt = (
            "You are the Scene Understanding Supervisor for Team 2.\n"
            "You coordinate multimodal scene understanding through expert agents and a verifier.\n"
            f"Available modalities: {', '.join(modalities)}\n"
            f"User requirement: {user_req}\n"
        )

        return {
            "system_prompt_messages": sys_prompt,
            "user_requirement": user_req,
            "team1_instruction": instruction,
            "team1_constraints": constraints,
            "text_address": text_addr,
            "audio_address": audio_addr,
            "photo_address": photo_addr,
            "video_address": video_addr,
            "state": "think",
            "current_iteration": 0,
        }

    # ------------------------------------------------------------------
    # Think / Act / Observation nodes
    # ------------------------------------------------------------------

    def _think_node(self, state: Graph) -> dict:
        """
        Think: decide which tool to execute next.
        Sequential order: generate_tasks → dispatch_experts → invoke_verifier → final
        """
        iteration = state.get("current_iteration", 0) + 1
        _log(f"[Supervisor] Think (iteration {iteration})", indent=1)

        if not state.get("expert_tasks_generated"):
            _log("[Supervisor] → Generate expert tasks (Algorithm 1)", indent=2)
            return {"state": "act", "tools": "generate_expert_tasks", "current_iteration": iteration}

        if not state.get("experts_complete"):
            _log("[Supervisor] → Dispatch parallel experts", indent=2)
            return {"state": "act", "tools": "dispatch_parallel_experts", "current_iteration": iteration}

        if not state.get("verifier_invoked"):
            _log("[Supervisor] → Invoke verifier", indent=2)
            return {"state": "act", "tools": "invoke_verifier", "current_iteration": iteration}

        _log("[Supervisor] → All steps complete, proceeding to final", indent=2)
        return {"state": "final", "tools": "none", "current_iteration": iteration}

    def _act_node(self, state: Graph) -> dict:
        """Act: dispatch to selected tool (routing handles actual dispatch)."""
        tool = state.get("tools", "none")
        _log(f"[Supervisor] Act → dispatching: {tool}", indent=2)
        return {"state": "execute"}

    def _observation_node(self, state: Graph) -> dict:
        """Observe: check results and decide next step."""
        _log("[Supervisor] Observation", indent=1)

        # If verifier was invoked and failed, consider reflection
        if state.get("verifier_invoked") and not state.get("verification_passed"):
            issues = state.get("verification_result", {}).get("issues", [])
            if issues and state.get("reflection_count", 0) < state.get("max_reflect_retries", 2):
                _log(f"[Supervisor] Verification failed → reflect", indent=2)
                return {"state": "reflect"}
            elif issues:
                _log("[Supervisor] Verification failed, max retries reached → final", indent=2)
                return {"state": "final"}

        return {"state": "think"}

    # ------------------------------------------------------------------
    # Tool: Generate Expert Tasks (Algorithm 1)
    # ------------------------------------------------------------------

    async def _generate_expert_tasks_node(self, state: Graph) -> dict:
        """
        Read expert AgentProfiles and generate per-expert tasks.
        Implements Algorithm 1: DetermineNeed → ReadProfile → Match →
        ExtractInstruction → GenerateConstraints.
        Uses AsyncTaskCreator (migrated from multi_understanding._init_node).
        """
        _log("[Supervisor] Generating expert tasks (Algorithm 1)", indent=1)

        text_address = state.get("text_address", [])
        audio_address = state.get("audio_address", [])
        photo_address = state.get("photo_address", [])
        video_address = state.get("video_address", [])

        instruction = state.get("team1_instruction", state.get("user_requirement", ""))

        updates = {
            "expert_tasks_generated": True,
            "generated_text_task": "",
            "generated_audio_task": "",
            "generated_photo_task": "",
            "generated_video_task": "",
        }

        # DetermineNeed: only include experts whose modality has input
        try:
            from Team2.AgentProfile.text_agent_profile import TEXT_AGENT_PROFILE
            from Team2.AgentProfile.audio_agent_profile import AUDIO_AGENT_PROFILE
            from Team2.AgentProfile.photo_agent_profile import PHOTO_AGENT_PROFILE
            from Team2.AgentProfile.video_agent_profile import VIDEO_AGENT_PROFILE

            profiles_to_use = []
            if text_address:
                profiles_to_use.append(("text", TEXT_AGENT_PROFILE))
            if audio_address:
                profiles_to_use.append(("audio", AUDIO_AGENT_PROFILE))
            if photo_address:
                profiles_to_use.append(("photo", PHOTO_AGENT_PROFILE))
            if video_address:
                profiles_to_use.append(("video", VIDEO_AGENT_PROFILE))

            if profiles_to_use and TASK_CREATOR_IMPORT_SUCCESS and AsyncTaskCreator is not None:
                json_scene_data = [
                    {"关键帧": "0s", "背景": "generic background", "主体": "generic subject", "心情": "neutral"}
                ]

                task_creator = AsyncTaskCreator(llm=self.model)
                profiles_list = [p for _, p in profiles_to_use]

                task_strings = await task_creator.create_tasks_for_all_agents(
                    profiles=profiles_list,
                    user_requirement=instruction,
                    json_scene_data=json_scene_data,
                    num_tasks_per_agent=1,
                    save_to_file=False,
                )

                idx = 0
                for name, _ in profiles_to_use:
                    if idx < len(task_strings):
                        updates[f"generated_{name}_task"] = task_strings[idx]
                        _log(f"[Supervisor] Generated task for {name}: {task_strings[idx][:80]}...", indent=2)
                        idx += 1

                _log(f"[Supervisor] Task generation complete ({len(profiles_to_use)} experts)", indent=2)
            elif profiles_to_use:
                # Fallback: simple default tasks
                _log("[Supervisor] AsyncTaskCreator unavailable, using default tasks", indent=2)
                if text_address:
                    updates["generated_text_task"] = f"Analyze text files {text_address} and return a JSON dict with background, background style, subject, and subject mood"
                if audio_address:
                    updates["generated_audio_task"] = f"Analyze audio files {audio_address} and return a list of time-segmented descriptions"
                if photo_address:
                    updates["generated_photo_task"] = f"Analyze image files {photo_address} and return a JSON dict with background, background style, subject, and subject mood"
                if video_address:
                    updates["generated_video_task"] = f"Analyze video files {video_address} and return a list of keyframe analysis results"

        except Exception as e:
            _log(f"[Supervisor] Task generation failed: {e}, using defaults", indent=2)
            if text_address:
                updates["generated_text_task"] = f"Analyze text files {text_address} and return a JSON dict with background, background style, subject, and subject mood"
            if audio_address:
                updates["generated_audio_task"] = f"Analyze audio files {audio_address} and return a list of time-segmented descriptions"
            if photo_address:
                updates["generated_photo_task"] = f"Analyze image files {photo_address} and return a JSON dict with background, background style, subject, and subject mood"
            if video_address:
                updates["generated_video_task"] = f"Analyze video files {video_address} and return a list of keyframe analysis results"

        return updates

    # ------------------------------------------------------------------
    # Tool: Dispatch Parallel Experts
    # ------------------------------------------------------------------

    async def _dispatch_experts_node(self, state: Graph) -> dict:
        """
        Execute modality experts in parallel via asyncio.gather().
        Each expert runs its own Think-Act-Observe-Reflect loop internally.
        """
        _log("[Supervisor] Dispatching parallel experts", indent=1)

        text_address = state.get("text_address", [])
        audio_address = state.get("audio_address", [])
        photo_address = state.get("photo_address", [])
        video_address = state.get("video_address", [])

        if not self.agents_available:
            _log("[Supervisor] Expert agents not available, returning empty results", indent=2)
            return {"experts_complete": True, "text_result": {}, "audio_result": [], "photo_result": {}, "video_result": []}

        tasks = []
        task_labels = []

        if text_address:
            tasks.append(self._execute_text(state))
            task_labels.append("text")
        if audio_address:
            tasks.append(self._execute_audio(state))
            task_labels.append("audio")
        if photo_address:
            tasks.append(self._execute_photo(state))
            task_labels.append("photo")
        if video_address:
            tasks.append(self._execute_video(state))
            task_labels.append("video")

        _log(f"[Supervisor] Launching {len(tasks)} expert(s) in parallel: {', '.join(task_labels)}", indent=2)

        updates = {
            "experts_complete": True,
            "text_result": {},
            "audio_result": [],
            "photo_result": {},
            "video_result": [],
        }

        if not tasks:
            _log("[Supervisor] No modalities to process", indent=2)
            return updates

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300.0,
            )

            for label, result in zip(task_labels, results):
                if isinstance(result, Exception):
                    _log(f"[Supervisor] Expert '{label}' failed: {result}", indent=2)
                    updates[f"{label}_result"] = {} if label in ("text", "photo") else []
                else:
                    updates[f"{label}_result"] = result
                    result_desc = f"{len(result)} items" if isinstance(result, list) else ("non-empty" if result else "empty")
                    _log(f"[Supervisor] Expert '{label}' complete: {result_desc}", indent=2)

        except asyncio.TimeoutError:
            _log("[Supervisor] Expert execution timed out (>300s)", indent=2)
        except Exception as e:
            _log(f"[Supervisor] Expert execution failed: {e}", indent=2)

        return updates

    async def _execute_text(self, state: Graph) -> Dict:
        """Execute TextAgent for all text files."""
        task_str = state.get("generated_text_task", "")
        text_address = state.get("text_address", [])
        all_results = {}
        for path in text_address:
            try:
                response = await self.text_agent.ainvoke(user_input=task_str, text_path=path)
                result = response.get("text_result", {})
                all_results[os.path.basename(path)] = result
            except Exception as e:
                _log(f"[Supervisor] TextAgent error on {path}: {e}", indent=3)
        if len(text_address) == 1:
            return all_results.get(os.path.basename(text_address[0]), {})
        return all_results

    async def _execute_audio(self, state: Graph) -> List:
        """Execute AudioAgent for all audio files."""
        task_str = state.get("generated_audio_task", "")
        audio_address = state.get("audio_address", [])
        all_results = []
        for path in audio_address:
            try:
                response = await self.audio_agent.ainvoke(user_input=task_str, audio_path=path)
                result = response.get("audio_result", [])
                all_results.extend(result)
            except Exception as e:
                _log(f"[Supervisor] AudioAgent error on {path}: {e}", indent=3)
        return all_results

    async def _execute_photo(self, state: Graph) -> Dict:
        """Execute PhotoAgent for all image files."""
        task_str = state.get("generated_photo_task", "")
        photo_address = state.get("photo_address", [])
        all_results = {}
        for path in photo_address:
            try:
                response = await self.photo_agent.ainvoke(user_input=task_str, photo_path=path)
                result = response.get("photo_result", {})
                all_results[os.path.basename(path)] = result
            except Exception as e:
                _log(f"[Supervisor] PhotoAgent error on {path}: {e}", indent=3)
        if len(photo_address) == 1:
            return all_results.get(os.path.basename(photo_address[0]), {})
        return all_results

    async def _execute_video(self, state: Graph) -> List:
        """Execute VideoAgent for all video files."""
        task_str = state.get("generated_video_task", "")
        video_address = state.get("video_address", [])
        all_results = []
        for path in video_address:
            try:
                response = await self.video_agent.ainvoke(user_input=task_str, video_path=path)
                result = response.get("video_result", [])
                all_results.extend(result)
            except Exception as e:
                _log(f"[Supervisor] VideoAgent error on {path}: {e}", indent=3)
        return all_results

    # ------------------------------------------------------------------
    # Tool: Invoke Verifier
    # ------------------------------------------------------------------

    async def _invoke_verifier_node(self, state: Graph) -> dict:
        """
        Forward expert results + context to SceneVerifier.
        The Verifier is independent — it receives all inputs directly.
        """
        _log("[Supervisor] Invoking SceneVerifier", indent=1)

        if self.verifier is None:
            _log("[Supervisor] SceneVerifier not available, skipping verification", indent=2)
            return {
                "verifier_invoked": True,
                "verification_passed": True,
                "json_scene_result": [],
                "verification_result": {"passed": True, "issues": [], "warnings": ["Verifier not available"]},
            }

        expert_results = {
            "text_result": state.get("text_result", {}),
            "audio_result": state.get("audio_result", []),
            "photo_result": state.get("photo_result", {}),
            "video_result": state.get("video_result", []),
        }

        try:
            result = await self.verifier.ainvoke(
                expert_results=expert_results,
                user_requirement=state.get("user_requirement", ""),
                team1_instruction=state.get("team1_instruction", ""),
                team1_constraints=state.get("team1_constraints", []),
            )

            verification = result.get("verification", {})
            passed = verification.get("passed", False)

            _log(f"[Supervisor] Verifier result: {'PASSED' if passed else 'FAILED'}", indent=2)
            if not passed:
                for issue in verification.get("issues", []):
                    _log(f"[Supervisor]   Issue: {issue}", indent=3)

            return {
                "json_scene_result": result.get("json_scene_result", []),
                "verification_result": verification,
                "verification_passed": passed,
                "verifier_invoked": True,
            }

        except Exception as e:
            _log(f"[Supervisor] Verifier execution failed: {e}", indent=2)
            return {
                "verifier_invoked": True,
                "verification_passed": False,
                "json_scene_result": [],
                "verification_result": {"passed": False, "issues": [f"Verifier exception: {e}"], "warnings": []},
            }

    # ------------------------------------------------------------------
    # Reflect node
    # ------------------------------------------------------------------

    async def _reflect_node(self, state: Graph) -> dict:
        """
        Reflect: analyze verification failure, decide whether to retry.
        Based on the Reflexion framework.
        """
        _log("[Supervisor] Reflect", indent=1)

        verification = state.get("verification_result", {})
        issues = verification.get("issues", [])
        warnings_list = verification.get("warnings", [])
        reflection_count = state.get("reflection_count", 0) + 1

        observation = (
            f"Verification failed with {len(issues)} issues and {len(warnings_list)} warnings.\n"
            f"Issues: {json.dumps(issues, ensure_ascii=False)}\n"
            f"Warnings: {json.dumps(warnings_list, ensure_ascii=False)}"
        )

        # Get historical reflections
        history_reflections = ""
        if self.reflection_memory:
            history_reflections = self.reflection_memory.get_summary()

        # Build reflection prompt
        if REFLECTION_IMPORT_SUCCESS and build_reflection_prompt:
            reflect_prompt = build_reflection_prompt(
                task_type="scene_supervision",
                task_description="Multimodal scene understanding coordination and verification",
                observation=observation,
                history_reflections=history_reflections,
            )
        else:
            reflect_prompt = f"""You are a self-reflection expert. Analyze the verification results:

Task: Scene understanding supervision
Observation: {observation}
History: {history_reflections or "None"}

Return JSON:
{{
    "analysis": "Root cause analysis",
    "strengths": ["..."],
    "weaknesses": ["..."],
    "improvement": "What to do differently",
    "quality": "high/medium/low",
    "should_retry": true/false
}}"""

        try:
            messages = [SystemMessage(content=reflect_prompt)]
            response = await self.model.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            if REFLECTION_IMPORT_SUCCESS and parse_reflection_result:
                reflection_result = parse_reflection_result(response_text)
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        reflection_result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        reflection_result = {"quality": "low", "should_retry": False}
                else:
                    reflection_result = {"quality": "low", "should_retry": False}

        except Exception as e:
            _log(f"[Supervisor] Reflection failed: {e}", indent=2)
            reflection_result = {"quality": "low", "should_retry": False}

        # Save to memory
        if self.reflection_memory:
            self.reflection_memory.add_reflection(
                task_description="Scene supervision",
                observation=observation[:200],
                reflection=reflection_result.get("analysis", ""),
                improvement=reflection_result.get("improvement", ""),
                quality=reflection_result.get("quality", "low"),
                iterations=reflection_count,
            )

        should_retry = reflection_result.get("should_retry", False)
        quality = reflection_result.get("quality", "low")

        _log(
            f"[Supervisor] Reflection #{reflection_count}: quality={quality}, retry={should_retry}",
            indent=2)

        if should_retry and reflection_count < self.max_reflect_retries:
            # Reset verifier flag to re-invoke
            return {
                "reflection_result": reflection_result,
                "reflection_count": reflection_count,
                "verifier_invoked": False,
                "state": "think",
            }

        # Accept current result
        return {
            "reflection_result": reflection_result,
            "reflection_count": reflection_count,
            "state": "final",
        }

    # ------------------------------------------------------------------
    # Final node
    # ------------------------------------------------------------------

    def _final_node(self, state: Graph) -> dict:
        """Output final results."""
        _log("[Supervisor] Final node", indent=1)

        final_result = {
            "text_result": state.get("text_result", {}),
            "audio_result": state.get("audio_result", []),
            "photo_result": state.get("photo_result", {}),
            "video_result": state.get("video_result", []),
            "json_scene_result": state.get("json_scene_result", []),
        }

        final_answer = json.dumps(final_result, ensure_ascii=False, separators=(',', ':'))

        return {
            "final_answer": final_answer,
            "json_scene_result": state.get("json_scene_result", []),
            "complete": True,
        }

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def ainvoke_from_packet(self, task_packet: Dict[str, Any]) -> dict:
        """
        Primary entry: receive Team 1 task_packet.

        Args:
            task_packet: {
                "team_name": "Team2",
                "instruction": str,
                "constraints": List[str],
                "modalities": List[str],
                "modality_addresses": {
                    "text_address": List[str],
                    "audio_address": List[str],
                    "photo_address": List[str],
                    "video_address": List[str],
                },
                "user_requirement": str,
            }
        """
        addrs = task_packet.get("modality_addresses", {})

        initial_state = {
            "task_packet": task_packet,
            "user_requirement": task_packet.get("user_requirement", ""),
            "team1_instruction": task_packet.get("instruction", ""),
            "team1_constraints": task_packet.get("constraints", []),
            "text_address": addrs.get("text_address", []),
            "audio_address": addrs.get("audio_address", []),
            "photo_address": addrs.get("photo_address", []),
            "video_address": addrs.get("video_address", []),
            # Expert tasks
            "generated_text_task": "",
            "generated_audio_task": "",
            "generated_photo_task": "",
            "generated_video_task": "",
            "expert_tasks_generated": False,
            # Expert results
            "text_result": {},
            "audio_result": [],
            "photo_result": {},
            "video_result": [],
            "experts_complete": False,
            # Verifier
            "json_scene_result": [],
            "verification_result": {},
            "verification_passed": False,
            "verifier_invoked": False,
            # Control
            "state": "start",
            "tools": "none",
            "current_iteration": 0,
            "complete": False,
            "final_answer": "",
            # Reflection
            "reflection_result": {},
            "reflection_count": 0,
            "max_reflect_retries": self.max_reflect_retries,
            # Messages
            "global_messages": [],
            "system_prompt_messages": "",
            "think_message": [],
            "action_message": [],
            "observation_message": [],
        }

        _log("=" * 60, indent=0)
        _log("[SceneUnderstandingSupervisor] Starting pipeline", indent=0)
        _log("=" * 60, indent=0)

        response = await self.graph.ainvoke(initial_state, config={"recursion_limit": 50})

        return {
            "final_answer": response.get("final_answer", ""),
            "json_scene_result": response.get("json_scene_result", []),
            "verification_result": response.get("verification_result", {}),
            "complete": response.get("complete", False),
            # Individual expert results
            "text_result": response.get("text_result", {}),
            "audio_result": response.get("audio_result", []),
            "photo_result": response.get("photo_result", {}),
            "video_result": response.get("video_result", []),
        }

    async def ainvoke(
        self,
        user_input: str,
        text_address: Optional[List[str]] = None,
        audio_address: Optional[List[str]] = None,
        photo_address: Optional[List[str]] = None,
        video_address: Optional[List[str]] = None,
    ) -> dict:
        """
        Backward-compatible entry point (same signature as MultiUnderstandingAgent).

        Constructs a task_packet internally and delegates to ainvoke_from_packet.
        """
        if not any([text_address, audio_address, photo_address, video_address]):
            final_result = {
                "text_result": {},
                "audio_result": [],
                "photo_result": {},
                "video_result": [],
                "json_scene_result": [],
                "error": "No input addresses provided for any modality.",
            }
            return {
                "final_answer": json.dumps(final_result, ensure_ascii=False, separators=(',', ':')),
                "json_scene_result": [],
                "complete": True,
            }

        packet = {
            "team_name": "Team2",
            "instruction": user_input,
            "constraints": [],
            "modalities": [],
            "modality_addresses": {
                "text_address": text_address or [],
                "audio_address": audio_address or [],
                "photo_address": photo_address or [],
                "video_address": video_address or [],
            },
            "user_requirement": user_input,
        }
        return await self.ainvoke_from_packet(packet)

    def invoke(
        self,
        user_input: str,
        text_address: Optional[List[str]] = None,
        audio_address: Optional[List[str]] = None,
        photo_address: Optional[List[str]] = None,
        video_address: Optional[List[str]] = None,
    ) -> dict:
        """Synchronous wrapper for ainvoke."""
        return asyncio.run(self.ainvoke(
            user_input=user_input,
            text_address=text_address,
            audio_address=audio_address,
            photo_address=photo_address,
            video_address=video_address,
        ))


# ──────────────────────────────────────────────────────────────────────────────
# Standalone execution for testing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def _header(title: str, char: str = "=", width: int = 70):
        print(f"\n{char * width}")
        print(f"  {title}")
        print(f"{char * width}")

    def _print_expert_result(name: str, result):
        """Pretty-print one expert's output."""
        _header(f"Expert Result: {name}", char="-", width=60)
        if not result:
            print("  (empty — no result returned)")
            return
        if isinstance(result, list):
            print(f"  Scenes: {len(result)}")
            for i, scene in enumerate(result):
                if isinstance(scene, str):
                    try:
                        scene = json.loads(scene)
                    except (json.JSONDecodeError, ValueError):
                        print(f"    [{i}] {scene[:200]}")
                        continue
                if isinstance(scene, dict):
                    print(f"    [{i}]")
                    for k, v in scene.items():
                        if not str(k).startswith("_"):
                            print(f"      {k}: {v}")
                else:
                    print(f"    [{i}] {str(scene)[:200]}")
        elif isinstance(result, dict):
            for k, v in result.items():
                if not str(k).startswith("_"):
                    print(f"    {k}: {v}")
        else:
            print(f"  {str(result)[:500]}")

    def _print_fused_scenes(scenes: list):
        """Pretty-print final fused 9-field scenes."""
        _header("Final Fused Scene Result (Verifier Output)", char="=", width=70)
        NINE_FIELDS = ["关键帧", "主体", "主体心情", "主体声音内容", "主体声音风格",
                       "背景", "背景风格", "背景声音内容", "背景声音风格"]
        print(f"  Total scenes: {len(scenes)}")
        for i, scene in enumerate(scenes):
            print(f"\n  Scene {i}:")
            for field in NINE_FIELDS:
                val = scene.get(field, "")
                print(f"    {field}: {val}")


    async def _end_to_end_test():
        """
        Definitive end-to-end test: scene_17 with all 4 modalities.
        Prints individual expert results + final fused output.
        """
        _header("Team 2 End-to-End Test — scene_17 (4 modalities)", char="=", width=70)

        # --- Locate input files ---
        input_dir = os.path.join(PROJECT_ROOT, "Input", "scene_17")
        text_path = os.path.join(input_dir, "scene_17.txt")
        audio_path = os.path.join(input_dir, "scene_17.mp3")
        photo_path = os.path.join(input_dir, "scene_17.png")
        video_path = os.path.join(input_dir, "scene_17.mp4")

        print(f"\n  Input directory: {input_dir}")
        modalities_found = []
        for label, path in [("Text", text_path), ("Audio", audio_path),
                            ("Photo", photo_path), ("Video", video_path)]:
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            status = f"{size:,} bytes" if exists else "NOT FOUND"
            print(f"    {label}: {os.path.basename(path)} — {status}")
            if exists:
                modalities_found.append(label)

        if not modalities_found:
            print("\n  ERROR: No input files found. Aborting.")
            return

        print(f"\n  Modalities available: {', '.join(modalities_found)}")

        # --- Build task packet ---
        packet = {
            "team_name": "Team2",
            "instruction": (
                "Analyze all available modalities (text, audio, image, video) "
                "and produce unified scene descriptions for background music generation."
            ),
            "constraints": [
                "Maintain temporal consistency across scenes",
                "Ensure mood-voice alignment",
                "Preserve the original atmosphere of the input content",
            ],
            "modalities": [m.lower() for m in modalities_found],
            "modality_addresses": {
                "text_address": [text_path] if os.path.exists(text_path) else [],
                "audio_address": [audio_path] if os.path.exists(audio_path) else [],
                "photo_address": [photo_path] if os.path.exists(photo_path) else [],
                "video_address": [video_path] if os.path.exists(video_path) else [],
            },
            "user_requirement": "Generate background music for a beauty/makeup tutorial video",
        }

        # --- Run full pipeline ---
        _header("Running Supervisor Pipeline", char="-", width=60)
        supervisor = SceneUnderstandingSupervisor()
        result = await supervisor.ainvoke_from_packet(packet)

        # --- Print individual expert results ---
        _header("Individual Expert Understanding Results", char="=", width=70)

        _print_expert_result("Text (TextAgent)", result.get("text_result"))
        _print_expert_result("Audio (AudioAgent)", result.get("audio_result"))
        _print_expert_result("Photo (PhotoAgent)", result.get("photo_result"))
        _print_expert_result("Video (VideoAgent)", result.get("video_result"))

        # --- Print final fused result ---
        fused = result.get("json_scene_result", [])
        _print_fused_scenes(fused)

        # --- Print verification ---
        verification = result.get("verification_result", {})
        _header("Verification Result", char="-", width=60)
        passed = verification.get("passed", False)
        issues = verification.get("issues", [])
        warnings = verification.get("warnings", [])
        print(f"  Passed: {passed}")
        if issues:
            print(f"  Issues ({len(issues)}):")
            for issue in issues:
                print(f"    - {issue}")
        if warnings:
            print(f"  Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"    - {w}")

        # --- Summary ---
        _header("Summary", char="=", width=70)
        print(f"  Pipeline complete: {result.get('complete', False)}")
        print(f"  Expert results: text={'present' if result.get('text_result') else 'empty'}, "
              f"audio={'present' if result.get('audio_result') else 'empty'}, "
              f"photo={'present' if result.get('photo_result') else 'empty'}, "
              f"video={'present' if result.get('video_result') else 'empty'}")
        print(f"  Fused scenes: {len(fused)}")
        print(f"  Verification: {'PASSED' if passed else 'FAILED'}")
        print(f"{'=' * 70}\n")

    asyncio.run(_end_to_end_test())
