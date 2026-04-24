"""
Task Generator Module

基于子智能体的角色、工具、知识生成任务列表
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 添加项目根目录到路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from .task_profile import (
    AgentProfile,
    AgentRole,
    AgentTool,
    AgentKnowledge,
    KnowledgeDomain,
)
from promptStrategy.JSONSchema import JSONSchema

logger = logging.getLogger(__name__)


class SceneAnalysis(BaseModel):
    """json场景分析数据结构"""
    emotional_tone: List[str] = Field(
        default_factory=list,
        description="场景情感标签，从json_scene的'主体心情'字段提取"
    )
    key_subjects: List[str] = Field(
        default_factory=list,
        description="场景主体列表，从'主体'字段提取"
    )
    key_moments: List[Dict[str, str]] = Field(
        default_factory=list,
        description="关键时刻，包含'关键帧'和'主体心情'"
    )
    background_info: str = Field(
        default="",
        description="场景背景信息，从第一帧的'背景'字段提取"
    )
    music_style_recommendation: List[str] = Field(
        default_factory=list,
        description="推荐音乐风格，基于AgentProfile.knowledge生成"
    )
    language_recommendation: str = Field(
        default="Auto",
        description="推荐歌词语言（中文/英文/Auto）"
    )


class ToolUsageGuide(BaseModel):
    """工具使用指南"""
    tool_name: str = Field(..., description="工具名称")
    usage_purpose: str = Field(..., description="使用目的，来自tool.description")
    dependencies: List[str] = Field(
        default_factory=list,
        description="依赖的其他工具列表"
    )
    execution_order: int = Field(
        default=1,
        description="执行顺序（1表示第一个执行）"
    )
    parameters_guide: Dict[str, str] = Field(
        default_factory=dict,
        description="参数使用指南，key为参数名，value为使用说明"
    )


class Precaution(BaseModel):
    """注意事项/验证规则"""
    category: str = Field(..., description="类别，如'格式校验'、'重试机制'")
    rule_description: str = Field(..., description="规则描述，来自knowledge.rules或constraints")
    validation_method: str = Field(..., description="校验方法说明")
    error_handling: str = Field(..., description="错误处理和重试策略")


class Task(BaseModel):
    """
    增强版任务定义

    包含场景分析、最佳实践、工具使用说明、注意事项
    """
    # 基础信息
    task_id: str = Field(..., description="任务唯一标识")
    description: str = Field(..., description="任务基本描述")

    # json场景分析
    scene_analysis: Optional[SceneAnalysis] = Field(
        None,
        description="从json_scene提取的场景分析数据"
    )

    # 最佳实践（3-5条，来自AgentProfile.best_practices）
    best_practices: List[str] = Field(
        default_factory=list,
        description="任务执行的最佳实践，来自AgentProfile.best_practices"
    )

    # 工具使用说明（3-5条，包含依赖关系）
    tool_usage_guides: List[ToolUsageGuide] = Field(
        default_factory=list,
        description="工具使用指南，基于AgentProfile.tools生成"
    )

    # 注意事项（3-5条，包含格式校验和重试机制）
    precautions: List[Precaution] = Field(
        default_factory=list,
        description="注意事项，基于AgentProfile.knowledge.rules和constraints生成"
    )

    # 兼容性字段
    required_tools: List[str] = Field(default_factory=list)
    required_knowledge: List[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)


class TaskList(BaseModel):
    """
    任务列表

    Attributes:
        agent_id: 关联的智能体ID
        tasks: 任务列表
        metadata: 元数据
    """
    agent_id: str = Field(..., description="关联的智能体ID")
    tasks: List[Task] = Field(default_factory=list, description="任务列表")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="任务列表的元数据"
    )

    def add_task(self, task: Task):
        """添加任务到列表"""
        self.tasks.append(task)

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_tasks_by_tool(self, tool_name: str) -> List[Task]:
        """获取使用指定工具的所有任务"""
        return [t for t in self.tasks if tool_name in t.required_tools]

    def sort_by_priority(self):
        """按优先级排序任务"""
        self.tasks.sort(key=lambda x: x.priority, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()


class TaskGenerator:
    """
    任务生成器

    基于智能体的角色、工具和知识生成任务列表
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None, temperature: float = 0.7):
        """
        初始化任务生成器

        Args:
            llm: 语言模型实例
            temperature: 温度参数
        """
        if llm is None:
            from dotenv import load_dotenv
            load_dotenv()
            self.llm = ChatOpenAI(
                model="qwen3-max",
                api_key=os.getenv("MCP_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=temperature,
            )
        else:
            self.llm = llm

    def _generate_best_practices(
        self,
        profile: AgentProfile,
        num_practices: int = 3,
    ) -> List[str]:
        """
        从AgentProfile.best_practices选择最佳实践

        严格规则：
        - 只能从profile.best_practices中选择
        - 不允许凭空创建
        - 最多选择num_practices条（不超过5条）
        """
        if not profile.best_practices:
            return []

        limit = min(num_practices, 5)
        return profile.best_practices[:limit]

    def _generate_tool_usage_guides(
        self,
        profile: AgentProfile,
        required_tools: List[str],
    ) -> List[ToolUsageGuide]:
        """
        基于AgentProfile.tools生成工具使用指南

        包含依赖关系分析：
        - 依赖字段优先使用 AgentTool.dependencies
        - 若 dependencies 为空，则根据参数名做启发式依赖推断
        """
        guides: List[ToolUsageGuide] = []
        tool_map = {t.name: t for t in profile.tools}

        for i, tool_name in enumerate(required_tools, 1):
            tool = tool_map.get(tool_name)
            if not tool:
                continue

            dependencies: List[str] = []

            # 显式依赖
            if getattr(tool, "dependencies", None):
                dependencies.extend(tool.dependencies)

            # 基于参数名的启发式依赖（完全由 AgentProfile.tools 中已有参数名驱动）
            for param in tool.parameters:
                param_name = param.get("name", "")
                if param_name in ["json_scene", "gt_lyric", "audio_type", "descriptions"]:
                    if param_name not in dependencies:
                        dependencies.append(param_name)

            guide = ToolUsageGuide(
                tool_name=tool.name,
                usage_purpose=tool.description,
                dependencies=dependencies,
                execution_order=i,
                parameters_guide={
                    p.get("name", ""): p.get("description", "")
                    for p in tool.parameters
                    if p.get("name")
                },
            )
            guides.append(guide)

        return guides[:5]

    def _generate_precautions(
        self,
        profile: AgentProfile,
        num_precautions: int = 3,
    ) -> List[Precaution]:
        """
        基于AgentProfile.knowledge.rules和constraints生成注意事项

        包含输出格式校验和重试机制
        """
        precautions: List[Precaution] = []

        # 从 knowledge.rules 提取格式校验规则
        for knowledge in profile.knowledge:
            for rule in knowledge.rules[:2]:
                precaution = Precaution(
                    category="格式校验",
                    rule_description=rule,
                    validation_method=f"检查输出是否符合：{rule}",
                    error_handling=f"如果不符合“{rule}”，重新生成并确保符合此规则",
                )
                precautions.append(precaution)
                if len(precautions) >= num_precautions:
                    break
            if len(precautions) >= num_precautions:
                break

        # 从 constraints 提取约束条件
        if len(precautions) < num_precautions:
            for constraint in profile.constraints:
                precaution = Precaution(
                    category="约束条件",
                    rule_description=constraint,
                    validation_method=f"验证是否满足：{constraint}",
                    error_handling="违反约束时，调整任务参数或生成逻辑并重试",
                )
                precautions.append(precaution)
                if len(precautions) >= num_precautions:
                    break

        return precautions[:5]

    def _build_system_prompt(self, profile: AgentProfile) -> str:
        """
        构建系统提示词

        Args:
            profile: 智能体配置文件

        Returns:
            系统提示词字符串
        """
        prompt = f"""你是一个专业的任务规划专家，负责为子智能体生成合适的任务列表。

## 智能体信息

**智能体ID**: {profile.agent_id}
**角色名称**: {profile.role.name}
**角色描述**: {profile.role.description}

**职责**:
"""
        for i, resp in enumerate(profile.role.responsibilities, 1):
            prompt += f"{i}. {resp}\n"

        prompt += f"\n**专业领域**:\n"
        for i, exp in enumerate(profile.role.expertise, 1):
            prompt += f"{i}. {exp}\n"

        prompt += f"\n## 可用工具\n"
        for tool in profile.tools:
            prompt += f"\n### {tool.name}\n"
            prompt += f"- **功能**: {tool.description}\n"
            prompt += f"- **函数签名**: {tool.function_signature}\n"
            if tool.parameters:
                prompt += f"- **参数**:\n"
                for param in tool.parameters:
                    prompt += f"  - {param.get('name')} ({param.get('type')}): {param.get('description')}\n"
            prompt += f"- **返回值**: {tool.returns}\n"
            if tool.usage_example:
                prompt += f"- **使用示例**: {tool.usage_example}\n"

        prompt += f"\n## 知识库\n"
        for knowledge in profile.knowledge:
            prompt += f"\n### {knowledge.domain.value}\n"
            if knowledge.concepts:
                prompt += f"**核心概念**:\n"
                for concept in knowledge.concepts:
                    prompt += f"- {concept}\n"
            if knowledge.rules:
                prompt += f"**规则约束**:\n"
                for rule in knowledge.rules:
                    prompt += f"- {rule}\n"

        prompt += f"\n## 约束条件\n"
        for i, constraint in enumerate(profile.constraints, 1):
            prompt += f"{i}. {constraint}\n"

        prompt += f"\n## 最佳实践\n"
        for i, practice in enumerate(profile.best_practices, 1):
            prompt += f"{i}. {practice}\n"

        return prompt

    def _build_task_generation_prompt(
        self,
        profile: AgentProfile,
        user_requirement: str,
        context: Optional[Dict[str, Any]] = None,
        json_scene_data: Optional[List[Dict[str, Any]]] = None,
        num_tasks: int = 3,
    ) -> str:
        """
        构建任务生成提示词

        Args:
            profile: 智能体配置文件
            user_requirement: 用户需求描述
            context: 额外上下文信息
            num_tasks: 需要生成的任务数量

        Returns:
            完整的提示词
        """
        system_prompt = self._build_system_prompt(profile)

        user_prompt = f"""
## 用户需求

{user_requirement}
"""

        if context:
            user_prompt += f"\n## 上下文信息\n"
            for key, value in context.items():
                user_prompt += f"- **{key}**: {value}\n"

        # 场景分析部分：仅当提供 json_scene_data 时添加
        if json_scene_data:
            from task.scene_analyzer import SceneAnalyzer

            scene_analysis = SceneAnalyzer.extract_scene_analysis(
                json_scene=json_scene_data,
                profile=profile,
            )
            if scene_analysis:
                user_prompt += (
                    f"\n## 场景分析\n"
                    f"{json.dumps(scene_analysis.model_dump(), ensure_ascii=False)}\n"
                )

        # 基于 AgentProfile 构造模板数据（完全来源于 profile，不允许 LLM 自创）
        best_practices = self._generate_best_practices(profile, 3)
        if best_practices:
            user_prompt += f"\n## 最佳实践模板（必须从以下内容中选择，不允许凭空创建）\n"
            for i, practice in enumerate(best_practices, 1):
                user_prompt += f"{i}. {practice}\n"

        required_tools = [t.name for t in profile.tools]
        tool_guides = self._generate_tool_usage_guides(profile, required_tools)
        if tool_guides:
            user_prompt += f"\n## 工具使用指南模板（基于 AgentProfile.tools，不允许添加未列出的工具）\n"
            for guide in tool_guides:
                user_prompt += f"- {guide.tool_name}: {guide.usage_purpose}\n"
                if guide.dependencies:
                    user_prompt += f"  依赖: {', '.join(guide.dependencies)}\n"

        precautions = self._generate_precautions(profile, 3)
        if precautions:
            user_prompt += (
                f"\n## 注意事项模板（包含格式校验和重试机制，内容必须来自 knowledge.rules 或 constraints）\n"
            )
            for precaution in precautions:
                user_prompt += f"- [{precaution.category}] {precaution.rule_description}\n"
                user_prompt += f"  校验方法: {precaution.validation_method}\n"
                user_prompt += f"  错误处理: {precaution.error_handling}\n"

        user_prompt += f"""
## 任务要求

请根据上述智能体的角色、工具和知识，为该智能体生成 {num_tasks} 个具体的任务。

**任务生成要求**:
1. 任务必须贴近智能体的角色定位和职责
2. 任务必须严格依据智能体的可用工具
3. 任务必须符合智能体的知识领域
4. 每个任务都应该是具体的、可执行的
5. 任务描述要清晰明确，避免模糊不清
6. 任务之间可以有依赖关系，但也应该能够独立理解
7. 场景分析（如提供）、最佳实践、工具说明和注意事项必须严格基于上文给出的模板，不得自行创造新内容

**输出格式**:
请严格按照以下JSON格式输出任务列表：
"""

        # 定义JSON Schema（增强版 Task 结构）
        task_schema = {
            "tasks": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                minItems=num_tasks,
                maxItems=num_tasks,
                items=JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "task_id": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="任务唯一标识，格式：task_001、task_002等",
                        ),
                        "description": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="任务的基本描述",
                        ),
                        "scene_analysis": JSONSchema(
                            type=JSONSchema.Type.OBJECT,
                            properties={
                                "emotional_tone": JSONSchema(
                                    type=JSONSchema.Type.ARRAY,
                                    description="场景情感标签列表",
                                ),
                                "key_subjects": JSONSchema(
                                    type=JSONSchema.Type.ARRAY,
                                    description="场景主体列表",
                                ),
                                "key_moments": JSONSchema(
                                    type=JSONSchema.Type.ARRAY,
                                    description="关键时刻列表，每项包含时间和描述",
                                ),
                                "background_info": JSONSchema(
                                    type=JSONSchema.Type.STRING,
                                    description="场景背景信息",
                                ),
                                "music_style_recommendation": JSONSchema(
                                    type=JSONSchema.Type.ARRAY,
                                    description="推荐音乐风格列表",
                                ),
                                "language_recommendation": JSONSchema(
                                    type=JSONSchema.Type.STRING,
                                    description="推荐歌词语言（中文/英文/Auto）",
                                ),
                            },
                        ),
                        "best_practices": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            minItems=3,
                            maxItems=5,
                            description="任务执行最佳实践，必须从提供的最佳实践模板中选择",
                        ),
                        "tool_usage_guides": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            minItems=3,
                            maxItems=5,
                            description="工具使用指南，基于 AgentProfile.tools 生成",
                        ),
                        "precautions": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            minItems=3,
                            maxItems=5,
                            description="注意事项列表，基于 knowledge.rules 与 constraints 生成",
                        ),
                        # 兼容性字段
                        "required_tools": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            items=JSONSchema(type=JSONSchema.Type.STRING),
                            description="完成此任务需要使用的工具名称列表（兼容旧字段）",
                        ),
                        "required_knowledge": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            items=JSONSchema(type=JSONSchema.Type.STRING),
                            description="完成此任务需要的知识领域列表（兼容旧字段）",
                        ),
                        "priority": JSONSchema(
                            type=JSONSchema.Type.INTEGER,
                            description="任务优先级，1-10之间，10为最高优先级",
                        ),
                    },
                ),
                description="任务列表",
            )
        }

        # 转换schema为可序列化格式
        def schema_to_dict(obj):
            if isinstance(obj, JSONSchema):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: schema_to_dict(v) for k, v in obj.items()}
            return obj

        user_prompt += f"\n```json\n{json.dumps(schema_to_dict(task_schema), indent=2, ensure_ascii=False)}\n```\n"

        return system_prompt + "\n" + user_prompt

    async def generate_tasks_async(
        self,
        profile: AgentProfile,
        user_requirement: str,
        context: Optional[Dict[str, Any]] = None,
        json_scene_data: Optional[List[Dict[str, Any]]] = None,
        num_tasks: int = 3,
    ) -> TaskList:
        """
        异步生成任务列表
        Args:
            profile: 智能体配置文件
            user_requirement: 用户需求描述
            context: 额外上下文信息
            json_scene_data: json_scene 场景数据（可选）
            num_tasks: 需要生成的任务数量
        Returns:
            任务列表对象
        """
        prompt = self._build_task_generation_prompt(
            profile, user_requirement, context, json_scene_data, num_tasks
        )

        messages = [
            SystemMessage(content="你是一个专业的任务规划专家。"),
            HumanMessage(content=prompt)
        ]

        response = await self.llm.ainvoke(messages)

        # 解析响应并做来源校验
        tasks = self._parse_task_response(response, profile.agent_id, profile)

        return tasks

    def generate_tasks(
        self,
        profile: AgentProfile,
        user_requirement: str,
        context: Optional[Dict[str, Any]] = None,
        json_scene_data: Optional[List[Dict[str, Any]]] = None,
        num_tasks: int = 3,
    ) -> TaskList:
        """
        同步生成任务列表

        Args:
            profile: 智能体配置文件
            user_requirement: 用户需求描述
            context: 额外上下文信息
            json_scene_data: json_scene 场景数据（可选）
            num_tasks: 需要生成的任务数量

        Returns:
            任务列表对象
        """
        prompt = self._build_task_generation_prompt(
            profile, user_requirement, context, json_scene_data, num_tasks
        )

        messages = [
            SystemMessage(content="你是一个专业的任务规划专家。"),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        # 解析响应并做来源校验
        tasks = self._parse_task_response(response, profile.agent_id, profile)

        return tasks

    def _parse_task_response(self, response: AIMessage, agent_id: str, profile: AgentProfile) -> TaskList:
        """
        解析LLM返回的任务响应

        Args:
            response: LLM响应
            agent_id: 智能体ID

        Returns:
            任务列表对象
        """
        content = response.content.strip()

        # 提取JSON内容
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        try:
            data = json.loads(content)
            tasks_data = data.get("tasks", [])

            task_list = TaskList(agent_id=agent_id)

            # 准备验证源数据
            best_practices_source: Set[str] = set(profile.best_practices or [])
            tool_names_source: Set[str] = {t.name for t in profile.tools}
            rules_source: Set[str] = set()
            for k in profile.knowledge:
                rules_source.update(k.rules or [])
            constraints_source: Set[str] = set(profile.constraints or [])

            for task_data in tasks_data:
                task = Task(**task_data)

                # 1. 最佳实践验证：仅保留 profile.best_practices 中存在的条目
                if task.best_practices:
                    task.best_practices = [
                        p for p in task.best_practices if p in best_practices_source
                    ]

                # 2. 工具验证：仅保留 AgentProfile.tools 中存在的工具
                if task.required_tools:
                    task.required_tools = [
                        t for t in task.required_tools if t in tool_names_source
                    ]
                if task.tool_usage_guides:
                    filtered_guides: List[ToolUsageGuide] = []
                    for guide in task.tool_usage_guides:
                        if guide.tool_name in tool_names_source:
                            filtered_guides.append(guide)
                    task.tool_usage_guides = filtered_guides

                # 3. 注意事项验证：仅保留 knowledge.rules 或 constraints 中存在的规则
                if task.precautions:
                    filtered_precautions: List[Precaution] = []
                    for p in task.precautions:
                        if (
                            p.rule_description in rules_source
                            or p.rule_description in constraints_source
                        ):
                            filtered_precautions.append(p)
                    task.precautions = filtered_precautions

                task_list.add_task(task)

            return task_list

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"解析任务响应失败: {e}")
            logger.error(f"响应内容: {content}")
            # 返回空任务列表
            return TaskList(agent_id=agent_id)

    def validate_tasks(
        self,
        tasks: TaskList,
        profile: AgentProfile
    ) -> Tuple[bool, List[str]]:
        """
        验证任务列表是否符合智能体配置

        Args:
            tasks: 任务列表
            profile: 智能体配置文件

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []

        # 获取可用工具和知识领域
        available_tools = {tool.name for tool in profile.tools}
        available_domains = {k.domain.value for k in profile.knowledge}

        for task in tasks.tasks:
            # 验证工具
            for tool_name in task.required_tools:
                if tool_name not in available_tools:
                    errors.append(
                        f"任务 {task.task_id} 使用了不可用的工具: {tool_name}"
                    )

            # 验证知识领域
            for domain in task.required_knowledge:
                if domain not in available_domains:
                    errors.append(
                        f"任务 {task.task_id} 需要不可用的知识领域: {domain}"
                    )

        return len(errors) == 0, errors


# 示例使用
async def main():
    """测试任务生成器"""
    from task.task_profile import EXAMPLE_POP_GT_LYRIC_PROFILE

    generator = TaskGenerator()

    user_requirement = """
    我需要为一个温馨家居场景生成音乐歌词。场景描述：
    - 室内环境，光线柔和
    - 年轻女性与宠物狗亲密互动
    - 氛围温暖、轻松、充满爱意

    请生成3个具体的任务来完成这个需求。
    """

    context = {
        "场景类型": "家居生活",
        "情感基调": "温馨、幸福",
        "目标风格": "轻松愉悦的流行音乐"
    }

    print("=" * 80)
    print("开始生成任务...")
    print("=" * 80)

    tasks = await generator.generate_tasks_async(
        profile=EXAMPLE_POP_GT_LYRIC_PROFILE,
        user_requirement=user_requirement,
        context=context,
        num_tasks=1
    )

    print(f"\n生成的任务列表 (共 {len(tasks.tasks)} 个任务):\n")
    for i, task in enumerate(tasks.tasks, 1):
        print(f"任务 {i}: {task.task_id}")
        print(f"  描述: {task.description}")
        print(f"  所需工具: {', '.join(task.required_tools)}")
        print(f"  所需知识: {', '.join(task.required_knowledge)}")
        print(f"  输入上下文: {task.input_context}")
        print(f"  期望输出: {task.expected_output}")
        print(f"  优先级: {task.priority}")
        if task.constraints:
            print(f"  约束条件:")
            for constraint in task.constraints:
                print(f"    - {constraint}")
        print()

    # 验证任务
    is_valid, errors = generator.validate_tasks(tasks, EXAMPLE_POP_GT_LYRIC_PROFILE)
    if is_valid:
        print("✓ 任务验证通过！")
    else:
        print("✗ 任务验证失败:")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
