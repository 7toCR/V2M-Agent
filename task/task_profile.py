"""
Task Profile Module

定义子智能体的角色、工具、知识的数据结构
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class ToolCategory(str, Enum):
    """工具分类"""
    DATA_PROCESSING = "数据处理"
    CONTENT_GENERATION = "内容生成"
    ANALYSIS = "分析"
    AUDIO_ANALYSIS = "音频分析"
    VIDEO_ANALYSIS = "视频分析"
    IMAGE_ANALYSIS = "图像分析"
    TEXT_ANALYSIS = "文本分析"
    COORDINATION = "协调"
    EXTRACTION = "提取"
    TRANSFORMATION = "转换"
    VALIDATION = "验证"
    OTHER = "其他"


class KnowledgeDomain(str, Enum):
    """知识领域分类"""
    MUSIC = "音乐"
    VIDEO = "视频"
    AUDIO = "音频"
    TEXT = "文本"
    IMAGE = "图像"
    SCENE = "场景理解"
    EMOTION = "情感分析"
    STYLE = "风格识别"
    MULTIMODAL = "多模态理解"
    SEMANTIC = "语义理解"
    VISUAL = "视觉理解"
    COLOR = "色彩分析"
    COMPOSITION = "构图分析"
    SOUND = "声音识别"
    RHYTHM = "节奏分析"
    TIME = "时间分段"
    TEMPORAL = "时序分析"
    OTHER = "其他"


class AgentRole(BaseModel):
    """
    智能体角色定义

    Attributes:
        name: 角色名称
        description: 角色描述
        responsibilities: 职责列表
        expertise: 专业领域
    """
    name: str = Field(..., description="角色名称，例如：'音乐歌词生成专家'")
    description: str = Field(..., description="角色的详细描述")
    responsibilities: List[str] = Field(
        default_factory=list,
        description="该角色的主要职责列表"
    )
    expertise: List[str] = Field(
        default_factory=list,
        description="该角色的专业领域列表"
    )


class AgentTool(BaseModel):
    """
    智能体工具定义

    Attributes:
        name: 工具名称
        function_signature: 函数签名
        description: 工具描述
        parameters: 参数列表
        returns: 返回值描述
        category: 工具分类
        usage_example: 使用示例
        dependencies: 依赖的其他工具名称列表（可选）
    """
    name: str = Field(..., description="工具名称，例如：'pop_gt_lyric'")
    function_signature: str = Field(
        ...,
        description="函数签名，例如：'def pop_gt_lyric(self, state: Graph) -> dict'"
    )
    description: str = Field(..., description="工具的功能描述")
    parameters: List[Dict[str, str]] = Field(
        default_factory=list,
        description="参数列表，每个参数包含 name、type、description"
    )
    returns: str = Field(..., description="返回值描述")
    category: ToolCategory = Field(
        default=ToolCategory.OTHER,
        description="工具分类"
    )
    usage_example: Optional[str] = Field(
        None,
        description="工具使用示例"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="该工具依赖的其他工具名称列表"
    )


class AgentKnowledge(BaseModel):
    """
    智能体知识定义

    Attributes:
        domain: 知识领域
        concepts: 核心概念列表
        rules: 规则和约束列表
        examples: 示例列表
        references: 参考资料列表
    """
    domain: KnowledgeDomain = Field(..., description="知识领域")
    concepts: List[str] = Field(
        default_factory=list,
        description="该领域的核心概念列表"
    )
    rules: List[str] = Field(
        default_factory=list,
        description="该领域的规则和约束列表"
    )
    examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="示例列表，每个示例包含输入和期望输出"
    )
    references: List[str] = Field(
        default_factory=list,
        description="参考资料列表"
    )


class AgentProfile(BaseModel):
    """
    智能体完整配置文件

    Attributes:
        agent_id: 智能体唯一标识
        description: 智能体总体描述
        role: 角色定义
        tools: 工具列表
        knowledge: 知识列表
        constraints: 约束条件列表
        best_practices: 最佳实践列表
        resources: 资源列表（文档、模型、数据等）
        run_methods: 运行方式说明
        command: 典型运行命令或入口
        guide_book: 参考指南名称或链接
    """
    agent_id: str = Field(..., description="智能体唯一标识符")
    description: str = Field(
        default="",
        description="智能体总体描述，说明核心目标与定位"
    )
    role: AgentRole = Field(..., description="智能体角色")
    tools: List[AgentTool] = Field(
        default_factory=list,
        description="智能体可用工具列表"
    )
    knowledge: List[AgentKnowledge] = Field(
        default_factory=list,
        description="智能体知识库"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="智能体的约束条件"
    )
    best_practices: List[str] = Field(
        default_factory=list,
        description="智能体的最佳实践"
    )
    resources: List[str] = Field(
        default_factory=list,
        description="可用资源（文档、数据、模型、外部系统等）"
    )
    run_methods: List[str] = Field(
        default_factory=list,
        description="运行方式或调用说明"
    )
    command: Optional[str] = Field(
        default=None,
        description="典型运行命令或入口"
    )
    guide_book: Optional[str] = Field(
        default=None,
        description="指南或手册名称/链接"
    )

    def get_tool_by_name(self, tool_name: str) -> Optional[AgentTool]:
        """根据工具名称获取工具"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def get_knowledge_by_domain(self, domain: KnowledgeDomain) -> List[AgentKnowledge]:
        """根据领域获取知识"""
        return [k for k in self.knowledge if k.domain == domain]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()



EXAMPLE_POP_GT_LYRIC_PROFILE = AgentProfile(
    agent_id="pop_gt_lyric_agent",
    description="根据多模态场景描述生成符合情感和氛围的流行歌词的专家型智能体。",
    role=AgentRole(
        name="流行音乐歌词生成专家",
        description="专注于根据场景描述生成符合情感和氛围的流行音乐歌词",
        responsibilities=[
            "分析场景的情感基调和氛围",
            "生成符合场景的歌词内容",
            "确保歌词结构完整（intro、verse、chorus、outro等）",
            "保持歌词风格与场景一致性"
        ],
        expertise=[
            "流行音乐歌词创作",
            "情感表达",
            "叙事结构",
            "韵律和节奏"
        ]
    ),
    tools=[
        AgentTool(
            name="pop_gt_lyric",
            function_signature="def pop_gt_lyric(self, state: Graph) -> dict",
            description="根据json_scene生成对应的prompt_gt_lyric字符串",
            parameters=[
                {
                    "name": "json_scene",
                    "type": "List[Dict[str, Any]]",
                    "description": "场景数据数组，包含时间段、主体声音内容、主体声音风格、环境声音内容、环境声音风格"
                }
            ],
            returns="歌词内容列表，匹配场景的情感、叙事和氛围，格式：[[intro-short] ; [verse] 句子1.句子2... ; [chorus] ... ; [outro-short]]",
            category=ToolCategory.CONTENT_GENERATION,
            usage_example="根据温馨家居场景生成轻松愉悦的歌词"
        )
    ],
    knowledge=[
        AgentKnowledge(
            domain=KnowledgeDomain.MUSIC,
            concepts=[
                "歌词结构：intro、verse、chorus、bridge、outro",
                "情感表达：happy、sad、romantic、melancholic等",
                "叙事方式：第一人称、第三人称、时间顺序、情感递进"
            ],
            rules=[
                "歌词必须包含完整的结构标签",
                "每个部分用分号(;)分隔",
                "每个句子用句号(.)分隔",
                "风格需要与场景情感一致",
                "支持中文、英文或中英混合歌词"
            ],
            examples=[
                {
                    "input": {
                        "scene": "温馨家居场景，女性与宠物狗互动",
                        "emotion": "幸福、温暖"
                    },
                    "output": "[intro-short] ; [verse] Lazy Sunday morning light.You and me, everything feels right ; [chorus] In your eyes I see home.Never want to be alone ; [outro-short]"
                }
            ],
            references=[
                "流行音乐歌词创作指南",
                "情感音乐理论"
            ]
        )
    ],
    constraints=[
        "必须严格遵循歌词结构格式",
        "不得生成不符合场景情感的内容",
        "歌词长度要适中，避免过长或过短",
        "保持语言风格统一"
    ],
    best_practices=[
        "首先分析场景的核心情感和氛围",
        "根据场景选择合适的语言（中文/英文）",
        "确保歌词的情感递进与场景变化一致",
        "使用具象化的描述增强画面感"
    ],
    resources=[
        "内部歌词结构规范文档",
        "示例歌词数据集",
        "情感词典与情绪映射表"
    ],
    run_methods=[
        "在流水线中调用 `pop_gt_lyric` 工具处理完整 json_scene",
        "单步调试可使用 __main__ 示例打印输出检查格式"
    ],
    command="pop_gt_lyric(json_scene)",
    guide_book="STREAMING_TASK_CREATOR_GUIDE.md"
)
if __name__ == "__main__":
    # 演示打印：GT-Lyric Agent 全量信息
    profile = EXAMPLE_POP_GT_LYRIC_PROFILE
    print("=" * 60)
    print("GT-Lyric Agent Profile")
    print("=" * 60)
    print(f"智能体ID: {profile.agent_id}")
    print(f"描述: {profile.description}")
    print(f"角色名称: {profile.role.name}")
    print(f"角色描述: {profile.role.description}")
    print(f"\n职责 ({len(profile.role.responsibilities)} 项):")
    for i, resp in enumerate(profile.role.responsibilities, 1):
        print(f"  {i}. {resp}")
    print(f"\n工具 ({len(profile.tools)} 个):")
    for tool in profile.tools:
        print(f"  - {tool.name}: {tool.description}")
    print(f"\n知识领域 ({len(profile.knowledge)} 个):")
    for k in profile.knowledge:
        print(f"  - {k.domain.value}: {len(k.concepts)} 个核心概念")
    print(f"\n约束 ({len(profile.constraints)} 项):")
    for i, c in enumerate(profile.constraints, 1):
        print(f"  {i}. {c}")
    print(f"\n最佳实践 ({len(profile.best_practices)} 项):")
    for i, b in enumerate(profile.best_practices, 1):
        print(f"  {i}. {b}")
    print(f"\n资源 ({len(profile.resources)} 项):")
    for i, r in enumerate(profile.resources, 1):
        print(f"  {i}. {r}")
    print(f"\n运行方式 ({len(profile.run_methods)} 项):")
    for i, rm in enumerate(profile.run_methods, 1):
        print(f"  {i}. {rm}")
    print(f"\nCOMMAND: {profile.command}")
    print(f"Guide Book: {profile.guide_book}")
