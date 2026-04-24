"""
Task Generation Examples

展示如何使用 task 模块为不同的子智能体生成任务
"""

import os
import sys
import asyncio
import json

# 添加项目根目录到路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from task.task_profile import (
    AgentProfile,
    AgentRole,
    AgentTool,
    AgentKnowledge,
    ToolCategory,
    KnowledgeDomain,
    EXAMPLE_POP_GT_LYRIC_PROFILE,
    EXAMPLE_POP_AUDIO_TYPE_PROFILE,
)
from task.task_generator import TaskGenerator


# POPDescriptionAgent 配置
POP_DESCRIPTION_PROFILE = AgentProfile(
    agent_id="pop_description_agent",
    role=AgentRole(
        name="音乐描述生成专家",
        description="根据场景、歌词和音乐风格生成详细的音乐描述，包括情感、性别、乐器、音色、BPM等",
        responsibilities=[
            "分析场景和歌词的情感基调",
            "选择合适的音乐参数（情感、性别、类型、乐器、音色、BPM）",
            "生成符合场景的音乐描述字符串",
            "确保描述的多样性和独特性"
        ],
        expertise=[
            "音乐情感分析",
            "乐器和音色选择",
            "音乐参数匹配",
            "BPM节奏控制"
        ]
    ),
    tools=[
        AgentTool(
            name="pop_descriptions",
            function_signature="def pop_descriptions(self, state: Graph) -> dict",
            description="根据json_scene生成对应的prompt_descriptions字符串列表",
            parameters=[
                {
                    "name": "json_scene",
                    "type": "List[Dict[str, Any]]",
                    "description": "场景数据数组"
                },
                {
                    "name": "emotion",
                    "type": "str",
                    "description": "情感类型：sad、emotional、angry、happy、uplifting、intense、romantic、melancholic"
                },
                {
                    "name": "gender",
                    "type": "str",
                    "description": "性别：female、male"
                },
                {
                    "name": "genre",
                    "type": "str",
                    "description": "音乐类型：pop、rock、jazz、blues等"
                },
                {
                    "name": "instrument",
                    "type": "str",
                    "description": "乐器组合：如'piano and drums'、'acoustic guitar'等"
                },
                {
                    "name": "timbre",
                    "type": "str",
                    "description": "音色：dark、bright、warm、rock、soft、vocal"
                },
                {
                    "name": "bpm",
                    "type": "int",
                    "description": "每分钟节拍数"
                }
            ],
            returns="描述字符串列表，格式如：['female, dark, pop, sad, piano and drums, the bpm is 125.']",
            category=ToolCategory.CONTENT_GENERATION,
            usage_example="female, bright, pop, happy, acoustic guitar and harmonica, the bpm is 72."
        )
    ],
    knowledge=[
        AgentKnowledge(
            domain=KnowledgeDomain.MUSIC,
            concepts=[
                "情感类型与场景的对应关系",
                "音乐类型的特点和适用场景",
                "乐器组合的情感表达",
                "音色与氛围的关系",
                "BPM与节奏感的匹配"
            ],
            rules=[
                "描述必须包含：性别、音色、类型、情感、乐器、BPM",
                "各参数用逗号分隔",
                "BPM格式为：'the bpm is XX.'",
                "确保参数组合的协调性",
                "鼓励多样性，避免重复"
            ],
            examples=[
                {
                    "input": {
                        "scene": "温馨家居",
                        "lyric": "轻松愉悦",
                        "audio_type": "Pop"
                    },
                    "output": "female, bright, pop, happy, acoustic guitar and harmonica, the bpm is 72."
                }
            ]
        )
    ],
    constraints=[
        "参数必须从预定义的列表中选择",
        "描述格式必须严格遵循",
        "情感、乐器、音色必须与场景和歌词匹配",
        "避免使用相同的参数组合"
    ],
    best_practices=[
        "首先分析场景的核心情感",
        "参考歌词和音乐风格选择合适的参数",
        "确保各参数之间协调一致",
        "鼓励从不同维度进行组合创新"
    ]
)


# POPIdxAgent 配置
POP_IDX_PROFILE = AgentProfile(
    agent_id="pop_idx_agent",
    role=AgentRole(
        name="音乐作品索引生成专家",
        description="根据场景、歌词、音乐风格和描述生成唯一的音乐作品标识符",
        responsibilities=[
            "分析歌词内容和语言",
            "根据当前时间生成时间戳",
            "创建具有描述性的歌曲名字",
            "确保索引的唯一性"
        ],
        expertise=[
            "命名规范",
            "时间戳生成",
            "语言识别",
            "内容提取"
        ]
    ),
    tools=[
        AgentTool(
            name="pop_idx",
            function_signature="def pop_idx(self, state: Graph) -> dict",
            description="根据当前时间生成现代音乐歌曲名字列表",
            parameters=[
                {
                    "name": "gt_lyric",
                    "type": "str",
                    "description": "歌词内容"
                },
                {
                    "name": "audio_type",
                    "type": "str",
                    "description": "音乐风格"
                },
                {
                    "name": "descriptions",
                    "type": "str",
                    "description": "音乐描述"
                }
            ],
            returns="索引列表，格式如：['2025-12-17-14-15-夏天', '2025-12-17-14-15-蝉鸣']",
            category=ToolCategory.CONTENT_GENERATION,
            usage_example="2025-12-17-14-15-Lazy-Sunday"
        )
    ],
    knowledge=[
        AgentKnowledge(
            domain=KnowledgeDomain.MUSIC,
            concepts=[
                "时间戳格式：YYYY-MM-DD-HH-MM",
                "歌曲名字命名规范",
                "中英文语言识别",
                "主题提取"
            ],
            rules=[
                "格式：{时间戳}-{歌曲名字}",
                "当歌词为英文时，生成英文名字",
                "当歌词为中文时，生成中文名字",
                "当歌词为中英文混杂时，根据内容占比决定",
                "BGM则完全随机"
            ],
            examples=[
                {
                    "input": {
                        "lyric": "Lazy Sunday morning light...",
                        "language": "English"
                    },
                    "output": "2025-12-17-14-15-Lazy-Sunday"
                },
                {
                    "input": {
                        "lyric": "夏日蝉鸣...",
                        "language": "Chinese"
                    },
                    "output": "2025-12-17-14-15-夏天"
                }
            ]
        )
    ],
    constraints=[
        "必须包含时间戳",
        "歌曲名字要简洁有意义",
        "名字语言要与歌词匹配",
        "确保索引唯一性"
    ],
    best_practices=[
        "首先识别歌词的主要语言",
        "提取歌词的核心主题或情感",
        "生成简洁且富有表现力的名字",
        "确保时间戳格式正确"
    ]
)


async def example_1_generate_lyric_tasks():
    """示例1：为 POPGtLyricAgent 生成任务"""
    print("=" * 80)
    print("示例1：为 POPGtLyricAgent 生成任务")
    print("=" * 80)

    generator = TaskGenerator()

    user_requirement = """
    我需要为一个温馨的宠物主题场景生成音乐歌词。
    场景描述：年轻女性与她的小狗在家中温馨互动，充满爱意和幸福感。
    情感基调：温暖、轻松、充满爱
    """

    context = {
        "场景关键帧数": "6个",
        "场景时长": "约30秒",
        "情感变化": "持续温馨，略带甜蜜",
        "目标语言": "英文为主"
    }

    # 示例 json_scene（真实使用时可从 scene.jsonl 读取）
    json_scene = [
        {
            "关键帧": "0s",
            "主体": "年轻女性和小狗",
            "主体心情": "女性略带惊讶，小狗好奇亲昵",
            "背景": "室内环境，光线柔和"
        },
        {
            "关键帧": "10s",
            "主体": "年轻女性和小狗",
            "主体心情": "女性幸福放松，小狗强烈依恋",
            "背景": "室内环境，温馨家居氛围"
        },
    ]

    tasks = await generator.generate_tasks_async(
        profile=EXAMPLE_POP_GT_LYRIC_PROFILE,
        user_requirement=user_requirement,
        context=context,
        json_scene_data=json_scene,
        num_tasks=1,
    )

    print(f"\n生成的任务列表 (共 {len(tasks.tasks)} 个):\n")
    for i, task in enumerate(tasks.tasks, 1):
        print(f"\n【任务】{task.task_id}")
        print(f"描述: {task.description}")

        if task.scene_analysis:
            print(f"\n【场景分析】")
            print(f"  情感标签: {', '.join(task.scene_analysis.emotional_tone)}")
            print(f"  主体: {', '.join(task.scene_analysis.key_subjects)}")
            print(f"  推荐风格: {', '.join(task.scene_analysis.music_style_recommendation)}")
            print(f"  推荐语言: {task.scene_analysis.language_recommendation}")

        print(f"\n【最佳实践】")
        for idx, practice in enumerate(task.best_practices, 1):
            print(f"  {idx}. {practice}")

        print(f"\n【工具使用说明】")
        for guide in task.tool_usage_guides:
            print(f"  - {guide.tool_name} (执行顺序: {guide.execution_order})")
            print(f"    目的: {guide.usage_purpose}")
            if guide.dependencies:
                print(f"    依赖: {', '.join(guide.dependencies)}")

        print(f"\n【注意事项】")
        for precaution in task.precautions:
            print(f"  - [{precaution.category}] {precaution.rule_description}")
            print(f"    校验: {precaution.validation_method}")
            print(f"    错误处理: {precaution.error_handling}")

    # 保存到文件
    output_file = os.path.join(CURRENT_DIR, "examples", "lyric_tasks.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"任务已保存到: {output_file}\n")


async def example_2_generate_description_tasks():
    """示例2：为 POPDescriptionAgent 生成任务"""
    print("=" * 80)
    print("示例2：为 POPDescriptionAgent 生成任务")
    print("=" * 80)

    generator = TaskGenerator()

    user_requirement = """
    已经生成了歌词和音乐风格，现在需要生成详细的音乐描述。
    要求：
    1. 描述要符合温馨家居场景的氛围
    2. 要体现轻松愉悦的情感
    3. 确保参数的多样性和协调性
    """

    context = {
        "已有歌词": "Lazy Sunday morning light, You and me...",
        "已有音乐风格": "Pop, Folk",
        "场景情感": "温馨、幸福、轻松"
    }

    tasks = await generator.generate_tasks_async(
        profile=POP_DESCRIPTION_PROFILE,
        user_requirement=user_requirement,
        context=context,
        num_tasks=3,
    )

    print(f"\n生成的任务列表 (共 {len(tasks.tasks)} 个):\n")
    for i, task in enumerate(tasks.tasks, 1):
        print(f"【任务 {i}】{task.task_id}")
        print(f"  描述: {task.description}")
        print(f"  期望输出: {task.expected_output}")
        if task.constraints:
            print(f"  约束条件:")
            for constraint in task.constraints:
                print(f"    - {constraint}")
        print()

    # 保存到文件
    output_file = os.path.join(CURRENT_DIR, "examples", "description_tasks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"任务已保存到: {output_file}\n")


async def example_3_multi_agent_workflow():
    """示例3：完整的多智能体工作流任务生成"""
    print("=" * 80)
    print("示例3：完整的多智能体工作流任务生成")
    print("=" * 80)

    generator = TaskGenerator()

    # 场景描述
    scene_context = """
    一个温馨的家居场景视频，包含6个关键帧，总时长30秒。
    主要内容：年轻女性与她的小狗在家中亲密互动。
    情感基调：温暖、幸福、轻松愉悦。
    """

    # 为每个智能体生成任务
    agents = [
        ("歌词生成", EXAMPLE_POP_GT_LYRIC_PROFILE),
        ("音乐风格选择", EXAMPLE_POP_AUDIO_TYPE_PROFILE),
        ("音乐描述生成", POP_DESCRIPTION_PROFILE),
        ("索引生成", POP_IDX_PROFILE),
    ]

    all_tasks = {}

    for agent_name, profile in agents:
        print(f"\n正在为【{agent_name}智能体】生成任务...")

        requirement = f"""
        场景描述：{scene_context}

        作为{profile.role.name}，你需要完成你职责范围内的任务。
        请确保任务：
        1. 贴近你的角色定位
        2. 严格使用你的可用工具
        3. 符合你的知识领域
        """

        tasks = await generator.generate_tasks_async(
            profile=profile,
            user_requirement=requirement,
            context={"场景": scene_context},
            num_tasks=2,
        )

        all_tasks[agent_name] = tasks

        # 验证任务
        is_valid, errors = generator.validate_tasks(tasks, profile)
        if is_valid:
            print(f"  ✓ 任务验证通过")
        else:
            print(f"  ✗ 任务验证失败:")
            for error in errors:
                print(f"    - {error}")

    # 保存所有任务
    output_file = os.path.join(CURRENT_DIR, "examples", "workflow_tasks.json")
    workflow_data = {
        agent_name: tasks.to_dict()
        for agent_name, tasks in all_tasks.items()
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(workflow_data, f, indent=2, ensure_ascii=False)

    print(f"\n所有任务已保存到: {output_file}")

    # 打印任务摘要
    print("\n" + "=" * 80)
    print("任务生成摘要")
    print("=" * 80)
    for agent_name, tasks in all_tasks.items():
        print(f"\n{agent_name}: {len(tasks.tasks)} 个任务")
        for task in tasks.tasks:
            print(f"  - {task.description[:60]}...")


def example_4_streaming_task_creator():
    """
    示例4：使用StreamingTaskCreator处理多个agent_profile，支持流式输出

    特点：
    - 一次性处理多个agent配置
    - 实时显示LLM生成过程（token-by-token）
    - 自动保存任务到JSON文件
    - 完整的进度指示和错误处理
    """
    from task.task_create import StreamingTaskCreator

    print("=" * 80)
    print("示例4：使用StreamingTaskCreator处理多个Agent配置")
    print("=" * 80)

    # 创建流式任务生成器实例
    creator = StreamingTaskCreator()

    # 准备多个agent配置
    profiles = [
        EXAMPLE_POP_GT_LYRIC_PROFILE,
        EXAMPLE_POP_AUDIO_TYPE_PROFILE,
        POP_DESCRIPTION_PROFILE,
        POP_IDX_PROFILE,
    ]

    # 用户需求
    user_requirement = """
    我需要为一个温馨的家庭场景生成完整的音乐生成流程。
    场景描述：年轻女性与她的宠物狗在客厅里温馨互动，阳光透过窗户洒入房间，氛围非常舒适放松。
    目标：
    1. 生成符合场景的中文歌词
    2. 选择合适的音乐风格
    3. 生成详细的音乐描述
    4. 生成音乐作品索引
    """

    # 执行任务生成（带流式输出）
    # 这会在控制台实时显示LLM生成过程
    results = creator.create_tasks_for_profiles(
        profiles=profiles,
        user_requirement=user_requirement,
        num_tasks_per_agent=2,  # 每个agent生成2个任务
        save_files=True,  # 自动保存到JSON文件
    )

    # 显示结果摘要
    print("\n" + "=" * 80)
    print("生成结果摘要")
    print("=" * 80)
    for agent_id, task_list in results.items():
        print(f"\n{agent_id}: {len(task_list.tasks)} 个任务已生成")
        for task in task_list.tasks[:2]:  # 仅显示前2个任务
            print(f"  - {task.task_id}: {task.description[:60]}...")
            print(f"    最佳实践: {len(task.best_practices)} 条")
            print(f"    工具说明: {len(task.tool_usage_guides)} 条")
            print(f"    注意事项: {len(task.precautions)} 条")

    print("\n任务已保存到: {}".format(creator.output_dir))


async def main():
    """运行所有示例"""
    print("\n开始运行任务生成示例...\n")

    # 运行示例1
    await example_1_generate_lyric_tasks()

    #print("\n" + "-" * 80 + "\n")

    # 运行示例2
    #await example_2_generate_description_tasks()

    #print("\n" + "-" * 80 + "\n")

    # 运行示例3
    #await example_3_multi_agent_workflow()

    print("\n" + "=" * 80)
    print("所有示例运行完成！")
    print("=" * 80)


def run_streaming_example():
    """独立运行流式任务生成示例"""
    example_4_streaming_task_creator()


if __name__ == "__main__":
    asyncio.run(main())
