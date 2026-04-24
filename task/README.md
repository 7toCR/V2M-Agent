# Task Generation Module

这个文件夹包含了一套完整的任务生成系统，用于根据子智能体的角色、工具和知识自动生成任务列表。

## 核心功能

1. **任务配置定义** (`task_profile.py`)
   - 定义智能体的角色、工具、知识结构
   - 提供预定义的配置示例

2. **任务生成器** (`task_generator.py`)
   - 基于智能体配置自动生成任务列表
   - 支持同步和异步生成
   - 内置任务验证机制
   - 适用于单个agent的任务生成

3. **流式任务生成器** (`task_create.py`) NEW
   - 支持多个agent_profile的并行处理
   - 实时显示LLM生成过程（Token-by-Token流式输出）
   - 自动保存结果到JSON文件
   - 适用于批量处理多个agent的任务

4. **使用示例** (`examples.py`)
   - TaskGenerator的完整使用示例
   - StreamingTaskCreator的多agent工作流演示

## 快速开始

### 安装依赖

确保已安装项目依赖：
```bash
uv sync
```

### 基本使用

```python
from task.task_profile import EXAMPLE_POP_GT_LYRIC_PROFILE
from task.task_generator import TaskGenerator

# 创建任务生成器
generator = TaskGenerator()

# 定义用户需求
user_requirement = """
我需要为一个温馨家居场景生成音乐歌词。
场景描述：年轻女性与宠物狗互动，氛围温暖轻松。
"""

# 生成任务
tasks = generator.generate_tasks(
    profile=EXAMPLE_POP_GT_LYRIC_PROFILE,
    user_requirement=user_requirement,
    num_tasks=3
)

# 查看生成的任务
for task in tasks.tasks:
    print(f"任务: {task.description}")
    print(f"工具: {task.required_tools}")
    print(f"知识: {task.required_knowledge}")
```

### 运行示例

```bash
# 运行完整示例
uv run task/examples.py

# 或直接使用Python
python task/examples.py
```

## 核心概念

### AgentProfile（智能体配置）

智能体配置包含以下核心要素：

- **Role（角色）**: 智能体的名称、描述、职责和专业领域
- **Tools（工具）**: 智能体可用的工具列表，包括函数签名、参数、返回值
- **Knowledge（知识）**: 智能体的知识库，包括核心概念、规则、示例
- **Constraints（约束）**: 智能体的约束条件
- **Best Practices（最佳实践）**: 智能体的最佳实践指南

### Task（任务）

增强后的任务定义包含：

- **task_id**: 任务唯一标识
- **description**: 任务基本描述
- **scene_analysis**: 场景分析结果（可选，来自 `json_scene`）
  - `emotional_tone`: 场景情感标签列表
  - `key_subjects`: 场景主体列表
  - `key_moments`: 关键时刻（时间 + 描述）
  - `background_info`: 场景背景信息
  - `music_style_recommendation`: 推荐音乐风格（仅从 `AgentProfile.knowledge` 中抽取）
  - `language_recommendation`: 推荐歌词语言（中文 / 英文 / Auto）
- **best_practices**: 任务执行的最佳实践（3-5 条，严格来自 `AgentProfile.best_practices`）
- **tool_usage_guides**: 工具使用说明（3-5 条，基于 `AgentProfile.tools` 和工具依赖）
- **precautions**: 注意事项 / 验证规则（3-5 条，严格来自 `knowledge.rules` 和 `constraints`）
- **required_tools**: 需要使用的工具列表（兼容旧字段）
- **required_knowledge**: 需要的知识领域列表（兼容旧字段）
- **priority**: 任务优先级 (1-10)

### TaskGenerator（任务生成器）

任务生成器根据以下原则生成任务：

1. **角色贴近性**: 任务必须贴近智能体的角色定位和职责
2. **工具严格性**: 任务必须严格依据智能体的可用工具
3. **知识符合性**: 任务必须符合智能体的知识领域
4. **可执行性**: 每个任务都应该是具体的、可执行的
5. **清晰性**: 任务描述清晰明确，避免模糊不清

## 文件结构

```
task/
├── __init__.py                # 模块初始化
├── task_profile.py            # 任务配置定义
├── task_generator.py          # 任务生成器（单agent，LLM invoke方式）
├── task_create.py             # 流式任务生成器（多agent，LLM stream方式）NEW
├── scene_analyzer.py          # 场景分析工具
├── examples.py                # 使用示例
├── README.md                  # 本文件
├── output/                    # 生成的任务文件（运行task_create.py后生成）
│   ├── pop_gt_lyric_agent_tasks.json
│   ├── pop_audio_type_agent_tasks.json
│   ├── pop_description_agent_tasks.json
│   └── pop_idx_agent_tasks.json
└── examples/                  # 生成的示例任务（运行examples.py后生成）
    ├── lyric_tasks.json
    ├── description_tasks.json
    └── workflow_tasks.json
```

## 预定义配置

模块提供了以下预定义的智能体配置：

1. **EXAMPLE_POP_GT_LYRIC_PROFILE** - 流行音乐歌词生成专家
2. **EXAMPLE_POP_AUDIO_TYPE_PROFILE** - 音乐风格选择专家
3. **POP_DESCRIPTION_PROFILE** - 音乐描述生成专家（在examples.py中）
4. **POP_IDX_PROFILE** - 音乐作品索引生成专家（在examples.py中）

## 自定义配置

你可以创建自己的智能体配置：

```python
from task.task_profile import (
    AgentProfile, AgentRole, AgentTool, AgentKnowledge,
    ToolCategory, KnowledgeDomain
)

# 创建自定义配置
my_agent_profile = AgentProfile(
    agent_id="my_agent",
    role=AgentRole(
        name="我的智能体",
        description="智能体描述",
        responsibilities=["职责1", "职责2"],
        expertise=["专业领域1", "专业领域2"]
    ),
    tools=[
        AgentTool(
            name="my_tool",
            function_signature="def my_tool(self, param: str) -> str",
            description="工具描述",
            parameters=[
                {
                    "name": "param",
                    "type": "str",
                    "description": "参数描述"
                }
            ],
            returns="返回值描述",
            category=ToolCategory.CONTENT_GENERATION
        )
    ],
    knowledge=[
        AgentKnowledge(
            domain=KnowledgeDomain.MUSIC,
            concepts=["概念1", "概念2"],
            rules=["规则1", "规则2"]
        )
    ],
    constraints=["约束1", "约束2"],
    best_practices=["最佳实践1", "最佳实践2"]
)
```

## 任务验证

生成的任务会自动验证：

```python
# 验证任务
is_valid, errors = generator.validate_tasks(tasks, profile)

if is_valid:
    print("任务验证通过！")
else:
    print("任务验证失败:")
    for error in errors:
        print(f"  - {error}")
```

## 注意事项

1. 确保环境变量 `MCP_API_KEY` 已设置
2. 任务生成使用 LLM，可能需要一定时间
3. 生成的任务数量由 `num_tasks` 参数控制
4. 可以通过 `context` 参数提供额外的上下文信息

## 集成到现有Agent

你可以将生成的任务集成到现有的Agent中：

```python
# 在POPAgent的_task_gt_lyric_node中使用
from task.task_generator import TaskGenerator
from task.task_profile import EXAMPLE_POP_GT_LYRIC_PROFILE

generator = TaskGenerator()

# 生成任务
tasks = generator.generate_tasks(
    profile=EXAMPLE_POP_GT_LYRIC_PROFILE,
    user_requirement=user_requirement,
    context={"json_scene": json_scene},
    num_tasks=piece
)

# 使用生成的任务
for task in tasks.tasks:
    # 执行任务...
    pass
```

## 流式任务生成器（StreamingTaskCreator）

### 概述

`task_create.py` 中的 **StreamingTaskCreator** 是一个高级任务生成器，专门支持**多个agent_profile的并行处理**和**LLM流式输出**。

### 主要特性

1. **多Agent支持**: 一次性处理多个agent_profile，为每个agent生成任务
2. **流式输出**: 实时显示LLM生成过程（Token-by-Token），增强用户体验
3. **自动保存**: 生成完成后自动保存为JSON文件
4. **进度提示**: 清晰的进度显示和错误处理
5. **格式化输出**: 任务包含：描述、最佳实践、工具使用说明、注意事项

### 快速开始

#### 方式1：使用命令行

```bash
# 直接运行task_create.py中的main()
uv run -m task.task_create

# 或使用Python
python task/task_create.py
```

#### 方式2：在代码中使用

```python
from task.task_create import StreamingTaskCreator
from task.task_profile import EXAMPLE_POP_GT_LYRIC_PROFILE, EXAMPLE_POP_AUDIO_TYPE_PROFILE
from task.examples import POP_DESCRIPTION_PROFILE, POP_IDX_PROFILE

# 创建生成器实例
creator = StreamingTaskCreator()

# 准备要处理的agent配置列表
profiles = [
    EXAMPLE_POP_GT_LYRIC_PROFILE,
    EXAMPLE_POP_AUDIO_TYPE_PROFILE,
    POP_DESCRIPTION_PROFILE,
    POP_IDX_PROFILE,
]

# 用户需求
user_requirement = """
我需要为一个温馨的家庭场景生成完整的音乐生成流程。
场景描述：年轻女性与宠物狗互动，氛围温暖轻松。
"""

# 执行任务生成（带流式输出）
results = creator.create_tasks_for_profiles(
    profiles=profiles,
    user_requirement=user_requirement,
    num_tasks_per_agent=3,  # 每个agent生成3个任务
    save_files=True,         # 自动保存到JSON文件
)

# 访问生成的任务
for agent_id, task_list in results.items():
    print(f"{agent_id}: {len(task_list.tasks)} 个任务")
    for task in task_list.tasks:
        print(f"  - {task.description}")
        print(f"    最佳实践: {task.best_practices}")
        print(f"    工具说明: {task.tool_usage_guides}")
        print(f"    注意事项: {task.precautions}")
```

### API文档

#### StreamingTaskCreator

```python
class StreamingTaskCreator:
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        temperature: float = 0.7
    )

    def create_tasks_for_profiles(
        self,
        profiles: List[AgentProfile],
        user_requirement: str,
        context: Optional[Dict[str, Any]] = None,
        json_scene_data: Optional[List[Dict[str, Any]]] = None,
        num_tasks_per_agent: int = 3,
        save_files: bool = True,
    ) -> Dict[str, TaskList]
```

**参数说明**:
- `profiles`: 要处理的agent配置列表
- `user_requirement`: 用户需求描述，用于指导任务生成
- `context`: 额外上下文信息（可选）
- `json_scene_data`: 场景分析数据（可选），用于增强任务相关性
- `num_tasks_per_agent`: 每个agent生成的任务数量（默认3个）
- `save_files`: 是否自动保存结果到JSON文件（默认True）

**返回值**:
- `Dict[str, TaskList]`: 每个agent_id对应的任务列表

### 输出格式

生成的任务文件保存在 `task/output/` 目录，每个agent一个JSON文件：

```
task/output/
├── pop_gt_lyric_agent_tasks.json
├── pop_audio_type_agent_tasks.json
├── pop_description_agent_tasks.json
└── pop_idx_agent_tasks.json
```

每个JSON文件的结构：

```json
{
  "agent_id": "pop_gt_lyric_agent",
  "tasks": [
    {
      "task_id": "task_001",
      "description": "任务描述...",
      "best_practices": ["实践1", "实践2", "实践3"],
      "tool_usage_guides": [
        {
          "tool_name": "pop_gt_lyric",
          "usage_purpose": "生成歌词",
          "dependencies": [],
          "execution_order": 1,
          "parameters_guide": { ... }
        }
      ],
      "precautions": [
        {
          "category": "格式校验",
          "rule_description": "歌词必须符合结构规范",
          "validation_method": "检查格式",
          "error_handling": "重新生成"
        }
      ],
      "required_tools": ["pop_gt_lyric"],
      "required_knowledge": ["音乐创作"],
      "priority": 5
    }
  ],
  "metadata": {
    "generated_at": "2026-01-07T16:00:00",
    "num_tasks": 1
  }
}
```

### 流式输出示例

执行任务生成时，会实时显示LLM的生成过程：

```
================================================================================
🚀 开始流式生成任务列表
================================================================================
📋 准备处理 4 个agent配置
📝 用户需求: 我需要为一个温馨家居场景生成音乐...
================================================================================

[1/4] 正在处理: 流行音乐歌词生成专家
─────────────────────────────────────────────────────────────────────────────
🤖 生成中: 流行音乐歌词生成专家
─────────────────────────────────────────────────────────────────────────────

{
  "tasks": [
    {
      "task_id": "task_001",
      "description": "为温馨家居场景生成中文流行歌词，融合女性视角和宠物互动主题...",
      ...
    }
  ]
}

✅ 已生成 3 个任务并保存
💾 保存路径: task/output/pop_gt_lyric_agent_tasks.json

================================================================================
📊 任务生成完成
================================================================================
✅ 流行音乐歌词生成专家        3 个任务
✅ 音乐风格选择专家           3 个任务
✅ 音乐描述生成专家           3 个任务
✅ 音乐作品索引生成专家       3 个任务
================================================================================
📈 总计: 12 个任务，跨 4 个agent
💾 输出目录: D:\...\task\output\
================================================================================
```

### 运行示例

项目还提供了完整的使用示例函数：

```python
from task.examples import example_4_streaming_task_creator, run_streaming_example

# 方式1：直接调用示例函数
example_4_streaming_task_creator()

# 方式2：使用快捷函数
run_streaming_example()
```

### 与TaskGenerator的区别

| 特性 | TaskGenerator | StreamingTaskCreator |
|------|---------------|----------------------|
| 输入 | 单个agent配置 | 多个agent配置列表 |
| LLM方式 | invoke()（阻塞） | stream()（流式） |
| 显示方式 | 无实时显示 | 实时Token显示 |
| 文件保存 | 手动保存 | 自动保存 |
| 使用场景 | 单agent任务生成 | 批量多agent处理 |

### 错误处理

StreamingTaskCreator具有健壮的错误处理：

- **LLM API错误**: 显示错误信息并继续处理下一个agent
- **JSON解析错误**: 记录日志，继续处理
- **文件写入错误**: 显示错误，继续处理其他agent

```python
try:
    # 生成任务
    results = creator.create_tasks_for_profiles(...)
except Exception as e:
    print(f"生成失败: {e}")
```

### 高级用法

#### 自定义LLM实例

```python
from langchain_openai import ChatOpenAI

custom_llm = ChatOpenAI(
    model="qwen-max",
    api_key="your-api-key",
    temperature=0.5
)

creator = StreamingTaskCreator(llm=custom_llm)
```

#### 提供场景数据

```python
# 从scene.jsonl读取场景数据
json_scene_data = [
    {"关键帧": "0s", "背景": "室内", "主体": "女性", "主体心情": "开心"},
    {"关键帧": "5s", "背景": "客厅", "主体": "宠物狗", "主体心情": "活泼"},
]

results = creator.create_tasks_for_profiles(
    profiles=profiles,
    user_requirement=user_requirement,
    json_scene_data=json_scene_data,
    num_tasks_per_agent=3
)
```

## 贡献

欢迎贡献新的智能体配置或改进任务生成逻辑。

