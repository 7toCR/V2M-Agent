"""
Async Task Creator Module

单次异步LLM调用为多个agent_profile生成任务列表
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage

from .task_profile import AgentProfile
from tools.tools import parse_tasks_from_json, extract_tasks_array_from_response

logger = logging.getLogger(__name__)


class AsyncTaskCreator:
    """
    异步任务生成器，支持一次性为多个agent_profile生成任务

    特点：
    - 单次异步LLM调用
    - 所有agents共享同一份scene数据
    - 返回格式化的任务字符串列表
    - 完全异步操作
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None, temperature: float = 0.7):
        """
        初始化异步任务生成器

        Args:
            llm: 语言模型实例，如果为None则创建默认实例
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

        # 创建输出目录
        self.output_dir = os.path.join(CURRENT_DIR, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    async def create_tasks_for_all_agents(
        self,
        profiles: List[AgentProfile],
        user_requirement: str,
        json_scene_data: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        num_tasks_per_agent: int = 3,
        save_to_file: bool = True,
    ) -> List[str]:
        """
        单次异步LLM调用为所有agents生成任务

        Args:
            profiles: agent配置列表
            user_requirement: 用户需求
            json_scene_data: 场景数据（所有agents共享）
            context: 额外上下文信息
            num_tasks_per_agent: 每个agent生成的任务数
            save_to_file: 是否保存结果到文件

        Returns:
            List[str] - 格式化的任务字符串列表
        """
        print("\n" + "=" * 80)
        print("[*] 开始生成任务（异步模式）")
        print("=" * 80)
        print(f"[-] 将处理 {len(profiles)} 个agent配置")
        print(f"[-] 场景数据帧数: {len(json_scene_data)}")
        print(f"[-] 每个agent生成 {num_tasks_per_agent} 个任务")
        print("=" * 80 + "\n")

        # 构建综合提示词
        prompt = self._build_combined_prompt(
            profiles=profiles,
            user_requirement=user_requirement,
            json_scene_data=json_scene_data,
            context=context,
            num_tasks_per_agent=num_tasks_per_agent,
        )

        # 构建消息列表
        messages = [
            SystemMessage(
                content="你是一个专业的任务规划专家，为多个智能体生成具体的任务列表。"
            ),
            HumanMessage(content=prompt),
        ]

        # 异步调用LLM（一次性调用，产生所有agents的任务）
        print("[...] 正在调用LLM生成任务...")
        response_content = await self._call_llm_async(messages)
        print("[v] LLM调用完成\n")

        # 解析响应为任务
        task_strings = self._parse_response_to_tasks(response_content, profiles)
        print(f"[v] 解析完成: {len(task_strings)} 个任务\n")

        # 保存结果
        if save_to_file:
            await self._save_results(task_strings, profiles)

        # 打印总结
        self._print_summary(len(profiles), len(task_strings))

        return task_strings

    def _build_combined_prompt(
        self,
        profiles: List[AgentProfile],
        user_requirement: str,
        json_scene_data: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        num_tasks_per_agent: int,
    ) -> str:
        """
        为所有agents构建综合提示词

        Args:
            profiles: agent配置列表
            user_requirement: 用户需求
            json_scene_data: 场景数据
            context: 额外上下文
            num_tasks_per_agent: 每个agent的任务数

        Returns:
            完整的提示词字符串
        """
        prompt_lines = ["## 用户需求和场景\n"]

        # 合并user_requirement和scene数据
        prompt_lines.append(f"用户需求:\n{user_requirement}\n")

        if json_scene_data:
            prompt_lines.append("场景数据:")
            for i, frame in enumerate(json_scene_data):
                prompt_lines.append(f"  关键帧 {i}: {json.dumps(frame, ensure_ascii=False)}")
            prompt_lines.append("")

        if context:
            prompt_lines.append("额外上下文:")
            for key, value in context.items():
                prompt_lines.append(f"  {key}: {value}")
            prompt_lines.append("")

        # 添加所有agent配置
        prompt_lines.append("\n## 需要为以下智能体生成任务\n")

        for i, profile in enumerate(profiles, 1):
            prompt_lines.append(f"### {i}. {profile.role.name} ({profile.agent_id})")
            prompt_lines.append(f"描述: {profile.role.description}\n")

            prompt_lines.append("职责:")
            for resp in profile.role.responsibilities:
                prompt_lines.append(f"  - {resp}")
            prompt_lines.append("")

            prompt_lines.append("专业领域:")
            for exp in profile.role.expertise:
                prompt_lines.append(f"  - {exp}")
            prompt_lines.append("")

            if profile.tools:
                prompt_lines.append("可用工具:")
                for tool in profile.tools:
                    prompt_lines.append(f"  - {tool.name}: {tool.description}")
                prompt_lines.append("")

            if profile.knowledge:
                prompt_lines.append("知识库:")
                for knowledge in profile.knowledge:
                    prompt_lines.append(f"  - {knowledge.domain.value}")
                    if knowledge.concepts:
                        for concept in knowledge.concepts:
                            prompt_lines.append(f"    * {concept}")
                prompt_lines.append("")

            if profile.constraints:
                prompt_lines.append("约束条件:")
                for constraint in profile.constraints:
                    prompt_lines.append(f"  - {constraint}")
                prompt_lines.append("")

            if profile.best_practices:
                prompt_lines.append("最佳实践:")
                for practice in profile.best_practices:
                    prompt_lines.append(f"  - {practice}")
                prompt_lines.append("")

            prompt_lines.append("-" * 80 + "\n")

        # 任务生成要求
        prompt_lines.append("\n## 任务生成要求\n")
        prompt_lines.append(f"""
1. 为每个智能体生成 {num_tasks_per_agent} 个具体的、可执行的任务
2. 所有智能体共享上述场景数据和用户需求
3. 任务必须贴近各智能体的角色定位和职责
4. 最佳实践、工具使用说明、注意事项必须严格基于对应AgentProfile中的内容，不得凭空创造
5. 生成的任务之间可以有依赖关系
6. 确保任务清晰明确

## 输出格式

请返回一个JSON数组，每个元素是一个任务对象，结构如下：

```json
[
  {{
    "task_id": "task_001",
    "description": "任务描述，清晰具体",
    "best_practices": [
      "实践1（必须来自相应AgentProfile.best_practices）",
      "实践2",
      "实践3"
    ],
    "tool_usage_guides": [
      {{
        "tool_name": "工具名称",
        "usage_purpose": "使用目的",
        "dependencies": ["依赖工具1", "依赖工具2"],
        "parameters_guide": {{
          "参数名1": "参数说明",
          "参数名2": "参数说明"
        }}
      }}
    ],
    "precautions": [
      {{
        "category": "类别（格式校验/约束条件等）",
        "rule_description": "规则描述（必须来自knowledge.rules或constraints）",
        "validation_method": "校验方法",
        "error_handling": "错误处理方式"
      }}
    ],
    "required_tools": ["工具1", "工具2"],
    "required_knowledge": ["知识领域1"],
    "priority": 5
  }},
  ...更多任务...
]
```

请确保JSON格式正确，能够被正确解析。
""")

        return "\n".join(prompt_lines)

    async def _call_llm_async(self, messages: List[Any]) -> str:
        """
        异步调用LLM一次性生成所有agents的任务

        Args:
            messages: LLM消息列表

        Returns:
            LLM响应内容字符串
        """
        # 使用ainvoke进行异步调用
        response = await self.llm.ainvoke(messages)
        return response.content

    def _parse_response_to_tasks(
        self,
        response_content: str,
        profiles: List[AgentProfile],
    ) -> List[str]:
        """
        解析LLM响应并转换为格式化的任务字符串列表

        Args:
            response_content: LLM响应内容
            profiles: agent配置列表

        Returns:
            List[str] - 格式化的任务字符串
        """
        # 从响应中提取tasks数组
        tasks_data = extract_tasks_array_from_response(response_content)

        if not tasks_data:
            logger.warning("Failed to extract tasks from LLM response")
            return []

        # 转换为格式化字符串列表
        task_strings = parse_tasks_from_json({"tasks": tasks_data})

        return task_strings

    async def _save_results(
        self,
        task_strings: List[str],
        profiles: List[AgentProfile],
    ) -> str:
        """
        保存结果到文件

        Args:
            task_strings: 格式化的任务字符串列表
            profiles: agent配置列表

        Returns:
            保存的文件路径
        """
        # 生成文件名（包含所有agent ID）
        agent_ids = "_".join([p.agent_id[:10] for p in profiles])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tasks_{agent_ids}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)

        # 保存为文本文件（便于直接查看）
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"任务生成时间: {datetime.now().isoformat()}\n")
            f.write(f"处理的Agents: {', '.join([p.agent_id for p in profiles])}\n")
            f.write(f"总任务数: {len(task_strings)}\n")
            f.write("=" * 80 + "\n\n")
            f.write("\n".join(task_strings))

        logger.info(f"Saved task results to {filepath}")
        print(f"[v] 结果已保存: {filepath}")

        return filepath

    def _print_summary(
        self,
        num_agents: int,
        num_tasks: int,
    ) -> None:
        """
        打印最终的任务生成总结

        Args:
            num_agents: agent数量
            num_tasks: 生成的任务数
        """
        print("\n" + "=" * 80)
        print("[=] 任务生成完成")
        print("=" * 80)
        print(f"[v] 处理Agents: {num_agents} 个")
        print(f"[v] 生成任务: {num_tasks} 个")
        print(f"[-] 输出目录: {self.output_dir}")
        print("=" * 80 + "\n")


async def main():
    """
    命令行入口点示例

    演示如何使用AsyncTaskCreator处理多个agent_profile
    """
    from task.task_profile import (
        EXAMPLE_POP_GT_LYRIC_PROFILE,
        EXAMPLE_POP_AUDIO_TYPE_PROFILE,
    )
    from task.examples import POP_DESCRIPTION_PROFILE, POP_IDX_PROFILE

    # 创建生成器实例
    creator = AsyncTaskCreator()

    # 准备agent配置列表
    profiles = [
        EXAMPLE_POP_GT_LYRIC_PROFILE,
        EXAMPLE_POP_AUDIO_TYPE_PROFILE,
        POP_DESCRIPTION_PROFILE,
        POP_IDX_PROFILE,
    ]

    # 准备场景数据（所有agents共享）
    json_scene_data = [
        {
            "关键帧": "0s",
            "背景": "温馨家庭客厅",
            "背景风格": "温暖舒适",
            "主体": "年轻女性和宠物狗",
            "主体心情": "开心、放松",
        },
        {
            "关键帧": "5s",
            "背景": "阳光透过窗户洒入",
            "背景风格": "明亮温暖",
            "主体": "互动玩耍",
            "主体心情": "幸福、亲密",
        },
    ]

    # 用户需求
    user_requirement = """
    我需要为一个温馨的家庭场景生成完整的音乐生成流程。

    场景描述：
    年轻女性与她的宠物狗在客厅里温馨互动，阳光透过窗户洒入房间，
    氛围非常舒适放松。这是一个充满爱意和幸福感的时刻。

    任务目标：
    1. 生成符合场景的中文流行歌词，融合女性视角和宠物互动主题
    2. 选择最合适的音乐风格和音色
    3. 生成详细的音乐描述参数
    4. 生成对应的音乐作品索引
    """

    # 执行任务生成（一次性异步调用）
    task_strings = await creator.create_tasks_for_all_agents(
        profiles=profiles,
        user_requirement=user_requirement,
        json_scene_data=json_scene_data,
        num_tasks_per_agent=1,
        save_to_file=True,
    )

    # 显示生成的任务
    print("\n" + "=" * 80)
    print("生成的任务列表")
    print("=" * 80 + "\n")
    for i, task_str in enumerate(task_strings, 1):
        print(f"[任务 {i}]")
        print(task_str)
        print()

    print(f"\n[v] 总共生成 {len(task_strings)} 个任务！")


if __name__ == "__main__":
    asyncio.run(main())
