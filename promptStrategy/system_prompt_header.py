from pydantic import BaseModel, Field
from typing import List

DEFAULT_Agent_NAME = "POPMusic-helper"
DEFAULT_Agent_ROLE = (
    "一位专注于流行音乐创作的数字助手："
    "具备深厚的音乐理论知识，熟悉各类音乐风格和制作流程。"
    "我擅长为AI音乐生成工具构建高质量的提示词，能够精准描述旋律、和声、节奏和情感。"
    "我对流行音乐趋势有敏锐洞察力，能根据需求创作出符合时代潮流和市场偏好的音乐提示词。"
    "我注重实用性和创造性，善于利用现有工具和技术解决音乐创作中的问题。"
)

DEFAULT_AGENT_GOALS = [
    "生成唯一的音乐作品ID，确保每个作品都有唯一的标识符",
    "根据给定的主题或情感创作音乐作品歌词，确保歌词与主题相关且有感染力",
    "为音乐作品编写详细描述，包括创作背景、情感表达、音乐风格等信息",
    "确定音乐作品的类型和风格（如流行、摇滚、电子、R&B等）",
    "将音乐作品的完整信息（包括ID、歌词、描述、类型等）整理为JSONL格式文件",
    "利用生成的JSONL文件作为输入，创建或触发音乐生成过程，最终产出完整的音乐作品"
]


class SystemPromptHeader(BaseModel):
    """
    用于定义Agent助手的个性配置文件。

    属性：
        agent_name (str): Agent助手的名称
        agent_role (str): Agent助手的角色描述
        agent_goals (list): Agent需要完成的目标列表
    """

    agent_name: str = DEFAULT_Agent_NAME
    agent_role: str = DEFAULT_Agent_ROLE
    """agent_role应该符合以下格式：我是{agent_name}, {agent_role}"""
    agent_goals: List[str] = Field(default_factory=lambda: DEFAULT_AGENT_GOALS.copy())


# 示例使用
if __name__ == "__main__":
    # 使用默认配置创建Profile
    header = SystemPromptHeader()

    print(f"Agent名称: {header.agent_name}")
    print(f"Agent角色: {header.agent_role}")
    print(f"Agent目标列表:")
    for i, goal in enumerate(header.agent_goals, 1):
        print(f"  {i}. {goal}")

    # 自定义配置示例
    custom_header = SystemPromptHeader(
        agent_name="定制音乐构建师",
        agent_role="我是定制音乐构建师，专注于个性化音乐创作，能够根据用户的具体需求创作独特的音乐作品。",
        agent_goals=[
            "为特定活动创作主题音乐",
            "根据用户描述的情感生成音乐",
            "创建品牌专属的音乐标识"
        ]
    )

    print(f"Agent名称: {custom_header.agent_name}")
    print(f"Agent角色: {custom_header.agent_role}")
    print(f"Agent目标列表:")
    for i, goal in enumerate(custom_header.agent_goals, 1):
        print(f"  {i}. {goal}")