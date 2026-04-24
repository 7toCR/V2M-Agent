from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SystemPromptBody(BaseModel):
    """An object that contains the basic directives for the AI prompt.

    Attributes:
        constraints (list): A list of constraints that the AI should adhere to.
        resources (list): A list of resources that the AI can utilize.
        best_practices (list): A list of best practices that the AI should follow.
    """

    resources: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    best_practices: list[str] = Field(default_factory=list)

    def __add__(self, other: SystemPromptBody) -> SystemPromptBody:
        return SystemPromptBody(
            resources=self.resources + other.resources,
            constraints=self.constraints + other.constraints,
            best_practices=self.best_practices + other.best_practices,
        ).model_copy(deep=True)



# ============================================================
# 新增：动态约束生成器
# 基于论文 Algorithm 1: Task Prompt Generation based on Agent Profile
# GenerateConstraints(H, P_i) - 基于用户需求和 Agent Profile 动态生成约束
# ============================================================

class DynamicConstraintsGenerator:
    """动态约束生成器
    
    基于论文 Algorithm 1 的 GenerateConstraints(H, P_i) 机制
    根据用户需求 H 和 Agent Profile P_i 动态生成约束
    """
    
    # 基础约束模板
    BASE_CONSTRAINTS = [
        "必须严格遵循JSON格式要求",
        "所有字段值必须为字符串类型",
        "不得遗漏任何必需字段",
        "时间标签必须按顺序排列",
    ]
    
    # 音乐相关约束
    MUSIC_CONSTRAINTS = [
        "歌词必须与场景情感保持一致",
        "音频类型必须从指定列表中选择",
        "BPM值必须在合理范围内(60-200)",
        "音乐风格必须与场景氛围匹配",
    ]
    
    # 多模态相关约束
    MULTIMODAL_CONSTRAINTS = [
        "不同模态的信息必须保持一致性",
        "时间戳必须与内容对应",
        "主体描述必须与视频/图片一致",
        "声音风格必须与场景情绪匹配",
    ]
    
    @classmethod
    def generate_constraints(cls, user_requirement: str, agent_profile) -> list[str]:
        """根据用户需求和 Agent Profile 动态生成约束
        
        Args:
            user_requirement: 用户需求描述
            agent_profile: Agent Profile 对象
            
        Returns:
            list[str]: 动态生成的约束列表
        """
        constraints = list(cls.BASE_CONSTRAINTS)
        
        # 根据用户需求中的关键词添加特定约束
        user_lower = user_requirement.lower()
        
        # 音乐相关需求
        if any(kw in user_lower for kw in ["音乐", "歌曲", "prompt", "audio", "music"]):
            constraints.extend(cls.MUSIC_CONSTRAINTS)
        
        # 多模态理解需求
        if any(kw in user_lower for kw in ["视频", "音频", "图片", "text", "video", "audio", "photo", "image"]):
            constraints.extend(cls.MULTIMODAL_CONSTRAINTS)
        
        # 添加 Agent Profile 中的特定约束
        if agent_profile and hasattr(agent_profile, 'constraints'):
            # 保留原始约束，添加动态约束
            original_constraints = agent_profile.constraints
            if isinstance(original_constraints, list):
                constraints.extend(original_constraints)
        
        # 去重
        seen = set()
        unique_constraints = []
        for c in constraints:
            if c not in seen:
                seen.add(c)
                unique_constraints.append(c)
        
        return unique_constraints
    
    @classmethod
    def generate_resources(cls, user_requirement: str, agent_profile) -> list[str]:
        """根据用户需求和 Agent Profile 动态生成资源列表
        
        Args:
            user_requirement: 用户需求描述
            agent_profile: Agent Profile 对象
            
        Returns:
            list[str]: 动态生成的资源列表
        """
        resources = []
        
        # 根据模态添加资源
        user_lower = user_requirement.lower()
        
        if any(kw in user_lower for kw in ["视频", "video"]):
            resources.append("视频分析模型 (Qwen3-VL-Plus)")
        
        if any(kw in user_lower for kw in ["音频", "audio"]):
            resources.append("音频分析模型 (Qwen3-Omni-Flash)")
        
        if any(kw in user_lower for kw in ["图片", "photo", "image"]):
            resources.append("图像分析模型")
        
        if any(kw in user_lower for kw in ["文本", "text"]):
            resources.append("文本分析模型")
        
        if any(kw in user_lower for kw in ["音乐", "歌曲", "audio", "music"]):
            resources.append("音乐生成模型 (SongGeneration/NotaGen)")
            resources.append("音乐风格分类器")
        
        # 添加 Agent Profile 中的原始资源
        if agent_profile and hasattr(agent_profile, 'resources'):
            original_resources = agent_profile.resources
            if isinstance(original_resources, list):
                resources.extend(original_resources)
        
        # 去重
        seen = set()
        unique_resources = []
        for r in resources:
            if r not in seen:
                seen.add(r)
                unique_resources.append(r)
        
        return unique_resources
    
    @classmethod
    def determine_needed_modalities(cls, user_requirement: str) -> list[str]:
        """根据用户需求确定需要的模态
        
        Args:
            user_requirement: 用户需求描述
            
        Returns:
            list[str]: 需要处理的模态列表
        """
        modalities = []
        user_lower = user_requirement.lower()
        
        if any(kw in user_lower for kw in ["文本", "text"]):
            modalities.append("text")
        if any(kw in user_lower for kw in ["音频", "audio"]):
            modalities.append("audio")
        if any(kw in user_lower for kw in ["图片", "photo", "image"]):
            modalities.append("photo")
        if any(kw in user_lower for kw in ["视频", "video"]):
            modalities.append("video")
        
        return modalities if modalities else ["text"]  # 默认至少需要文本


