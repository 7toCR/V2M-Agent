"""
Scene Analyzer Module

负责从 json_scene 结构中提取结构化的场景分析信息，为任务生成提供输入。
"""

from typing import List, Dict, Any, Optional, Set

from .task_profile import AgentProfile, KnowledgeDomain
from .task_generator import SceneAnalysis


def extract_emotion_keywords(text: str) -> Set[str]:
    """
    从描述文本中提取情感关键词。

    为了满足“不得凭空生成”的要求，这里仅做极简分词：
    - 以常见标点拆分
    - 去除空白
    - 返回原始片段集合，而不引入外部情感词典
    """
    if not text:
        return set()

    # 使用最简单的基于标点的切分，完全基于原始文本
    separators = ["，", "。", "、", ",", ".", "；", ";", "！", "?", "？", " "]
    tmp = [text]
    for sep in separators:
        parts: List[str] = []
        for seg in tmp:
            parts.extend(seg.split(sep))
        tmp = parts

    return {seg.strip() for seg in tmp if seg.strip()}


def extract_subject_keywords(text: str) -> Set[str]:
    """
    从主体描述中提取主体关键词。

    同样不引入任何外部知识，仅通过标点/空白切分并保留原始片段。
    """
    if not text:
        return set()

    separators = ["，", "。", "、", ",", ".", "和", "与", " ", "；", ";", "及"]
    tmp = [text]
    for sep in separators:
        parts: List[str] = []
        for seg in tmp:
            parts.extend(seg.split(sep))
        tmp = parts

    return {seg.strip() for seg in tmp if seg.strip()}


def recommend_styles(emotions: Set[str], music_knowledge: List[Any]) -> List[str]:
    """
    基于 AgentKnowledge.concepts / rules 中已有的文字，推荐音乐风格。

    约束：
    - 仅从 knowledge.concepts / knowledge.rules 里“原样抽取”可能的风格名字
    - 不引入任何未在知识中出现过的新风格
    """
    if not music_knowledge:
        return []

    styles: List[str] = []

    def extract_style_token(line: str) -> Optional[str]:
        """
        从一行概念/规则中抽取潜在风格标记：
        - 取第一个空格/中文冒号/英文冒号之前的片段
        - 仅在该片段为非空时返回
        """
        if not line:
            return None
        for sep in ["：", ":", " "]:
            if sep in line:
                token = line.split(sep, 1)[0].strip()
                return token or None
        return line.strip() or None

    seen: Set[str] = set()
    for k in music_knowledge:
        # 从 concepts 中抽取
        for c in getattr(k, "concepts", []):
            token = extract_style_token(c)
            if token and token not in seen:
                seen.add(token)
                styles.append(token)
        # 从 rules 中抽取（有些 profile 在规则里会枚举风格）
        for r in getattr(k, "rules", []):
            token = extract_style_token(r)
            if token and token not in seen:
                seen.add(token)
                styles.append(token)

    # 与 emotions 做一个简单的相关性过滤：这里不做额外推理，只是保留全部 styles
    # 这样可以确保完全来源于 AgentProfile.knowledge
    return styles


def recommend_language(json_scene: List[Dict[str, Any]]) -> str:
    """
    基于 json_scene 内容推荐歌词语言。

    约束：
    - 仅基于 json_scene 文本本身（是否含有中文/英文字符）做启发式判断
    - 不引入外部知识
    """
    has_chinese = False
    has_alpha = False

    for frame in json_scene:
        for key in ["主体", "主体心情", "背景"]:
            val = frame.get(key, "")
            if not isinstance(val, str):
                continue
            for ch in val:
                if "\u4e00" <= ch <= "\u9fff":
                    has_chinese = True
                if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
                    has_alpha = True

    if has_chinese and not has_alpha:
        return "中文"
    if has_alpha and not has_chinese:
        return "英文"
    # 同时存在或都不存在时，返回 Auto
    return "Auto"


class SceneAnalyzer:
    """场景分析器，从json_scene提取信息"""

    @staticmethod
    def extract_scene_analysis(
        json_scene: Optional[List[Dict[str, Any]]],
        profile: AgentProfile,
    ) -> Optional[SceneAnalysis]:
        """
        从json_scene提取场景分析

        提取逻辑：
        1. emotional_tone: 从所有帧的"主体心情"字段提取情感关键词
        2. key_subjects: 从"主体"字段提取主体对象
        3. key_moments: 保存每个关键帧的时间和心情
        4. background_info: 取第一帧的"背景"字段
        5. music_style_recommendation: 基于profile.knowledge推荐
        6. language_recommendation: 基于情感和场景内容判断
        """
        if not json_scene:
            return None

        scene_analysis = SceneAnalysis()

        # 1. 提取情感标签
        emotions: Set[str] = set()
        for frame in json_scene:
            mood = frame.get("主体心情", "")
            emotions.update(extract_emotion_keywords(mood))
        scene_analysis.emotional_tone = list(emotions)

        # 2. 提取主体
        subjects: Set[str] = set()
        for frame in json_scene:
            subject = frame.get("主体", "")
            subjects.update(extract_subject_keywords(subject))
        scene_analysis.key_subjects = list(subjects)

        # 3. 提取关键时刻（最多5个）
        scene_analysis.key_moments = [
            {
                "时间": frame.get("关键帧", ""),
                "描述": frame.get("主体心情", ""),
            }
            for frame in json_scene[:5]
        ]

        # 4. 背景信息
        if json_scene:
            first = json_scene[0]
            bg = first.get("背景", "")
            if isinstance(bg, str):
                scene_analysis.background_info = bg

        # 5. 音乐风格推荐（基于profile.knowledge）
        music_knowledge = profile.get_knowledge_by_domain(KnowledgeDomain.MUSIC)
        if music_knowledge:
            scene_analysis.music_style_recommendation = recommend_styles(
                emotions, music_knowledge
            )

        # 6. 语言推荐
        scene_analysis.language_recommendation = recommend_language(json_scene)

        return scene_analysis


