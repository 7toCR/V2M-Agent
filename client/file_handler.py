"""
文件处理工具 - 用于读取和保存文件
"""

import os
from typing import Optional


def read_lyric_file(file_path: str) -> Optional[str]:
    """
    读取lyric.jsonl文件内容
    
    Args:
        file_path: lyric.jsonl文件的路径
        
    Returns:
        文件内容字符串，如果文件不存在或读取失败则返回None
    """
    try:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 检查文件内容是否为空
        if not content:
            print(f"⚠️ 警告: 文件内容为空: {file_path}")
            print(f"   可能所有音频文件都已存在，无需生成")
            return None
        
        print(f"✅ 成功读取文件: {file_path} (内容长度: {len(content)} 字符)")
        return content
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None


def save_audio_file(file_path: str, content: bytes) -> bool:
    """
    保存下载的音频文件
    
    Args:
        file_path: 保存文件的完整路径
        content: 文件内容（字节）
        
    Returns:
        是否保存成功
    """
    try:
        # 确保目录存在
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        print(f"✅ 成功保存音频文件: {file_path}")
        return True
    except Exception as e:
        print(f"❌ 保存音频文件失败: {e}")
        return False


def get_output_dir(file_path: str) -> str:
    """
    从文件路径提取输出目录
    
    Args:
        file_path: 文件路径，例如 "sample/output/shanghai/lyric.jsonl"
        
    Returns:
        输出目录路径，例如 "sample/output/shanghai"
    """
    return os.path.dirname(os.path.abspath(file_path))


def ensure_dir_exists(dir_path: str) -> bool:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        dir_path: 目录路径
        
    Returns:
        是否成功
    """
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ 创建目录失败: {e}")
        return False

