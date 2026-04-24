"""
推理执行器 - 执行推理命令并管理结果文件
"""

import os
import sys
import subprocess
import time
import glob
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# NOTE: This repo may be run on Windows (GBK console) and Linux containers.
# Some consoles crash on emoji output. Use safe_print for user-facing logs.
def safe_print(message: str):
    try:
        print(message)
    except UnicodeEncodeError:
        # Best-effort: strip unsupported characters (e.g., emoji) for GBK consoles.
        try:
            print(message.encode("ascii", "ignore").decode("ascii"))
        except Exception:
            # Last resort
            print(str(message))

# 支持多种导入方式
try:
    # 方式1: 绝对导入（优先）
    from server.task_manager import InferenceTask, TaskStatus
except ImportError:
    try:
        # 方式2: 相对导入（当作为模块导入时）
        from .task_manager import InferenceTask, TaskStatus
    except ImportError:
        # 方式3: 直接导入（当在server目录中直接运行时）
        from task_manager import InferenceTask, TaskStatus


# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # server目录的父目录就是项目根目录
    project_root = os.path.dirname(current_dir)
    
    # 如果当前目录包含 autodl-tmp/SongGeneration，直接使用当前工作目录
    # 这样可以处理从 autodl-tmp/SongGeneration/ 直接运行 server.py 的情况
    try:
        cwd = os.getcwd()
        # 检查当前工作目录是否包含 autodl-tmp/SongGeneration
        if "autodl-tmp" in cwd and "SongGeneration" in cwd:
            # 找到 SongGeneration 目录
            parts = cwd.split(os.sep)
            if "SongGeneration" in parts:
                songgen_idx = parts.index("SongGeneration")
                songgen_path = os.sep.join(parts[:songgen_idx + 1])
                if os.path.exists(songgen_path):
                    print(f"检测到SongGeneration目录，使用: {songgen_path}")
                    return songgen_path
    except Exception as e:
        print(f"检测工作目录时出错，使用默认路径: {e}")
    
    return project_root

# 配置路径
PROJECT_ROOT = get_project_root()
# 使用绝对路径 - 同时兼容 sample/lyric.jsonl 与 sample/lyrics.jsonl
LYRIC_FILE_PATH = os.path.join(PROJECT_ROOT, "sample", "lyrics.jsonl")
LYRIC_FILE_PATH_ALT = os.path.join(PROJECT_ROOT, "sample", "lyric.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "sample", "output")
# 音频输出目录 - 优先使用项目内的路径
AUDIO_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "sample", "output", "audios")
# 如果项目根目录包含 autodl-tmp，也尝试那个路径
if "autodl-tmp" in PROJECT_ROOT:
    alt_audio_dir = os.path.join(PROJECT_ROOT, "sample", "output", "audios")
    if os.path.exists(alt_audio_dir):
        AUDIO_OUTPUT_DIR = alt_audio_dir

# 尝试多个可能的generate.sh路径
POSSIBLE_GENERATE_SCRIPTS = [
    os.path.join(PROJECT_ROOT, "generate.sh"),  # 项目根目录
    os.path.join(PROJECT_ROOT, "scripts", "generate.sh"),  # scripts目录
    "generate.sh",  # 当前目录或PATH中
    os.path.join(os.path.expanduser("~"), "generate.sh"),  # 用户主目录
]

def find_generate_script():
    """查找generate.sh脚本"""
    for script_path in POSSIBLE_GENERATE_SCRIPTS:
        if os.path.exists(script_path) and os.path.isfile(script_path):
            abs_path = os.path.abspath(script_path)
            safe_print(f"[OK] 找到generate.sh脚本: {abs_path}")
            return abs_path
    # 如果都找不到，返回第一个（可能是相对路径，会在运行时检查）
    # NOTE: Avoid emoji here; Windows GBK console may crash on import.
    safe_print(f"[WARN] 未找到generate.sh脚本，将尝试使用: {POSSIBLE_GENERATE_SCRIPTS[0]}")
    return POSSIBLE_GENERATE_SCRIPTS[0]

GENERATE_SCRIPT = find_generate_script()


@dataclass(frozen=True)
class CommandResult:
    cmd: List[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str

    def short_error(self, max_chars: int = 4000) -> str:
        """
        Build a human-friendly error message for propagating to client UI.
        """
        out = (self.stdout or "").strip()
        err = (self.stderr or "").strip()
        combined = "\n".join([x for x in [out, err] if x])
        if not combined:
            combined = "(no stdout/stderr captured)"
        if len(combined) > max_chars:
            combined = combined[: max_chars - 20] + "\n...(truncated)"

        hints = []
        # 137 is commonly OOM-kill / SIGKILL in containers
        if self.returncode in (137, 9) or "Killed" in combined:
            hints.append("Hint: exit=137/\"Killed\" usually means OOM (out-of-memory) or the process was SIGKILL'ed by the system.")
            hints.append("Try: enable --low_mem and/or disable flash-attn; reduce concurrent jobs; ensure enough GPU/CPU RAM.")

        hint_text = ("\n" + "\n".join(hints)) if hints else ""
        return (
            f"Command failed (exit={self.returncode}).\n"
            f"cmd: {' '.join(self.cmd)}\n"
            f"cwd: {self.cwd}\n"
            f"---- output ----\n{combined}{hint_text}"
        )


def ensure_dir_exists(dir_path: str):
    """确保目录存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def write_lyric_file(content: str, file_path: str = None) -> bool:
    """
    将内容写入lyric文件
    
    Args:
        content: 文件内容
        file_path: 文件路径（如果为None或相对路径，会使用LYRIC_FILE_PATH或转换为绝对路径）
        
    Returns:
        是否成功
    """
    try:
        # 如果没有提供路径，使用默认路径
        if file_path is None:
            file_path = LYRIC_FILE_PATH
        
        # 确保使用绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.join(PROJECT_ROOT, file_path)
        file_path = os.path.abspath(file_path)
        
        # 确保目录存在
        dir_path = os.path.dirname(file_path)
        ensure_dir_exists(dir_path)
        
        # 如果文件已存在，先删除以确保清空原有内容
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🗑️ 已删除旧文件: {file_path}")
            except Exception as e:
                print(f"⚠️ 删除旧文件失败: {e}")
        
        # 写入文件（'w' 模式会自动创建新文件）
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 写入文件: {file_path}（已清空原有内容）")
        return True
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def clear_output_dir(output_dir: str = OUTPUT_DIR) -> bool:
    """清空输出目录 sample/output（保留目录本身）"""
    try:
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(PROJECT_ROOT, output_dir)
        output_dir = os.path.abspath(output_dir)
        ensure_dir_exists(output_dir)

        # 统计要删除的文件和目录
        file_count = 0
        dir_count = 0
        
        # 递归删除所有文件/子目录
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                try:
                    file_path = os.path.join(root, name)
                    os.remove(file_path)
                    file_count += 1
                except Exception as e:
                    print(f"⚠️ 删除文件失败: {os.path.join(root, name)} - {e}")
            for name in dirs:
                try:
                    dir_path = os.path.join(root, name)
                    os.rmdir(dir_path)
                    dir_count += 1
                except Exception as e:
                    # 可能非空或权限问题，尝试强制删除
                    try:
                        import shutil
                        shutil.rmtree(dir_path, ignore_errors=True)
                        dir_count += 1
                    except Exception:
                        print(f"⚠️ 删除目录失败: {dir_path} - {e}")
        
        if file_count > 0 or dir_count > 0:
            print(f"✅ 已清空输出目录: {output_dir}")
            print(f"   删除文件: {file_count} 个")
            print(f"   删除目录: {dir_count} 个")
        else:
            print(f"✅ 输出目录已为空: {output_dir}")
        
        return True
    except Exception as e:
        print(f"❌ 清空输出目录失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_generate_command(
    model: str,
    lyric_file: str,
    output_dir: str,
    bgm: bool = False,
    extra_args: Optional[Sequence[str]] = None,
    env_overrides: Optional[dict] = None,
) -> Tuple[bool, CommandResult]:
    """
    运行generate.sh命令
    
    Args:
        model: 模型名称，例如 "songgeneration_large"
        lyric_file: lyric文件路径（绝对路径）
        output_dir: 输出目录（绝对路径）
        bgm: 是否添加--bgm参数
        
    Returns:
        (是否成功, CommandResult)
    """
    try:
        # 重新查找脚本（在项目根目录中）
        script_found = None
        for script_path in POSSIBLE_GENERATE_SCRIPTS:
            abs_path = os.path.abspath(script_path) if not os.path.isabs(script_path) else script_path
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                script_found = abs_path
                break
        
        # 如果还是找不到，尝试在项目根目录直接查找
        if script_found is None:
            project_script = os.path.join(PROJECT_ROOT, "generate.sh")
            if os.path.exists(project_script):
                script_found = project_script
        
        if script_found is None:
            print(f"❌ generate.sh脚本不存在")
            print(f"   项目根目录: {PROJECT_ROOT}")
            print(f"   尝试查找的位置:")
            for path in POSSIBLE_GENERATE_SCRIPTS:
                abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
                exists = "✅" if os.path.exists(abs_path) else "❌"
                print(f"   {exists} {abs_path}")
            return False
        
        script_path = script_found
        
        # 确保脚本有执行权限
        if not os.access(script_path, os.X_OK):
            print(f"⚠️ generate.sh脚本没有执行权限，尝试添加...")
            try:
                os.chmod(script_path, 0o755)
            except Exception as e:
                print(f"⚠️ 无法添加执行权限: {e}")
        
        # 将绝对路径转换为相对于项目根目录的相对路径
        # 脚本期望使用相对路径：sample/lyrics.jsonl 和 sample/output
        def get_relative_path(abs_path):
            """将绝对路径转换为相对于项目根目录的相对路径"""
            try:
                rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
                return rel_path
            except ValueError:
                # 如果路径不在项目根目录下，返回原路径
                return abs_path
        
        # 转换为相对路径（脚本期望相对路径）
        rel_lyric_file = get_relative_path(lyric_file)
        rel_output_dir = get_relative_path(output_dir)
        
        # 获取脚本相对于项目根目录的相对路径
        try:
            rel_script = os.path.relpath(script_path, PROJECT_ROOT)
            # 如果相对路径包含 ..，说明不在项目目录下，使用脚本名
            if ".." in rel_script:
                rel_script = "generate.sh"
        except ValueError:
            # 如果无法计算相对路径，使用脚本名
            rel_script = "generate.sh"
        
        # 构建命令 - 使用相对路径（脚本期望的格式）
        cmd = ["bash", rel_script, model, rel_lyric_file, rel_output_dir]
        if bgm:
            cmd.append("--bgm")
        if extra_args:
            cmd.extend(list(extra_args))
        
        print(f"🚀 执行命令: {' '.join(cmd)}")
        print(f"   工作目录: {PROJECT_ROOT}")
        print(f"   脚本绝对路径: {script_path}")
        print(f"   脚本相对路径: {rel_script}")
        print(f"   lyric文件 (相对路径): {rel_lyric_file}")
        print(f"   输出目录 (相对路径): {rel_output_dir}")
        if bgm:
            print(f"   ⚙️  生成类型: BGM (背景音乐)")
        else:
            print(f"   ⚙️  生成类型: Mixed (完整音频)")
        
        # 执行命令，设置工作目录为项目根目录（脚本必须在项目根目录执行）
        child_env = os.environ.copy()
        if env_overrides:
            child_env.update({k: str(v) for k, v in env_overrides.items()})

        result = subprocess.run(
            cmd,
            check=False,  # 不自动抛出异常，我们需要检查退出码和输出
            capture_output=True,
            text=True,
            timeout=3600,  # 1小时超时
            cwd=PROJECT_ROOT,  # 在项目根目录执行
            env=child_env,
        )
        
        cmd_result = CommandResult(
            cmd=cmd,
            cwd=PROJECT_ROOT,
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

        # 检查命令是否真正成功
        if result.returncode != 0:
            print(f"❌ 命令执行失败，退出码: {result.returncode}")
            if result.stdout:
                print(f"标准输出: {result.stdout}")
            if result.stderr:
                print(f"标准错误: {result.stderr}")
            return False, cmd_result
        
        # 检查输出中是否有错误信息
        output_text = (result.stdout or "") + (result.stderr or "")
        if "CUDA is not available" in output_text and "exit" in output_text.lower():
            print(f"⚠️ 检测到CUDA不可用，推理可能未执行")
            # 即使CUDA不可用，如果命令返回成功，我们也认为执行了
            # 但需要检查是否真的生成了文件
        
        print(f"✅ 命令执行成功")
        if result.stdout:
            # 打印关键输出信息（过滤掉冗余信息）
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 0:
                # 提取关键信息：处理进度、模型加载、生成完成等
                key_lines = []
                for line in output_lines:
                    line_lower = line.lower()
                    # 保留关键信息行
                    if any(keyword in line_lower for keyword in [
                        'process', 'cost', 's', 'transformer', 'checkpoint', 
                        'loaded', 'successfully', 'generation', 'complete',
                        'saved', 'save', 'audio', 'wav'
                    ]):
                        # 过滤掉tensor显示等冗余信息
                        if 'tensor([' not in line and 'device=' not in line:
                            key_lines.append(line)
                
                if key_lines:
                    print(f"📊 关键输出信息:")
                    # 只显示最后的关键信息行（最多15行）
                    for line in key_lines[-15:]:
                        print(f"  {line}")
                else:
                    # 如果没有关键信息，显示最后几行
                    print(f"📊 输出 (最后5行):")
                    for line in output_lines[-5:]:
                        print(f"  {line}")
        
        return True, cmd_result
    except subprocess.TimeoutExpired:
        print(f"❌ 命令执行超时")
        return False, CommandResult(
            cmd=["bash", "generate.sh", model, lyric_file, output_dir] + (["--bgm"] if bgm else []),
            cwd=PROJECT_ROOT,
            returncode=124,
            stdout="",
            stderr="TimeoutExpired",
        )
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        import traceback
        traceback.print_exc()
        return False, CommandResult(
            cmd=["bash", "generate.sh", model, lyric_file, output_dir] + (["--bgm"] if bgm else []),
            cwd=PROJECT_ROOT,
            returncode=1,
            stdout="",
            stderr=str(e),
        )


def _extract_idx_from_filename(filename: str, is_bgm: bool = False) -> str:
    """
    从文件名中提取idx
    文件名格式：时间戳-idx.flac 或 时间戳-idx-bgm.flac
    """
    # 去掉扩展名
    name = os.path.splitext(filename)[0]
    
    # 如果是-bgm版本，先去掉-bgm后缀
    if is_bgm and name.endswith('-bgm'):
        name = name[:-4]  # 去掉 '-bgm'
    
    # 分割文件名
    parts = name.split('-')
    
    # 检查前5个部分是否是时间戳格式（都是数字）
    # 时间戳格式：YYYY-MM-DD-HH-MM
    if len(parts) >= 6:
        # 检查前5个部分是否都是数字（可能是时间戳）
        is_timestamp = True
        for i in range(min(5, len(parts))):
            if not parts[i].isdigit():
                is_timestamp = False
                break
        
        if is_timestamp:
            # 与客户端/lyric.jsonl 约定保持一致：idx 包含时间戳（保留完整 name）
            # 例如：2026-01-14-11-44-SeeYouNextTime -> 2026-01-14-11-44-SeeYouNextTime
            idx = name
        else:
            # 不是时间戳格式，整个name就是idx（但要去掉-bgm）
            idx = name
    else:
        # 少于6个部分，整个name就是idx
        idx = name
    
    return idx


def _filter_files_by_indices(filenames: List[str], expected_indices: List[str]) -> List[str]:
    """
    根据期望的idx列表过滤文件名
    
    Args:
        filenames: 文件名列表
        expected_indices: 期望的idx列表
        
    Returns:
        匹配的文件名列表
    """
    matched_files = []
    expected_set = set(expected_indices)
    
    for filename in filenames:
        # 检查无-bgm版本
        idx = _extract_idx_from_filename(filename, is_bgm=False)
        if idx in expected_set:
            matched_files.append(filename)
            continue
        
        # 检查-bgm版本
        if filename.endswith('-bgm.flac'):
            idx = _extract_idx_from_filename(filename, is_bgm=True)
            if idx in expected_set:
                matched_files.append(filename)
    
    return matched_files


def scan_audio_files(audio_dir: str = AUDIO_OUTPUT_DIR, max_wait: int = 30, wait_interval: float = 1.0, expected_indices: Optional[List[str]] = None) -> List[str]:
    """
    扫描音频文件目录，返回所有.flac文件
    会等待文件生成，最多等待max_wait秒
    
    Args:
        audio_dir: 音频文件目录
        max_wait: 最大等待时间（秒）
        wait_interval: 等待间隔（秒）
        expected_indices: 期望的idx列表，如果提供，只返回匹配的文件
        
    Returns:
        音频文件名列表（仅文件名，不包含路径）
    """
    audio_files = []
    
    try:
        safe_print(f"[SCAN] 开始扫描音频文件目录: {audio_dir}")
        if expected_indices:
            safe_print(f"   期望找到 {len(expected_indices)} 个idx对应的文件")
        
        # 等待目录创建或文件生成
        waited = 0
        scan_count = 0
        while waited < max_wait:
            if os.path.exists(audio_dir):
                scan_count += 1
                # 递归查找所有.flac文件（包括子目录）
                pattern = os.path.join(audio_dir, "**", "*.flac")
                files = glob.glob(pattern, recursive=True)
                
                if files:
                    # 只返回文件名（去重）
                    all_files = list(set([os.path.basename(f) for f in files]))
                    # 如果提供了expected_indices，只返回匹配的文件
                    if expected_indices:
                        audio_files = _filter_files_by_indices(all_files, expected_indices)
                        safe_print(f"[SCAN] 扫描 #{scan_count}: 找到 {len(audio_files)}/{len(all_files)} 个匹配的音频文件（期望 {len(expected_indices)} 个idx）")
                        # 诊断 + 回退：有 flac 但完全匹配不到 idx 时，避免“生成了但客户端拿不到列表”
                        if not audio_files and all_files:
                            try:
                                sample_file = all_files[0]
                                extracted = _extract_idx_from_filename(
                                    sample_file,
                                    is_bgm=sample_file.endswith("-bgm.flac"),
                                )
                                safe_print("[DIAG] 目录里存在 .flac 但按 idx 过滤后为空，可能是 idx 规则不一致")
                                safe_print(f"   expected_indices 示例: {expected_indices[:5]}")
                                safe_print(f"   all_files 示例: {all_files[:5]}")
                                safe_print(f"   sample_file: {sample_file}")
                                safe_print(f"   extracted_idx(sample_file): {extracted}")
                            except Exception:
                                pass
                            # 回退：先返回全部文件名（客户端后续仍可按文件名下载）
                            audio_files = all_files
                            safe_print(f"[WARN] 回退: 返回全部 {len(audio_files)} 个音频文件（未按 idx 过滤）")
                        if len(audio_files) < len(expected_indices):
                            # 提取已找到文件的idx
                            found_indices = set()
                            for f in audio_files:
                                if f.endswith('-bgm.flac'):
                                    found_indices.add(_extract_idx_from_filename(f, is_bgm=True))
                                else:
                                    found_indices.add(_extract_idx_from_filename(f, is_bgm=False))
                            missing = set(expected_indices) - found_indices
                            if missing:
                                safe_print(f"   [WARN] 仍缺少 {len(missing)} 个idx的文件: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
                    else:
                        audio_files = all_files
                        safe_print(f"[SCAN] 扫描 #{scan_count}: 找到 {len(audio_files)} 个音频文件")
                    
                    # 显示找到的文件位置（前几个）
                    if files and scan_count == 1:
                        safe_print("   文件位置示例:")
                        for f in files[:3]:
                            # Windows may have different drives (e.g., C: temp vs D: project),
                            # which makes commonpath/relpath raise ValueError.
                            try:
                                rel_path = os.path.relpath(f, PROJECT_ROOT) if os.path.commonpath([f, PROJECT_ROOT]) else f
                            except ValueError:
                                rel_path = f
                            safe_print(f"     - {rel_path}")
                        if len(files) > 3:
                            safe_print(f"     ... 还有 {len(files) - 3} 个文件")
                    
                    # 如果找到了期望数量的文件，提前返回
                    if expected_indices and len(audio_files) >= len(expected_indices):
                        safe_print("[OK] 已找到所有期望的文件，提前返回")
                        return audio_files
                    
                    # 如果找到了文件，再等待一下确保文件完全写入
                    if audio_files:
                        time.sleep(2)
                        return audio_files
            
            time.sleep(wait_interval)
            waited += wait_interval
        
        # 如果等待后仍然没有找到，尝试直接扫描
        if os.path.exists(audio_dir):
            safe_print("[WARN] 等待超时，进行最终扫描...")
            # 递归查找所有.flac文件
            pattern = os.path.join(audio_dir, "**", "*.flac")
            files = glob.glob(pattern, recursive=True)
            all_files = list(set([os.path.basename(f) for f in files]))
            # 如果提供了expected_indices，只返回匹配的文件
            if expected_indices:
                audio_files = _filter_files_by_indices(all_files, expected_indices)
                safe_print(f"[SCAN] 最终扫描: 找到 {len(audio_files)}/{len(all_files)} 个匹配的音频文件（期望 {len(expected_indices)} 个idx）")
            else:
                audio_files = all_files
                safe_print(f"[SCAN] 最终扫描: 找到 {len(audio_files)} 个音频文件")
        
        if not audio_files:
            safe_print(f"[WARN] 音频目录不存在或没有找到文件: {audio_dir}")
            # 尝试检查其他可能的目录
            alt_dirs = [
                os.path.join(PROJECT_ROOT, "sample", "output", "audios"),
                os.path.join(OUTPUT_DIR, "audios"),
                OUTPUT_DIR,  # 直接在output目录下
            ]
            safe_print("   尝试在备用目录中查找...")
            for alt_dir in alt_dirs:
                if os.path.exists(alt_dir) and alt_dir != audio_dir:
                    pattern = os.path.join(alt_dir, "**", "*.flac")
                    files = glob.glob(pattern, recursive=True)
                    if files:
                        all_files = list(set([os.path.basename(f) for f in files]))
                        # 如果提供了expected_indices，只返回匹配的文件
                        if expected_indices:
                            audio_files = _filter_files_by_indices(all_files, expected_indices)
                            safe_print(f"[SCAN] 在备用目录 {alt_dir} 找到 {len(audio_files)}/{len(all_files)} 个匹配的音频文件")
                        else:
                            audio_files = all_files
                            safe_print(f"[SCAN] 在备用目录 {alt_dir} 找到 {len(audio_files)} 个音频文件")
                        break
        else:
            safe_print(f"[OK] 扫描完成: 找到 {len(audio_files)} 个音频文件")
        
        return audio_files
    except Exception as e:
        safe_print(f"[ERR] 扫描音频文件失败: {e}")
        import traceback
        traceback.print_exc()
        return audio_files


def get_audio_file_path(filename: str, audio_dir: str = AUDIO_OUTPUT_DIR) -> Optional[str]:
    """
    获取音频文件的完整路径（会在多个可能的位置查找）
    
    Args:
        filename: 文件名
        audio_dir: 音频文件目录
        
    Returns:
        完整路径，如果文件不存在则返回None
    """
    # 首先在指定目录查找
    file_path = os.path.join(audio_dir, filename)
    if os.path.exists(file_path):
        return file_path
    
    # 如果找不到，尝试在其他可能的位置递归查找
    search_dirs = [
        audio_dir,
        os.path.join(PROJECT_ROOT, "sample", "output", "audios"),
        os.path.join(OUTPUT_DIR, "audios"),
        OUTPUT_DIR,
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        # 递归查找文件
        pattern = os.path.join(search_dir, "**", filename)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]  # 返回第一个匹配的文件
    
    return None


def modify_lyric_file_for_bgm(file_path: str) -> bool:
    """
    修改lyric文件，给每个json的idx字段添加-bgm后缀
    
    Args:
        file_path: lyric文件路径
        
    Returns:
        是否成功
    """
    try:
        import json
        
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 修改每一行
        modified_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # 如果idx不包含-bgm，则添加
                if 'idx' in item and not item['idx'].endswith('-bgm'):
                    item['idx'] = f"{item['idx']}-bgm"
                modified_lines.append(json.dumps(item, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"⚠️ 解析JSON行失败: {e}")
                # 保留原行
                modified_lines.append(line)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in modified_lines:
                f.write(line + '\n')
        
        print(f"✅ 已修改lyric文件，添加-bgm后缀到 {len(modified_lines)} 条记录")
        return True
    except Exception as e:
        print(f"❌ 修改lyric文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_inference_task(task: InferenceTask, lyric_content: str, generate_type: str = "both", output_dir: Optional[str] = None) -> bool:
    """
    执行推理任务
    
    Args:
        task: 推理任务对象
        lyric_content: lyric文件内容
        generate_type: 生成类型，"normal"（只生成无-bgm）、"bgm"（只生成-bgm）、"both"（两阶段都生成）
        output_dir: 输出目录（服务器相对路径，例如 "sample/output/scene_17"），如果为None则使用默认OUTPUT_DIR
        
    Returns:
        是否成功
    """
    try:
        # 设置任务状态为运行中
        task.set_status(TaskStatus.RUNNING)
        task.set_progress(0)
        task.set_phase(0)
        
        # 统一使用 sample/output 作为输出目录（无论客户端传入什么）
        actual_output_dir = OUTPUT_DIR  # 使用 sample/output
        print(f"📂 输出目录: {actual_output_dir} (统一使用 sample/output)")
        
        # 从lyric_content中提取idx列表，用于过滤音频文件
        import json
        expected_indices = []
        try:
            for line in lyric_content.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    idx = str(item.get('idx', ''))
                    if idx:
                        # 如果idx包含-bgm后缀，去掉它（因为我们要匹配文件名）
                        if idx.endswith('-bgm'):
                            idx = idx[:-4]
                        expected_indices.append(idx)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"⚠️ 提取idx列表失败: {e}，将扫描所有音频文件")
            expected_indices = None
        
        if expected_indices:
            print(f"📋 期望生成 {len(expected_indices)} 个音频文件（idx列表）")
        
        # 1. 写入lyric文件到 sample/lyrics.jsonl（清空原有内容）
        # 注意：无论客户端上传的文件名是什么（lyrics.jsonl 或 lyric.jsonl），都写入到 sample/lyrics.jsonl
        print(f"📝 步骤 1/6: 将上传的内容写入到 sample/lyrics.jsonl（清空原有内容）...")
        if not write_lyric_file(lyric_content, LYRIC_FILE_PATH):
            task.set_status(TaskStatus.FAILED, "写入lyric文件失败")
            return False
        print(f"✅ 已写入到: {LYRIC_FILE_PATH}")
        # 兼容：也写入 sample/lyric.jsonl（清空原有内容）
        write_lyric_file(lyric_content, LYRIC_FILE_PATH_ALT)
        task.set_progress(10)

        # 2. 清空输出目录（清空所有原有内容）- 确保在每次任务开始前都清空
        print(f"🧹 步骤 2/6: 清空输出目录 {actual_output_dir}（确保每次任务开始前都清空）...")
        if not clear_output_dir(actual_output_dir):
            task.set_status(TaskStatus.FAILED, "清空输出目录失败")
            return False
        task.set_progress(15)
        
        # 根据generate_type决定执行流程
        if generate_type == "bgm":
            # 只生成-bgm版本，跳过第一阶段
            print(f"🎵 跳过第一阶段，直接生成-bgm版本...")
            task.set_progress(45)
            
            # 修改lyric文件，给每个idx添加-bgm后缀
            print(f"📝 修改lyric文件（添加-bgm后缀）...")
            if not modify_lyric_file_for_bgm(LYRIC_FILE_PATH):
                task.set_status(TaskStatus.FAILED, "修改lyric文件失败")
                return False
            modify_lyric_file_for_bgm(LYRIC_FILE_PATH_ALT)
            task.set_progress(50)
            
            # 执行-bgm推理
            print(f"🎵 执行-bgm推理...")
            env_overrides = {
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128"),
            }
            ok, res = run_generate_command(
                "songgeneration_large",
                LYRIC_FILE_PATH,
                actual_output_dir,
                bgm=True,
                extra_args=[],
                env_overrides=env_overrides,
            )
            if not ok:
                if res.returncode in (137, 9) or "Killed" in (res.stderr + res.stdout):
                    print("⚠️ 检测到可能的 OOM/Killed，尝试使用 --low_mem 重试(BGM)...")
                    ok, res2 = run_generate_command(
                        "songgeneration_large",
                        LYRIC_FILE_PATH,
                        actual_output_dir,
                        bgm=True,
                        extra_args=["--low_mem"],
                        env_overrides=env_overrides,
                    )
                    if ok:
                        res = res2
                    else:
                        print("⚠️ --low_mem 仍失败，尝试禁用 flash-attn 再重试(BGM)...")
                        ok, res3 = run_generate_command(
                            "songgeneration_large",
                            LYRIC_FILE_PATH,
                            actual_output_dir,
                            bgm=True,
                            extra_args=["--low_mem", "--not_use_flash_attn"],
                            env_overrides=env_overrides,
                        )
                        if ok:
                            res = res3
                        else:
                            res = res3
                
                if not ok:
                    task.set_status(TaskStatus.FAILED, "BGM推理失败\n" + res.short_error())
                    return False
            task.set_progress(90)
            
            # 扫描生成的文件（扫描 sample/output 目录，递归查找所有 .flac 文件）
            print(f"📂 扫描音频文件（扫描 {actual_output_dir} 目录）...")
            time.sleep(2)
            # 直接扫描 sample/output 目录，递归查找所有 .flac 文件，只返回匹配expected_indices的文件
            phase2_files = scan_audio_files(audio_dir=actual_output_dir, max_wait=30, wait_interval=1.0, expected_indices=expected_indices)
            if not phase2_files:
                print(f"⚠️ 第一次扫描未找到文件，等待5秒后重试...")
                time.sleep(5)
                phase2_files = scan_audio_files(audio_dir=actual_output_dir, max_wait=10, wait_interval=0.5, expected_indices=expected_indices)
            
            task.set_phase(2)
            task.set_phase_files(2, phase2_files)
            for filename in phase2_files:
                task.add_result_file(filename)
            
            task.set_progress(100)
            task.set_status(TaskStatus.COMPLETED)
            print(f"✅ 推理任务完成（BGM版本）")
            print(f"   生成文件数: {len(phase2_files)}")
            return True
        
        elif generate_type == "normal":
            # 只生成无-bgm版本
            print(f"🎵 步骤 3/6: 执行推理（生成无-bgm版本）...")
            env_overrides = {
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128"),
            }
            ok, res = run_generate_command(
                "songgeneration_large",
                LYRIC_FILE_PATH,
                actual_output_dir,
                bgm=False,
                extra_args=[],
                env_overrides=env_overrides,
            )
            if not ok:
                # Retry #1: low_mem
                if res.returncode in (137, 9) or "Killed" in (res.stderr + res.stdout):
                    print("⚠️ 检测到可能的 OOM/Killed，尝试使用 --low_mem 重试...")
                    ok, res2 = run_generate_command(
                        "songgeneration_large",
                        LYRIC_FILE_PATH,
                        actual_output_dir,
                        bgm=False,
                        extra_args=["--low_mem"],
                        env_overrides=env_overrides,
                    )
                    if ok:
                        res = res2
                    else:
                        # Retry #2: low_mem + not_use_flash_attn
                        print("⚠️ --low_mem 仍失败，尝试禁用 flash-attn 再重试...")
                        ok, res3 = run_generate_command(
                            "songgeneration_large",
                            LYRIC_FILE_PATH,
                            actual_output_dir,
                            bgm=False,
                            extra_args=["--low_mem", "--not_use_flash_attn"],
                            env_overrides=env_overrides,
                        )
                        if ok:
                            res = res3
                        else:
                            res = res3

                if not ok:
                    task.set_status(TaskStatus.FAILED, "推理失败\n" + res.short_error())
                    return False
            task.set_progress(90)
            
            # 扫描生成的文件（扫描 sample/output 目录，递归查找所有 .flac 文件）
            print(f"📂 扫描音频文件（扫描 {actual_output_dir} 目录）...")
            time.sleep(2)
            # 直接扫描 sample/output 目录，递归查找所有 .flac 文件，只返回匹配expected_indices的文件
            phase1_files = scan_audio_files(audio_dir=actual_output_dir, max_wait=30, wait_interval=1.0, expected_indices=expected_indices)
            if not phase1_files:
                print(f"⚠️ 第一次扫描未找到文件，等待5秒后重试...")
                time.sleep(5)
                phase1_files = scan_audio_files(audio_dir=actual_output_dir, max_wait=10, wait_interval=0.5, expected_indices=expected_indices)
            
            task.set_phase(1)
            task.set_phase_files(1, phase1_files)
            for filename in phase1_files:
                task.add_result_file(filename)
            
            task.set_progress(100)
            task.set_status(TaskStatus.COMPLETED)
            print(f"✅ 推理任务完成（无-bgm版本）")
            print(f"   生成文件数: {len(phase1_files)}")
            return True
        
        else:
            # generate_type == "both": 两阶段生成流程
            # 3. 执行第一次推理（不带--bgm）
            print(f"🎵 步骤 3/6: 执行第一次推理（生成完整音频）...")
            env_overrides = {
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128"),
            }
            ok, res = run_generate_command(
                "songgeneration_large",
                LYRIC_FILE_PATH,
                actual_output_dir,
                bgm=False,
                extra_args=[],
                env_overrides=env_overrides,
            )
            if not ok:
                # Retry #1: low_mem
                if res.returncode in (137, 9) or "Killed" in (res.stderr + res.stdout):
                    print("⚠️ 检测到可能的 OOM/Killed，尝试使用 --low_mem 重试...")
                    ok, res2 = run_generate_command(
                        "songgeneration_large",
                        LYRIC_FILE_PATH,
                        actual_output_dir,
                        bgm=False,
                        extra_args=["--low_mem"],
                        env_overrides=env_overrides,
                    )
                    if ok:
                        res = res2
                    else:
                        # Retry #2: low_mem + not_use_flash_attn
                        print("⚠️ --low_mem 仍失败，尝试禁用 flash-attn 再重试...")
                        ok, res3 = run_generate_command(
                            "songgeneration_large",
                            LYRIC_FILE_PATH,
                            actual_output_dir,
                            bgm=False,
                            extra_args=["--low_mem", "--not_use_flash_attn"],
                            env_overrides=env_overrides,
                        )
                        if ok:
                            res = res3
                        else:
                            res = res3

                if not ok:
                    task.set_status(TaskStatus.FAILED, "第一次推理失败\n" + res.short_error())
                    return False
            task.set_progress(45)

        # 4. 扫描第一次生成的文件并进入等待继续（让客户端先下载）
        print(f"📂 步骤 4/6: 扫描第一阶段音频文件（扫描 {actual_output_dir} 目录）...")
        time.sleep(2)
        # 直接扫描 sample/output 目录，递归查找所有 .flac 文件，只返回匹配expected_indices的文件
        phase1_files = scan_audio_files(audio_dir=actual_output_dir, max_wait=30, wait_interval=1.0, expected_indices=expected_indices)
        task.set_phase(1)
        task.set_phase_files(1, phase1_files)
        for filename in phase1_files:
            task.add_result_file(filename)
        task.set_progress(50)

        print("⏸️ 第一阶段完成，等待客户端确认继续生成BGM...")
        task.require_continue()
        if not task.wait_for_continue(timeout=6 * 3600):  # 最长等待6小时
            task.set_status(TaskStatus.FAILED, "等待客户端继续超时")
            return False

        # 客户端已完成 phase1 下载：不清空输出目录，保留已有文件
        # print(f"🧹 phase1已确认下载，清空输出目录以开始BGM阶段...")
        # if not clear_output_dir(actual_output_dir):
        #     task.set_status(TaskStatus.FAILED, "清空输出目录失败（phase2前）")
        #     return False
        print(f"📂 phase1已确认下载，继续生成BGM阶段（不清空输出目录）...")
        
        # 5. 修改lyric文件，给每个idx添加-bgm后缀
        print(f"📝 步骤 5/6: 修改lyric文件（添加-bgm后缀）...")
        if not modify_lyric_file_for_bgm(LYRIC_FILE_PATH):
            task.set_status(TaskStatus.FAILED, "修改lyric文件失败")
            return False
        # 同步修改 alt 文件（sample/lyric.jsonl）
        modify_lyric_file_for_bgm(LYRIC_FILE_PATH_ALT)
        task.set_progress(60)
        
        # 6. 执行第二次推理（带--bgm）
        print(f"🎵 步骤 6/6: 执行第二次推理（生成BGM版本）...")
        ok, res = run_generate_command(
            "songgeneration_large",
            LYRIC_FILE_PATH,
            actual_output_dir,
            bgm=True,
            extra_args=[],
            env_overrides=env_overrides,
        )
        if not ok:
            if res.returncode in (137, 9) or "Killed" in (res.stderr + res.stdout):
                print("⚠️ 检测到可能的 OOM/Killed，尝试使用 --low_mem 重试(BGM)...")
                ok, res2 = run_generate_command(
                    "songgeneration_large",
                    LYRIC_FILE_PATH,
                    actual_output_dir,
                    bgm=True,
                    extra_args=["--low_mem"],
                    env_overrides=env_overrides,
                )
                if ok:
                    res = res2
                else:
                    print("⚠️ --low_mem 仍失败，尝试禁用 flash-attn 再重试(BGM)...")
                    ok, res3 = run_generate_command(
                        "songgeneration_large",
                        LYRIC_FILE_PATH,
                        actual_output_dir,
                        bgm=True,
                        extra_args=["--low_mem", "--not_use_flash_attn"],
                        env_overrides=env_overrides,
                    )
                    if ok:
                        res = res3
                    else:
                        res = res3

            if not ok:
                task.set_status(TaskStatus.FAILED, "第二次推理失败\n" + res.short_error())
                return False
        task.set_progress(90)

        # 扫描第二阶段生成的文件（扫描 sample/output 目录，递归查找所有 .flac 文件）
        print(f"📂 扫描第二阶段音频文件（扫描 {actual_output_dir} 目录）...")
        time.sleep(2)
        # 直接扫描 sample/output 目录，递归查找所有 .flac 文件，只返回匹配expected_indices的文件
        phase2_files = scan_audio_files(audio_dir=actual_output_dir, max_wait=30, wait_interval=1.0, expected_indices=expected_indices)
        if not phase2_files:
            print(f"⚠️ 第二阶段第一次扫描未找到文件，等待5秒后重试...")
            time.sleep(5)
            phase2_files = scan_audio_files(audio_dir=actual_output_dir, max_wait=10, wait_interval=0.5, expected_indices=expected_indices)

        task.set_phase(2)
        task.set_phase_files(2, phase2_files)
        for filename in phase2_files:
            task.add_result_file(filename)

        task.set_progress(100)

        task.set_status(TaskStatus.COMPLETED)
        print(f"✅ 推理任务完成")
        print(f"   phase1: {len(task.phase1_files)} files, phase2: {len(task.phase2_files)} files")
        
        return True
        
    except Exception as e:
        error_msg = f"推理任务执行出错: {str(e)}"
        print(f"❌ {error_msg}")
        task.set_status(TaskStatus.FAILED, error_msg)
        return False

