from __future__ import annotations

import atexit
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from typing import Optional, TextIO


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_SPACE_RE = re.compile(r"\s+")


class LiveStatus:
    """Render a compact, continuously refreshed one-line runtime status."""

    _refresh_interval = 0.4
    _plain_interval = 5.0
    _detail_limit = 30
    _theme_limit = 12
    _left_separator = "  "
    _middle_gap = 2
    _right_separator = " "
    _frames = ("o....", ".o...", "..o..", "...o.", "....o")
    _ascii_frames = _frames
    _colors = (31, 33, 32, 36, 34, 35)

    _lock = threading.RLock()
    _stop_event = threading.Event()
    _thread: Optional[threading.Thread] = None
    _active = False
    _started_at: Optional[float] = None
    _theme = "初始化"
    _detail = ""
    _tick = 0
    _last_render_len = 0
    _last_render_at = 0.0
    _last_plain_at = 0.0
    _last_plain_key: Optional[tuple[str, str, int]] = None
    _output_stream: Optional[TextIO] = None

    @classmethod
    def start(
        cls,
        theme: str = "初始化",
        detail: str = "",
        *,
        reset_elapsed: bool = True,
    ) -> None:
        if not cls._enabled():
            return

        with cls._lock:
            if reset_elapsed or cls._started_at is None:
                cls._started_at = time.time()
            cls._theme = cls._clean_theme(theme) or cls._theme
            cls._detail = cls._clean_detail(detail, cls._detail_limit)
            cls._active = True
            cls._stop_event.clear()

            if cls._dynamic_enabled():
                if cls._thread is None or not cls._thread.is_alive():
                    cls._thread = threading.Thread(
                        target=cls._run,
                        name="LiveStatus",
                        daemon=True,
                    )
                    cls._thread.start()
                cls._render_locked(force=True)
            else:
                cls._emit_plain_locked(force=True)

    @classmethod
    def update(
        cls,
        theme: Optional[str] = None,
        detail: Optional[str] = None,
        *,
        force: bool = False,
    ) -> None:
        if not cls._enabled():
            return

        with cls._lock:
            if not cls._active:
                cls.start(theme or cls._theme, detail or "", reset_elapsed=True)
                return

            if theme:
                cls._theme = cls._clean_theme(theme) or cls._theme
            if detail is not None:
                cls._detail = cls._clean_detail(detail, cls._detail_limit)

            if cls._dynamic_enabled():
                now = time.time()
                if force or now - cls._last_render_at >= 0.1:
                    cls._render_locked(force=force)
            else:
                cls._emit_plain_locked(force=force)

    @classmethod
    def finish(cls, final_detail: Optional[str] = None) -> None:
        with cls._lock:
            if final_detail:
                cls._detail = cls._clean_detail(final_detail, cls._detail_limit)
                if cls._active and cls._dynamic_enabled():
                    cls._render_locked(force=True)

            was_active = cls._active
            cls._active = False
            cls._stop_event.set()

        thread = cls._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)

        with cls._lock:
            cls._thread = None
            if not was_active:
                cls._last_render_len = 0

    @classmethod
    def print_line(cls, message: str = "") -> None:
        """Route runtime messages into the active live status row."""

        text = str(message)
        with cls._lock:
            if cls._active:
                detail = cls._clean_detail(text, cls._detail_limit)
                if not detail:
                    return
                cls._theme = cls.infer_theme(text) or cls._theme
                cls._detail = detail
                cls._render_locked(force=True)
            elif text:
                cls._safe_write(text + "\n")
            cls._safe_flush()

    @classmethod
    def set_output_stream(cls, stream: Optional[TextIO]) -> None:
        """Pin live rendering to a real stream while stdout/stderr are captured."""

        with cls._lock:
            cls._output_stream = stream

    @classmethod
    def infer_theme(cls, text: str) -> Optional[str]:
        lower = str(text).lower()
        if any(token in lower for token in ("client", "server", "ssh", "socket", "download", "upload", "客户端", "服务端", "服务器", "隧道", "下载", "上传")):
            return "客户端生成"
        if any(token in lower for token in ("music", "lyric", "composer", "stylist", "prompt", "音乐", "歌词", "bgm")):
            return "音乐生成"
        if any(token in lower for token in ("video", "视频", "关键帧")):
            return "视频理解"
        if any(token in lower for token in ("audio", "音频", "声音", "声", "分段")):
            return "音频理解"
        if any(token in lower for token in ("photo", "image", "图片", "图像")):
            return "图片理解"
        if any(token in lower for token in ("text", "文本")):
            return "文本理解"
        if any(token in lower for token in ("scene", "场景", "verifier", "fusion", "融合")):
            return "场景理解"
        if any(token in lower for token in ("requirement", "user", "用户", "需求", "指令", "约束")):
            return "需求解析"
        return None

    @classmethod
    def _run(cls) -> None:
        while not cls._stop_event.wait(cls._refresh_interval):
            with cls._lock:
                if not cls._active:
                    continue
                cls._tick += 1
                cls._render_locked()

    @classmethod
    def _render_locked(cls, *, force: bool = False) -> None:
        if not cls._active or not cls._dynamic_enabled():
            return

        now = time.time()
        if not force and now - cls._last_render_at < cls._refresh_interval * 0.75:
            return

        line = cls._format_line(color=True)
        cls._clear_line_locked()
        cls._safe_write(line)

        cls._last_render_len = cls._visible_width(line)
        cls._last_render_at = now
        cls._safe_flush()

    @classmethod
    def _emit_plain_locked(cls, *, force: bool = False) -> None:
        now = time.time()
        elapsed = cls._elapsed_seconds()
        key = (cls._theme, cls._detail, elapsed // 10)
        if not force and key == cls._last_plain_key:
            return
        if not force and now - cls._last_plain_at < cls._plain_interval:
            return

        cls._last_plain_key = key
        cls._last_plain_at = now
        line = cls._format_line(color=False)
        cls._clear_line_locked()
        cls._safe_write(line)
        cls._last_render_len = cls._visible_width(line)
        cls._safe_flush()

    @classmethod
    def _format_line(cls, *, color: bool) -> str:
        elapsed = cls._elapsed_seconds()
        theme = cls._clean_theme(cls._theme) or "运行中"
        detail = cls._clean_detail(cls._detail, cls._detail_limit) or "处理中"
        frame = cls._frame(color=color)

        width = max(39, cls._terminal_width() - 1)
        right_plain = f"{cls._frame(color=False)}{cls._right_separator}{elapsed}s"
        right = f"{frame}{cls._right_separator}{elapsed}s"
        prefix = f"{theme}{cls._left_separator}"

        detail_width = max(
            0,
            width
            - cls._visible_width(prefix)
            - cls._middle_gap
            - cls._visible_width(right_plain),
        )
        detail = cls._clip_display(detail, detail_width)

        left = f"{prefix}{detail}"
        gap_width = max(
            cls._middle_gap,
            width - cls._visible_width(left) - cls._visible_width(right_plain),
        )
        return f"{left}{' ' * gap_width}{right}"

    @classmethod
    def _frame(cls, *, color: bool) -> str:
        unicode_ok = cls._unicode_enabled()
        frames = cls._frames if unicode_ok else cls._ascii_frames
        frame = frames[cls._tick % len(frames)]
        if not color or not cls._color_enabled():
            return frame

        code = cls._colors[cls._tick % len(cls._colors)]
        return f"\033[{code}m{frame}\033[0m"

    @classmethod
    def _elapsed_seconds(cls) -> int:
        started_at = cls._started_at or time.time()
        return max(0, int(time.time() - started_at))

    @classmethod
    def _clear_line_locked(cls) -> None:
        if cls._ansi_clear_enabled():
            cls._safe_write("\r\x1b[2K")
            return

        clear_width = max(cls._last_render_len, cls._terminal_width())
        if clear_width:
            cls._safe_write("\r" + (" " * max(0, clear_width - 1)) + "\r")
        else:
            cls._safe_write("\r")

    @classmethod
    def _clean_text(cls, value: object, limit: int) -> str:
        text = _SPACE_RE.sub(" ", str(value or "")).strip()
        return cls._clip(text, limit)

    @classmethod
    def _clean_theme(cls, value: object) -> str:
        text = _SPACE_RE.sub(" ", str(value or "")).strip()
        aliases = {
            "用户理解": "需求解析",
            "Team1": "需求解析",
            "Team2": "场景理解",
            "Team3": "音乐生成",
        }
        text = aliases.get(text, text)
        return cls._clip(text, cls._theme_limit)

    @classmethod
    def _clean_detail(cls, value: object, limit: int) -> str:
        text = str(value or "").replace("\r", " ").replace("\n", " ")
        text = _SPACE_RE.sub(" ", text).strip()
        if not text:
            return ""

        if cls._is_noise(text):
            return ""

        detail = cls._canonical_detail(text)
        return cls._clip(detail, limit)

    @staticmethod
    def _is_noise(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if set(stripped) <= {"=", "-", "_", " "}:
            return True
        lower = stripped.lower()
        noise_tokens = (
            "[debug]",
            "traceback",
            "final_answer",
            "lyricist_result:",
            "composer_result:",
            "stylist_result:",
            "observation",
            "content=",
            "response:",
            "result:",
            "additional_kwargs=",
            "response_metadata=",
            "usage_metadata=",
            "token_usage",
            "system_fingerprint",
            "logprobs",
            "id='lc_run",
            "during handling of the above exception",
        )
        if any(token in lower for token in noise_tokens):
            return True
        if len(stripped) > 80 and any(mark in stripped for mark in ("{", "}", "[", "]", ":")):
            return True
        return False

    @classmethod
    def _canonical_detail(cls, text: str) -> str:
        lower = text.lower()
        if lower.startswith("final_answer"):
            return "生成结果"
        if lower.startswith(("lyricist_result:", "composer_result:", "stylist_result:")):
            return "专家结果"
        if lower.startswith("observation"):
            return "解析结果"
        if "ssh 隧道握手失败" in lower or "error reading ssh protocol banner" in lower:
            return "SSH 隧道握手失败"
        if "ssh" in lower and any(token in lower for token in ("失败", "failed", "error", "exception")):
            return "SSH 隧道连接失败"

        mappings = (
            (("parsing requirement", "解析用户需求"), "解析用户需求"),
            (("determining modalities", "识别输入模态"), "识别输入模态"),
            (("determining needed teams", "规划协作团队"), "规划协作团队"),
            (("reading team profiles", "读取团队配置"), "读取团队配置"),
            (("matching requirements", "匹配任务能力"), "匹配任务能力"),
            (("extracting instructions", "提取任务指令"), "提取任务指令"),
            (("generating constraints", "生成约束条件"), "生成约束条件"),
            (("generating team 2 packet",), "生成场景理解任务包"),
            (("generating team 3 packet",), "生成音乐生成任务包"),
            (("all needed packets ready", "verify"), "校验任务包"),
            (("strategic planning", "musical blueprint"), "生成音乐蓝图"),
            (("calling llm", "调用llm", "调用模型", "正在调用llm生成任务"), "调用模型生成专家任务"),
            (("generating expert tasks", "生成专家任务", "开始生成任务", "任务生成完成"), "生成专家任务"),
            (("llm调用完成", "llm call complete", "llm调用完成"), "专家任务生成完成"),
            (("解析完成", "parse complete", "解析专家任务"), "解析专家任务"),
            (("validation passed", "校验通过"), "校验通过"),
            (("validation", "verifier", "校验"), "校验结果"),
            (("reflection node started", "反思"), "反思校验结果"),
            (("sceneunderstandingsupervisor",), "融合多模态场景"),
            (("requirementsupervisor",), "解析用户需求"),
            (("musicgenerationsupervisor",), "生成音乐提示"),
            (("download dir", "下载目录"), "准备下载生成音频"),
            (("upload", "上传"), "上传歌词文件"),
            (("download", "下载"), "下载生成音频"),
            (("ssh", "隧道"), "连接生成服务"),
            (("socketio", "服务器", "server"), "连接生成服务"),
            (("progress=", "status=", "phase="), text),
        )
        for needles, replacement in mappings:
            if any(needle in lower for needle in needles):
                return replacement

        if lower.startswith(("thought", "think")):
            return "分析任务上下文"
        if lower.startswith("task:"):
            return "执行专家任务"
        if lower.startswith("action"):
            if "video" in lower or "视频" in lower:
                return "调用视频理解"
            if "audio" in lower or "音频" in lower or "声音" in lower:
                return "调用音频理解"
            if "photo" in lower or "image" in lower or "图片" in lower:
                return "调用图片理解"
            if "text" in lower or "文本" in lower:
                return "调用文本理解"
            return "执行专家动作"
        if lower.startswith("observation"):
            return "解析理解结果"

        return text

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        suffix = "..."
        if limit <= len(suffix):
            return text[:limit]
        return text[: limit - len(suffix)] + suffix

    @classmethod
    def _clip_display(cls, text: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if cls._visible_width(text) <= max_width:
            return text

        suffix = "..."
        suffix_width = cls._visible_width(suffix)
        if max_width <= suffix_width:
            return "." * max_width

        target_width = max_width - suffix_width
        used_width = 0
        chars: list[str] = []
        for char in text:
            char_width = cls._char_width(char)
            if used_width + char_width > target_width:
                break
            chars.append(char)
            used_width += char_width
        return "".join(chars) + suffix

    @classmethod
    def _visible_width(cls, text: str) -> int:
        visible = _ANSI_RE.sub("", str(text or ""))
        return sum(cls._char_width(char) for char in visible)

    @staticmethod
    def _char_width(char: str) -> int:
        if not char:
            return 0
        if unicodedata.combining(char):
            return 0
        category = unicodedata.category(char)
        if category.startswith("C"):
            return 0
        return 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1

    @staticmethod
    def _terminal_width() -> int:
        return max(40, shutil.get_terminal_size(fallback=(120, 20)).columns)

    @staticmethod
    def _enabled() -> bool:
        return os.getenv("NINGBGM_LIVE_STATUS", "1").lower() not in {"0", "false", "no"}

    @staticmethod
    def _dynamic_enabled() -> bool:
        return True

    @classmethod
    def _ansi_clear_enabled(cls) -> bool:
        stream = cls._output_stream or sys.stdout
        if not bool(getattr(stream, "isatty", lambda: False)()):
            return False
        if os.name != "nt":
            return True

        term = os.getenv("TERM", "").lower()
        return bool(
            os.getenv("WT_SESSION")
            or os.getenv("ANSICON")
            or os.getenv("ConEmuANSI", "").upper() == "ON"
            or "xterm" in term
            or "ansi" in term
            or "vt100" in term
        )

    @classmethod
    def _color_enabled(cls) -> bool:
        if os.getenv("NO_COLOR"):
            return False
        term = os.getenv("TERM", "")
        stream = cls._output_stream or sys.stdout
        stream_is_tty = hasattr(stream, "isatty") and stream.isatty()
        return stream_is_tty and term.lower() != "dumb"

    @staticmethod
    def _unicode_enabled() -> bool:
        encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
        return "utf" in encoding or encoding in {"", "cp65001"}

    @staticmethod
    def _safe_write(text: str) -> None:
        stream = LiveStatus._output_stream or sys.stdout
        try:
            stream.write(text)
        except UnicodeEncodeError:
            stream.write(text.encode("ascii", "ignore").decode("ascii"))

    @staticmethod
    def _safe_flush() -> None:
        stream = LiveStatus._output_stream or sys.stdout
        try:
            stream.flush()
        except Exception:
            pass


atexit.register(LiveStatus.finish)
