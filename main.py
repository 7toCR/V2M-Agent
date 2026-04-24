from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.live_status import LiveStatus

DEFAULT_INPUT_DIR = PROJECT_ROOT / "Input" / "scene_17"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "scene_17"
DEFAULT_REQUIREMENT = (
    "请综合分析 scene_17 的文本、音频、图片和视频内容，完成统一的多模态场景理解，"
    "再基于场景情绪、画面节奏、环境声音和叙事氛围生成适合短视频使用的中文流行背景音乐 prompt。"
    "要求音乐贴合场景变化，兼顾画面氛围、主体情绪、听感节奏和可用于后续音乐生成模型的结构化字段。"
)

TEXT_EXTENSIONS = {".txt", ".json", ".jsonl", ".md", ".doc", ".docx", ".pdf"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".wma"}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


class _LiveOutputStream:
    """Convert noisy runtime writes into LiveStatus updates."""

    def __init__(self, original: Any, *, is_error: bool = False) -> None:
        self._original = original
        self._is_error = is_error
        self._buffer = ""
        self.encoding = getattr(original, "encoding", None)
        self.errors = getattr(original, "errors", None)

    def write(self, text: object) -> int:
        chunk = str(text or "")
        if not chunk:
            return 0

        self._buffer += chunk.replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume(line)

        if len(self._buffer) > 80:
            self._consume(self._buffer)
            self._buffer = ""

        return len(chunk)

    def flush(self) -> None:
        if self._buffer.strip():
            self._consume(self._buffer)
        self._buffer = ""
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._original.fileno()

    def _consume(self, line: str) -> None:
        detail = _runtime_status_detail(line, is_error=self._is_error)
        if not detail:
            return
        theme = LiveStatus.infer_theme(line) or ("客户端生成" if self._is_error else None)
        LiveStatus.update(theme=theme, detail=detail, force=True)


@contextmanager
def _capture_runtime_output() -> Any:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_proxy = _LiveOutputStream(original_stdout)
    stderr_proxy = _LiveOutputStream(original_stdout, is_error=True)
    LiveStatus.set_output_stream(original_stdout)
    sys.stdout = stdout_proxy
    sys.stderr = stderr_proxy
    try:
        yield
    finally:
        stdout_proxy.flush()
        stderr_proxy.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        LiveStatus.set_output_stream(None)


def _runtime_status_detail(line: str, *, is_error: bool = False) -> str:
    text = " ".join(str(line or "").split())
    if not text:
        return ""

    stripped = text.strip()
    if set(stripped) <= {"=", "-", "_", " "}:
        return ""

    lower = stripped.lower()
    if lower.startswith("final_answer") or "final_answer" in lower:
        return ""
    if lower.startswith(("lyricist_result:", "composer_result:", "stylist_result:", "observation")):
        return ""
    if any(token in lower for token in ("content=", "response:", "result:", "additional_kwargs=", "response_metadata=", "usage_metadata=")):
        return ""
    if len(stripped) > 80 and any(mark in stripped for mark in ("{", "}", "[", "]", ":")):
        return ""
    if "error reading ssh protocol banner" in lower:
        return "SSH 隧道握手失败"
    if "ssh" in lower and any(token in lower for token in ("失败", "failed", "error", "exception")):
        return "SSH 隧道连接失败"
    if "traceback" in lower or lower.startswith(("file ", "raise ", "self.", "buf =")):
        return ""
    if lower in {"eoferror"} or "during handling of the above exception" in lower:
        return ""
    if "[debug]" in lower:
        return ""

    return stripped[:30]


def _summarize_inputs(multimodal_input: dict[str, list[str]]) -> str:
    labels = {
        "text_address": "文本",
        "audio_address": "音频",
        "photo_address": "图片",
        "video_address": "视频",
    }
    parts = [
        f"{label}{len(multimodal_input.get(key, []))}"
        for key, label in labels.items()
        if multimodal_input.get(key)
    ]
    return "输入 " + " ".join(parts)


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _format_run_paths(input_dir: Path, output_dir: Path) -> str:
    lines = [
        "Resolved paths:",
        f"  input_dir:  {input_dir}",
        f"  output_dir: {output_dir}",
    ]
    if not _is_within_directory(output_dir, PROJECT_ROOT):
        lines.extend(
            [
                "Warning: output directory is outside the current project root.",
                f"  project_root: {PROJECT_ROOT}",
            ]
        )
    return "\n".join(lines)


def _format_saved_outputs_message(output_dir: Path) -> str:
    return f"Saved outputs to:\n  {output_dir}"


def _print_direct(message: str) -> None:
    stream = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
    stream.write(message)
    if not message.endswith("\n"):
        stream.write("\n")
    stream.flush()


def discover_inputs(input_dir: str | Path) -> dict[str, list[str]]:
    """Discover modality files under input_dir with case-insensitive suffixes."""
    base = Path(input_dir).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {base}")

    multimodal_input: dict[str, list[str]] = {
        "text_address": [],
        "audio_address": [],
        "photo_address": [],
        "video_address": [],
    }

    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        path_str = str(path)
        if suffix in TEXT_EXTENSIONS:
            multimodal_input["text_address"].append(path_str)
        elif suffix in AUDIO_EXTENSIONS:
            multimodal_input["audio_address"].append(path_str)
        elif suffix in PHOTO_EXTENSIONS:
            multimodal_input["photo_address"].append(path_str)
        elif suffix in VIDEO_EXTENSIONS:
            multimodal_input["video_address"].append(path_str)

    if not any(multimodal_input.values()):
        raise FileNotFoundError(f"No supported modality files found in: {base}")

    return multimodal_input


def _modalities_from_input(multimodal_input: dict[str, list[str]]) -> list[str]:
    modalities: list[str] = []
    if multimodal_input.get("text_address"):
        modalities.append("text")
    if multimodal_input.get("audio_address"):
        modalities.append("audio")
    if multimodal_input.get("photo_address"):
        modalities.append("photo")
    if multimodal_input.get("video_address"):
        modalities.append("video")
    return modalities


def build_team2_fallback_packet(
    requirement: str,
    multimodal_input: dict[str, list[str]],
) -> dict[str, Any]:
    """Build a deterministic Team2 packet only when explicitly requested."""
    return {
        "team_name": "Team2",
        "instruction": (
            "Analyze all available modalities from scene_17 and produce a unified "
            "nine-field scene representation for downstream music generation. "
            "Preserve temporal order, align audio with the visual timeline, merge "
            "text and image context into the video skeleton, and return verified "
            "scene descriptions."
        ),
        "constraints": [
            "Use all available text, audio, photo, and video inputs.",
            "Preserve temporal ordering across scenes.",
            "Attach audio evidence to the visual timeline instead of replacing it.",
            "Output structured scene descriptions suitable for Team3 music generation.",
        ],
        "modalities": _modalities_from_input(multimodal_input),
        "modality_addresses": multimodal_input,
        "user_requirement": requirement,
        "parsed_requirement": {
            "objectives": ["multimodal scene understanding", "music prompt generation"],
            "constraints": ["scene-aligned music", "structured downstream output"],
            "modality_hints": _modalities_from_input(multimodal_input),
        },
    }


def build_team3_fallback_packet(
    requirement: str,
    json_scene: list[dict[str, Any]],
    piece: int,
) -> dict[str, Any]:
    """Build a deterministic Team3 packet only when explicitly requested."""
    return {
        "team_name": "Team3",
        "instruction": (
            "Generate structured music prompts from the Team2 scene results. "
            "Create a musical blueprint, run lyric, composition, and audio-type "
            "experts, then verify that the prompts match scene mood and timeline."
        ),
        "constraints": [
            "Generate exactly the requested number of prompt records when possible.",
            "Each prompt must contain idx, gt_lyric, descriptions, and auto_prompt_audio_type.",
            "Keep the music aligned with the scene mood, tempo, and narrative arc.",
        ],
        "json_scene": json_scene,
        "user_requirement": requirement,
        "piece": piece,
    }


async def run_pipeline(
    requirement: str,
    multimodal_input: dict[str, list[str]],
    piece: int = 2,
    *,
    allow_fallback_packets: bool = False,
) -> dict[str, Any]:
    """Run Team1, Team2, and Team3 in sequence."""
    from Team1.supervisor.requirement_supervisor import RequirementSupervisor
    from Team2.supervisor.scene_understanding_supervisor import SceneUnderstandingSupervisor
    from Team3.supervisor.music_generation_supervisor import MusicGenerationSupervisor

    LiveStatus.start("需求解析", "解析用户需求")
    started_at = time.time()
    result: dict[str, Any] = {
        "requirement": requirement,
        "multimodal_input": multimodal_input,
        "piece": piece,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    LiveStatus.update("需求解析", "生成任务包", force=True)
    team1 = RequirementSupervisor()
    team1_result = await team1.ainvoke(
        user_requirement=requirement,
        multimodal_input=multimodal_input,
    )
    analysis = team1_result.get("requirement_analysis_result") or {}
    team2_packet = analysis.get("team2_task_packet")
    team3_packet = analysis.get("team3_task_packet")

    if not team2_packet:
        if not allow_fallback_packets:
            raise RuntimeError("Team1 did not produce team2_task_packet.")
        LiveStatus.update("需求解析", "补全场景理解任务包", force=True)
        team2_packet = build_team2_fallback_packet(requirement, multimodal_input)

    result["team1_result"] = team1_result
    result["team2_task_packet"] = team2_packet
    result["team3_task_packet_from_team1"] = team3_packet

    LiveStatus.update("场景理解", "融合多模态场景", force=True)
    team2 = SceneUnderstandingSupervisor()
    team2_result = await team2.ainvoke_from_packet(team2_packet)
    json_scene = team2_result.get("json_scene_result") or []
    if not json_scene:
        raise RuntimeError("Team2 did not produce json_scene_result.")

    result["team2_result"] = team2_result
    result["json_scene_result"] = json_scene

    LiveStatus.update("音乐生成", "生成音乐提示", force=True)
    if not team3_packet:
        if not allow_fallback_packets:
            raise RuntimeError("Team1 did not produce team3_task_packet.")
        LiveStatus.update("音乐生成", "补全音乐生成任务包", force=True)
        team3_packet = build_team3_fallback_packet(requirement, json_scene, piece)
    else:
        team3_packet = dict(team3_packet)
        team3_packet["json_scene"] = json_scene
        team3_packet["piece"] = piece
        team3_packet.setdefault("user_requirement", requirement)

    team3 = MusicGenerationSupervisor()
    team3_result = await team3.ainvoke_from_packet(team3_packet)
    pop_prompt_result = team3_result.get("pop_prompt_result") or []
    if not pop_prompt_result:
        raise RuntimeError("Team3 did not produce pop_prompt_result.")

    result["team3_task_packet"] = team3_packet
    result["team3_result"] = team3_result
    result["pop_prompt_result"] = pop_prompt_result
    result["complete"] = True
    result["elapsed_seconds"] = round(time.time() - started_at, 2)
    result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return result


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Any) -> None:
    if rows is None:
        rows = []
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        rows = [rows]

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def save_outputs(result: dict[str, Any], output_dir: str | Path, scene_name: str = "scene_17") -> None:
    """Persist final and intermediate pipeline outputs."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    _write_json(out / "team1_result.json", result.get("team1_result", {}))
    _write_json(out / "team2_task_packet.json", result.get("team2_task_packet", {}))
    _write_json(out / "team2_result.json", result.get("team2_result", {}))
    _write_json(out / "team3_task_packet.json", result.get("team3_task_packet", {}))
    _write_json(out / "team3_result.json", result.get("team3_result", {}))

    _write_jsonl(out / "scene.jsonl", result.get("json_scene_result", []))
    _write_jsonl(out / "lyric.jsonl", result.get("pop_prompt_result", []))

    team2_result = result.get("team2_result", {})
    _write_jsonl(out / "text.jsonl", team2_result.get("text_result", {}))
    _write_jsonl(out / "audio.jsonl", team2_result.get("audio_result", []))
    _write_jsonl(out / "photo.jsonl", team2_result.get("photo_result", {}))
    _write_jsonl(out / "video.jsonl", team2_result.get("video_result", []))

    meta = {
        "scene_name": scene_name,
        "requirement": result.get("requirement", ""),
        "multimodal_input": result.get("multimodal_input", {}),
        "piece": result.get("piece"),
        "complete": result.get("complete", False),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "scene_count": len(result.get("json_scene_result", []) or []),
        "prompt_count": len(result.get("pop_prompt_result", []) or []),
    }
    _write_json(out / "pipeline_meta.json", meta)


def _print_discovered_inputs(multimodal_input: dict[str, list[str]]) -> None:
    print("Discovered inputs:")
    for key in ("text_address", "audio_address", "photo_address", "video_address"):
        values = multimodal_input.get(key, [])
        print(f"  {key}: {len(values)}")
        for value in values:
            print(f"    - {value}")


def _check_environment() -> None:
    missing = [name for name in ("MCP_API_KEY", "DASHSCOPE_API_KEY") if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise EnvironmentError(
            f"Missing required API key environment variable(s): {joined}. "
            "Set them in .env or in the current shell before running the full pipeline."
        )


def _validate_generated_lyric_file(path: Path, not_before: float) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Generated lyric file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Generated lyric path is not a file: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Generated lyric file is empty: {path}")

    # Allow a small timestamp tolerance for filesystems with coarse mtime precision.
    if path.stat().st_mtime < not_before - 2:
        raise RuntimeError(f"Generated lyric file was not refreshed: {path}")


def run_server_inference(
    lyric_path: Path,
    output_dir: Path,
    *,
    generate_type: str,
    auto_ssh: bool | None,
    config_path: str | None,
) -> None:
    """Upload the generated lyric file to the server and download generated audio."""
    from client.client import MusicGenerationClient

    LiveStatus.start("客户端生成", f"准备歌词 {lyric_path.name} type={generate_type}")
    LiveStatus.update("客户端生成", "连接生成服务", force=True)

    client = MusicGenerationClient(auto_ssh=auto_ssh, config_path=config_path)
    ok = client.process_lyric_file(
        file_path=str(lyric_path),
        output_dir=str(output_dir),
        auto_disconnect=True,
        generate_type=generate_type,
    )
    if not ok:
        raise RuntimeError("客户端生成失败")


def _print_api_connection_help(exc: APIConnectionError) -> None:
    print("\nFailed to connect to the DashScope OpenAI-compatible API.")
    print(f"  error: {exc}")
    print("  endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1")
    print("\nCheck network access, firewall rules, and proxy environment variables:")
    print("  PowerShell: Get-ChildItem Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:NO_PROXY -ErrorAction SilentlyContinue")
    print("If a stale proxy is configured, clear it for this shell and rerun:")
    print("  Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue")
    print("  Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Team1 -> Team2 -> Team3 scene_17 end-to-end test.",
    )
    parser.add_argument(
        "--test-scene-17",
        action="store_true",
        help="Use the built-in scene_17 input and output paths.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing multimodal input files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where JSON/JSONL outputs will be saved.",
    )
    parser.add_argument(
        "--requirement",
        default=DEFAULT_REQUIREMENT,
        help="User requirement passed to Team1.",
    )
    parser.add_argument(
        "--piece",
        type=int,
        default=2,
        help="Number of music prompt records requested from Team3.",
    )
    parser.add_argument(
        "--allow-fallback-packets",
        action="store_true",
        help="Continue with deterministic packets if Team1 misses Team2/Team3 packets.",
    )
    parser.add_argument(
        "--skip-server-inference",
        action="store_true",
        help="Only run Team1/Team2/Team3 and save JSON/JSONL outputs; skip client/server audio generation.",
    )
    parser.add_argument(
        "--generate-type",
        choices=["normal", "bgm", "both"],
        default="both",
        help="Server inference generation mode.",
    )
    parser.add_argument(
        "--client-config-path",
        default=None,
        help="Optional path to the client/server connection config.",
    )
    parser.add_argument(
        "--no-auto-ssh",
        action="store_true",
        help="Disable automatic SSH tunnel creation and connect directly to the configured server URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered inputs and the constructed requirement; do not call LLM agents.",
    )
    return parser.parse_args()


async def async_main() -> None:
    load_dotenv()
    args = parse_args()

    input_dir = DEFAULT_INPUT_DIR if args.test_scene_17 else Path(args.input_dir)
    output_dir = DEFAULT_OUTPUT_DIR if args.test_scene_17 else Path(args.output_dir)
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    saved_output_dir: Path | None = None

    if args.dry_run:
        multimodal_input = discover_inputs(input_dir)
        _print_discovered_inputs(multimodal_input)
        print()
        print(_format_run_paths(input_dir, output_dir))
        print(f"\nConstructed requirement:\n  {args.requirement}")
        print("\nDry run complete. No agent pipeline was executed.")
        return

    print(_format_run_paths(input_dir, output_dir))

    with _capture_runtime_output():
        try:
            LiveStatus.start("准备输入", "扫描输入文件")
            multimodal_input = discover_inputs(input_dir)
            LiveStatus.update("准备输入", _summarize_inputs(multimodal_input), force=True)

            _check_environment()
            result = await run_pipeline(
                requirement=args.requirement,
                multimodal_input=multimodal_input,
                piece=args.piece,
                allow_fallback_packets=args.allow_fallback_packets,
            )

            save_started_at = time.time()
            LiveStatus.update("保存结果", "写入结构化文件", force=True)
            save_outputs(result, output_dir, input_dir.name)
            saved_output_dir = output_dir
            _print_direct("\n" + _format_saved_outputs_message(saved_output_dir))
            lyric_path = output_dir / "lyric.jsonl"

            scene_count = len(result.get("json_scene_result", []) or [])
            prompt_count = len(result.get("pop_prompt_result", []) or [])
            LiveStatus.update("保存结果", f"场景{scene_count} 音乐提示{prompt_count}", force=True)

            if args.skip_server_inference:
                LiveStatus.finish("流程完成 已跳过客户端生成")
                return

            _validate_generated_lyric_file(lyric_path, save_started_at)
            run_server_inference(
                lyric_path,
                output_dir,
                generate_type=args.generate_type,
                auto_ssh=False if args.no_auto_ssh else None,
                config_path=args.client_config_path,
            )
            LiveStatus.finish("客户端生成完成")
        except APIConnectionError:
            LiveStatus.finish("API 连接失败")
            raise SystemExit(1) from None
        except SystemExit:
            raise
        except Exception as exc:
            detail = str(exc) or "流程执行失败"
            LiveStatus.finish(detail)
            raise SystemExit(1) from None


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
