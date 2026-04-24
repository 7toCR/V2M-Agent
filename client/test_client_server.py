"""
End-to-end smoke test for client/server pipeline with auto SSH tunnel.

Default scenario:
- lyric file: D:\\github\\NingBGM\\Output\\scene_17\\lyric.jsonl
- generate type: both
- auto ssh: enabled
"""

import argparse
import os
import sys
from typing import Optional


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_LYRIC_PATH = r"D:\github\NingBGM\Output\scene_17\lyric.jsonl"


def run_e2e_test(
    lyric_path: str,
    generate_type: str = "both",
    auto_disconnect: bool = True,
    config_path: Optional[str] = None,
) -> int:
    try:
        from client.client import MusicGenerationClient
    except Exception as e:
        print(f"[ERROR] failed to import MusicGenerationClient: {e}")
        return 3

    if not os.path.exists(lyric_path):
        print(f"[ERROR] lyric file not found: {lyric_path}")
        return 2

    output_dir = os.path.dirname(lyric_path)
    client = MusicGenerationClient(auto_ssh=True, config_path=config_path)

    print("[INFO] starting e2e test")
    print(f"[INFO] lyric_path={lyric_path}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] generate_type={generate_type}")
    if config_path:
        print(f"[INFO] config_path={config_path}")

    ok = client.process_lyric_file(
        file_path=lyric_path,
        output_dir=output_dir,
        auto_disconnect=auto_disconnect,
        generate_type=generate_type,
    )

    if ok:
        print("[OK] e2e test passed")
        return 0

    print("[ERROR] e2e test failed")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Client/Server e2e test with auto SSH connection, inference and download.",
    )
    parser.add_argument(
        "--lyric-path",
        default=DEFAULT_LYRIC_PATH,
        help="Path to lyric.jsonl used for test.",
    )
    parser.add_argument(
        "--generate-type",
        choices=["normal", "bgm", "both"],
        default="both",
        help="Generation mode for inference.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional config path. Defaults to client/config_client_server.json if present.",
    )
    parser.add_argument(
        "--keep-connection",
        action="store_true",
        help="Keep connection after test finishes.",
    )

    args = parser.parse_args()
    return run_e2e_test(
        lyric_path=args.lyric_path,
        generate_type=args.generate_type,
        auto_disconnect=not args.keep_connection,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
