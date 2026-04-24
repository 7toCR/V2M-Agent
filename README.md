# NingBGM

NingBGM is a multimodal pipeline that analyzes text, audio, images, and video, then produces structured scene understanding results and music-generation prompts. The simplest open-source workflow is to run the local Team1 -> Team2 -> Team3 pipeline and stop before remote audio generation.

## What This Repository Does

- Team1 converts a user request into task packets.
- Team2 merges multimodal inputs into structured scene descriptions.
- Team3 turns scene results into music prompts and lyric-style output.
- The optional client/server audio generation stage exists in the codebase, but server deployment documentation is intentionally left blank for now.

## Requirements

- Python 3.12 or newer
- `uv`
- Valid API keys for the configured model providers

## Quick Start

### 1. Clone the repository

```powershell
git clone https://github.com/7toCR/V2M-Agent.git
cd NingBGM
```

### 2. Create a virtual environment and install dependencies

```powershell
uv venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Create a `.env` file

Copy `.env.example` to `.env` in the repository root:

```env
MCP_API_KEY=your_api_key
DASHSCOPE_API_KEY=your_api_key
```

These two variables are required for the full local pipeline.

### 4. Optional: update model configuration

If you want to change models, base URLs, or token limits, edit these files:

- `Team1/config_requirement_supervisor.json`
- `Team2/config_scene_understanding.json`
- `Team3/config_music_generation.json`

The default configs use DashScope-compatible endpoints, but you can replace them with your own compatible setup.

If you plan to use the optional remote generation stage later, copy `client/config_client_server.example.json` to `client/config_client_server.json` and fill in your own server and SSH values.

### 5. Run a smoke test with the built-in sample

This command only checks input discovery and argument resolution:

```powershell
uv run --no-sync python main.py --test-scene-17 --dry-run
```

### 6. Run the local pipeline

For open-source users, this is the recommended command because it skips the remote server stage and only generates structured outputs locally:

```powershell
uv run --no-sync python main.py `
  --input-dir ".\Input\scene_17" `
  --output-dir ".\Output\scene_17" `
  --requirement "Analyze the provided text, audio, image, and video files, build a unified multimodal scene understanding result, and generate structured background music prompts that match the mood, pacing, and narrative arc." `
  --piece 2 `
  --skip-server-inference
```

## Prepare Your Own Input

Put files for one scene in a single folder. The pipeline scans files recursively and groups them by extension.

Example:

```text
Input/
  scene_001/
    scene.txt
    scene.mp3
    scene.png
    scene.mp4
```

Supported file types:

- Text: `.txt`, `.json`, `.jsonl`, `.md`, `.doc`, `.docx`, `.pdf`
- Audio: `.mp3`, `.wav`, `.flac`, `.aac`, `.m4a`, `.ogg`, `.wma`
- Image: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`, `.svg`
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`

## Output

The pipeline writes results to the directory passed to `--output-dir`.

Typical files:

- `team1_result.json`
- `team2_task_packet.json`
- `team2_result.json`
- `team3_task_packet.json`
- `team3_result.json`
- `scene.jsonl`
- `lyric.jsonl`
- `text.jsonl`
- `audio.jsonl`
- `photo.jsonl`
- `video.jsonl`
- `pipeline_meta.json`

## Useful CLI Options

- `--test-scene-17`: use the built-in sample input and output paths
- `--dry-run`: only inspect inputs and print the resolved requirement
- `--piece`: control how many music prompt records Team3 should generate
- `--skip-server-inference`: skip remote audio generation and keep the run fully local
- `--allow-fallback-packets`: continue with deterministic fallback packets if Team1 does not produce downstream packets
- `--generate-type {normal,bgm,both}`: generation mode for the remote server stage
- `--client-config-path`: custom path to the client/server config file
- `--no-auto-ssh`: connect directly to the configured server URL without creating an SSH tunnel

## Server Deployment

TODO

## Notes

- If the run fails with missing environment variables, check `.env` first.
- If you only want reproducible open-source results, use `--skip-server-inference`.
- If you set `--output-dir` outside the repository, the program will still write there and print the resolved path.
