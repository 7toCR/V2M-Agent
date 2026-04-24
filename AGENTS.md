# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the entrypoint for the local Team1 -> Team2 -> Team3 pipeline. `Team1/`, `Team2/`, and `Team3/` contain each stage's agent profiles, supervisors, verifiers, and JSON model configs. `client/` and `server/` support the optional remote music-generation stage. Shared helpers live in `tools/`, task-generation utilities and examples live in `task/`, and prompt builders live in `promptStrategy/`. Use `Input/scene_*` for sample inputs and `Output/scene_*` for generated artifacts; do not commit generated outputs, secrets, or local config copies.

## Build, Test, and Development Commands
- `uv venv` and `.\.venv\Scripts\activate`: create and enter the Python 3.12 environment.
- `uv pip install -r requirements.txt`: install project dependencies.
- `uv run --no-sync python main.py --test-scene-17 --dry-run`: smoke test input discovery and CLI wiring.
- `uv run --no-sync python main.py --input-dir .\Input\scene_17 --output-dir .\Output\scene_17 --skip-server-inference`: run the local pipeline without remote audio generation.
- `uv run --no-sync client/test_client_server.py --help`: inspect the optional client/server e2e test entrypoint after local outputs exist.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for modules, functions, and variables, `PascalCase` for classes, and concise docstrings where behavior is not obvious. Keep new files aligned with the current package layout, for example `Team2/verifier/...` or `tools/...`. Prefer explicit type hints on public functions, and keep JSON config names stage-specific, such as `config_scene_understanding.json`.

## Testing Guidelines
This repo currently relies on smoke tests and scenario scripts rather than a full pytest suite. Add focused tests near the feature you change when practical, and use deterministic sample data under `Input/` or `task/examples/`. Name new test files `test_*.py`. Before opening a PR, run the dry-run command above and any stage-specific script you touched. There is no documented coverage target yet.

## Commit & Pull Request Guidelines
Current history uses short subjects like `first commit` and `upload`; keep commit titles short, imperative, and more descriptive than that, for example `add fallback packet validation`. Keep PRs narrow in scope. Include the pipeline stage affected, required config or `.env` changes, exact commands run, and sample output paths if behavior changes. Never include API keys, `.env`, or generated `Output/` artifacts.
