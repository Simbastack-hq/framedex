# framedex — AGENTS.md

## Nature of this repo

A **Claude Code skill** that describes the `vidx` toolchain. The authoritative instruction file consumed by Claude Code sessions is `SKILL.md` (not this file). This repo is a proper Python package with `src/` layout — install via `uv pip install -e .`.

## Structure

```
pyproject.toml                 # deps, ruff/mypy config, entry points
.pre-commit-config.yaml        # pre-commit hooks (ruff check + format, mypy)
.devcontainer/devcontainer.json # VS Code dev container
scripts/
  setup.py                     # system binaries + model pre-download (not a package installer)
src/framedex/
  __init__.py                  # package init, version from pyproject.toml via importlib.metadata
  index_videos.py              # main pipeline (~1530 lines), entrypoint for `vidx`
  face_db.py                   # face detection + SQLite face DB module
  trip_summary.py              # `vidx-summary` — recursive per-folder summaries
  master_index.py              # `vidx-master` — drive-level _INDEX.md + _INDEX.json
  query.py                     # `vidx-query` — filter sidecars by metadata
```

## Developer commands

```bash
uv pip install -e .            # editable install (changes take effect immediately)
uv run ruff check src/framedex # lint
uv run ruff format src/framedex # format
uv run mypy src/framedex       # type check
pre-commit install             # install pre-commit hooks
```

## Dependencies

- **System binaries** (required): `ffmpeg`, `ffprobe`, `exiftool` — macOS, install via `brew`
- **Python deps** in `pyproject.toml`:
  `whisperx`, `anthropic`, `requests`, `PyYAML`, `insightface`, `onnxruntime`, `opencv-python-headless`
- `scripts/setup.py` verifies system binaries and pre-downloads Whisper + insightface models

## Testing / verification

- **No tests** — there is no test framework, test directory, or test script.
- **Only way to validate changes**: run the scripts on actual video files with `--dry-run` or `--max-files N`

## Platform assumptions

- macOS — references `brew`, `/Volumes/` mount points, `~/.zshrc`, `~/.claude/`
- Python >= 3.10

## Key conventions

- Sidecar format: `[filename].description.md` with YAML frontmatter
- Face DB: `~/.framedex/faces.db` (centralized SQLite, not per-drive)
- Default vision backend: `cli` (Claude Max CLI, `claude -p`)
- Default vision model: `claude-haiku-4-5` (aliased as `haiku` in `--vision-model`)
- Default whisper model: `large-v3-turbo`
- Context file: `.video-context.md` at scan root for vision priors + Whisper proper-noun biasing

## When editing scripts

- `index_videos.py` defers heavy imports (`whisperx`, `yaml`, `insightface`) until after setup is verified — keep this pattern for any new heavy deps
- `anthropic` SDK is imported lazily only on the `--backend api` code path (via `TYPE_CHECKING` guard)
- The script uses `subprocess` for `ffmpeg`, `ffprobe`, `exiftool`, and `claude -p` — always preserve the env scrubbing that strips `ANTHROPIC_API_KEY` from the `claude -p` subprocess
- All imports are proper package imports (`from framedex import face_db`) — no `sys.path` hacks
