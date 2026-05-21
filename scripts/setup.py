#!/usr/bin/env python3
"""
One-time setup for the framedex skill.

Verifies system binaries (ffmpeg, ffprobe, exiftool) and pre-downloads the
default Whisper + insightface models so the first `fdx` run doesn't stall
mid-index. Pyannote model download is NOT automated because it requires
accepting terms on Hugging Face manually.

Run after `uv pip install -e .`:

    python3 scripts/setup.py

Flags:
    --whisper-model SIZE    Default 'large-v3-turbo'.
    --skip-model-download   Don't pre-download the Whisper model.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import textwrap


def _install_hint(name: str) -> str:
    system = platform.system()
    hints = {
        "Darwin": f"brew install {name}",
        "Linux": f"apt install {name}  (or your distro's package manager)",
        "Windows": f"choco install {name}  (or scoop/winget)",
    }
    return hints.get(system, f"install {name} and add to PATH")


REQUIRED_BINARIES = {
    "ffmpeg": _install_hint("ffmpeg"),
    "ffprobe": _install_hint("ffmpeg"),
    "exiftool": _install_hint("exiftool"),
}


def check_binaries() -> list[tuple[str, str]]:
    missing = []
    for name, install in REQUIRED_BINARIES.items():
        if shutil.which(name) is None:
            missing.append((name, install))
    return missing


def predownload_whisper(model_size: str) -> None:
    print(
        f"Pre-downloading Whisper model '{model_size}' (this can take a few minutes)..."
    )
    import whisperx

    # device="cpu" works everywhere; we just want the model files on disk.
    whisperx.load_model(model_size, device="cpu", compute_type="int8")
    print("Whisper model ready.")


def predownload_insightface() -> None:
    """Pre-download buffalo_l face models (~200MB) + verify install."""
    print("Pre-downloading insightface buffalo_l face models (~200MB)...")
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("insightface ready.")
        return True
    except Exception as e:
        print(f"insightface init failed: {e}")
        print("Face detection will be disabled until this is resolved.")
        print("Workaround: pass --no-faces to fdx.")
        return False


def print_pyannote_instructions() -> None:
    print(
        textwrap.dedent("""
        ─── Speaker diarization setup (one-time, manual) ─────────────────────

        Diarization (who-said-what) uses pyannote models from Hugging Face,
        which require you to:

        1. Create a free HF account: https://huggingface.co/join
        2. Accept the terms for BOTH models (click "Agree" on each page):
              https://huggingface.co/pyannote/speaker-diarization-3.1
              https://huggingface.co/pyannote/segmentation-3.0
        3. Create a read token: https://huggingface.co/settings/tokens
        4. Export the token:

              export HF_TOKEN=hf_yourTokenHere

        Add the export to your ~/.zshrc so it persists.

        Skipping diarization entirely (no HF account required) is fine — pass
        --no-diarize to fdx and you'll still get full transcripts,
        just without speaker labels.
        ──────────────────────────────────────────────────────────────────────
    """).rstrip()
    )


def print_anthropic_instructions() -> None:
    print(
        textwrap.dedent("""
        ─── Anthropic API key setup ────────────────────────────────────────

        For Claude vision (scene descriptions), set one of:

            export ANTHROPIC_API_KEY=sk-ant-...
            echo "sk-ant-..." > ~/.claude/credentials/anthropic-key.txt

        Get a key at: https://console.anthropic.com/settings/keys
        ──────────────────────────────────────────────────────────────────────
    """).rstrip()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--whisper-model",
        default="large-v3-turbo",
        help="Whisper model to pre-download (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip the Whisper model pre-download step",
    )
    args = parser.parse_args()

    print("== framedex setup ==\n")

    missing = check_binaries()
    if missing:
        print("Missing required binaries:")
        for name, install in missing:
            print(f"  {name}: install with `{install}`")
        return 1
    print("ffmpeg / ffprobe / exiftool: OK\n")

    if not args.skip_model_download:
        try:
            predownload_whisper(args.whisper_model)
        except Exception as e:
            print(f"Whisper model download failed: {e}", file=sys.stderr)
            print("You can retry later or pass --skip-model-download.")
            # Don't hard-fail — user can still run the script with on-demand download
            print()
        predownload_insightface()
        print()
    else:
        print(
            "Skipping Whisper + insightface pre-download (will download on first run).\n"
        )

    print_pyannote_instructions()
    print_anthropic_instructions()

    print("\nSetup complete. Try:")
    print("  fdx /Volumes/YOUR-SSD --dry-run         # see count, no API calls")
    print("  fdx /Volumes/YOUR-SSD --max-files 5     # test batch of 5")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
