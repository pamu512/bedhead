"""Model download helper.

MediaPipe task models are fetched on first run into <repo>/models/ (gitignored).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"

MODELS: dict[str, str] = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
    "selfie_multiclass_256x256.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    ),
}


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[bedhead] downloading {dest.name} ...")

    def hook(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        if total > 0:
            pct = min(100, done * 100 // total)
            print(f"\r[bedhead]   {pct:3d}% ({done // 1024}/{total // 1024} KiB)", end="")
        else:
            print(f"\r[bedhead]   {done // 1024} KiB", end="")

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.rename(dest)


def ensure_models() -> dict[str, Path]:
    """Make sure required MediaPipe models exist locally; download once if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in MODELS.items():
        dest = MODEL_DIR / name
        if not dest.exists():
            _download(url, dest)
        paths[name] = dest
    return paths
