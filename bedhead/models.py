"""Model download helper.

The MediaPipe face-landmarker task file is fetched once into the user cache
directory (never the repo, never site-packages) and verified by a pinned
sha256 so a tampered or truncated download can never reach the pipeline.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path


def _cache_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "bedhead"
    if os.name == "nt":
        return Path.home() / "AppData" / "Local" / "bedhead" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "bedhead"


MODEL_DIR = _cache_dir() / "models"

# filename -> (url, sha256). Hashes pin the exact bytes Google publishes
# today; if you intentionally bump a model version, recompute and update them.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
_SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
_SEGMENTER_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"

MODELS: dict[str, tuple[str, str]] = {
    "face_landmarker.task": (_MODEL_URL, _MODEL_SHA256),
    "selfie_multiclass_256x256.tflite": (_SEGMENTER_URL, _SEGMENTER_SHA256),
}

# selfie-multiclass label space (verified empirically against the model on a
# real-face photo: class-3 centroid lands mid-face, class-2 lands at the
# shoulders/chest, class-4 on the crown).
SELFIE_CLASSES = {
    0: "background",
    1: "body-skin",
    2: "clothes",
    3: "face-skin",
    4: "hair",
    5: "others",
}
FACE_SKIN_CLASS = 3
HAIR_CLASS = 4
BACKGROUND_CLASS = 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, sha256: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[bedhead] downloading {dest.name} ...")

    def hook(blocks: int, block_size: int, total: int) ->    None:
        done = blocks * block_size
        if total > 0:
            pct = min(100, done * 100 // total)
            print(f"\r[bedhead]   {pct:3d}% ({done // 1024}/{total // 1024} KiB)", end="")
        else:
            print(f"\r[bedhead]   {done // 1024} KiB", end="")

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    got = _sha256(tmp)
    if got != sha256:
        tmp.unlink()
        raise RuntimeError(
            f"{dest.name}: sha256 mismatch (got {got}, expected {sha256}); download discarded"
        )
    tmp.rename(dest)


def ensure_models(needed: set[str] | None = None) -> dict[str, Path]:
    """Make sure required MediaPipe models exist locally; download + verify once.

    `needed` filters by filename so callers only fetch what they use
    (the tracker never triggers the 16 MiB segmenter download).
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, (url, sha256) in MODELS.items():
        if needed is not None and name not in needed:
            continue
        dest = MODEL_DIR / name
        if not dest.exists() or _sha256(dest) != sha256:
            _download(url, sha256, dest)
        paths[name] = dest
    return paths
