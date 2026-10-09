"""Presets: one dataclass, JSON-serializable, drives every retouch strength."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from typing import Any


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _clamp01(value: float) -> float:
    value = float(value)
    if math.isnan(value) or value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


@dataclass
class Preset:
    """All strengths are 0.0..1.0. `intensity` is the global multiplier."""

    intensity: float = 0.7          # global dial (Off..Subtle..Full rescue maps to 0/0.4/0.8)
    skin: float = 0.6               # mesh-aware skin smoothing
    under_eye: float = 0.5          # brighten lower-orbit band
    shine: float = 0.6              # tame oily highlights on skin
    teeth: float = 0.35             # whiten inside inner lips when mouth is open
    hairline: float = 0.0           # EXPERIMENTAL: soften stray strands along hairline band
    clothes: float = 0.5             # crease/wrinkle softening on the garment mask
    stain: float = 0.4               # pull faint stains/shading toward garment color
    logo_blur: float = 0.0           # blur logo/text prints on the garment (redaction)
    soft_light: float = 0.25        # gentle exposure lift + warmth, NVIDIA-brightness style
    color_match: float = 0.0        # reference color match (Reinhard LAB transfer, 0=off)
    vibrance: float = 0.0           # saturation lift weighted to dull pixels (anti-webcam-gray)
    studio_light: float = 0.0       # relight the person only (Apple Studio Light class)
    background_darken: float = 0.0  # darken background for subject pop
    detail_boost: float = 0.0       # face-region unsharp (webcams are soft)
    face_lift: float = 0.0          # face-oval exposure lift (dim-room closer)
    eye_light: float = 0.0          # brighten eye region for an awake look
    background_strength: float = 0.0  # blur/darken background (needs segmentation)
    background_mode: str = "blur"   # "off" | "blur" | "dark"
    show_original: bool = False     # A/B bypass (preview-only; virtual camera keeps retouched)

    def scaled(self, name: str) -> float:
        return _clamp01(getattr(self, name)) * _clamp01(self.intensity)

    def save(self, path: str) -> None:
        """Atomically write the preset so a concurrent reader never sees a torn file."""
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".preset.", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> Preset:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(  # noqa: TRY004 - callers/tests expect ValueError for any bad JSON
                f"preset file must contain a JSON object, got {type(data).__name__}")
        valid = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k not in valid:
                continue
            if k == "show_original":
                if not isinstance(v, bool):
                    raise ValueError(f"preset field 'show_original' must be a boolean, got {v!r}")
                kwargs[k] = v
            elif k == "background_mode":
                mode = str(v).lower()
                if mode not in ("off", "blur", "dark"):
                    raise ValueError(f"background_mode must be off|blur|dark, got {v!r}")
                kwargs[k] = mode
            elif not _is_finite_number(v):
                raise ValueError(f"preset field {k!r} must be a finite number, got {v!r}")
            else:
                kwargs[k] = float(v)
        return cls(**kwargs)

    def describe(self) -> str:
        bg = f"bg={self.background_mode}:{self.background_strength:.2f}" \
            if self.background_strength > 0 else "bg=off"
        return (
            f"intensity={self.intensity:.2f} skin={self.skin:.2f} under_eye={self.under_eye:.2f} "
            f"shine={self.shine:.2f} teeth={self.teeth:.2f} hairline={self.hairline:.2f} "
            f"clothes={self.clothes:.2f} stain={self.stain:.2f} logo_blur={self.logo_blur:.2f} "
            f"soft_light={self.soft_light:.2f} studio={self.studio_light:.2f} "
            f"eye_light={self.eye_light:.2f} {bg}"
        )


# Named profiles used by the UI / CLI shortcuts.
PRESETS: dict[str, dict[str, Any]] = {
    "subtle": {"intensity": 0.4},
    "rescue": {"intensity": 0.8},
    "studio": {
        "intensity": 0.5,
        "soft_light": 0.2,
        "studio_light": 0.6,
        "eye_light": 0.4,
        "background_darken": 0.85,
        "background_strength": 0.85,
        "background_mode": "blur",
    },
    "focus": {
        "intensity": 0.35,
        "background_strength": 0.9,
        "background_mode": "dark",
    },
}


# Background modes accepted by Preset.background_mode (kept in one place for
# the panel combobox and load-time validation).
BACKGROUND_MODES = ("off", "blur", "dark")
