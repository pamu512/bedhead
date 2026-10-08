"""Presets: one dataclass, JSON-serializable, drives every retouch strength."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


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
    soft_light: float = 0.25        # gentle exposure lift + warmth, NVIDIA-brightness style
    show_original: bool = False     # A/B bypass (also bypasses virtual camera output)

    def scaled(self, name: str) -> float:
        return _clamp01(getattr(self, name)) * _clamp01(self.intensity)

    def save(self, path: str) -> None:
        dest = Path(path)
        payload = json.dumps(asdict(self), indent=2)
        tmp = dest.with_name(dest.name + ".part")
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str) -> "Preset":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: preset JSON must be an object")
        valid = {f.name for f in fields(cls)}
        kwargs: dict = {}
        for key, value in data.items():
            if key not in valid:
                continue
            if key == "show_original":
                if not isinstance(value, bool):
                    raise ValueError(f"{path}: show_original must be a boolean")
            elif not _is_finite_number(value):
                raise ValueError(f"{path}: {key} must be a finite number")
            kwargs[key] = value
        return cls(**kwargs)

    def describe(self) -> str:
        return (
            f"intensity={self.intensity:.2f} skin={self.skin:.2f} under_eye={self.under_eye:.2f} "
            f"shine={self.shine:.2f} teeth={self.teeth:.2f} hairline={self.hairline:.2f} "
            f"clothes={self.clothes:.2f} stain={self.stain:.2f} "
            f"soft_light={self.soft_light:.2f}"
        )


# Named profiles used by the UI / CLI shortcuts.
PRESETS: dict[str, dict[str, float]] = {
    "subtle": {"intensity": 0.4},
    "rescue": {"intensity": 0.8},
}
