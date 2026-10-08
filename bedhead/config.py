"""Presets: one dataclass, JSON-serializable, drives every retouch strength."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Preset:
    """All strengths are 0.0..1.0. `intensity` is the global multiplier."""

    intensity: float = 0.7          # global dial (Off..Subtle..Full rescue maps to 0/0.4/0.8)
    skin: float = 0.6               # mesh-aware skin smoothing
    under_eye: float = 0.5          # brighten lower-orbit band
    shine: float = 0.6              # tame oily highlights on skin
    teeth: float = 0.35             # whiten inside inner lips when mouth is open
    hairline: float = 0.0           # EXPERIMENTAL: soften stray strands along hairline band
    soft_light: float = 0.25        # gentle exposure lift + warmth, NVIDIA-brightness style
    show_original: bool = False     # A/B bypass (also bypasses virtual camera output)

    def scaled(self, name: str) -> float:
        return max(0.0, min(1.0, getattr(self, name))) * max(0.0, min(1.0, self.intensity))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Preset":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def describe(self) -> str:
        return (
            f"intensity={self.intensity:.2f} skin={self.skin:.2f} under_eye={self.under_eye:.2f} "
            f"shine={self.shine:.2f} teeth={self.teeth:.2f} hairline={self.hairline:.2f} "
            f"soft_light={self.soft_light:.2f}"
        )


# Named profiles used by the UI / CLI shortcuts.
PRESETS: dict[str, dict[str, float]] = {
    "subtle": {"intensity": 0.4},
    "rescue": {"intensity": 0.8},
}
