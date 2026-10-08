"""Optional Tk control panel: sliders over a shared preset, live while bedhead runs.

Run alongside the CLI:  bedhead --cam  (then)  python -m bedhead.panel
The panel edits ~/.bedhead/preset.json; the pipeline hot-reloads it each second.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .config import Preset

PRESET_PATH = Path.home() / ".bedhead" / "preset.json"


class Panel(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Bed Head — control panel")
        self.minsize(360, 420)
        self.preset = Preset()
        self._load()

        frm = ttk.Frame(self, padding=16)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Bed Head", font=("system", 22, "bold")).pack(anchor="w")
        ttk.Label(frm, text="look presentable on video calls", foreground="#777").pack(anchor="w")

        self.vars: dict[str, tk.DoubleVar] = {}
        rows = (
            ("intensity", "Intensity (global)"),
            ("skin", "Skin smoothing"),
            ("under_eye", "Under-eye brighten"),
            ("shine", "Shine control"),
            ("teeth", "Teeth whitening"),
            ("hairline", "Hairline soften (exp.)"),
            ("clothes", "Clothes: crease softening"),
            ("stain", "Clothes: stain fade"),
            ("logo_blur", "Clothes: blur logos/text"),
            ("soft_light", "Soft light"),
        )
        for name, label in rows:
            row = ttk.Frame(frm)
            row.pack(fill="x", pady=6)
            ttk.Label(row, text=label, width=22).pack(side="left")
            var = tk.DoubleVar(value=getattr(self.preset, name))
            self.vars[name] = var
            ttk.Scale(row, from_=0.0, to=1.0, variable=var, command=lambda _v, n=name: self._on_slide(n)) .pack(
                side="left", fill="x", expand=True
            )
            val = ttk.Label(row, text=f"{var.get():.2f}", width=5)
            val.pack(side="left", padx=(8, 0))
            var.trace_add("write", lambda *_a, n=name, l=val: l.config(text=f"{self.vars[n].get():.2f}"))

        self.status = ttk.Label(frm, text="auto-saving to ~/.bedhead/preset.json", foreground="#7a4a3a")
        self.status.pack(anchor="w", pady=(14, 0))
        ttk.Button(frm, text="Reset to defaults", command=self._reset).pack(anchor="w", pady=8)

    def _load(self) -> None:
        if PRESET_PATH.exists():
            try:
                self.preset = Preset.load(str(PRESET_PATH))
            except Exception:  # noqa: BLE001
                pass

    def _on_slide(self, name: str) -> None:
        setattr(self.preset, name, float(self.vars[name].get()))
        self._save()

    def _reset(self) -> None:
        self.preset = Preset()
        for name, var in self.vars.items():
            var.set(getattr(self.preset, name))
        self._save()

    def _save(self) -> None:
        PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.preset.save(str(PRESET_PATH))


if __name__ == "__main__":
    Panel().mainloop()
