"""Tier B ANE/GPU port: convert inswapper_128.onnx to CoreML + profile.

Conversion path (the direct coremltools ONNX converter is GONE in
coremltools >= 8, and the 7.x wheels lack the frontend on py3.12/arm64 --
both verified the hard way on 2026-10-25):

    ONNX --onnx2torch--> torch module --jit.trace--> coremltools --> mlprogram

Candidates produced and profiled (real inputs: live portrait crop +
emap-projected arcface latent, the exact insightface INSwapper.get
contract):

    mlprogram-fp32   - full-graph ML Program, fp32
    mlprogram-fp16   - fp16 compute precision (GPU sweet spot; measured
                       16.7-18 ms swap on M-series vs 97-271 ms CPU/ANE)

Each candidate is checked for numeric agreement (corr > 0.99) against the
ONNX CPU reference before latency counts. GPU (ComputeUnit.ALL) is the
deployment configuration recorded in bedhead/tierb.py.

Run:  .venv/bin/python scripts/tierb_coreml_port.py [--frames 60]
Requires: onnx2torch, torch, coremltools (>=8 fine; no ONNX frontend needed)
Writes models/inswapper_128_coreml_mlprogram_fp{16,32}.mlpackage (gitignored,
regenerated on demand) + docs/tierb-coreml-<date>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "models" / "inswapper_128.onnx"
OUT_DIR = ROOT / "models"
DOCS = ROOT / "docs"
PORTRAIT = ROOT / "../one-piece-looks/apps/client/public/fixtures/face-portrait.jpg"


def real_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(target_blob, latent, aimg) exactly as insightface INSwapper.get builds them.

    Contract (verified against insightface.model_zoo.inswapper source):
      blob  = blobFromImage(aimg, 1.0/255.0, 128, mean=(0,0,0), swapRB=True)
              i.e. RGB channels scaled to [0,1]
      emap  = graph.initializer[-1]  (512x512)
      latent = (normed_embedding @ emap), renormalized to unit norm
    """
    import onnx
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    from onnx import numpy_helper

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = cv2.imread(str(PORTRAIT))
    if img is None:
        raise SystemExit(f"portrait fixture missing: {PORTRAIT}")
    faces = app.get(img)
    if not faces:
        raise SystemExit("no face in portrait fixture")
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    aimg, _ = face_align.norm_crop2(img, f.kps, 128)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, (128, 128),
                                 (0.0, 0.0, 0.0), swapRB=True)
    emap = numpy_helper.to_array(onnx.load(str(ONNX_PATH)).graph.initializer[-1])
    if emap.shape != (512, 512):
        raise SystemExit(f"unexpected emap shape {emap.shape}")
    latent = f.normed_embedding.reshape(1, -1) @ emap
    latent /= np.linalg.norm(latent)
    return blob.astype(np.float32), latent.astype(np.float32), aimg


def onnx_reference(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return sess.run(["output"], {"target": target, "source": source})[0]


def convert(kind: str) -> Path:
    """ONNX -> torch -> trace -> CoreML (the working two-hop path)."""
    import coremltools as ct
    import onnx2torch
    import torch

    torch.manual_seed(0)
    module = onnx2torch.convert(str(ONNX_PATH)).eval()
    x = torch.randn(1, 3, 128, 128)
    s = torch.randn(1, 512)
    with torch.no_grad():
        traced = torch.jit.trace(module, (x, s))

    precision = {"mlprogram-fp32": ct.precision.FLOAT32,
                 "mlprogram-fp16": ct.precision.FLOAT16}[kind]
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="target", shape=(1, 3, 128, 128)),
            ct.TensorType(name="source", shape=(1, 512)),
        ],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS13,
    )
    dest = OUT_DIR / f"inswapper_128_coreml_{kind.replace('-', '_')}.mlpackage"
    mlmodel.save(str(dest))
    return dest


def profile(path: Path, target: np.ndarray, source: np.ndarray,
            ref: np.ndarray, frames: int, units_label: str) -> dict:
    import coremltools as ct

    units = ct.ComputeUnit.ALL if units_label == "ALL" else ct.ComputeUnit.CPU_ONLY
    mlmodel = ct.models.MLModel(str(path), compute_units=units)
    for _ in range(5):  # warm-up: compilation + placement
        mlmodel.predict({"target": target, "source": source})
    times: list[float] = []
    corr = max_abs = float("nan")
    for i in range(frames):
        t0 = time.perf_counter()
        out = mlmodel.predict({"target": target, "source": source})
        times.append((time.perf_counter() - t0) * 1000)
        if i == 0:
            pred = np.asarray(next(iter(out.values()))).astype(np.float32)
            if pred.shape != ref.shape:
                pred = pred.reshape(ref.shape)
            max_abs = float(np.abs(pred - ref).max())
            corr = float(np.corrcoef(pred.ravel(), ref.ravel())[0, 1])
    return {
        "path": str(path.name),
        "compute_units": units_label,
        "latency_ms_mean": statistics.mean(times),
        "latency_ms_p95": statistics.quantiles(times, n=20)[18],
        "max_abs_err_vs_onnx": max_abs,
        "corr_vs_onnx": corr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    target, source, _ = real_inputs()
    print("== ONNX CPU reference (real inputs)")
    ref = onnx_reference(target, source)
    print(f"   ref: mean {ref.mean():+.4f} std {ref.std():.4f} "
          f"(all-zero ref = bad inputs)")

    results = []
    for kind in ("mlprogram-fp32", "mlprogram-fp16"):
        print(f"== convert {kind} (onnx -> torch -> CoreML)")
        try:
            path = convert(kind)
        except Exception as e:  # noqa: BLE001
            print(f"   CONVERT FAILED: {e.__class__.__name__}: {e}")
            results.append({"path": kind, "error": f"{e.__class__.__name__}: {e}"})
            continue
        for units in ("ALL", "CPU_ONLY"):
            print(f"   profile {path.name} [{units}]")
            try:
                r = profile(path, target, source, ref, args.frames, units)
            except Exception as e:  # noqa: BLE001
                print(f"   PROFILE FAILED: {e.__class__.__name__}: {e}")
                results.append({"path": kind, "compute_units": units,
                                "error": f"{e.__class__.__name__}: {e}"})
                continue
            print(f"   mean {r['latency_ms_mean']:6.1f} ms | p95 {r['latency_ms_p95']:6.1f} | "
                  f"corr {r['corr_vs_onnx']:.4f} | maxerr {r['max_abs_err_vs_onnx']:.4f}")
            results.append(r)

    dest = DOCS / f"tierb-coreml-{date.today()}.json"
    dest.write_text(json.dumps({
        "date": str(date.today()),
        "conversion": "onnx -> torch (onnx2torch) -> jit.trace -> coremltools mlprogram",
        "onnx_reference_stats": {"mean": float(ref.mean()), "std": float(ref.std())},
        "frames": args.frames,
        "results": results,
    }, indent=2))
    print(f"saved: {dest}")
    ok = [r for r in results if "corr_vs_onnx" in r and r["corr_vs_onnx"] > 0.99]
    if ok:
        winner = min(ok, key=lambda r: r["latency_ms_mean"])
        print(f"\nbest valid: {winner['path']} [{winner['compute_units']}] "
              f"at {winner['latency_ms_mean']:.1f} ms/frame")
    else:
        print("\nno candidate both ran and matched the ONNX output (corr > 0.99)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
