"""Tier B ANE port, step 1: convert inswapper_128.onnx to CoreML + profile.

Uses REAL model inputs (live portrait crop + emap-projected arcface latent,
replicating insightface's INSwapper.get preprocessing exactly) so the ONNX
reference produces the true in-distribution output, then converts to three
CoreML candidates and measures warm per-frame latency + numeric agreement:

  mlprogram-fp32 / mlprogram-fp16 / neuralnetwork-fp16

Run:  .venv/bin/python scripts/tierb_coreml_port.py [--frames 60]
Writes models/inswapper_128_coreml_*.mlpackage + docs/tierb-coreml-<date>.json
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
      emap  = graph.initializer[-1]  (512x512, present in the facefusion file)
      latent = (normed_embedding @ emap), renormalized to unit norm
    """
    import onnx
    from onnx import numpy_helper
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = cv2.imread(str(PORTRAIT))
    faces = app.get(img)
    assert faces, "no face in portrait fixture"
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    from insightface.utils import face_align

    aimg, _ = face_align.norm_crop2(img, f.kps, 128)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, (128, 128),
                                 (0.0, 0.0, 0.0), swapRB=True)
    emap = numpy_helper.to_array(onnx.load(str(ONNX_PATH)).graph.initializer[-1])
    assert emap.shape == (512, 512), f"unexpected emap shape {emap.shape}"
    latent = f.normed_embedding.reshape(1, -1) @ emap
    latent /= np.linalg.norm(latent)
    return blob.astype(np.float32), latent.astype(np.float32), aimg


def onnx_reference(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return sess.run(["output"], {"target": target, "source": source})[0]


def convert(kind: str) -> Path:
    import coremltools as ct

    precision = {
        "mlprogram-fp32": ct.precision.FLOAT32,
        "mlprogram-fp16": ct.precision.FLOAT16,
        "neuralnetwork-fp16": None,
    }[kind]
    kwargs = dict(
        model=str(ONNX_PATH),
        inputs=[
            ct.TensorType(name="target", shape=(1, 3, 128, 128)),
            ct.TensorType(name="source", shape=(1, 512)),
        ],
        source="onnx",
    )
    if kind.startswith("mlprogram"):
        mlmodel = ct.convert(
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=precision,
            **kwargs,
        )
    else:
        mlmodel = ct.convert(**kwargs)
    dest = OUT_DIR / f"inswapper_128_coreml_{kind.replace('-', '_')}.mlpackage"
    mlmodel.save(str(dest))
    return dest


def profile(path: Path, target: np.ndarray, source: np.ndarray,
            ref: np.ndarray, frames: int) -> dict:
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(path))
    for _ in range(5):  # warm-up: compilation + placement
        mlmodel.predict({"target": target, "source": source})
    times: list[float] = []
    max_abs = corr = float("nan")
    for i in range(frames):
        t0 = time.perf_counter()
        out = mlmodel.predict({"target": target, "source": source})
        times.append((time.perf_counter() - t0) * 1000)
        if i == 0:
            pred = np.asarray(list(out.values())[0]).astype(np.float32)
            if pred.shape != ref.shape:
                pred = pred.reshape(ref.shape)
            max_abs = float(np.abs(pred - ref).max())
            corr = float(np.corrcoef(pred.ravel(), ref.ravel())[0, 1])
    return {
        "path": str(path.name),
        "latency_ms_mean": statistics.mean(times),
        "latency_ms_p95": statistics.quantiles(times, n=20)[18],
        "latency_ms_min": min(times),
        "max_abs_err_vs_onnx": max_abs,
        "corr_vs_onnx": corr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    target, source, aimg = real_inputs()
    print("== ONNX CPU reference (real inputs)")
    ref = onnx_reference(target, source)
    print(f"   ref: mean {ref.mean():+.4f} std {ref.std():.4f} "
          f"min {ref.min():+.3f} max {ref.max():+.3f}  (all-zero ref = bad inputs)")

    results = []
    for kind in ("mlprogram-fp32", "mlprogram-fp16", "neuralnetwork-fp16"):
        print(f"== convert {kind}")
        try:
            path = convert(kind)
        except Exception as e:  # noqa: BLE001
            print(f"   CONVERT FAILED: {e.__class__.__name__}: {e}")
            results.append({"path": kind, "error": f"{e.__class__.__name__}: {e}"})
            continue
        print(f"   {path.name}")
        try:
            r = profile(path, target, source, ref, args.frames)
        except Exception as e:  # noqa: BLE001
            print(f"   PROFILE FAILED: {e.__class__.__name__}: {e}")
            results.append({"path": kind, "error": f"{e.__class__.__name__}: {e}"})
            continue
        print(f"   mean {r['latency_ms_mean']:.1f} ms | p95 {r['latency_ms_p95']:.1f} ms"
              f" | corr {r['corr_vs_onnx']:.4f} | maxerr {r['max_abs_err_vs_onnx']:.4f}")
        results.append(r)

    dest = DOCS / f"tierb-coreml-{date.today()}.json"
    dest.write_text(json.dumps({
        "date": str(date.today()),
        "onnx_reference_stats": {"mean": float(ref.mean()), "std": float(ref.std())},
        "frames": args.frames,
        "results": results,
    }, indent=2))
    print(f"saved: {dest}")
    ok = [r for r in results if "latency_ms_mean" in r and r["corr_vs_onnx"] > 0.99]
    if ok:
        winner = min(ok, key=lambda r: r["latency_ms_mean"])
        print(f"\nbest valid: {winner['path']} at {winner['latency_ms_mean']:.1f} ms/frame")
        budget = 45.0 - 12.0  # minus align/freq-composite/paste overhead (~12 ms)
        if winner["latency_ms_mean"] < budget:
            print(f"=> swap inference UNDER the ~{budget:.0f} ms share of the 45 ms budget")
        else:
            print(f"=> still above the ~{budget:.0f} ms inference share; see JSON")
    else:
        print("\nno candidate both ran and matched the ONNX output (corr > 0.99)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
