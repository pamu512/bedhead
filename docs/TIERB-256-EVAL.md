# Tier B 256px model evaluation: hyperswap_1a_256 (CLOSED)

Date: 2026-10-09. Verdict: **not usable in bedhead today**; investigation
closed with findings below. The shipped 128px INSwapper + frequency-separable
composite (native-res texture, HF = raw exactly) remains the Tier B path.

## What was tried

- Model: hyperswap_1a_256.onnx (facefusion models-3.3.0, 384 MB,
  sha256 c0e98a8a...), interface `source[1,512] + target[1,3,256,256] ->
  output[1,3,256,256] (+ unused mask)`.
- Ran facefusion's EXACT preprocessing contract (fetched from their source):
  target `(RGB/255 - 0.5)/0.5` CHW from an arcface_128-template 256px crop;
  decode `out*0.5+0.5` clip, RGB->BGR.

## Findings

1. **Performance**: 3.4 s/frame warm CPU (9.1 s first-run). 46x the current
   128px INSwapper (74 ms). Offline-only on Apple Silicon CPU; CoreML/ANE
   port would be a requirement before any live use.
2. **Embedding mismatch (root cause of failure)**: with our insightface
   buffalo arcface unit-norm embedding, output is a subtly retouched copy of
   the input (pixel corr 0.987, mean diff 5/255; normed == zeros control
   within 4.7) -- the source embedding is effectively ignored. With the RAW
   (unnormalized) embedding, the decoder is driven far out of distribution:
   magenta cast, merged eye sockets, clipped posterized histograms; the SCRFD
   detector finds no face. facefusion evidently pairs hyperswap with a
   different embedding producer; porting that is possible but only worth it
   if a real-time inference path exists (see 1).

## If reopened

- Identify the exact arcface/embedding model facefusion uses for hyperswap
  (their `embedding_norm` path) and reproduce its embedding space.
- Profile a CoreML-converted graph on ANE before any further work; if that
  lands under ~100 ms/frame, the 256px quality gain (HF texture ceiling)
  justifies the adapter. Otherwise stay on INSwapper_128 + freq composite.
