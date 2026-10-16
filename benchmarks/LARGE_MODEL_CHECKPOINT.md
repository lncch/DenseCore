# Larger-model CPU checkpoint

2026-10-15, Apple M4 macOS arm64, Python 3.13.7, PyTorch 2.14.1, NumPy 2.5.4,
float32 CPU, one requested thread.
Runtime remains f16fbf7; benchmark base 835b221 with the recorded harness hash.

## Model and measured results

Linear(256,1024), ReLU, Linear(1024,256): 525,568 parameters, compared with 12,448
in the prior small MLP. Deterministic shared weights/data, MSE and SGD lr=.001.
All eight pre-timing numerical gates and 24 final-process checks pass. Three
fresh processes per engine/batch, alternating order. Values are median microseconds
with ranges of process medians, not confidence intervals. RSS is whole-process MiB.

| Batch | Engine | Inference us (range) | Training us (range) | Peak RSS MiB |
| --- | --- | --- | --- | --- |
| 1 | densecore | 17.00 (16.81-19.23) | 328.44 (303.54-375.25) | 64.30 |
| 1 | densecore_compiled | 22.79 (20.13-93.25) | 650.74 (474.08-888.02) | 64.28 |
| 1 | torch_eager | 40.13 (24.74-45.42) | 364.96 (298.84-592.15) | 292.02 |
| 1 | torch_optimized_eager | 17.72 (17.52-17.82) | 311.01 (278.01-462.30) | 220.39 |
| 32 | densecore | 140.08 (120.36-187.74) | 466.98 (435.76-692.91) | 65.52 |
| 32 | densecore_compiled | 119.37 (107.27-182.39) | 752.78 (630.74-885.89) | 68.09 |
| 32 | torch_eager | 153.60 (112.33-160.82) | 721.49 (583.50-740.80) | 296.41 |
| 32 | torch_optimized_eager | 145.09 (107.64-207.93) | 534.48 (523.72-694.10) | 224.00 |

Compilation loses training latency sharply against DenseCore eager at both batches,
and against the optimized Torch control. At batch 32, compiled inference has the
lowest median in this sweep. DenseCore eager training loses to optimized Torch at
batch 1 but has a lower median at batch 32. Small differences with overlapping
ranges are not statistically significant evidence. No universal win follows.

## Binary validation protocol

Parent runs now write initial output and complete 20-step weight snapshots as
NPZ arrays under a fresh owned directory. Checksums protect integrity; separate
checks require matching keys/shapes, finite values and allclose at rtol=1e-4,
atol=1e-6. A review found matching infinities could pass np.allclose; a failing
regression reproduced it before explicit finite checks were added. Shape changes,
payload tampering, exact output paths and overwrite protection are tested.
Legacy direct-child JSON output remains supported.

All numerical values are still checked. Binary storage avoids enormous retained
Python-float lists and JSON strings; no weight sampling substitutes for full
validation. Preserve the payload directory alongside the result JSON. Payload
paths are local provenance and can be relocated with explicit path adjustment.

First-inference conversion now uses ndarray copy rather than tolist, so cold fields
cannot be compared directly with earlier runs. Whole-process RSS also changes with
binary rather than Python-list validation snapshots; do not call that a runtime
memory optimization. Steady inference/training loop boundaries are unchanged:
100 inference warmups, five 500-iteration inference intervals, training after 40
updates with three 200-step intervals. Optimized Torch uses reusable inference
buffers/output clone and post-accumulate SGD hooks, not torch.compile. No fused
larger-model result or multithread/GPU result is claimed.

## Reproduction and provenance

```sh
PYTHONPATH="$PWD/python" .venv/bin/python benchmarks/benchmark_cpu.py --include-densecore-compile --features 256 1024 256 --batches 1 32 --output /tmp/densecore-large-model-new.json
```

Choose a new output path. Preserve its sibling generated arrays directory. Actual
raw outputs are outside Git under ../../work/large-model-nwBwA7: large-model.json,
its payload directory and provenance.json. The intentionally versioned compact
summary records gates, intervals, payload hashes, source/native build identity and
raw JSON checksum. Full binary arrays remain outside Git.

Validation: native code unchanged; Rust debug 89 passed, six existing manual
benchmarks ignored, fmt and Clippy with warnings denied pass. Full Python suite: 825 passed, two existing NumPy empty-mean reference warnings.
Six focused harness regressions pass, including legacy direct-child output.
Independent read-only review found no remaining benchmark gate blocker after the
finite-value correction. Review did not rerun timings/tests. No packaging/runtime API
changed. No version bump, push, merge, tag or publication.

Three-second, one-millisecond macOS samples of separate warmed eager and compiled
processes completed successfully. Both workloads ran 20,000 steps after warmup.
Compiled traces include BackwardPlan backward plus compiled-VJP gradient extraction
and memmove frames. This identifies a copy boundary to investigate, not a proven
sole cause or quantified allocation cost. Raw traces and profile status remain
beside the benchmark JSON; workload script is under work/milestone-audit relative
to workspace root. No runtime/optimizer optimization was made.

Source archive verifies exact harness/test bytes and excludes raw NPZ arrays.
Archive SHA256: 4ef11c0b8c5d24152f48a9ea6302a7c5283176e6cf9b38b89a34ae6c72799fd3.
It precedes this final validation text. Activated fresh-source environment
installation passes all six focused harness regressions without Torch.

Remaining gates: isolate compiled-training regression, thread sweep, application
startup and complete dependency footprint. No speculative optimizer/kernel rewrite
was made from aggregate timings alone.
