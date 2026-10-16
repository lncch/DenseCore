# Current CPU measurement checkpoint

Date: 2026-10-15. Apple M4, macOS arm64, Python 3.13.7. Runtime commit f16fbf7;
initial benchmark tree 221da20. DenseCore package metadata 0.10.0, PyTorch 2.14.1,
NumPy 2.5.4. This is a workload-specific checkpoint, not a release or universal
framework comparison.

## Matched workload and results

Float32 CPU MLP: Linear(64,128), ReLU, Linear(128,32); MSE and SGD lr=.001.
Identical deterministic weights, inputs and targets. One thread via environment
controls; Torch thread/inter-op counts explicitly one. Three fresh processes per
engine/batch with alternating order. Table shows microsecond medians and the range
of process medians, not confidence intervals. Peak RSS is whole-process MiB.

| Batch | Engine | Inference us (range) | Training us (range) | Peak RSS MiB |
| --- | --- | --- | --- | --- |
| 1 | densecore | 5.55 (5.53-6.51) | 54.10 (50.13-76.17) | 40.39 |
| 1 | densecore_compiled | 8.75 (8.67-8.78) | 52.93 (52.70-53.07) | 40.28 |
| 1 | torch_eager | 12.84 (12.52-14.37) | 117.66 (104.21-118.33) | 281.78 |
| 1 | torch_optimized_eager | 6.69 (6.69-6.76) | 97.65 (96.14-98.47) | 211.20 |
| 32 | densecore | 10.53 (10.23-13.19) | 56.68 (55.15-64.48) | 40.45 |
| 32 | densecore_compiled | 14.35 (13.23-15.42) | 66.32 (61.19-71.50) | 40.53 |
| 32 | torch_eager | 17.27 (17.27-17.76) | 113.44 (109.90-115.16) | 281.52 |
| 32 | torch_optimized_eager | 11.44 (11.43-11.78) | 107.52 (100.65-115.54) | 210.69 |
| 1024 | densecore | 92.88 (92.66-123.23) | 396.33 (360.38-590.89) | 46.83 |
| 1024 | densecore_compiled | 99.03 (94.48-111.59) | 484.07 (437.89-484.46) | 49.80 |
| 1024 | torch_eager | 92.01 (91.24-107.16) | 385.12 (379.33-446.44) | 286.08 |
| 1024 | torch_optimized_eager | 87.03 (85.93-88.48) | 363.43 (360.60-365.47) | 216.53 |

DenseCore eager has lower median latency for these batch-1/32 runs; batch-1024
loses against optimized Torch eager. DenseCore compiled inference is slower than
its own eager path at every measured batch; compiled training also loses to its
eager path at batches 32/1024. Retain those losses. Small inference differences
with overlapping ranges are not evidence of statistical significance. No
GPU, energy, large-model, multithread or torch.compile result is claimed.

All 12 pre-timing numerical gates and 36 timed-process final gates pass. Gates
check weight/data hashes, initial outputs/loss, weights after 20 steps and final
loss tolerance. They do not prove per-step gradient equivalence or arbitrary
model correctness. The larger case increases batch, not parameter count.

## Measurement boundaries

Steady inference excludes NumPy/list output conversion. Inference has 100 warmups
then five 500-iteration intervals per process. Training starts timing after 40
updates, then three 200-step intervals. GC is disabled within timed intervals and
restored. DenseCore compilation wrapper setup and first execution are separate;
lazy specialization belongs to first execution. First-execution fields include
output conversion and are not dispatch-only latency.

Optimized Torch eager uses reusable inference buffers, an output clone and
post-accumulate SGD hooks. It remains numerically gated against ordinary Torch.
No torch.compile run is included. Import fields measure in-process NumPy/framework
import and thread setup, not interpreter/application startup. Lazy optimizer
imports enter setup. RSS includes imports, validation lists, buffers, caches and
work, not isolated tensor or training memory. Tracemalloc ownership measurements
are separate in HOOK_VALIDATION.md and exclude native allocations.

Installed distribution-file totals: DenseCore 1,610,473 bytes (fresh wheel), Torch
579,769,425 bytes (development environment). These exclude transitive dependencies
and download sizes and compare frameworks with different feature coverage. They
are not full-environment installation footprints.

## Reproduction and provenance

Run from a built development environment with optional Torch installed. Choose
new output paths; the benchmark now refuses overwrite and accepts positive batch
lists. Original defaults remain batches 1/32 and three repetitions.

```sh
PYTHONPATH="$PWD/python" .venv/bin/python benchmarks/benchmark_cpu.py --include-densecore-compile --output /tmp/densecore-cpu-small-new.json
PYTHONPATH="$PWD/python" .venv/bin/python benchmarks/benchmark_cpu.py --include-densecore-compile --batches 1024 --output /tmp/densecore-cpu-large-batch-new.json
```

These are reproduction commands, not new timing claims for a different machine.
Actual outputs and source/native-extension hashes remain outside Git under
../../work/current-cpu-bench-c5y49x: cpu.json, cpu-batch1024.json, provenance.json
and distribution-footprint.json. The second sweep changes only parent CLI batch
selection and overwrite validation; child timing logic is unchanged. Native build
hash was captured before any further runtime build. No raw generated data is
accidentally staged. The ledger links this checkpoint rather than treating old
version benchmark files as current evidence.

Remaining Milestone 2 gates: genuinely larger models, controlled thread sweeps,
application startup, complete dependency footprint, allocation profiling and
optimization of the hook ownership regression. Results support only this measured
small-model subset. Do not declare the whole milestone complete.


## Current optimizer and fused-training sweep

The separate native-training protocol uses the same float32 MLP, batches 1/32,
SGD, momentum .9 and Adam at lr=.001. Three fresh processes per configuration;
all 30 pre-timing gates and 90 timed-process final checks pass. Initial 12-step
loss/weights/optimizer-state gates are followed by 50 warmup updates and five
300-step timed intervals. Final scalar-loss checks do not prove full optimizer-
state equivalence after every timed update.

The control named torch_optimized_eager in this sweep is Torch optimizers with
foreach=True, zero_grad(set_to_none=True), ordinary model forward/backward. It is
not the manual post-accumulate SGD control in the earlier CPU sweep. Do not
combine those control labels into a claim about the fastest possible Torch path.
Fused execution has its documented detached-input/target and supported-module
restrictions; these are not arbitrary eager API equivalents. Uncached fused is
an explicit workspace-cache ablation.

| Optimizer | Batch | Engine | Training us (process range) | Peak RSS MiB |
| --- | --- | --- | --- | --- |
| sgd | 1 | densecore | 54.65 (49.95-70.47) | 38.58 |
| sgd | 1 | densecore_compiled | 53.15 (51.71-55.10) | 38.53 |
| sgd | 1 | densecore_fused | 34.84 (33.63-37.13) | 38.25 |
| sgd | 1 | densecore_fused_uncached | 36.39 (35.94-42.80) | 38.30 |
| sgd | 1 | torch_optimized_eager | 125.81 (117.08-178.33) | 278.27 |
| sgd | 32 | densecore | 55.09 (54.46-56.04) | 38.50 |
| sgd | 32 | densecore_compiled | 64.31 (62.43-65.82) | 38.61 |
| sgd | 32 | densecore_fused | 41.94 (39.93-42.20) | 38.34 |
| sgd | 32 | densecore_fused_uncached | 45.54 (44.23-49.29) | 38.44 |
| sgd | 32 | torch_optimized_eager | 116.01 (114.69-136.99) | 282.06 |
| momentum | 1 | densecore | 51.71 (50.36-51.91) | 38.86 |
| momentum | 1 | densecore_compiled | 56.43 (55.11-58.06) | 38.84 |
| momentum | 1 | densecore_fused | 37.20 (37.07-38.26) | 38.78 |
| momentum | 1 | densecore_fused_uncached | 40.99 (39.23-63.44) | 38.67 |
| momentum | 1 | torch_optimized_eager | 128.10 (125.83-134.41) | 281.91 |
| momentum | 32 | densecore | 59.35 (57.42-69.56) | 39.05 |
| momentum | 32 | densecore_compiled | 64.78 (63.45-86.24) | 38.97 |
| momentum | 32 | densecore_fused | 44.60 (42.83-53.19) | 38.83 |
| momentum | 32 | densecore_fused_uncached | 53.85 (49.15-55.64) | 39.00 |
| momentum | 32 | torch_optimized_eager | 137.98 (136.07-172.80) | 279.41 |
| adam | 1 | densecore | 56.67 (55.94-57.27) | 39.38 |
| adam | 1 | densecore_compiled | 62.21 (60.96-62.55) | 39.45 |
| adam | 1 | densecore_fused | 43.10 (42.80-43.17) | 39.17 |
| adam | 1 | densecore_fused_uncached | 45.51 (45.32-45.51) | 39.22 |
| adam | 1 | torch_optimized_eager | 177.84 (176.79-184.88) | 283.50 |
| adam | 32 | densecore | 63.47 (61.61-63.72) | 39.39 |
| adam | 32 | densecore_compiled | 70.56 (69.36-81.86) | 39.67 |
| adam | 32 | densecore_fused | 51.25 (49.32-58.83) | 39.33 |
| adam | 32 | densecore_fused_uncached | 61.50 (52.89-66.46) | 39.27 |
| adam | 32 | torch_optimized_eager | 182.37 (176.98-190.02) | 283.72 |

Fused DenseCore has lower medians than these foreach Torch controls on all six
small-model configurations. This does not erase the batch-1024 losses above or
establish large-model superiority. Ranges are descriptive with three processes,
not confidence intervals. Whole-process peak RSS has the same scope caveat.

Reproduce into a new path:

```sh
PYTHONPATH="$PWD/python" .venv/bin/python benchmarks/benchmark_native_training.py --repeats 3 --output /tmp/densecore-current-optimizers-new.json
```

Raw optimizer evidence remains beside the CPU artifacts as training-optimizers.json.
The intentionally versioned compact summary includes gates, process intervals,
source hashes and raw artifact checksums. Full raw artifacts stay outside Git.

A construction profile (10,000 scalar adds, cProfile) confirms owner-slot creation
and identity synchronization costs. Files remain in work/milestone-audit relative
to workspace root. No speculative runtime optimization was made from that profile;
the material no-hook construction regression remains open.
