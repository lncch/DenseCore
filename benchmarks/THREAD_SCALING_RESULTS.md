# Requested CPU thread settings

2026-10-15. Benchmark implementation 00db6af; runtime a2fb278, package metadata
0.10.0. Apple M4, local macOS arm64, Python 3.13.7 and Torch 2.14.1. Actual
platform, versions, native binary hash and benchmark source hash are retained in
thread-sweep-2026-10-15-summary.json.

## Controls and gates

The parent passes --threads to every validation/timed child and sets BLAS/OpenMP
environment controls before launching it. Each child configures those controls
before importing NumPy/framework code. Torch intra-op uses the requested value;
inter-op stays one. Child rows record requested values, environment and Torch
getters. The parent rejects inconsistent reported settings. Default remains one.
DenseCore does not expose active Accelerate worker counts, so these are requested
settings, not proof of equal or fully occupied worker pools. Small workloads may
use fewer workers. No CPU affinity or operating-system cache eviction was applied.

Models are float32 ReLU MLPs: small 64-128-32 (12,448 parameters), larger
256-1024-256 (525,568 parameters), each at batch 1/32. Engine controls are DenseCore
eager/compiled, Torch eager SGD and the existing optimized Torch control. That
optimized control uses buffered inference and post-accumulate SGD updates; it is
not the foreach-optimizer control used by benchmark_native_training.py. Torch
compiler mode was not included in this sweep.

Each configuration validates all engines before timing, using identical initial
weights/data and full initial-output/20-step weight snapshots, with finite/shape/
checksum/allclose gates. All 48 pre-timing gates and 144 final process checks pass.
A separate post-sweep check compares each two/four-thread run with its same-engine,
same-model/batch one-thread reference: all 96 payload/hash/first/final-loss checks
pass. Those cross-setting checks run after measurement; they are not prechecks.

Three fresh processes per engine/model/batch/thread configuration. Engine order
alternates within each configuration. Configurations ran serially in this fixed
order: small/1, large/4, small/2, large/2, small/4, large/1. No simultaneous benchmark
or build was started by this task; unrelated host activity was not controlled.
Configuration order is not randomized; thermal/background drift
can affect apparent scaling. These are descriptive local results, not confidence
intervals or universal thread-count recommendations.

## Training microseconds

Each cell is the median of three process medians, with minimum/maximum process
medians in parentheses. Each process contains three 200-step training blocks after
warmup. Losing results remain visible.

| Model/batch | Requested threads | DenseCore eager | DenseCore compiled | Torch eager | Torch optimized |
| --- | --- | --- | --- | --- | --- |
| small/1 | 1 | 46.7 (46.1-46.7) | 51.4 (50.9-51.5) | 104.3 (103.8-106.7) | 95.4 (94.2-95.7) |
| small/1 | 2 | 46.7 (46.7-48.0) | 51.3 (51.0-53.1) | 103.1 (102.5-105.3) | 94.7 (94.2-94.7) |
| small/1 | 4 | 47.7 (47.1-47.7) | 52.3 (51.4-52.5) | 103.7 (103.4-103.8) | 94.8 (93.8-94.9) |
| small/32 | 1 | 53.7 (52.6-53.7) | 60.9 (59.8-61.7) | 108.9 (108.5-109.7) | 99.4 (98.1-100.1) |
| small/32 | 2 | 52.6 (52.5-52.6) | 60.6 (59.6-62.3) | 108.9 (108.5-112.5) | 101.3 (99.3-102.3) |
| small/32 | 4 | 52.9 (52.5-53.8) | 60.7 (60.1-61.4) | 111.8 (109.5-112.6) | 102.2 (99.1-102.6) |
| large/1 | 1 | 283.0 (282.3-382.6) | 400.9 (400.8-504.3) | 288.8 (287.7-308.1) | 275.4 (275.4-275.6) |
| large/1 | 2 | 283.3 (282.8-284.2) | 407.7 (403.4-486.9) | 273.3 (269.2-273.8) | 259.3 (256.9-260.2) |
| large/1 | 4 | 287.0 (282.9-301.1) | 412.2 (401.3-444.9) | 264.2 (263.5-270.9) | 250.6 (249.9-255.1) |
| large/32 | 1 | 468.6 (451.7-504.6) | 599.3 (593.3-633.3) | 387.4 (386.1-447.9) | 374.6 (371.8-428.1) |
| large/32 | 2 | 414.0 (387.7-526.9) | 666.4 (512.3-670.0) | 387.0 (379.2-450.9) | 366.8 (365.3-426.1) |
| large/32 | 4 | 492.5 (454.5-588.3) | 609.2 (604.2-650.4) | 375.0 (372.4-555.3) | 360.3 (359.4-363.3) |

DenseCore eager wins the measured small-model training medians against both Torch
controls at all requested settings. For the larger model, batch-32 training loses
to both controls at all settings. Compiled DenseCore training is slower than its
eager path in every displayed case. Requested two threads reduces the larger
batch-32 eager median in this sweep, but four does not preserve that improvement.
This does not establish consistent DenseCore parallel scaling.

## Inference microseconds

Same process-median/min/max convention; five 500-call inference blocks per process.

| Model/batch | Requested threads | DenseCore eager | DenseCore compiled | Torch eager | Torch optimized |
| --- | --- | --- | --- | --- | --- |
| small/1 | 1 | 5.6 (5.5-5.6) | 8.8 (8.6-8.8) | 12.6 (12.5-12.7) | 6.7 (6.6-6.8) |
| small/1 | 2 | 5.6 (5.6-5.6) | 8.8 (8.7-9.1) | 12.6 (12.5-12.7) | 6.7 (6.6-6.7) |
| small/1 | 4 | 5.6 (5.6-5.7) | 8.8 (8.8-8.9) | 12.5 (12.5-12.5) | 6.7 (6.6-6.7) |
| small/32 | 1 | 10.0 (10.0-10.2) | 13.4 (13.3-13.4) | 17.3 (17.2-17.6) | 11.4 (11.4-11.5) |
| small/32 | 2 | 9.9 (9.9-10.1) | 13.3 (13.1-13.5) | 17.3 (17.3-17.6) | 11.5 (11.3-11.6) |
| small/32 | 4 | 10.0 (9.9-10.0) | 13.5 (13.3-13.5) | 17.1 (17.1-17.3) | 11.4 (11.4-11.6) |
| large/1 | 1 | 16.7 (16.6-18.9) | 20.3 (19.8-20.6) | 23.9 (23.7-24.0) | 17.4 (17.4-17.4) |
| large/1 | 2 | 16.6 (16.5-18.9) | 20.3 (19.9-20.4) | 24.1 (23.5-24.4) | 17.5 (17.3-17.6) |
| large/1 | 4 | 16.6 (16.5-16.6) | 20.3 (20.1-20.4) | 23.8 (23.6-23.9) | 17.4 (17.4-17.5) |
| large/32 | 1 | 113.2 (104.2-140.3) | 109.6 (107.4-126.2) | 105.7 (105.4-105.9) | 104.1 (103.6-105.0) |
| large/32 | 2 | 103.8 (103.6-104.0) | 107.1 (106.9-107.3) | 105.9 (105.8-118.8) | 104.1 (104.1-104.3) |
| large/32 | 4 | 104.2 (104.0-119.2) | 107.5 (106.8-107.6) | 105.5 (105.5-106.1) | 104.6 (104.0-104.7) |

Compare matching rows and engine definitions. Do not combine these timings with
historical one-thread results as if runtime, harness and machine state were fixed.
Whole-process RSS remains available in raw results but includes imports, validation
payloads, caches and temporary allocations; it is not isolated tensor memory.

## Reproduction and provenance

After installing development dependencies and building the extension, invoke the
parent once for each requested setting/model. This exact larger-model command was
one of the completed runs (choose a new output path for another run):

```sh
PYTHONPATH="$PWD/python" .venv/bin/python benchmarks/benchmark_cpu.py --include-densecore-compile --features 256 1024 256 --batches 1 32 --threads 2 --output "$PWD/../../work/cpu-thread-sweep-lutx1bwa/large-t2.json"
```

The existing artifact is protected against overwrite. Small-model runs use
--features 64 128 32. Full command arrays, original benchmark script and serial
orchestrator are preserved under work/cpu-thread-sweep-lutx1bwa relative to workspace
root, with raw logs, JSON and NPZ snapshots. Source hash matches the recorded
benchmark implementation. All 144 process timing rows, ranges and validation
payload checksums are deliberately versioned in the compact summary; binary
payloads and build artifacts remain outside Git.

Three new regression tests failed before --threads existed, then passed after
implementation: child controls with equivalent snapshots, Torch configured counts,
and rejecting zero threads before artifact creation. Nine benchmark tests and the
full 978-test Python suite pass, with two existing NumPy reference warnings.
Independent read-only review found no required blocker. Runtime and native kernels
are unchanged; prior native test/build evidence remains the runtime baseline.

Next evidence gap: separate actual tensor/activation allocation behavior from
whole-process RSS. Measure only well-defined owned storage/workspace capacities
or clearly bounded process deltas; preserve limitations before proposing more
compiled-training allocation changes. Linux and GPU validation remain outstanding.

Updated wheel/source builds pass README metadata, 30 license files and scratch
exclusions. From the unpacked source, 24 benchmark/archive tests pass and one
optional Torch test skips in the installed-wheel environment. Package files and
exact SHA256 receipts stay under work/cpu-thread-sweep-lutx1bwa/packages and
package-receipt.json. Archives precede this final validation note. No raw numerical
arrays, executables or release artifacts are committed; no push or publication.
