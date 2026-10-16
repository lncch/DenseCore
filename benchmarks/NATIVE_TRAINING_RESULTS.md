# Native compiled training, Optimization 5

API: compiled_train_step(model,x,target,optimizer). CPU float32, compiled built-in Linear/ReLU/Sequential, mean MSE, SGD/momentum/Adam. Existing optimizer state and Parameter objects retained. Native explicit gradients preserve eager leaf .grad; returned loss detached. Unsupported inputs raise. Shared parameter objects deduplicate into one slot; different objects aliasing native input are rejected. Native publication under held GIL closes the Python assignment-loop race. Ordinary concurrent model/optimizer mutation remains unsupported, and fatal allocation failures/process interruption are not recoverable transactions.

## Protocol

Mac CPU, matched64→128→32 network, batches1/32, initialweight/data seeds20261014. Optimizers learningrate.001; momentum.9; Adam defaults. Compare Densecore eager, ordinary compiled, fused and PyTorch optimized eager with foreach optimizer. No hooks that change update timing. All arms run same zero-grad-equivalent fresh-gradient step, forward/MSE/backward/update. Fused preserves previously stored gradients, which start None in benchmark. Five fresh processes per cell, alternating engine order. Twelve losses plus weights/optimizer state after12steps parity gate rtol1e-4/atol1e-6 before timing. 50 warmupsteps, five blocks of300steps, GCdisabled only during timing. Final-loss gate rtol1e-3/atol1e-5; all hashes and gates pass. PeakRSS includes imports/setup/validation/timing, not tensor-only memory.

## Initial result and accepted guard refinement

First implementation slower than ordinary compiled in all six cells. Instrumented Python profile over20000steps: nativecall.387seconds, signaturevalidation.171 cumulative, repeated parameter traversal.162 cumulative. These instrumented times are diagnostic, not public throughput.

Strict registry validation now shares existing signature traversal. Avoid repeated named_parameters walk and redundant trainability/leaf property crossings; native still checks leaf/trainable/version conditions. Cache immutable slot/publication metadata. Registry order, extra child/parameter, replaced/frozen/shared parameters and dispatch guards covered by tests.

Guarded run medians, microseconds per complete step:

| Optimizer / batch | Eager | Ordinary compiled | Fused | PyTorch foreach |
|---|---:|---:|---:|---:|
| sgd / 1 | 17.71 | 23.02 | 20.85 | 61.84 |
| sgd / 32 | 21.60 | 26.71 | 26.94 | 64.05 |
| momentum / 1 | 19.04 | 25.80 | 22.70 | 71.16 |
| momentum / 32 | 23.68 | 28.48 | 28.13 | 73.40 |
| adam / 1 | 22.86 | 28.53 | 26.48 | 98.80 |
| adam / 32 | 25.94 | 32.22 | 31.25 | 99.56 |

Fused improves ordinary compiled batch1 by roughly8–14%; batch32 results range from a small regression to3% improvement. Eager remains faster in every cell. First and guarded runs show substantial host-wide control drift; absolute across-run differences are not causal speedup evidence. Compare engines within each recorded run. Process-level min/max and all block timings remain in raw JSON.

## Next measured bottleneck

Temporary native Instant timers isolated workspacecreation1.625µs(batch1)/2.333µs(batch32), versus train_step_unlocked15.792/19.750µs, after100warmups and1000samples. Timers/printing removed from production source. This creates evidence for full workspace reuse specific to detached fused steps under executor lock, rather than reintroducing discarded layout-only caching for live autograd outputs. Subsequent results recorded separately.

Raw: native-training-baseline.json, native-training-guarded.json, baseline-audit.json, python-profile.json, allocation-profile.json. Controls and numerical gates are preserved. This is a narrow CPU MSE workload, not broad PyTorch superiority.

## Publication transaction refinement

Native optimizer preparation now stages parameter snapshots and moment/state updates without committing versions. Before commit, binding allocates all Python native handles, detached loss, output list/tuple and setter argument tuples. Publication uses validated built-in Tensor slot descriptor and preallocated arguments. Rust discard-prepared-update regression verifies abort leaves versions/state unchanged.

GC is paused with CPython C API while locks are held and restored after lock guards drop, including error paths. This avoids GC callbacks reentering the optimizer mutex during precommit Python allocation. CPython3.11 allocation can collect synchronously ([official source](https://raw.githubusercontent.com/python/cpython/v3.11.15/Modules/gcmodule.c));3.13 schedules collection at evaluation boundaries ([official source](https://raw.githubusercontent.com/python/cpython/v3.13.7/Python/gc.c)). Tests preserve initially enabled/disabled GC mode. Ordinary GIL Python only.

Staged benchmark raw: native-training-staged.json. All24pre-timing and120process/final-loss gates passed. Fused versus ordinary compiled ratios remain favorable in recorded cells; eager remains faster. Final workspace experiment will provide current-code comparisons rather than extrapolating these historical measurements.

## Optimization 6, full fused workspace reuse

Each shape/gradient specialization owns one optional training workspace, borrowed under executor lock. Up to four specializations remain cached. Only detached fused steps reuse this workspace; ordinary live autograd outputs retain their own independent workspace. Saved input handles and readiness flags clear after optimizer preparation, including optimizer failure. Forward/backward errors can retain bounded saved handles until next forward or eviction.

Same-build A/B uses the identical public fused wrapper and native training binding, changing only the private executor constructor cache flag. Five fresh processes per engine/cell, alternating order, same 30 pre-timing gates and150timedprocess/final-loss gates. All passed. No numeric or optimizer math changed.

| Optimizer / batch | Eager µs | Ordinary compiled µs | Fused cached µs | Fused uncached µs | Torch foreach µs |
|---|---:|---:|---:|---:|---:|
| sgd / 1 | 17.20 | 23.36 | 18.69 | 20.93 | 62.43 |
| sgd / 32 | 21.70 | 27.04 | 23.38 | 26.81 | 65.17 |
| momentum / 1 | 19.52 | 25.11 | 20.62 | 23.08 | 71.88 |
| momentum / 32 | 23.45 | 28.28 | 25.61 | 28.28 | 74.02 |
| adam / 1 | 23.30 | 28.25 | 24.89 | 26.35 | 98.26 |
| adam / 32 | 29.86 | 35.44 | 31.09 | 32.69 | 121.94 |

Cached fused median improves uncached in every measured cell, approximately5–15%, and ordinary compiled approximately10–25%. Eager remains faster in all six median comparisons. RSS around38–39MiB remains comparable to uncached; process RSS alone cannot measure precise buffer savings. Adam timing has notable process variation, especially batch32 and one batch1 outlier. Report min/max rather than treating median as guaranteed latency.

Accepted: full fused workspace reuse. Initial slower fused prototype and all intermediate results preserved. Discarded layout-only cache experiment remains discarded. Native optimizer-state copy was sampled but not changed in this mission; future optimization requires its own matched experiment. Raw final: native-training-workspace.json. Reproduce with benchmark_native_training.py; output refuses overwrite.
