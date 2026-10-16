# Densecore 0.10 native training engineering report

Date: 2026-10-14. Branch: feat/native-training-step. Base:308c7bd. Work remains local; no remote push, package publication, merge or user-data deletion. MIT source license retained. Original untracked benchmark/profile/backup files preserved.

## Delivered

Added dc.compiled_train_step(model,x,target,optimizer), extending existing unfinished Rust implementation rather than replacing it. The call executes compiled forward, mean MSE, backward and one stateful optimizer update, then returns a detached scalar. Python Parameter identity is stable; native handles carry current parameter versions. SGD, momentum SGD and Adam preserve state across successive fused and ordinary steps.

Added the next measured improvement: one reusable full training workspace per cached shape/gradient specialization for detached fused calls. At most four native plans/workspaces remain cached. Ordinary compiled autograd outputs continue owning their own workspaces. No numerical kernel or optimizer math changed.

## API decisions

Use existing capture and signature guards. Add target at slot P+1; shift only intermediate references, keeping x and parameter input slots fixed. Strict registry checks share the signature walk, avoiding a second named_parameters traversal. Cache executor, parameter slots and publication metadata. Recapture supported structural/mode changes; reject unsupported dispatch or registration.

Accepted public subset: built-in Linear/ReLU/Sequential already enabled with compile(model), mean MSE, exact registered trainable Parameters, detached float32 CPU x/target, built-in SGD/Adam in matching identity/order. Shared Parameter objects deduplicate into one graph slot. Distinct objects aliasing native inputs, frozen parameters, unsupported subclasses/forward methods, LayerNorm, differentiable x/target and disabled gradient mode raise. No silent eager fallback in fused API.

Fused gradients are private and do not accumulate or clear existing eager .grad. Ordinary training remains unchanged. Users must not call backward or optimizer.step again for the same fused batch. Native versions invalidate overlapping older autograd graphs. Concurrent fused calls serialize; concurrent ordinary state mutation is outside supported contract. Ordinary GIL Python only.

```python
import densecore as dc
model = dc.compile(dc.nn.Sequential(dc.nn.Linear(64, 128), dc.nn.ReLU(), dc.nn.Linear(128, 32)))
optimizer = dc.optim.Adam(model.parameters(), lr=0.001)
x = dc.tensor([[0.1] * 64])
target = dc.tensor([[0.0] * 32])
loss = dc.compiled_train_step(model, x, target, optimizer)
```

## Execution and atomicity

```text
Python guards + cached capture
             |
Native locks: optimizer -> autograd engine -> executor
             |
Reused workspace: forward -> MSE -> backward
             |
Stage parameter snapshots + optimizer state
             |
Allocate Python handles, result, setter arguments
             |
Commit native versions/state -> publish Parameter slots
             |
Detached loss
```

Preserved existing optimizer-to-engine lock order; unlocked helpers prevent double engine locking. All parameter/shape/version/gradient validation and replacement/state preparation complete before native mutation. Py::new allocations originally occurred after optimizer commit; independent review exposed recoverable MemoryError publication risk. Binding now allocates Python wrappers, detached loss, list/result tuple and setter argument tuples before commit. A validated builtin Tensor member descriptor publishes under held GIL, with no Python bytecode between parameter assignments. Wrong descriptor owner/writability is checked before mutation.

GC allocation callbacks on CPython3.11 can reenter native locks. GcPause uses CPython C API to pause GC before acquiring mutexes and restores its original mode after every mutex guard drops, including errors. The Python3.11 installed-wheel callback test verifies optimizer inspection does not deadlock. Fatal Rust allocation/process interruption still cannot guarantee recovery. These limits are documented rather than presented as a general process-level transaction system.

## Files and responsibilities

- python/densecore/native_training.py: strict cached public runner, cooperative fused-call serialization and loss wrapper; __init__.py exports API.
- python/densecore/compilation.py: optional strict training registry checks within existing signature traversal; ordinary capture behavior unchanged.
- src/graph_api.rs: MSE graph binding, fused binding, specialization workspace cache, precommit Python allocation/publication, GC pause.
- src/graph_backward.rs and src/optim.rs: preparation/commit boundary, reusable workspace lifecycle; ordinary paths share existing transaction semantics.
- tests/test_native_training.py and Rust optimizer test: state/identity/failure/alias/guard/thread/trace/GC/workspace regressions. benchmarks/benchmark_native_training.py and new JSON/profile/report artifacts: matched evidence. README/CHANGELOG/version metadata and docs reports/manifests: local candidate packaging.

## Correctness evidence

Full source suite:288 Python tests passed on macOS with PyTorch oracle tests enabled. Release Rust:53 passed,6 manual benchmarks ignored in ordinary suite. Release clippy warnings denied, Rust formatting and diff checks passed. Independent reviewers examined lock order, publication staging, cache lifecycle and benchmark controls; findings fixed before completion.

Dedicated Python tests compare loss, parameters and optimizer state at each of12 consecutive steps for SGD, momentum and Adam, with weight decay and shared/unshared parameters. They continue through ordinary training on the same state. Other tests cover parameter identity, ordered optimizer matching, frozen/replaced/aliased parameters, shape failure, unsupported dispatch, mode changes, registry changes, overflow failure atomicity, stale retained graphs, preserved eager gradients, fused threads, Python trace publication visibility and GC mode restoration/callback reentry. Workspace checks cycle through more than four batch shapes, verify eviction/revisit parity, retain detached loss snapshots and recover after failed optimizer transaction. Rust discard-prepared-update test verifies an aborted prepared update changes neither versions nor state.

## Performance evidence

Apple M4, CPU float32, 64→128→32 network, mean MSE, learning rate .001, momentum .9 or Adam defaults. Final experiment uses five fresh processes per engine/cell, batches1/32, five300-step blocks after50warmups. Alternating engine order; requested single-thread environment; PyTorch intra/inter-op threads set to1. GC disabled only during timed blocks. Compare eager, ordinary compiled, fused cached, fused uncached and PyTorch foreach optimizers. Same initial arrays, data and update counts. No inference-only operation is substituted for a complete training step.

Thirty pre-timing gates passed: twelve losses plus weights/state after12steps, rtol1e-4/atol1e-6. All150 timed processes passed corresponding parity and final-loss/hash gates; final-loss rtol1e-3/atol1e-5. Unit tests separately compare weights/state after every step. Setup/import/first-call outside steady-state timings. Peak process RSS includes setup/imports/validation/timing and is not activation-only memory or an energy estimate.

Final medians: cached fused5–15% faster than same-build uncached fused,10–25% faster than ordinary compiled. Eager remains faster in all six median comparisons. Representative SGD: batch1 fused18.69µs, ordinary compiled23.36µs, eager17.20µs; batch32 fused23.38µs, compiled27.04µs, eager21.70µs. Cached process RSS37.98–39.22MiB, comparable to uncached38.02–39.33MiB. PyTorch controls around282–283MiB. Adam includes substantial process variation; full medians/min/max/raw blocks are preserved. These are narrow Mac workload findings, not broad PyTorch/GPU superiority.

Initial fused prototype was slower in every cell. Python profile found repeated parameter-tree traversal; guard refinement removed it. Temporary native timers then isolated fresh workspace creation1.625µs(batch1)/2.333µs(batch32), justifying full fused reuse. Same-build uncached control avoids attributing host-wide drift to code changes. The earlier layout-only cache experiment remains discarded. Native optimizer-state cloning was sampled as a future candidate, not changed without a new matched experiment.

Detailed tables/protocol: benchmarks/NATIVE_TRAINING_RESULTS.md. Final raw: benchmarks/native-training-workspace.json. Source/build provenance: docs/NATIVE_TRAINING_BENCHMARK_PROVENANCE.json. Benchmarks timed development0.9-tagged code corresponding to completed runtime implementation; subsequent0.10 bump changes version metadata only.

## Wheels and platforms

Local0.10 wheels, installed without Torch, each passed187 tests:

- macOS ARM64, CPython3.13: clean environment.
- Linux ARM64, CPython3.11.17, glibc2.36: isolated Docker build,53Rust tests and releaseclippy, installed-package suite including GC callback regression, compiled regression and pretrained letterT inference.
- Linux x86-64 under local emulation, CPython3.13.16, glibc2.36: same checks. Correctness/installability evidence, no native x86 speed claim.

Linux backend remains reference GEMM, macOS Accelerate. Python3.12 and other OS/architecture/version combinations remain unverified for0.10; prepared remote CI has not run. No wheel is bundled with Torch or dataset. Every inspected wheel contains28 dependency license texts plus the project MIT license (29 files total). Exact filenames/sizes/SHA256: docs/NATIVE_TRAINING_RELEASE_ARTIFACTS.json. Historical0.9 artifacts remain unchanged.

## Commits and decisions

- a94322c: feat: add transactional compiled MSE training step. Includes guard refinement, precommit allocation, GC safety, tests and preserved intermediate evidence.
- 0684351: perf: reuse serialized fused training workspaces. Includes same-build A/B benchmark, failure/eviction/retained-loss tests and evidence.
- The local release-preparation commit containing this report records0.10 metadata, wheel manifests and final verification. No merge or push.

Accepted: opt-in fused API, combined strict guard traversal, precommit publication preparation, full detached fused workspace reuse. Rejected: claiming initial slower fused prototype as a speedup, causal absolute comparisons across drifting runs, reintroducing old layout-only cache without evidence, changing optimizer math solely because sampling showed it.

## Remaining priorities

1. Profile remaining Python dispatch and native optimizer-state copying with paired controls; aim to beat eager consistently before making broader speed claims.
2. Extend fused training to LayerNorm and cross-entropy through explicit parity/failure gates, enabling existing real tabular classifier workload.
3. Optimize Linux math backend and measure native Linux workloads; correctness wheels do not establish Linux performance.
4. Add supported platform/version CI evidence before public release; GPU, broader tensor semantics and pretrained architectures remain later work.

Next concrete experiment: profile the current cached fused path with representative Adam training and compare a transactional state-copy reduction against unchanged eager/compiled controls. Keep fixed numeric tolerances and record unsuccessful attempts.

These wheel records are historical runtime-validation builds predating the documentation milestone. Current release-readiness artifacts are recorded separately in RELEASE_READINESS_ARTIFACTS.json.
