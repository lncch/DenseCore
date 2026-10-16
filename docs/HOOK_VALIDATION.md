# Gradient hook gate validation

2026-10-15. Base 92b3744, feat/v0.12-storage. Local macOS arm64, Python 3.13.7,
rustc 1.99.0. Package remains 0.10.0; no release declared.

## Correctness and review

Real first-order tensor hooks support late leaf/nonleaf registration, accumulated
node gradients, ordered callback snapshots, removal, None/exact-shape replacement,
retained graphs, custom Function ordering and supported parameter replacements.
Python callbacks remain outside native locks and Arc graph ownership. Weak node
identities pin allocation identity without retaining native values/recipes.

An independent read-only reviewer selected Weak slots over strong graph handles.
Implementation review found two bypasses: missing native fused rejection, then
stale markers after parameter replacement. Both were reproduced by failing tests
before fixes. Follow-up review found no remaining code blocker. Review itself ran
no tests; results below were executed directly.

Native tests prove hook yield/resume ordering and that consumed ancestors die
while Weak identity and output survive. Python tests cover late registration,
removal snapshots, invalid result dtype/shape/grad state, exceptions/retry, per-
callback mutation checks, compiled boundaries, optimizer/model/checkpoint/fused
replacement, GC cycles, deleted wrappers, retained/strided/Torch parity, empty/
detached boundaries, bounded concurrency and a 20,000-node graph.

Rust debug/release each passed 89 tests, 6 existing manual benchmarks ignored.
Formatting and Clippy all targets with warnings denied passed. Release extension
rebuilt. Final full Python suite: 819 passed.

Fresh activated source install, maturin 1.15.0 and --no-build-isolation, without
Torch: 675 passed, 35 skipped; three Torch-only modules excluded. Fresh wheel
hook/custom suites: 64 passed, 1 optional Torch case skipped. Both TENSOR_HOOKS.md
and CUSTOM_FUNCTION.md examples executed against the wheel. Two existing NumPy
empty-mean reference warnings remain. Archives contain the runtime and precede
this checkpoint and final plan/checkpoint documents, not a published release.

Artifact checker passed matching README, 30 license files and source exclusions.
Artifacts outside Git: ../../work/tensor-hooks-bJwIdN.
Wheel SHA256: ba7ccb481bd5e347609239b390c2b4df847ea7a7f61256cd33a31d4691be4ff7.
Source SHA256: ed9e797c771e55f8fbac8d619e122f913bc922ba971ec982db7fb990caef5902.

## Measured ownership cost

benchmarks/benchmark_hook_ownership.py runs separate fresh processes for timing
and traced Python allocation. Three samples per build, 3,000 scalar additions,
no hooks registered. Baseline: installed singleton-dimensions wheel under
../../work/singleton-dims-7owF3M/wheel-env; candidate: working extension/Python
sources. Raw samples: ../../work/milestone-audit/hook-owner-final.json. Baseline
predates the hook implementation; both package metadata versions read 0.10.0.
This is an ownership regression probe, not an equivalent PyTorch benchmark.

Median uninstrumented construction: baseline 7.778 ms, candidate 17.343 ms.
Backward: baseline 2.761 ms, candidate 2.755 ms. Traced Python peak: 2,828 versus
498,871 bytes. Whole-process peak RSS in allocation runs: 41,598,976 versus
42,893,312 bytes, including imports. These are not incremental tensor memory or
statistical equivalence claims. Lazy dictionary/lock allocation reduced the
initial probe's Python peak, but ordinary graph construction still regresses.
No performance improvement claimed. Milestone 2 must profile this cost on actual
networks and scalar-heavy graphs before making competitive claims.

## Scope and remaining milestone gates

Fused training rejects active parameter hooks. Nested hook backward rejects;
custom Function keeps its existing independent ordinary-backward subset.
Callbacks can retain captured Python tensors until removal or owner collection.
No higher-order/forward AD, inference mode or checkpoint wrapper claimed.
INFERENCE_CHECKPOINT_INVESTIGATION.md records concrete semantics and prerequisites
for the latter two. Linux/new remote CI and GPU remain unvalidated here.

No push, merge, tag, publication, user-data deletion or protected-branch changes.
Continue through milestone acceptance checks. Next independent task: current-
revision matched CPU benchmark gates, retaining regressions and losing results.
