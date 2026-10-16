# Compiled empty broadcast correction

## Defect and fix

Independent predicate review identified CompiledBroadcast evaluating index strides and its output product before recognizing an empty logical result. A native-backed Python regression tested zero dimensions at every position around [usize::MAX, 2]. Eight of twelve add/multiply inference/backward cases failed with shape product overflow or broadcast size overflow; four already passed.

CompiledBroadcast now uses TensorLayout's zero-aware checked count before computing strides. Empty descriptors store no unused strides and emit no logical indices. Existing forward, add-gradient and multiply-gradient loops consequently perform no element access. Nonempty shape/operand validation still runs through shared broadcast_shape, so overflowing nonempty shapes remain rejected. Nonempty arithmetic and the compact stride representation are unchanged.

Tests cover both gradient modes, both binary operations and every zero-axis position. Gradient-enabled cases require an explicit empty seed and check an empty input gradient plus a zero scalar gradient. Existing Rust compact-index/map-oracle tests continue to verify ordinary broadcasting and nonempty overflow rejection. This fixes the concrete gap recorded by predicate review; it is not a claim that every operator accepts all huge empty layouts.

## Continuation

After validation and local commits, continue the roadmap with the Python equality/hash/identity audit before comparison operator overloads, or another dependency-ready operator/autograd unit if that audit shows a larger compatibility risk. Named comparisons and boolean logical APIs already work. CPU float32 math and nondifferentiable bool masks remain the implemented dtype subset. GPU, boolean compiler lowering, custom autograd and distributed work remain incomplete. No version bump, push, merge, tag or publication is authorized.


## Validated checkpoint

Implementation: c54c3c1. Local macOS arm64, Python 3.13.7, rustc 1.99.0. Full Python suite: **648 passed**, including all twelve new native-backed cases. Two existing NumPy empty-mean reference warnings remain. Rust debug/release: **79 passed**, six manual benchmarks ignored; formatting and Clippy with warnings denied passed.

Fresh source install in an activated new environment with maturin 1.15.0 and --no-build-isolation, without PyTorch: **512 passed, 27 skipped**, same warnings. Three Torch-only modules were excluded and optional Torch cases skipped naturally. Direct wheel installation then passed **29 compiler tests**. The four recently added README examples (bmm, joining, bool storage, predicates) ran successfully against that wheel. Artifact checks matched README metadata and all 30 license files, with source scratch exclusions passing.

Archives remain outside Git at workspace directory work/compiled-empty-1792037217. Direct wheel SHA256: 72abc81532be74a9e5d99b38520752e68c22bceb83bfb117d94c77bf50708e02. Source archive SHA256: d2b8a880b81a5ac78843faa12d297bec07559cc56e7b66e6b19565b302bf464e. This validation section was added after those builds. No generated benchmark data or release artifacts were committed.

Protected feat/v0.11 remains at 4044e5c0a415e9e20e0b57cc91b588e5b82f4bbc. No push, merge, tag or publication occurred. Package metadata remains 0.10.0. Next continuation should start with the equality/hash/identity audit described above, preserving the clean working tree and this checkpoint. Autonomous hourly continuation remains authorized; no user continue message is required.
