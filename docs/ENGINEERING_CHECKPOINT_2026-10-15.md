# DenseCore engineering checkpoint, 2026-10-15

## Current status

Development branch: feat/v0.12-storage. Baseline: feat/v0.11 at 4044e5c. Implemented shared-storage/layout foundations, positive-step slicing with backward scatter, full-Jacobian numerical gradcheck, and a partial v0.13 operator registry. No v0.12 or later release is declared complete. Cargo and Python package metadata remain 0.10.0. Existing history was preserved; no push, merge, tag or package publication occurred.

The reported v0.11 baseline was verified rather than assumed: 388 Python tests and 53 Rust tests passed locally, formatting/Clippy passed, and remote CI run 38648968163 succeeded at 4044e5c, including the sixth milestone. Existing NumPy empty-mean reference warnings were identified.

## Commits created before this report

```text
f02ddec feat: add shared tensor storage and slicing views
cfa697c fix: skip unreachable empty softmax dimensions
f09fdb3 docs: document storage invariants and validation limits
8c5c590 feat: add full-Jacobian float32 gradcheck
0a0ac3d docs: describe numerical gradcheck contract
a7d405b fix: reject malformed gradients in gradcheck
ce987ad feat: register CPU float32 unary math kernels
9da310e docs: record unary dispatch scope and edge checks
6a55320 refactor: register binary kernels and shared shape rules
2e3f005 docs: record registry migration boundaries
```

This checkpoint document is committed separately after these implementation and documentation commits.

## Validation

Local platform: macOS arm64, Python 3.13.7, rustc 1.99.0. Final implementation checkpoint: 2e3f005.

- cargo fmt --check: pass.
- cargo test and cargo test --release: 62 passed each, six explicitly ignored manual benchmarks each; no failures.
- cargo clippy --all-targets -- -D warnings: pass.
- Release editable extension build plus PYTHONPATH=python Python suite: 464 passed, two existing NumPy empty-mean reference warnings.
- Fresh isolated source-distribution installation without PyTorch: 346 passed, nine PyTorch-only skips. Three pre-existing oracle/conversion files excluded as in CI.
- After uninstalling that package, installation of the built macOS arm64 wheel in the same isolated environment: 346 passed, nine PyTorch-only skips, same exclusions. Imports were checked to resolve inside the isolated environment and Torch was absent.
- Artifact inspection: README metadata matches, 30 wheel license files present, experimental/scratch data excluded from the source archive.
- Python examples in docs/GRADCHECK.md and docs/OPERATOR_DISPATCH.md executed successfully.

Current verified artifacts and installation logs are outside Git at:

```text
<repo>/../../work/storage-dispatch-validated.90kDq4
```

Wheel SHA256: d36149f5397abcf69054cc59c585987b41baae93204c68b9c34811a62e9ef97e.
Source archive SHA256: 84471b041c9be780408bee11de8e507555155caac6b47d0e126aab4b646c5cd8.

These artifacts were validated at the implementation checkpoint before this status-only document was added. They are local test candidates, not published release assets.

## Benchmarks

The earlier storage candidate was compared against 4044e5c on the same host with 1024x1024 float32 input, 100 view constructions per sample and seven samples. Median construction times: reshape 103652.50 to 3887.50 ns; transpose 6280717.92 to 1931.25 ns; permute 2725501.25 to 3910.41 ns. These exclude downstream materialization and are not whole-training speed claims.

The retained-view benchmark separately measured creation, views, slicing, cold contiguous materialization and process peak RSS. With 32 retained tensors, one local run recorded zero extra peak RSS for shared views and 102350848 extra peak bytes for materialized tensors. Initial construction already raises peak RSS, so these differences are not live allocation sizes. Reproduce with benchmarks/benchmark_layout.py.

A sequential three-process-repeat MLP comparison of the storage candidate against a separately built v0.11 wheel covered SGD/Adam, batches 1/32, eager/compiled/fused paths and alternating candidate order. Weights/data hashes and loss trajectories matched. Median candidate/baseline latency ratios ranged 0.791 to 1.081. Results were mixed. They predate subsequent registry migration, so no final-branch end-to-end speed improvement is claimed. A separate five-engine PyTorch benchmark passed numerical gates; overlapping exploratory timing samples are excluded from performance conclusions.

Raw measurements remain outside Git under work/v012-view-comparison.json, work/v012-layout-benchmark.json and work/v012-training-regression-reviewed.json. Generated benchmark data, wheels and source archives were not committed.

## Architecture and corrected defects

Immutable Arc-backed storage is separate from checked shape/strides/offset metadata and autograd Node identity. Views share backing allocations. Existing kernels consume logical row-major values; strided layouts materialize once into a shared cache. Parameter replacement retains immutable snapshots and existing logical version checks. No storage mutation API or unsafe Rust was added.

Basic indexing supports integers, positive-step slices and one ellipsis. Slice backward scatters into logical parent coordinates, preserving chained transpose/permutation gradients. Gradcheck compares full Jacobians using independent input copies and checks gradient shapes before NumPy assignment.

Native CPU/float32 metadata now feeds static unary and two-input kernel registrations. Unary negation/exp/log/sqrt/sin/cos have shared mathematical backward formulas outside backend registrations. Eager add/multiply/matmul use registered kernels; graph binary/matmul inference uses registered shape rules. Existing optimized CPU graph/workspace execution is preserved.

Independent read-only reviews produced concrete regression tests. Corrected empty-layout stride overflow, unreachable empty-softmax dimension overflow, gradcheck accepting wrong-shaped gradients, and sqrt backward underflowing a resolvable subnormal seed. Each defect was reproduced before correction.

See TENSOR_STORAGE.md, COMPATIBILITY.md, GRADCHECK.md and OPERATOR_DISPATCH.md for contracts and tradeoffs.

## Limitations

CPU float32 remains the only backend/dtype. GPU, mixed precision, distributed training and higher-order AD are not implemented. New unary math operations are eager only; compiler capture/lowering has not been extended for them. Reductions, normalization and losses still await registry migration. Batched matmul, concatenation, stacking and comparisons remain roadmap work.

Negative strides, advanced/boolean indexing, new-axis indexing and in-place mutation are unsupported. Parameter/view snapshot semantics differ from PyTorch mutable aliases. Non-contiguous reshape conservatively copies even when a more general stride-preserving reshape might exist. Strided materialization caches can retain extra buffers and small views can keep large allocations alive.

No new Linux/Python-matrix CI or GPU hardware validation was run for this local branch. Missing Cargo descriptive metadata still produces a packaging warning. These constraints prevent declaring a general-purpose or release-ready PyTorch alternative.

## Next

Migrate reductions into registered shape/device/dtype schemas with shared eager/compiler validation and existing numerical/gradient gates. Continue normalization/loss schemas afterward, then add device/dtype propagation to compiler IR before introducing device backends. Preserve the storage and alias safeguards throughout.
