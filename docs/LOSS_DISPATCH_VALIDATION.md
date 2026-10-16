# Loss dispatch validation, 2026-10-15

## Current status and commits

Implementation commit: d24dd5a, on feat/v0.12-storage. Existing mean MSE and cross-entropy forward operations now select registered eager CPU float32 kernels. Shared schemas validate eager operations and compiler inference. Package metadata remains 0.10.0; no new release is declared. Documentation is committed separately.

## Architecture and corrected defect

MseSpec requires identical nonempty shapes and checked element count, including scalar support. It is also used by MSE gradient validation. CrossEntropySpec validates rank-2 logits with positive class width, matching rank-1 label shape and checked dimensions. Execution mode explicitly preserves eager empty-batch NaN/empty-gradient behavior while rejecting compiled empty batches. Integer class metadata remains distinct from differentiable tensors.

Specialized CPU workspace forward uses the same loss arithmetic as eager execution. Compiled cross-entropy previously added a small log-normalizer to a large maximum before subtracting the target logit. A failing regression showed equal logits at positive and negative 1e30 returned zero instead of log(3). Shared shifted-logit arithmetic now preserves the correct result without clamping or hardcoded outputs.

MSE and cross-entropy VJP formulas are preserved. Compiled backward retains its existing softmax calculation and now shares finite/integer/range validation for class indices. This does not claim unified backward dispatch or backend-independent compiler execution. Empty MSE is now rejected during compiler shape inference rather than waiting for execution.

## Validation

- Full Python suite: 536 passed, with two existing NumPy empty-mean reference warnings.
- Rust debug/release suites: 68 passed each, six ignored manual benchmarks each. Formatting and Clippy pass.
- New loss test file: 18 cases covering both loss paths, strided inputs/labels, both MSE operand gradients, shared-storage full Jacobians, invalid-label workspace recovery, empty policies, NaN/infinity propagation and the large-logit cancellation regression.
- Existing fused MSE/cross-entropy training and optimizer failure-atomicity tests remain passing.
- Fresh isolated source installation without PyTorch: 404 passed, 23 oracle skips. Three pre-existing oracle/conversion files excluded as in CI.
- Installed macOS arm64 wheel in that isolated environment: 404 passed, 23 oracle skips, same exclusions. Package imports resolved inside the environment and Torch was absent.
- Artifact checks: README metadata matches, 30 wheel license files present, source scratch/experimental data excluded.

Independent read-only review found no correctness defects in schema consistency, empty-batch policy, stable arithmetic, label checks or failed-workspace recovery. Execution results above provide validation beyond that static review.

Artifacts/logs remain outside the Git repository under this workspace-relative path:

```text
work/loss-validated.8ao9mL
```

Wheel SHA256: cc1fab9001156bf1d5f2c38be3edb03ee6d140f2227331e755c81e062b2e697c.
Source archive SHA256: 9e204f4e739df473d287e4fa2012fc23b56235c08bae1804ff5e1b4aad221ea0.

Artifacts were built from the validated working tree before the implementation commit. They are local test candidates, not published assets. No generated data or binaries were committed. Existing user files were preserved.

## Benchmarks

No new timing measurements or speed claim. This migration fixes numerical correctness and shares forward semantics.

## Limitations

CPU float32 and mean reduction only. Public cross-entropy labels remain integer Python/NumPy metadata; the internal compiled label representation is detached float32 integer-valued metadata. Eager and compiled empty cross-entropy behavior intentionally differ as before. Other reduction modes and public integer tensor dtypes are unsupported.

Registry migration remains partial: Linear, ReLU and softmax-family paths still need migration. Linux/Python-matrix CI was not rerun because nothing was pushed. Existing Cargo descriptive-metadata packaging warning remains.

## Next

Migrate Linear shape validation and eager kernel selection into the registry, preserving optimized GEMM, optional bias and compiled training semantics. Compiler device/dtype propagation remains later work. No push, tags, publication or history rewriting occurred.
