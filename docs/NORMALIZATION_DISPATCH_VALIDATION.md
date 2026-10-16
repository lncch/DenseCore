# LayerNorm dispatch validation, 2026-10-15

## Current status and commits

Implementation commit: cc0ba1e. Branch: feat/v0.12-storage. Existing LayerNorm forward now uses registered eager CPU float32 dispatch with shared eager/compiler metadata validation. Package metadata remains 0.10.0; no new release is declared. Documentation is committed separately.

## Architecture

LayerNormSpec checks rank-2 input, positive feature width, checked element count, optional affine shapes and positive finite float32 epsilon. Eager dimensions and graph shape inference call that schema. Compiler contracts can additionally supply an expected feature width. Eager functional calls retain their Python normalized_shape check, while the Rust eager operation infers actual width from input shape.

Registration validates the input and all supplied affine operands against the CPU/float32 kernel key. Existing statistics, forward arithmetic and first-order backward formulas were preserved. Specialized compiled/fused workspace execution still uses existing CPU buffer kernels after metadata validation. There is no implicit fallback or GPU support.

## Validation

- Full Python suite: 518 passed; two existing NumPy empty-mean reference warnings.
- Rust debug/release suites: 67 passed each; six ignored manual benchmarks each. Formatting and Clippy pass.
- New Python coverage: 29 cases for strided input/affine VJPs against PyTorch, full numerical Jacobians, bias-only operation, empty-batch affine gradients, invalid shapes/epsilon, normalized-feature contracts and shared input/weight storage gradients. Two new Rust tests cover checked schema behavior and exact parity with the existing CPU forward kernel.
- Existing fused LayerNorm training tests passed as part of the full suite.
- Fresh source installation without PyTorch: 390 passed, 19 oracle skips. Three existing oracle/conversion files excluded as in CI.
- Installed macOS arm64 wheel in the same isolated environment: 390 passed, 19 oracle skips, same exclusions. Imports resolved inside that environment and Torch was absent.
- Artifact checks: README metadata matches, 30 wheel license files present, source scratch/experimental data excluded.

Independent read-only review found no concrete correctness defects. Actual tests cover the review's uninspected epsilon-conversion and compiled empty-batch backward cases. Subnormal/overflowing Python epsilon values that become zero/infinity in float32 are rejected.

Artifacts and install logs remain outside Git:

```text
<repo>/../../work/normalization-validated.Pp1lZ7
```

Wheel SHA256: 84755a76fb2a3a68ab9e9e353f5bce1fe29b2d806de4f01b724545a5e95dd910.
Source archive SHA256: d9146efd7a2d8386e0f12fda266e9a2dcf4b60451ef15f79126bccd55ffb3077.

Artifacts were built from the validated working tree before its implementation commit. They are local test candidates, not published assets. Generated binaries/data were not committed, and existing user files were preserved.

## Benchmarks

No new timing measurements or performance claim. Kernel arithmetic and backward formulas are unchanged.

## Limitations

CPU float32, rank-2 last-dimension LayerNorm only. This is not BatchNorm, arbitrary normalized-shape support or general compiled backend dispatch. Linux/Python-matrix CI was not rerun because nothing was pushed. Existing Cargo descriptive-metadata packaging warning remains.

## Next

Migrate MSE and cross-entropy loss schemas into registered dispatch with shared eager/compiler validation. Compiler device/dtype propagation remains later work. No push, tags, package publication or history rewriting occurred.
