# Reduction dispatch validation, 2026-10-15

## Current status and commits

Implementation commit: 885844f. Development branch remains feat/v0.12-storage. Sum/mean registration migration is validated locally; no release version was changed. This report and API documentation are committed separately.

## Architecture

A checked ReductionSpec handles whole-tensor versus explicit axes, duplicate/range checks, output shape and lazy mean divisor calculation. Eager reductions select registered CPU float32 kernels. Compiler whole-reduction shape validation uses the same schema; specialized CPU workspace execution shares whole-reduction arithmetic. Existing axis-gradient mappings and mathematical VJPs are preserved.

Output allocation capacity errors return Result errors. Empty means and empty backward maps skip unused divisor evaluation, fixing a reproduced overflow for shape [0, usize::MAX, 2] reduced over axes [1, 2]. The divisor itself remains mathematically defined: an empty axis list has divisor 1, and a selected zero dimension gives divisor 0. No fake zero divisor is substituted for an unrepresentable unused product.

## Validation

- Full Python suite: 489 passed, two existing NumPy empty-mean reference warnings.
- Rust debug/release suites: 65 passed each, six ignored manual benchmarks each. Formatting and Clippy pass. The final empty-axis divisor assertion was also run in the three-test reduction subset.
- New Python coverage: 25 tests for strided/offset views, axes/keepdims, full numerical Jacobians, eager/compiled PyTorch VJP parity, scalar/empty tensors and the huge empty-mean backward regression.
- Fresh isolated source installation without PyTorch: 369 passed, 11 oracle skips, with the same three oracle/conversion files excluded as CI.
- Installed macOS arm64 wheel in that isolated environment: 369 passed, 11 oracle skips, same exclusions. Package imports were verified inside the environment and Torch was absent.
- Artifact checks: README metadata matches, license payload present, scratch/experimental data excluded.

Independent read-only review identified the empty-output mean case. A failing Python regression reproduced it before correction. The review suggestion to redefine every empty-input divisor as zero was rejected because an empty axis list must retain divisor 1; unused work is skipped instead.

Local artifacts and installation logs remain outside Git:

```text
<repo>/../../work/reduction-validated.d66vMi
```

Wheel SHA256: e7cc7917904c8ee897d18476927c857b1eea40a15b7ee77bfdbc4003921e6e93.
Source archive SHA256: 0b53a8b47eb1873954e4ee8e5635f8b7713d941c9fd4e1a3b458f0a1753b0fa9.

These are local test artifacts built from the validated working tree, not published release assets. A final test-only divisor assertion does not change their runtime implementation. No generated benchmark data or release binaries were committed. Existing user files were preserved.

## Benchmarks and limitations

No new timing measurements or speed claim. Existing serial float32 accumulation order is preserved.

CPU float32 only. Compiler capture still supports whole-tensor sum/mean, not axis reductions. Specialized compiled CPU execution is not general backend dispatch. Linux/Python-matrix CI was not rerun because this branch was not pushed. Existing Cargo descriptive-metadata packaging warning remains.

## Next

Migrate normalization into registered schemas and shared eager/compiler validation. Loss schemas and compiler device/dtype propagation remain later tasks. No push, tags, package publication or history rewriting occurred.
