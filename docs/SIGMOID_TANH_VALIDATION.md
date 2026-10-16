# Sigmoid and tanh checkpoint

Date: 2026-10-15. Base: d45ce22. Branch: feat/v0.12-storage.
Package metadata remains 0.10.0; no new release declared.

## Implementation

Registered CPU float32 Tensor.sigmoid() and Tensor.tanh() follow the existing
unary-dispatch schema. Inputs read logical strided values; outputs are fresh
contiguous values. Sigmoid uses nonpositive exponential arguments through a
sign-dependent formula. Tanh uses Rust f32::tanh. Backward uses rounded float32
activation values, with ordinary first-order broadcast-free VJPs and existing
saved-version checks. Custom Function owners propagate through both methods.

Infinities saturate, NaNs propagate, and tanh preserves signed zero. Negative
sigmoid inputs can retain positive subnormal outputs that other implementations
flush to zero; equality is tolerance-based. Saturated float32 outputs can have
exactly zero gradients. No compiler capture nodes, activation modules, NumPy
activation ufunc bridge, higher-order AD or hardware backend was added.

## Evidence

Eight initial public tests failed for missing tensor methods. Final tests cover
strided full-Jacobian gradcheck, PyTorch forward/VJP parity, extreme/nonfinite
values, signed zero, subnormals, scalar/empty tensors, no_grad, boolean rejection,
shared/retained/custom graphs and parameter replacement. Nonfinite checks also
run without optional Torch. The existing native registry test iterates every
registration and verifies uniqueness, native metadata and strided outputs.

Local macOS arm64, Python 3.13.7, rustc 1.99.0:

- Rust debug/release: 87 passed each, 6 existing manual benchmarks ignored.
  Formatting and Clippy all targets with warnings denied passed.
- Release editable extension rebuilt. Full Python suite: 755 passed; two existing
  NumPy empty-mean reference warnings.
- Fresh activated source install, maturin 1.15.0, --no-build-isolation, no Torch:
  614 passed, 32 skipped; three Torch-only modules excluded.
- Fresh wheel activation/custom Function checks: 47 passed, 2 optional Torch
  cases skipped. New README example executed successfully against that wheel.
- Artifact checker passes matching README, 30 license files and scratch exclusions.
  Artifacts stay outside Git at ../../work/sigmoid-tanh-2sDSOz.
  Wheel SHA256: 7e23bba2ce56701c31fcdee6aa9339c836bef9fd76f733ba761621cd7a4c9f0c.
  Source SHA256: 82561d113e388fe06066b437575d6415ffcac5463c04bc9c2fbad735e32261a3.
  Archives precede the final dispatcher documentation and this validation report.

No performance improvement or Linux/GPU validation claimed. No optimizer/runtime
refactoring, push, merge, tag or publication. Generated data/artifacts not staged.

## Next dependency-ready unit

Add thin nn.Sigmoid and nn.Tanh modules and matching functional entry points only
if their interfaces follow existing module behavior. Verify Sequential use,
parameter registration, mode/state behavior and compiler eager fallback; do not
silently claim native activation capture. General gradient hooks still require
independent ownership critique as recorded in GRADIENT_HOOK_DESIGN.md. Continue
narrow correctness/operator work while that external review remains unavailable.
