# Singleton dimensions and fused-training boundary checkpoint

Date: 2026-10-15. Base: 95fed5d. Branch: feat/v0.12-storage.

Tensor.unsqueeze inserts one singleton axis; Tensor.squeeze removes all singleton
axes or selects one integer axis. Both normalize negative dimensions and reject
booleans/nonintegers and out-of-range axes. Scalar squeeze permits axes 0/-1 as
no-ops. Squeezing a nonsingleton axis returns self. Tuple squeeze axes are an
explicit unsupported subset. Bool dtype is preserved and remains nondifferentiable.

Operations compose existing reshape. Contiguous layouts share immutable storage;
strided layouts can materialize logical values. These methods do not claim
PyTorch's complete stride-preserving view behavior. Existing float32 VJPs,
custom Function ownership and version checks follow the reshape path.

The preceding compiler-boundary audit added six Sigmoid/Tanh/Flatten × SGD/Adam
cases. Fused training rejects those unsupported modules before changing native
parameter handles, parameter values, existing gradients or optimizer state.
An initial test setup omitted SGD's required lr; corrected before validation.
No runtime fix was needed for those boundaries.

Twelve singleton-dimension tests failed for missing methods before implementation.
Coverage includes shape/values/gradients, scalar/empty/bool tensors, axis errors,
custom owners and optional PyTorch parity on a permuted tensor.

Local macOS arm64, Python 3.13.7. Rust unchanged: debug 87 passed, 6 existing
manual benchmarks ignored; formatting and Clippy all targets with warnings denied
passed. Full Python suite: 793 passed, two existing NumPy empty-mean warnings.
Release editable extension rebuilt. Fresh activated source installation: 50 passed,
1 optional Torch case skipped. Fresh wheel: 12 passed, 1 optional Torch case
skipped. New README example executed against fresh wheel.

Artifact checker passes matching README, 30 license files and source scratch
exclusions. Archives outside Git: ../../work/singleton-dims-7owF3M.
Wheel SHA256: 2a894a1c6611b1185b4eecf21dbfccb52de106c54d61ef4b5baf8cb14f3dfa66.
Source SHA256: a52cd5f427f5318a708635d9fd05a95add301cf519ab19599a41952bab3512f3.
Archives precede this checkpoint. Generated data/artifacts not staged.

No new native kernel, compiler capture, performance claim or hardware validation.
Package version remains 0.10.0. No push, merge, tag or publication.

Next unit: inspect unresolved operator and autograd coverage before choosing the
next bounded implementation. General hook ownership remains subject to independent
critique under GRADIENT_HOOK_DESIGN.md. Do not substitute unsupported compiler
fallback or reshape copies for claims of native capture or general zero-copy views.
