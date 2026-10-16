# Sigmoid and tanh

Goal: extend the established CPU float32 unary registry with eager sigmoid/tanh
and first-order VJPs. No new compiler capture, module classes or dtype/backend.

Sigmoid forward uses 1/(1+exp(-x)) for nonnegative x and exp(x)/(1+exp(x))
otherwise. Tanh uses f32::tanh. VJPs use rounded forward values: g*y*(1-y)
and g*(1-y)*(1+y). Infinity saturates, NaN propagates; float32 saturation can
produce exact zero gradients. Existing immutable snapshots and owner propagation
remain unchanged. This follows existing registry interfaces, not new architecture.

- [x] Observe eight failing public tests for missing tensor methods.
- [x] Add registry kernels, VJP cases and Tensor methods through existing wrapper.
- [x] Validate strided/full-Jacobian, Torch, saturation, nonfinite, scalar/empty,
  shared/retained/custom graphs, no_grad, dtype rejection and parameter versions.
- [x] Run Rust format/debug/release/Clippy, full Python and fresh install checks.
- [x] Execute README example, record evidence and commit code/docs separately.
