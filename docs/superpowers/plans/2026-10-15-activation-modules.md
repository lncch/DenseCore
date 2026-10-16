# Activation module integration

Goal: expose existing sigmoid/tanh kernels through parameter-free nn modules and
functional entry points. Follow Module registration/mode/state behavior and
strict Tensor validation. Public compiler keeps existing capture subset and
falls back eagerly for unsupported activation modules.

- [x] Observe four failures for absent nn.Sigmoid/nn.Tanh.
- [x] Add functional.sigmoid/tanh and nn.Sigmoid/nn.Tanh wrappers.
- [x] Test functional equivalence, state/modes, gradcheck, invalid inputs and
  Sequential/compiler-fallback training with optimizer updates.
- [x] Run complete gates and fresh wheel/source checks; execute README example.
- [x] Commit implementation and documentation separately.

No new native kernels, dtype/device support, compiler capture or version change.
