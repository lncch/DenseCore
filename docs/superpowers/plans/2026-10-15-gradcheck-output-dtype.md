# Gradcheck output dtype correction

Goal: reject nondifferentiable boolean outputs before constructing a numerical
Jacobian or reporting a successful gradient check.

Scope: the existing evaluate boundary in python/densecore/autograd.py and
regressions in tests/test_gradcheck.py. No Rust math or optimizer changes.

- [x] Reproduce boolean-output acceptance in both raise_exception modes.
- [x] Reproduce empty boolean-output acceptance and dtype changes during
  finite-difference evaluation.
- [x] Require float32 output on every evaluate call, before NumPy conversion.
- [x] Run complete Rust/Python gates and fresh installed-package checks.
- [x] Update gradcheck semantics, validation checkpoint and next-unit design.
- [x] Commit implementation and documentation separately.

All four regression cases failed before the production change. Input validation
errors remain exceptions even when raise_exception=False; that option controls
Jacobian disagreement only. Existing float32 constant and empty outputs remain
supported and covered by existing tests.
