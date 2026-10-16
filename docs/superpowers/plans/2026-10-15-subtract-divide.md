# Float32 subtraction and division

Goal: add real registered CPU float32 subtraction/division and exact-shape
first-order broadcast VJPs, following existing add/multiply architecture.

Scope: ops.rs enum/kernel, dispatch.rs registration, autograd.rs binary VJPs,
lib.rs native methods and tensor.py arithmetic/NumPy bridge. No Python compiler capture,
new dtype promotion, in-place operation or backend expansion.

- [x] Write public numerical/gradient/invalid-input tests and observe seven
  failures because subtraction/division are unsupported.
- [x] Extend existing binary dispatch and gradient cases; use direct IEEE
  division rather than reciprocal multiplication to preserve forward behavior.
- [x] Check scalar/reflected, broadcast, strided, empty/shared/frozen parents,
  no_grad, nonfinite values, versioning and custom owner propagation.
- [x] Validate Rust format/debug/release/Clippy, full Python, fresh installations.
- [x] Document supported subset and actual results; separate code/docs commits.

Subtraction VJPs are g and -g. Division VJPs are g/b and -(g*(a/b))/b,
then existing broadcast maps reduce into each parent shape. The denominator
formula avoids explicitly squaring b, which can overflow for finite values.
Nonfinite arithmetic follows IEEE propagation; no domain guard is added.

Implementation discovery: BinaryOp is shared with native graph instructions.
Extend internal broadcast forwards/VJPs and operand pinning consistently, with
planned-workspace parity and frozen-parent masks. Public Python graph schema
continues to accept only its existing add/mul operation tags.
