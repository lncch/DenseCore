# Subtraction and division checkpoint

Date: 2026-10-15. Base: eacc526. Branch: feat/v0.12-storage.
Package metadata remains 0.10.0; no release milestone declared complete.

## Implementation

CPU float32 subtraction and true division use the existing registered binary
kernel family. Python operators include reflected real scalars and the strict
NumPy subtract/true_divide bridge. Checked right-aligned broadcasting and logical
strided inputs follow add/multiply. Results propagate custom Function owners.
No array fallback, mixed tensor dtype promotion, floor division or in-place
variant was added.

First-order VJPs reduce broadcast contributions into exact parent shapes.
Subtraction contributes g and -g. Division contributes g/b and -(g*(a/b))/b,
avoiding an explicit b*b that can overflow for large finite denominators.
IEEE zeros, infinities and NaNs propagate; domain checks do not replace IEEE
arithmetic. Empty/shared/frozen parents follow existing graph semantics.

BinaryOp is shared with native graph instructions. Extending it required matching
compiled broadcast formulas and division operand pinning. Native tests cover
broadcast shape combinations, planned workspace reuse and both frozen-parent
masks, including computed denominators whose forward slots can be reused.
The public Python graph schema and module capture subset remain unchanged;
subtraction/division are not new Python capture tags.

## Evidence

Seven initial public regressions failed because operators were absent. Final
coverage includes finite-difference Jacobians, PyTorch forward/VJP parity,
reflected NumPy scalars, empty dimensions, shared/frozen parents, invalid
shape/dtype/array operands, no_grad, version checks, custom owner propagation and
large-denominator gradients. The initial new Rust test had a helper/import scope
error, corrected before final Rust gates and source archive generation.

Local macOS arm64, Python 3.13.7, rustc 1.99.0:

- Rust fmt and Clippy all targets with warnings denied pass. Debug/release each
  pass 87 tests; 6 existing manual benchmarks ignored.
- Full Python suite: 745 passed. Two existing NumPy empty-mean reference warnings.
- Release editable extension rebuilt. Fresh activated source install without
  optional Torch: 606 passed, 30 skipped; three Torch-only modules excluded.
- Fresh wheel arithmetic/custom Function suites: 48 passed, 2 optional Torch
  cases skipped. New README Python example executed successfully against wheel.
- Source archive rebuilt after final Rust test corrections and reinstalled;
  targeted arithmetic regressions: 9 passed, 2 optional Torch cases skipped.
- Artifact checker verifies matching README, all 30 license files and source
  scratch exclusions. Final artifacts stay outside Git at
  ../../work/sub-div-Eq1xwk/final. No generated benchmark data staged.
  Final wheel SHA256: 5f7203c9c6d52dcb8526310ac80d6a75d606cffc799b32b07c867514b1b4d743.
  Final source SHA256: a5789e50ece2b9dc2a2a24757834d3c4b0d20aadb275acf0b5b467846f3d3d6e.
  These archives precede this validation document. Final wheel was reinstalled
  and its 48-test/2-skip check and README example rerun successfully.

No Linux/GPU validation or performance improvement claimed. No runtime/optimizer
refactoring. Protected branches preserved; nothing pushed, merged, tagged or
published. Independent hook ownership review remains unavailable as recorded in
GRADIENT_HOOK_DESIGN.md; this unit follows existing binary-dispatch architecture
without introducing hook ownership changes.

## Next dependency-ready unit

General gradient hooks still require independent ownership critique before broad
implementation. Meanwhile, remaining dense-network activation coverage can be
extended through the existing unary registry: investigate stable sigmoid/tanh
forwards and first-order VJPs with saturation/nonfinite tests, finite differences
and PyTorch parity. Do not claim new compiler capture or hardware support.
