# Autograd lifecycle coverage audit

2026-10-15. New regression coverage commit 3fc37db.
This audit narrows remaining foundation gates; it does not declare all operators
or the whole CPU foundation complete.

## Inspected coverage

Existing test_autograd.py checks shared unequal paths, retained/consumed graphs,
invalid seeds, empty gradients and leaf snapshot immutability. Per-operation suites
cover full Jacobians and strided VJPs for unary math, sigmoid/tanh, bmm,
concatenation, reductions, slicing, softmax and other supported families. These
checks are distributed across test files; their existence alone does not prove
every combination of operator, graph topology and layout.

Concrete coverage gap: the six registered elementary unary operations had strided
numerical/Jacobian tests, but no common cross-operation lifecycle matrix combining
shared paths with retention, invalid-seed retry and empty layouts. Sigmoid/tanh
and slicing already had some lifecycle checks. The new matrix makes those
combinations explicit for neg, exp, log, sqrt, sin, cos, sigmoid, tanh and ReLU.

## New checks

Eighteen cases combine each operation with nonempty and empty transposed/sliced
inputs. Analytic NumPy derivatives define expected leaf gradients, including
zeros in skipped columns and accumulation from duplicated shared output paths.
Each case checks retained backward, exact-shape seed rejection without gradient
publication, successful retry, a detached earlier gradient snapshot, disconnected
leaf state, and rejection after graph consumption without further accumulation.

All 18 new cases pass against the current implementation. No production defect
was reproduced, so no speculative runtime fix was added. Full Python suite now
passes 844 tests with two existing NumPy empty-mean reference warnings. Runtime
is a2fb278; its Rust debug/release gates remain 93 passed each, six ignored. The
new test file has no optional Torch dependency. No build or public API changed.

## Binary, matrix and reduction extension

Commit 904ef82 adds 71 lifecycle cases. Broadcast add/sub/mul/div cover strided
inputs, empty output, both differentiable inputs or either frozen operand, and
identical native parents. Matmul/bmm cover transposed operands, duplicate parents,
empty row/inner/column/batch dimensions and frozen operands. Axis reductions cover
shared strided paths, empty unreduced axes and empty reduced axes producing
nonempty output. Empty mean outputs remain NaN while their empty input and scalar
broadcast-parent gradients have the expected empty/zero values. Eager and native
compiled whole reductions cover retained/retried empty sum/mean.

NumPy VJPs are independent expected gradients. Invalid seed tests verify preflight
failure and retry. Parameter replacement tests across binary/matrix/reduction
families verify stale-version rejection without publishing an unrelated leaf's
gradient. Those version failures occur at preflight; existing custom/hook tests
cover exceptions after partial traversal. The new matrix does not claim to inject
late native VJP failures. Existing Rust transaction fault-injection tests supply
that separate engine boundary.

Independent read-only review checked VJP formulas and scope. It identified the
empty-reduced-axis coverage gap, now covered. Final review found no required
blocker. No runtime defect was reproduced by these lifecycle additions.

## Packaging correction and validation

Fresh sdist inspection reproduced an actual checker regression introduced by the
previous standalone Rust benchmark source addition: the checker allowed only
benchmark .py/.md files and rejected gemm_initialization_probe.rs. Commit cee3ebe
allows exactly that Rust source and requires its Python companion. Six regression
tests cover accepted sources, either missing companion and rejection of result
arrays, executables and unlisted Rust files. Before the fix, three failed and
three passed; afterward all six pass. No broad binary/result allowlist was added.

Current local full suite: 921 passed, two existing NumPy empty-mean warnings.
Rust debug/release: 93 passed each, six manual benchmarks ignored. Formatting and
Clippy all targets with warnings denied pass. New release wheel/source builds
pass README metadata, 30 license files and scratch exclusions.

Fresh activated source install, running tests from the unpacked source archive:
777 passed, 35 skipped without optional Torch; three Torch-only modules excluded.
Fresh wheel lifecycle/hook/compiler/custom suite: 172 passed, one optional Torch
case skipped. The source installation explicitly activates its virtual environment
before --no-build-isolation, preserving the maturin executable PATH.

Artifacts stay outside Git under work/lifecycle-packaging-41SMg1/validated relative
to workspace root. Wheel SHA256:
25306bd140a1b68b79f5cdcaefffc7415e4c8d609ff1af11fdabe7d052ad532f.
Source SHA256:
1c5f0165f85b2b7bcbdfe4e4e49223a72ae8b39fd8dc4165a52fe48878a64093.
The failed pre-fix source archive is preserved in the parent directory. Validated
archives precede these final validation notes and do not represent a published
release. Package metadata remains 0.10.0.

Compatibility text now states that custom Function outputs support tensor hooks.
README distinguishes absent module hooks from supported tensor hooks. Historical
transaction/hook design documents now point to implemented behavior while keeping
their original design rationale. No documentation examples were added or changed.

## Remaining gates

Local audited lifecycle and source/wheel gates pass for this scope. Full CPU
foundation is still not declared complete. At that checkpoint, the remaining task was Linear/LayerNorm/loss lifecycle
coverage and public compiled/fused boundaries. The completion record below
supersedes that next action.
Do not replace those gates with more convenience APIs.

Higher-order and forward AD remain unsupported. Linux remote CI, device-native
GPU execution, thread sweeps and complete startup/footprint measurements remain
outstanding. This audit makes no performance or hardware claim.

## Dense module and loss lifecycle completion

Commit 6334409 adds 44 cases for public eager/compiled Linear and LayerNorm,
strided MSE and cross entropy, late input-hook failures and fused callback
rejection. Linear and LayerNorm VJPs use independent NumPy formulas, including
empty batches, frozen inputs/weights and shared outputs. Public compiled tests
assert that a native executor exists, preventing silent eager fallback from
satisfying the compiled claim.

Late input hooks run after model/loss VJPs and deliberately raise. Tests preserve
existing gradients and values, remove the hook and successfully retry the same
graph. Fused Linear/LayerNorm tests first populate Adam state, then reject active
parameter hooks without changing optimizer state, parameter values or eager
gradients. Removing the hook permits the next fused step.

Independent read-only review found no required defect in these cases or additional
concrete local CPU foundation blocker within the reviewed scope. This is scoped
validation of supported behavior, not universal correctness or release readiness.
Inference mode and checkpoint recomputation retain their documented reasoned
deferrals; higher-order and forward AD remain unsupported.

Final local checks: 965 Python tests passed with two existing NumPy reference
warnings; Rust debug/release each passed 93 tests with six manual benchmarks
ignored; formatting and Clippy passed. The existing installed wheel from
work/lifecycle-packaging-41SMg1/validated was tested against the expanded lifecycle,
hook, fused LayerNorm and loss suites: 196 passed, five optional Torch cases skipped.
This is an additional installed-wheel test run, not a newly built archive. Runtime
source remains a2fb278; only tests changed after the preceding packaging checks.
No test/build/review process remains pending.

Next substantive gate: process startup and dependency-inclusive installed payload
measurements. Measure interpreter launch separately from import and first forward;
validate equivalent forward results before timings. Use installed wheel environments,
record interpreter/package/native identities, include empty-process and NumPy
controls, preserve repeated outcomes and distinguish declared dependency payload
from total disk usage or RAM. Linux remote CI remains unverified.
