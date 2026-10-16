# Custom Function validation checkpoint

Date: 2026-10-15. Branch: `feat/v0.12-storage`. Package metadata remains
0.10.0; this is an unreleased implementation checkpoint.

## Implemented

Single-output, tensor-only float32 `autograd.Function.apply`, detached saved
snapshots, first-order custom gradients and Python-owned callback records.
Resumable native backward yields with guards released, revalidates versions and
graph readiness, and commits leaf gradients atomically. Ordinary native backward
uses the same scheduler. No Python reference enters Rust graph ownership.

Regression coverage includes callback errors with retry, malformed gradient
count/type/shape/dtype, higher-order rejection, duplicate/frozen inputs, retained
and strided gradients, version invalidation, context phases and GC cycles.
A timeout-bounded subprocess verifies callback gradient reads and independent
ordinary backward. Recursive custom backward rejects. Ownership propagation is
covered through views, math, losses, LayerNorm, joins and compiled ReLU.
Native tests cover competing sessions, saved snapshots without graph ownership,
invalid-gradient retry and saved-version changes across suspension.

## Validation

- Rust formatting passed; debug and release each passed 86 tests with 6 existing
  manual benchmarks ignored. Clippy passed all targets with warnings denied.
- Full Python suite: 727 passed. Fresh source installation: 590 passed,
  28 skipped without optional Torch (three Torch-specific modules excluded).
  Fresh wheel custom-function and compiler regression suites: 54 passed.
- Executed the complete Python example in CUSTOM_FUNCTION.md against the working
  extension and a fresh wheel installation.
- Release wheel and source distribution built outside Git under
  `../../work/custom-function-hIlWuh`. Fresh source environment activated before
  installing with `--no-build-isolation`, making maturin executable discoverable.
- Artifact checker passed: wheel includes 30 license files and matching README;
  source archive excludes scratch files. No artifact or benchmark data staged.
- Artifact hashes: wheel `b7670c94e36b1fb5dba39cade308de97d946e21c263cc8ecef70797045505e03`;
  source `9d286653044724bd0ccfeec29599a10d7d61d10e16f3561253a1a3b0051fcd91`.
  Archives precede the final additions to regression tests and this checkpoint;
  they verify the same runtime implementation, not a release snapshot.

## Limits and next unit

Local macOS arm64 CPU validation only. No Linux or GPU validation claimed.
No speed/memory improvement claimed. Two existing NumPy empty-mean warnings
remain. An independent design critique informed ownership and session design;
a subsequent external implementation review could not run because Claude's
weekly usage limit was exhausted. Local regression checks are not a substitute
for that unavailable independent implementation review.

Next dependency-ready unit: design gradient hook ownership, callback ordering,
reentrancy and failure semantics before adding hooks. Custom compiler capture,
multi-output Function, higher-order gradients and forward AD remain unsupported.
No push, tag, release or publication authorized or performed.
