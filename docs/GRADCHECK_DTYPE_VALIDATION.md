# Gradcheck dtype and custom-backward boundary checkpoint

Date: 2026-10-15. Branch: feat/v0.12-storage. Base: ed9d6da.

## Correctness change

A comparison such as gradcheck(lambda x: x > 0, x) previously reported True at
locally constant points. Empty boolean outputs also passed vacuously. A boolean
output appearing only during numerical evaluation could produce a misleading
Jacobian mismatch instead of a supported-dtype error. Four regressions failed
before the correction. Every evaluate call now requires a float32 output before
NumPy conversion. Invalid outputs raise TypeError in both raise_exception modes;
existing float32 constant and empty outputs remain supported.

Additional custom Function regressions verify captured native parent identity
when forward replaces a Parameter, version revalidation after a callback updates
a saved Parameter, and concurrent same-context rejection using a timeout-bounded
subprocess. Callback optimizer side effects persist while the failed outer
backward publishes no leaf gradient. These tests passed without runtime changes.

## Validation

- Rust fmt passed; debug/release each passed 86 tests, 6 existing manual benchmarks
  ignored. Clippy all targets with warnings denied passed.
- Release editable extension rebuilt with maturin.
- Final full Python suite: 734 passed.
- Fresh activated source-install environment: 597 passed, 28 skipped without
  optional Torch; three Torch-only modules excluded. Fresh wheel gradcheck and
  custom Function suites: 65 passed. Both documentation Python examples in
  GRADCHECK.md and CUSTOM_FUNCTION.md executed successfully against that wheel.
- Two existing NumPy empty-mean reference warnings remain.
- One final full-suite invocation accidentally selected the source environment
  without Torch and failed collection. It was corrected by explicitly selecting
  the development .venv interpreter; no test or dependency requirement weakened.
- Artifact checker passed matching wheel README, 30 license files and source
  scratch exclusions. Artifacts remain outside Git under
  ../../work/gradcheck-dtype-zr2Uc7; no generated benchmarks or artifacts staged.
- Wheel SHA256: bbf8ca3f9997948838c4db3f5661f3c590f8722f7a25e50fe07440eac987d7af.
  Source SHA256: 243002644b0150f063b352f4a9be549e7e91fa9e4dd3a1421649a7888d4a1786.
  These archives contain the runtime/tests and GRADCHECK.md change but precede
  this checkpoint and the hook design investigation.

Local macOS arm64 CPU only. No performance improvement, Linux/GPU validation or
new release claimed. Version metadata remains 0.10.0. No push, merge, tag or
publication. Stable branches preserved.

Implementation and regression commits: see the local Git history immediately
before the documentation checkpoint.

## Next autonomous unit

Read GRADIENT_HOOK_DESIGN.md. Get independent critique of late registration,
GC-visible ownership, parameter handle replacement and fused-training behavior
before a broad hook implementation. External review remains unavailable: Claude
weekly quota exhausted; both configured free opencode models returned server
errors. No hook API, placeholder or completion claim added. Resume independent
review when service state changes; continue narrower correctness work meanwhile.
