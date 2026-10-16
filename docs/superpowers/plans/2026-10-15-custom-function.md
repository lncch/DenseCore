# Custom Function Implementation Plan

> Inline implementation under the autonomous mission. One writer; no push/publication.

**Goal:** Real CPU float32 single-output first-order Function.apply with safe Python callbacks.

**Architecture:** Numeric custom IDs and saved snapshots in Rust, resumable backward with optimistic validation, GC-visible Python owner DAG, callbacks outside locks.

**Spec:** docs/CUSTOM_FUNCTION_DESIGN.md

1. Add failing Rust session tests, then extract autograd_session.rs and integrate Custom recipes/saved_snapshot/has_custom. Native backward drains ordinary sessions and refuses callbacks. Validate unchanged Rust/Python suites.
2. Add failing public Function tests. Implement custom_api.rs bindings and function.py protocol/driver. Initialize and propagate owner DAG at every public wrapping site. Add deterministic has_custom/owner invariant tests.
3. Obtain independent code review; fix findings with regressions. Validate callback errors, version checks, overlapping consumption, subprocess deadlock checks and GC cycles.
4. Run formatting, Clippy, Rust debug/release, full Python, release build, fresh source/wheel installation and artifact inspection. Execute README example. Commit implementation and documentation separately, record exact results/limitations/next unit.

Do not add gradient hooks in this unit. They need their own ownership/reentrancy review after Function works.
