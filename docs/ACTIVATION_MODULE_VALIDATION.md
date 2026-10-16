# Activation module integration checkpoint

Date: 2026-10-15. Base: 1c3fb13. Branch: feat/v0.12-storage.

Added parameter-free nn.Sigmoid and nn.Tanh plus functional.sigmoid and
functional.tanh. Functions validate Tensor inputs and delegate to existing eager
CPU float32 kernels. Modules inherit ordinary registration, state and train/eval
behavior. Sequential models containing either module train through compiler eager
fallback. Native capture support is unchanged, and the regression checks that no
native executor is created for these models.

Four tests failed for missing classes before implementation. Regressions cover
functional/module equivalence, state/modes, full-Jacobian gradcheck, invalid
inputs, Sequential gradients and optimizer updates under fallback.

Validation: local macOS arm64, Python 3.13.7. Full Python suite passed 759 tests
with two existing NumPy empty-mean reference warnings. Rust unchanged; debug
passed 87 tests with 6 existing manual benchmarks ignored, formatting and Clippy
all targets with warnings denied passed. Release editable extension rebuilt.

Fresh activated source install: activation/module/compiler suites passed 30
tests, 2 optional Torch cases skipped. Fresh wheel: 12 passed, 2 optional Torch
cases skipped. Updated README example executed successfully against wheel. Artifact
checker passed matching README, 30 license files and source scratch exclusions.
Artifacts remain outside Git under ../../work/activation-modules-m5qmfj.
Wheel SHA256: 674070428baa20c187c0d8b0b3dd18dd833d33a9e0ad8ce14ee948cfca1247fa.
Source SHA256: 49d2bec0e8234cf4c08b6cc67ddfe7e59dc48200f3e6688a46a95cbb94f8aaf2.
Archives precede this checkpoint and final dispatcher documentation update.

No performance claim, Linux/GPU validation, version bump or new compiler capture.
No push, merge, tag or publication. Code and documentation committed separately.

Next unit: audit remaining dense operator coverage and select a bounded missing
operation with numerical and failure tests. General gradient hook ownership still
needs independent review under GRADIENT_HOOK_DESIGN.md. Do not treat eager
fallback as compiled activation execution.
