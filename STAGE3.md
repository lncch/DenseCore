# Stage 3 verification: dense model training

Date: 2026-10-14. Added registered dense Python modules, mean MSE and stable CrossEntropy, native Rust SGD/Adam, immutable versioned parameter updates and strict copied model/optimizer state dictionaries.

## Evidence

Final suite: 135 Python cases, including 55 executed PyTorch2.13.0 parity cases without skips; 11 Rust tests; clippy with warnings denied. Rebuilt CPython3.13/macOS arm64 release wheel passes 80 non-oracle tests outside source paths with PyTorch absent.

Parity includes supplied-weight module outputs/gradients, both loss gradients, extreme/nonfinite/empty CE inputs, twelve-step SGD/Adam parameter/moment comparisons, missing gradients, and fifteen-step complete model loops for both losses and optimizers. Optimizer update/state checks use rtol1e-5, atol1e-6. Model-plus-Adam resume reproduces uninterrupted parameter arrays exactly in the deterministic fixture.

Predeclared synthetic workflow gates passed after100 steps: heldout regression MSE1.7400436647108108e-10; heldout two-cluster classification accuracy100%, training CE fell0.55609536 to0.00003306194. See examples/PROTOCOL.md and results.json. These toy tasks are correctness demonstrations, not broad accuracy or performance results.

## Review and fixes

Independent design review accepted versioned immutable replacement with all-parent validation and compute-all-before-commit. Final code reviewer independently reran the original133 Python/11 Rust tests and clippy, then identified two defects: native replacement could partially invalidate dependent leaf inputs, and mixed bool/int labels bypassed dtype rejection. Both reproduced in failing regression tests. Replacement handles are now all prepared before version commits, and Python Boolean labels are rejected before coercion. Final135-case suite and rebuilt80-case wheel suite pass. The review's original blocking report is historical; no second review was dispatched after regression-backed fixes.

## Decisions and costs

- Continue local feat/dense-training checkout with one writer; no extra worktree or publishing. Cost if relocation is needed: move checkout.
- Saved values remain immutable; update creates a new leaf sharing a version token, preserving Python Parameter identity and gradient. Cost: copied values/gradients and single-thread training contract.
- No-grad derived operands retain version provenance, including conflicting generations. Cost: metadata overhead and conservative stale-expression rejection; this prevents old frozen weights from silently participating in backward.
- Native replacement commits after every prepared leaf succeeds. Cost: staging memory, traded for batch atomicity.
- Optimizer state is an own version1 ordered format, not an arbitrary PyTorch optimizer-file importer. Cost: conversion requires an explicit later adapter.

## Limits

Only float32 CPU, dense2D batches, mean losses and documented optimizer options. No hooks, buffers, parameter groups, closure/Nesterov/AMSGrad, compiled execution or GPU. Linux and other Python versions remain untested. Reference math loops/index maps/Python-list transfer remain; no speed claim applies to this public library. Private native integer-extraction error normalization remains deferred from Stage1; supported public entry points validate their inputs.

Wheel SHA-256: aebe786d70ad9f87e304d6aa4776ce2dcc10d6330de394a1729cc7b059d427bf
