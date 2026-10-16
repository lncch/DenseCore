# Boolean Foundation Implementation Plan

> **For agentic workers:** Use inline execution under the standing autonomous mission. Only one writer changes this working tree. Do not spawn implementation agents or ask the sleeping user to continue.

**Goal:** Add real CPU boolean tensors with shared checked views and explicit float-only boundaries.

**Architecture:** Share TensorStorage and layout logic through a sealed Element trait and TypedTensorData<T>. Keep TensorData as the f32 alias and autograd unchanged. A separate private _BoolTensor is exposed through the existing Python Tensor wrapper, with no gradient graph.

**Tech Stack:** Rust, PyO3, Python, NumPy.

**Spec:** docs/BOOLEAN_STORAGE_DESIGN.md

## Task 1: Typed immutable storage

- [x] Add failing Rust storage regressions in src/tensor.rs for bool dtype, views, reshape ownership, empty shapes and overflow. Observe missing BoolTensorData/DType::Bool failures.
- [x] Read independent design critique and resolve concrete issues before changing storage.
- [x] Generalize the existing storage/data implementation in src/tensor.rs; seal supported element types to f32 and bool. Keep every float32 caller statically typed.
- [x] Make TensorSpec metadata helpers generic over supported typed data. Explicitly reject Bool in compiler CPU-float32 lowering, including input-only graphs.
- [x] Run Rust tests, formatting and Clippy; verify all old float32 tests still pass.

## Task 2: Public boolean construction and views

- [x] Add tests/test_boolean_tensor.py; observe failures for bool construction, logical-not and truth behavior.
- [x] Add src/boolean_api.rs with immutable bool native storage, copying export, view/layout methods and nondifferentiable state. Register its PyO3 class in src/lib.rs.
- [x] Update python/densecore/tensor.py for explicit dtype="bool", native bool export, cross-dtype storage checks, supported views, logical-not and scalar truth.
- [x] Guard float-only tensor math and functional NN entry points. Reject boolean Tensor initialization of Parameter in nn.py. Expose logical_not in __init__.py. Keep cat/stack, optimizer, compiler and checkpoint Parameters float32-only.
- [x] Run boolean tests and the full Python suite. Add failing regressions for review findings before fixes.

## Task 3: Review, install and checkpoint

- [x] Obtain read-only code review and resolve findings with regression tests.
- [x] Run Rust debug/release tests, formatting, Clippy and a release extension build. Run full Python suite.
- [x] Build source/wheel outside Git. Inspect license/README/scratch payload, install into an activated fresh environment, run no-Torch and targeted wheel tests.
- [x] Commit implementation only after gates pass. Commit documentation separately: supported subset, limitations, tested example, actual results, next comparison/logical unit. No benchmark or GPU claim.

No push, merge, tag, publication, destructive cleanup or package version bump.
