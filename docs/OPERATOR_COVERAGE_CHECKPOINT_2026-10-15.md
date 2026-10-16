# Operator coverage checkpoint, 2026-10-15

## State

Branch: `feat/v0.12-storage`. Validated source/docs checkpoint: `d9dbd4f`. Package metadata remains 0.10.0; no release is declared. The protected `feat/v0.11` branch remains at `4044e5c0a415e9e20e0b57cc91b588e5b82f4bbc`. No push, merge, tag or publication occurred.

This supplements the earlier engineering checkpoint. Reductions, normalization, losses, Linear and activation forward paths now have static CPU float32 registrations and checked shared shape rules. The earlier report's outstanding migration list is historical.

## Completed units

- `aa4e7f3` / `9536bc2`: Linear registration and documentation.
- `a170069` / `4bbdc89`: ReLU/softmax/log-softmax registration and documentation.
- `89e5978` / `a5e836a`: differentiable rank-3 batched matrix multiplication and documentation.
- `fedca7d` / `d9dbd4f`: differentiable concatenation/stacking and documentation.

Implementation and documentation are separate commits. Existing optimized GEMM arithmetic was retained. New bmm calls that boundary once per batch with reusable matrix scratch; it is not a native batched BLAS implementation. Stack composes singleton reshape with registered concatenation. Shared-parent/view gradients accumulate by node identity. Existing parameter snapshot/version checks apply.

Independent read-only design and code reviews checked overflow, zero dimensions, aliases and first-order gradients. Code review found bare Tensor input silently iterating over rows for cat/stack. A failing regression reproduced both cases; the final API rejects bare Tensor input. No other actionable defect was reported in the reviewed new kernels and autograd paths.

## Verification

Local environment: macOS arm64, Python 3.13.7, rustc 1.99.0.

- Full editable Python suite: **582 passed**, two existing NumPy empty-mean reference warnings.
- Rust debug and release: **73 passed**, six manual benchmarks ignored. Formatting and Clippy with warnings denied passed.
- Fresh source install without PyTorch: **448 passed, 25 skipped**, same two reference warnings. Three Torch-only test modules were explicitly excluded; optional Torch cases in other modules skipped naturally.
- Fresh wheel targeted activation/bmm/cat/stack suite: **32 passed, two optional Torch cases skipped**.
- Both newly added README examples ran successfully against both editable code and the fresh wheel.

Source installation used an activated fresh virtual environment with maturin 1.15.0 and `--no-build-isolation`, exercising backend executable discovery through that environment's PATH. The wheel and source archive passed the repository artifact checker: README matched wheel metadata, all 30 license files matched, and scratch directories were excluded.

Validated archives are outside Git under `work/join-final-1792023126`, relative to the workspace root `i-w`. Direct wheel SHA256: `f80a1f38b6fb742ae4dc209e1c1bd05261da16f017d0af3eea668104b1c1d80b`. Source archive SHA256: `c2f331559d6893912907be11ab0bf62d752bab93ef7427138141382d08b6c738`. This report was written after archive validation and is not part of those archives. No generated benchmark data or release archives were committed.

## Limits and next work

CPU float32 remains the only backend/dtype. New bmm/cat/stack are eager only. Compilation remains the documented built-in module subset; specialized CPU workspaces do not become generic backend execution through these registrations. Concatenation gradients can retain a whole upstream allocation through immutable slices. Strided operands can retain cached logical materialization. No new performance claim is made.

Comparison/logical operators, general dtype inference, compiler device/dtype propagation, custom autograd functions, inference mode, GPU and distributed work remain incomplete. Do not declare v0.13 or later complete. Comparisons need an explicit boolean representation contract rather than float32 masks presented as full dtype support.

Next dependency-ready unit: inspect compiler graph metadata and add checked CPU/float32 device/dtype propagation and runtime consistency validation without pretending GPU support. Get an independent design review before broad changes. Preserve specialized workspaces and failure-atomic optimizers. Then design real boolean comparison/logical support with tests and documented scope. Read the full mission attachment before choosing later units.

Hourly autonomous continuation is active for this chat. It should resume from Git and this checkpoint, avoid duplicate concurrent work, and require no user message to continue. Usage and execution limits still apply.
