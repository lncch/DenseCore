# Stage 4 verification: file weights and checkpoints

Date:2026-10-14. Added float32 safetensors raw weights, versioned single-file model/optimizer resume and optional PyTorch tensor-state conversion. Architecture remains supplied by user code.

## Evidence

151 Python tests pass:55 numerical PyTorch parity cases and3 optional conversion cases execute without skips. Eleven Rust tests and clippy pass. Fresh installed macOS/CPython3.13 wheel executes93 non-oracle cases without PyTorch. File resume example matches an uninterrupted Adam update exactly.

Tests cover scalar/empty tensors, shared aliases, external format=pt weights, dtype/name/shape errors, SGD/Adam resume, optimizer order/version/buffer references, unused arrays, model-only metadata, interrupted serialization preserving destination and cleanup, conversion name maps and failed conversion preserving destination.

Independent reviewer reran149 cases at the pre-fix commit and adversarially checked malformed buffer references/shapes, Boolean steps, duplicate JSON, truncated files and failed os.replace. All rejected without changing trained target states. Two schema findings, unknown internal weight formats and unchecked model-only parameter metadata, were reproduced and fixed with RED→GREEN tests. External raw safetensors metadata stays accepted when exact names/shapes/float32 match; unknown densecore.* formats are rejected.

## Decisions

Model/optimizer preparation helpers stage validation before native model replacement and optimizer assignment. Cost: copied arrays and staged optimizer state. Atomic path replacement stages in the same destination directory; successful saves replace existing files. Failure/power-loss durability is not guaranteed by fsync, which is not used.

Checkpoint optimizer must cover registered model parameters in order. Cost: arbitrary subset/group checkpoint layouts are unsupported. JSON metadata records only primitive fields and buffer references; no model code or pickle is stored. Conversion lazily imports Torch and uses weights_only CPU loading; only dense float32 tensor state dictionaries or state_dict wrappers are accepted.

## Limits

Validated on Apple M4/macOS, Python3.13.7, NumPy2.5.4, safetensors0.8.0, PyO3 0.29.3. Linux and other Python versions remain unvalidated. No optimized math, compiled graph or speed claim is added in this stage. Own optimizer checkpoint format is not a PyTorch optimizer-state importer.

Wheel SHA-256: e87ee225628f0cf89686eaee499a6f46b851116752346172c0604ddc3899946c
