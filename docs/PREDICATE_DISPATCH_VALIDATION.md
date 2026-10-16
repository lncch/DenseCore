# Comparison and binary logical dispatch

This named-API checkpoint predates the subsequent [Python comparison operator unit](COMPARISON_OPERATOR_VALIDATION.md). Operator limitations below describe the original checkpoint.

The subsequent [compiled empty-broadcast correction](COMPILED_EMPTY_BROADCAST_VALIDATION.md) closes the specific compiler edge case recorded below.

## Supported API

Float32 Tensor methods and package functions eq/ne/lt/le/gt/ge compare logical values and return real boolean tensors. Real scalar right operands use the existing float32 scalar-construction path, including its rounding. Boolean Tensor methods and package functions logical_and/logical_or/logical_xor accept bool tensors or boolean scalar right operands. No numeric truth conversion or dtype promotion is applied by binary logical operators.

Python ==/!= keep object identity behavior; ordered comparison and bitwise logical operator overloads are not introduced. These are named APIs only. Boolean-input comparisons, boolean indexing/reductions, joining and compiler capture remain unsupported. Existing logical_not / ~ behavior remains unchanged.

## Typed dispatch and semantics

Static CPU predicate registrations contain operation name, input device/dtype and a scalar kernel. Rust function signatures enforce f32 comparison inputs versus bool logical inputs. Both families return BoolTensorData, so output dtype is fixed by actual typed storage rather than a metadata label. Input metadata checks use both operands. Shape inference and logical row-major indexing reuse the existing checked broadcasting rules.

Native float comparisons only borrow input values; they create no autograd Node or saved float handles. Results require no gradients even when both float operands do. Comparisons do not consume the input graph. Rust float comparison semantics preserve IEEE NaN behavior: eq and ordered comparisons with NaN are false; ne is true. Signed zeros compare equal. NaNs are not clamped or replaced.

Output allocation uses fallible reservation. Nonempty broadcasts currently use the existing two output-sized index maps, with additional usize memory cost. Strided inputs may materialize logical caches. No speed or memory improvement is claimed.

## Empty broadcasting correction

The shared eager shape count now uses TensorLayout's zero-aware checked count. Empty broadcast plans return before computing unreachable strides/index maps. A failing native-backed Python regression reproduced float addition raising shape product overflow for shape [usize::MAX, 2, 0]. The corrected float addition, float comparison and bool logical operation all return that empty logical shape. Nonempty output-size overflow remains rejected. This change does not expand compiler capture or claim every specialized workspace supports arbitrary empty shapes.

## Validation scope

Tests cover strided multi-axis broadcasting, package and method forms, scalar/empty results, invalid types/shapes, NaNs, infinities and signed zeros, nondifferentiability and continued input backward, plus installed PyTorch comparison parity. Rust coverage checks real output dtype, unknown operation rejection and broadcast output overflow. Independent review findings and final install gates are recorded below after completion.

Independent read-only review found no predicate-path defects. It confirmed a separate compiled-broadcast limitation: CompiledBroadcast still computes strides/products before recognizing huge empty layouts. The new eager regression does not prove that compiled edge case works. Correct that parity gap as the next bounded unit before adding more compiler coverage. The review also confirmed the existing index-map memory cost; no performance refactor is included here.


## Validated checkpoint

Implementation: 95a2112. macOS arm64, Python 3.13.7, rustc 1.99.0. Full Python suite: **636 passed**, two existing NumPy empty-mean reference warnings. Rust debug/release: **79 passed**, six manual benchmarks ignored. Formatting and Clippy with warnings denied passed. README predicate example executed successfully against editable code and the direct wheel.

Fresh source installation in an activated new environment, maturin 1.15.0, --no-build-isolation and no PyTorch: **500 passed, 27 skipped**, same reference warnings. Three Torch-only modules were excluded; other optional Torch tests skipped naturally. Direct wheel install then passed **52 targeted bool/predicate/compiler tests**, two optional Torch tests skipped. Artifact inspector confirmed README matching, 30 matching license files and source scratch exclusions.

Validated archives remain outside Git under workspace directory work/predicates-1792030794. Direct wheel SHA256: 1231927673b329d9a5f4707bb0ac99f16a51890199ffb29841bdb9498286ab22. Source archive SHA256: 1713987c44e5548f5f413ab40d61e22add1ddd0655016542b21927daf6d65b72. This validation section was added after building those archives. No generated benchmark data or release artifacts were committed.

Next unit: reproduce and correct huge-empty CompiledBroadcast inference/backward parity, preserving checked nonempty overflow. Then audit Python equality/hash/identity interactions before considering comparison operator overloads. General multi-dtype arithmetic, bool compiler lowering, GPU, custom autograd and distributed work remain incomplete. Package version remains 0.10.0. No push, merge, tag or publication occurred.
