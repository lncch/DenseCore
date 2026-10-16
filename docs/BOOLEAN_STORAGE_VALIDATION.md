# Boolean storage foundation

This is the foundation checkpoint. Subsequent named comparison and binary logical support is documented in [predicate validation](PREDICATE_DISPATCH_VALIDATION.md); the limitations below describe this original unit.

## Implemented contract

The shared storage and view implementation is generic over a sealed Element trait with f32 and bool implementations. TensorData remains the f32 alias used by all existing math/autograd kernels; BoolTensorData owns real Vec<bool> values. Rust stores these bool values as one byte per element, without bit packing. Layout validation, immutable Arc storage, strided logical-value caching, reshape and slicing rules are shared. Float32 kernel signatures and autograd nodes remain statically float32.

A separate private PyO3 _BoolTensor owns boolean data without an autograd Node. The existing public Tensor wrapper accepts explicit dtype="bool" and exports copied NumPy bool arrays. Construction accepts boolean/numeric inputs using nonzero truth conversion, including NaN; strings and object arrays are rejected. Default construction stays float32. No implicit dtype inference or promotion is added.

Boolean tensors support layout inspection, shared positive-step slicing/permutation/transpose, contiguous reshape, conservative non-contiguous reshape copying, detach and native logical-not via logical_not / ~. Grad is None, requires_grad is false, and backward/zero_grad raise. Boolean requires_grad=True is rejected even under no_grad. Logical-not allocates new bool storage with fallible reservation.

Float-only receiver calls use a common Python native-handle check. PyO3's typed argument boundary prevents bool handles from entering float kernels, optimizer handles and compiler execution. Functional neural-network receivers also validate float32. Boolean Tensor initialization of Parameter and boolean state-dictionary values are rejected; failed state loading preserves existing Parameters. Explicit conversion of numeric/boolean arrays through the existing default float32 constructor remains unchanged.

TensorSpec helpers can inspect either real typed representation. Rust tests now exercise dtype mismatch using genuine bool and float32 buffers. Compiler lowering explicitly rejects Bool, including input-only graphs. Device mismatch remains untestable because CPU is the only implemented device. The Python graph binding remains float-native-handle-only.

## Scalar truth

Python truth testing requires exactly one logical element for both float32 and bool tensors. Empty and multi-element tensors raise an ambiguity RuntimeError. Reading that single native value avoids NumPy array rank limits. A failing native-backed regression reproduced rank-100 scalar truth failing through NumPy reshape before that dependency was removed.

This changes prior float tensor truth behavior, which treated every Tensor object as true. Elementwise boolean combinations must not use Python and/or. Binary logical operations are not implemented in this unit.

## Design and review

The independent design critique accepted typed storage instead of a runtime storage enum or copied boolean layout code. Accepted corrections: explicit float-only receiver guards, cross-dtype shares_storage=False, Parameter/state-loading guards and explicit Bool compiler rejection. A narrow Parameter/state guard preserves existing default float construction behavior. Numeric boolean truth conversion is deliberate and checked against installed PyTorch; strict 0/1-only construction was not adopted.

See BOOLEAN_STORAGE_DESIGN.md and the boolean foundation implementation plan for scope. Comparisons and binary logical kernels remain the next unit. Logical-not currently uses the typed boolean native path; this foundation does not claim general multi-dtype operator dispatch or boolean compiler support.

## Limitations

CPU only. No boolean gradients, comparison operators, binary logical operators, boolean reductions/indexing, boolean cat/stack, boolean NN Parameters/checkpoints or compiler lowering. No cross-dtype storage sharing or promotion. Strided logical caches and view retention have the same memory tradeoffs as float32. No performance or hardware claim is added. Package version remains 0.10.0.

Gradcheck rejects boolean inputs explicitly, including nondifferentiable boolean constants alongside float32 inputs. A failing regression showed its previous reconstruction path silently converted such constants to float32. Boolean constants in numerical checks are not supported in this unit.

Independent code review found a missed eager Linear receiver guard and optimizer state buffers accepting bool values through Python-to-float conversion. Failing regressions reproduced both issues, including SGD momentum and both Adam moments. The final guards reject them before optimizer replacement or hyperparameter publication. Mixed float-left/bool-right operands continue to reject through PyO3 typed argument conversion; error messages can include private native class names. That boundary is safe and tested, without a message-unification refactor.


## Validated checkpoint and continuation

Implementation commit: 8d915a6. Local platform: macOS arm64, Python 3.13.7, rustc 1.99.0. Full Python suite: **619 passed**; two existing NumPy empty-mean reference warnings remain. Rust debug/release: **78 passed**, six manual benchmarks ignored. Formatting and Clippy with warnings denied passed. The new README boolean example ran against editable code and the fresh direct wheel.

Fresh source installation in an activated new environment used maturin 1.15.0 with --no-build-isolation, without PyTorch: **484 passed, 26 skipped**, same reference warnings. Three Torch-only modules were excluded; other optional Torch cases skipped naturally. Direct wheel installation then passed **46 targeted boolean/compiler/layout tests**, four optional Torch tests skipped. Artifact inspection matched README metadata and all 30 license files and confirmed source scratch exclusions.

Archives remain outside Git in workspace directory work/boolean-foundation-1792030505. Direct wheel SHA256: d05e7883d3b2ecd032868b4667bc5af6fb95a0724c783bbb1dbdcd314fb4bf8b. Source archive SHA256: 1243faa468ab1724e9117a80f024620430716c23a8e91116b6a71799a898ce3a. This validation section and final plan status were written after those builds. Generated benchmark data and release artifacts were not committed.

Protected feat/v0.11 remains at 4044e5c0a415e9e20e0b57cc91b588e5b82f4bbc. No push, merge, tag or publication occurred. Current version remains 0.10.0, and no later milestone is declared complete.

Next dependency-ready unit: registered float32 comparisons returning real boolean data, and boolean binary logical operations using shared checked broadcasting. Begin with named methods/functions; equality overloads need a separate identity/hash audit. Test arbitrary strided broadcasting, empty dimensions, NaN/signed-zero comparison semantics, nondifferentiability of grad-enabled float operands, invalid dtype combinations and no graph retention. Preserve this unit's float-only optimizer/compiler guards. Continue under the autonomous mission without asking the user to say continue.
