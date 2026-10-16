# Compiler metadata validation

## Contract

TensorSpec is an internal immutable logical tensor specification containing checked shape, device and dtype. ExecutionPlan stores one specification per node rather than a separate shape-only array. Compilation from actual tensor handles builds specifications from their native metadata. Legacy internal shape-only compilation explicitly assumes CPU float32 and delegates to the same typed entry point.

All currently lowered instructions preserve the common operand device and dtype. Compilation checks operand agreement. The compiler explicitly matches supported input device/dtype combinations before lowering, including input-only graphs. CPU float32 is the sole implemented combination. Adding a real enum variant requires a lowering decision; no wildcard silently routes a future device or dtype into CPU float32 workspaces. Internal compiled cross-entropy labels remain detached float32 class-index metadata, not a public integer dtype.

```text
native tensor handles
        |
        v
checked input specifications + gradient flags
        |
        v
shape inference + operand metadata propagation
        |
        v
immutable plan specifications
        |
        v
runtime validation -> CPU float32 workspace execution
```

Python executor cache entries compare borrowed shape/device/dtype metadata and gradient flags on hits, without creating owned specification copies. Inference retains its existing `enabled && requires_grad` flag policy; fused training keeps its existing gradient-enabled guard. The four-entry cache policy and live-output workspace ownership are unchanged.

Layout is deliberately excluded from TensorSpec. Contiguous and strided tensors with equal logical metadata can use the same specialization. Kernels continue to consume logical row-major values through the existing materialization cache. Layout is not a claim of contiguous physical storage.

## Runtime checks and corrected empty layouts

Direct plan execution and borrowed workspace execution share input-contract validation. Backward seeds retain the existing exact-output-shape error and additionally check the output device/dtype specification before resetting gradient buffers. Parameter-version and optimizer-transaction behavior are unchanged.

Workspace element counts now use TensorLayout's checked count rules. A zero dimension means zero elements before any unreachable product is evaluated. A regression reproduced inference failing with `workspace tensor size overflow` and gradient-enabled execution failing with `gradient size overflow` for shape `[usize::MAX, 2, 0]`. Native-backed Python tests now pass inference, forward and backward for that empty layout without allocating its unreachable dimensions.

Checked TensorSpec construction also rejects overflowing nonempty input-only graph shapes before execution. Rust tests exercise every existing instruction's derived shape and CPU float32 metadata, explicit typed compilation, shape rejection and workspace reuse. Python tests cover strided-layout changes, recovery after invalid shapes, and no-grad execution.

## Validation limits

At this metadata checkpoint, CPU and Float32 were the only Device/DType variants. The later [boolean storage unit](BOOLEAN_STORAGE_VALIDATION.md) adds a real Bool representation and explicit compiler rejection; it does not add bool lowering. Positive metadata propagation and shape rejection are testable; runtime device/dtype rejection branches could not be reached at that checkpoint. The boolean unit now tests dtype rejection with real typed storage; device rejection remains unreachable with CPU alone. No fake enum variants or metadata-only device support were added. Tests do not claim negative device/dtype coverage.

No public Python API, capture coverage, dtype promotion, GPU backend or performance result is added by this change. Specialized workspaces still execute CPU kernels directly. New unary math, bmm, cat and stack remain eager-only operations.


## Validated checkpoint

Implementation commit: `d72f986`. Local platform: macOS arm64, Python 3.13.7, rustc 1.99.0. Full Python suite: **585 passed**, with two existing NumPy empty-mean reference warnings. Rust debug/release: **76 passed**, six manual benchmarks ignored. Formatting and Clippy with warnings denied passed. Independent read-only design and code review reported no remaining actionable defect in this change; it explicitly identified unreachable device/dtype rejection paths as a test limitation.

Fresh source installation used an activated new virtual environment, maturin 1.15.0 and `--no-build-isolation`, without PyTorch. Result: **451 passed, 25 skipped**, with the same reference warnings; three Torch-only modules were excluded and other optional Torch cases skipped naturally. A subsequent installation of the direct wheel passed **35 compiler tests**. Wheel metadata matched README, all 30 license files matched, and source scratch exclusions passed.

Validated archives remain outside Git in workspace directory `work/compiler-metadata-1792026479`. Wheel SHA256: `bd2eb22e630552e879db9a7dc20aada0e820bce5bc4350612c47811c208391d3`. Source archive SHA256: `bf971a97eea0ccdb8820c69e71383f8c192f001ec5a2fb120a92101777b4d66e`. This validation section was added after building those archives. No generated benchmark data or release artifacts were committed. Package version remains 0.10.0; no push, merge, tag or publication occurred.

## Continuation

The prior operator-coverage checkpoint's compiler-metadata next unit is complete at the scope above. Comparison/logical operators still require a real boolean tensor representation and explicit nondifferentiable behavior. Next unit: design that storage/API contract against current autograd, checkpoint and dispatch assumptions; get an independent read-only critique, then implement the smallest coherent supported subset with failing regressions. Do not represent boolean masks as float32 while claiming boolean dtype support. Preserve the CPU-only float32 path and existing compiler/autograd behavior throughout. GPU, mixed precision, custom autograd and distributed work remain incomplete.
