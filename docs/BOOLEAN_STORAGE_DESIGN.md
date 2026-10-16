# Boolean storage design

## Scope and decision

Use one generic immutable storage/layout implementation, specialized at compile time for f32 and bool. Keep the existing TensorData name as the float32 alias. Add BoolTensorData for real boolean values. Rust Vec<bool> stores one bool per byte; this design does not bit-pack masks. Device remains CPU.

An enum inside TensorData would force every existing values() caller to handle dtype errors or risk a panic when a float kernel meets boolean storage. A separate copied layout implementation would duplicate slicing and overflow rules. Generic typed storage shares those rules while keeping float kernels and autograd statically restricted to float32.

The public Python Tensor wrapper can hold either private native handle. A private _BoolTensor owns BoolTensorData without an autograd Node. No bool requires_grad state is accepted. Native float operations continue to require _Tensor arguments; boolean handles cannot cross that boundary. Python validates unsupported boolean operations with TypeError rather than silently casting or leaking AttributeError.

## Initial public subset

Explicit tensor(data, dtype="bool") copies values into boolean native storage; numpy() copies back with NumPy bool dtype. Default construction remains float32. Boolean tensors support shape/layout inspection, basic positive-step slicing, permutation/transpose, reshape, contiguous materialization and detach with the same immutable view rules as float32 tensors. Grad is None, requires_grad is false, and backward raises. Mixed-dtype shares_storage returns false.

Logical-not is a native boolean operation and returns new boolean storage. Initial float math, reductions, neural-network modules, joining, optimizer/checkpoint Parameters and compiler capture remain float32-only. Parameter construction from a boolean Tensor rejects rather than silently creating a trainable float copy. Scalar truth is allowed only for a single logical element; empty and multi-element tensors raise ambiguity errors.

## Following unit

Add native float32 comparisons returning BoolTensorData and native boolean binary logical operations. Reuse checked broadcast shape/index rules. Start with named eq/ne/lt/le/gt/ge and logical_and/or/xor APIs. Audit Python hash/identity-dependent Parameter code before overloading equality operators. These results are nondifferentiable and never retain a float autograd graph.

## Validation and boundaries

Observe failing storage and public API regressions before implementation. Verify existing float32 tests remain unchanged. Exercise bool views, scalar/empty behavior, copy ownership, unsupported operations, compiler dtype rejection and bool requires_grad rejection. Run full Python and Rust debug/release tests, formatting, Clippy and fresh source/wheel installation. Keep implementation and validation documentation commits separate. No GPU, mixed precision, boolean indexing, boolean reduction, dtype promotion or performance claim is part of this unit.

This design follows the standing autonomous mission authorization. Independent critique was completed before implementation; accepted corrections and scope choices are recorded in BOOLEAN_STORAGE_VALIDATION.md.
