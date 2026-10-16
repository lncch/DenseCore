# Device and dtype readiness audit, 2026-10-16

Read-only audit of whether the current storage, dispatch and autograd design can take a GPU
device or a second floating dtype (float16/bfloat16) without a rewrite. No code changed.

## Findings

1. **Storage is host memory by type.** `TensorStorage<T>` is `Arc<Vec<T>>` (src/tensor.rs:197).
   There is no buffer abstraction, so a GPU allocation cannot be represented. Kernels read
   `TypedTensorData::values() -> &[T]` (src/tensor.rs, `values`), which assumes the CPU can
   address the data directly.

2. **Device is fixed.** `Device` has one variant, `Cpu` (src/tensor.rs:2), and
   `TypedTensorData::device()` returns `Device::Cpu` without consulting storage. The metadata is
   honest today but carries no information.

3. **The differentiable dtype is fixed at compile time.** `pub type TensorData = TypedTensorData<f32>`
   (src/tensor.rs:323) is what autograd nodes, graph values, saved tensors and optimizers hold.
   `Element` is sealed to `f32` and `bool`. Adding float16 means either making autograd generic
   over `T` or turning `TensorData` into a dtype enum. Both touch autograd.rs, graph*.rs and optim.rs.

4. **Eager dispatch keys are ready in shape, not in content.** Unary, binary, reduction, loss,
   LayerNorm, linear and activation families register `(operator, device, dtype)` entries
   (src/dispatch.rs, `UNARY_KERNELS`, `BINARY_KERNELS`). Lookup already rejects mismatches. New
   device/dtype kernels could be registered here once storage allows them.

5. **Compiled and fused paths bypass dispatch.** Graph workspaces run specialized CPU
   instructions on `Vec<f32>` buffers (src/graph_workspace.rs, src/graph_backward.rs; 41 `f32`
   uses across graph files). A GPU or float16 compiled path needs its own lowering; schemas are
   shared, kernels are not.

6. **Strided views copy.** Non-contiguous `values()` materializes a cached row-major copy. This
   is correct and simple on CPU, but on a GPU it would force a device-to-host round trip unless
   kernels learn to take strides.

## What this means

- Eager dispatch (finding 4) is the part closest to ready.
- Storage (findings 1, 2, 6) is the blocking piece for any GPU work. It should change before
  more CPU-only kernels are written against `values() -> &[T]`, because each new kernel adds to
  the migration later.
- Float16 (finding 3) is a separate, larger change to autograd and optimizers. It is not needed
  for CPU work and can wait until a target workload needs it.

## Suggested order (not started)

1. Introduce a storage enum (`Host(Arc<Vec<T>>)` now, device variants later) behind the
   existing `TensorStorage` API, with `values()` returning an error for non-host storage.
2. Make `device()` read from storage.
3. Only then add a GPU backend, starting with allocation, copies and one kernel.

Each step is a design decision for the project owner. This audit records evidence only.
