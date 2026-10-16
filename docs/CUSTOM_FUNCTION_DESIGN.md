# First-order custom Function design

## Boundary

Rust Custom recipes contain a monotonic numeric id, float32 input handles and detached saved snapshots only. No Python reference enters the Arc graph. This avoids Python refcount finalizers under ENGINE and invisible Python/Rust ownership cycles. Saved snapshots retain immutable values plus version/generation/provenance, not graph edges or mutable Python Parameter wrappers.

BackwardSession snapshots traversal/recipes under ENGINE, runs ordinary derivatives under ENGINE, and yields an id, detached upstream gradient and saved snapshots with all guards released. Python calls user backward under no_grad and resumes with detached float32 gradients or None. Every resume/commit rechecks versions and non-leaf readiness. Leaf publication and graph consumption use the existing atomic transaction. Legacy native backward drains this scheduler for ordinary graphs and rejects a Custom yield with an instruction to use Tensor.backward.

No persistent graph lease is held across callbacks. Two non-retaining sessions may snapshot a graph; after one consumes it, the other must fail validation without publishing. Python try/finally rejects recursive custom backward and concurrent use of one callback context, clears temporary saved wrappers and aborts failed sessions. Retaining a Python exception/traceback cannot block later backward.

## Python ownership

Tensor gains a GC-visible _contexts tuple DAG. Public differentiable wrappers propagate parent owner tuples by sharing, never flattening. A Function output owns (record, merged_input_owners). Normal results without custom ancestors use (). Detached, no_grad, gradient and bool results drop owners. Records and ctx are ordinary Python objects, so ctx.output = output cycles remain visible to Python GC. Root owners are walked iteratively once per backward into an id-to-record map; there is no global strong or weak callback registry.

Every constructor/wrapper site must initialize owners, including reshape, functional ops, Linear, cat, compiled outputs, optimizer snapshots and fused result raw construction. A native has_custom bit permits deterministic detection of missing owner propagation. Private raw native APIs do not manage Python callbacks.

## Protocol and saved tensors

Function subclasses implement static forward(ctx, *tensor_inputs) and backward(ctx, grad_output), invoked through Function.apply. Initial inputs and the single output are CPU float32 Tensors; no non-tensor arguments, multi-output, bool output, higher-order AD or compiler capture is claimed. Forward and backward run under no_grad. ctx.needs_input_grad reflects input flags. save_for_backward accepts float32 tensors during forward; Rust snapshots preserve saved versions and values. ctx.saved_tensors is available during forward/backward, with detached snapshots during backward. Context holds no saved tensor payload while idle; native recipes own it until consumption or graph destruction. User metadata attributes remain Python-managed.

Capture original native input handles before forward and check them before attach, so input replacement during forward cannot silently change parent identity. Callback errors and invalid gradient tuples/types/shapes leave engine-published leaf gradients and readiness unchanged. Callback side effects cannot be rolled back. None gradients are allowed. Unsupported higher-order returned gradients reject explicitly.

## Validation

Test resumable Rust gradients, invalid shapes, saved versions, optimistic consumption and saved-node graph release. Public tests cover full-Jacobian gradcheck, shared inputs/views, frozen inputs, failure/retry, original exceptions, no_grad, saved-version mutation, callback grad reads and nested ordinary backward without deadlock, recursive custom rejection, owner propagation through all public differentiable APIs/compiled submodules, GC-visible cycles and consumed payload release. Use subprocess timeouts for deadlock-sensitive tests.

Independent critique selected optimistic validation over leases, owner walking over a weak registry, and Rust detached snapshots over retaining mutable Python Tensor wrappers. No hooks or placeholder Function API is published before this protocol is implemented and validated.
