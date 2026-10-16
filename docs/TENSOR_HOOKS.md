# First-order tensor gradient hooks

`Tensor.register_hook(callback)` supports CPU float32 tensors requiring gradients.
It returns a handle with idempotent `remove()`. Register on an ancestor after
creating descendants, including a non-leaf tensor. The callback receives its
node's accumulated gradient once per backward, before that node's VJP or leaf
publication. Hooks on custom Function outputs run before their custom backward.

```python
import densecore as dc

x = dc.tensor(2., requires_grad=True)
y = x * x
loss = y + y
handle = y.register_hook(lambda gradient: gradient * 3)
loss.backward()
assert x.grad.numpy().item() == 24.
handle.remove()
```

Callbacks run under no_grad with no native engine/node/workspace guard held.
Return None to preserve the incoming gradient, or a detached float32 Tensor of
exactly the same shape to replace it. Strided replacement gradients are supported.
Higher-order gradients are rejected. Hooks on disconnected nodes do not run.
Retained traversals run hooks again. Existing leaf gradients are accumulated only
after the traversal succeeds; the hook sees this traversal's contribution.

Callbacks are snapshotted at backward entry in registration order. Removal or
registration during backward affects subsequent traversals, not that snapshot.
Ordering across unrelated nodes is unspecified. Concurrent invocation of the same
slot and all nested public backward calls from a tensor hook are rejected.
Custom Function retains its existing independent ordinary-backward behavior.

After each callback, the engine validates saved versions, graph readiness, detached
gradient state and exact shape before calling the next hook. Exceptions and invalid
results leave this backward's pending leaf updates unpublished and graph unconsumed.
Remove or correct the faulty hook to retry. Callback side effects such as optimizer
updates cannot be rolled back. Parameter hooks persist across supported optimizer,
model-state and checkpoint replacements.

Compiled public outputs/inputs/parameters support hooks through their eager graph
boundary. Internal activations without Python Tensor wrappers are not hook targets.
Fused compiled_train_step rejects active parameter hooks before mutation at both
public Python and native executor boundaries. Ordinary optimizer.step uses the
already-hooked gradients. Direct private native backward rejects active hooked
graphs; use public Tensor.backward. Arbitrary private native alias manipulation is
not a supported ownership API.

Each differentiable Python wrapper owns a mutable GC-visible slot, shared through
ancestor owner tuples. Slots hold native Weak<Node> identities, not graph-owning
handles; consumed native ancestors can release while output wrappers survive.
Callbacks and closures remain owned until removal or owner-DAG collection; closures
that capture tensors intentionally retain those tensors. Detached/no_grad outputs,
boolean tensors and exported gradients inherit no hook slots. No global strong
callback registry or Python reference inside native Arc graphs is used.

This ownership has a measured Python allocation and graph-construction cost even
when no hooks are registered. See HOOK_VALIDATION.md. It is not a performance win.
