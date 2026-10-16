# Custom first-order autograd

Subclass `densecore.autograd.Function` and implement static `forward` and
`backward` methods. Invoke `apply`, rather than calling `forward` directly.

```python
import numpy as np
import densecore as dc

class Square(dc.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, gradient):
        x, = ctx.saved_tensors
        return gradient * x * 2

x = dc.tensor([2., 3.], requires_grad=True)
Square.apply(x).sum().backward()
np.testing.assert_array_equal(x.grad.numpy(), [4., 6.])
assert dc.autograd.gradcheck(Square.apply, dc.tensor([.2, .3], requires_grad=True))
```

`apply` accepts one or more positional float32 Tensor inputs and returns one
float32 Tensor. Both callbacks run under `no_grad`. `ctx.needs_input_grad` is a
tuple of input flags. Return one detached float32 Tensor or `None` per input;
a single input may return its gradient directly. Gradient shapes must exactly
match inputs, including frozen inputs when a gradient is supplied. Duplicate
inputs accumulate contributions. `None` supplies no contribution.

`ctx.save_for_backward` accepts float32 tensors during forward. It stores
immutable numeric snapshots with version checks, without retaining their original
graph nodes. `ctx.saved_tensors` is available only during forward/backward;
returned tensors are detached. Direct saved Parameter handles and frozen-value provenance reject changed versions.
Snapshots preserve immutable computed values; they do not recursively attach the
original intermediate graph.
Plain context attributes can store metadata, but user-owned references retain
what they point to until the context is collected.

Native backward yields before Python callbacks and holds no engine guard across
them. Python owns callback records through a GC-visible graph. Callback failures,
invalid gradients and version errors leave this backward's leaf updates
unpublished and its graph unconsumed, allowing a corrected callback to retry.
User callback side effects, including an independent backward or optimizer step,
are not rolled back. Concurrent non-retaining backward sessions cannot commit
the same consumed graph twice. Concurrent use of the same callback context and
reentrant custom backward are rejected. Independent ordinary backward inside a
callback is supported under `enable_grad`.

`retain_graph=True` follows ordinary backward semantics. `no_grad` or inputs
without gradients produce detached outputs. The private native backward entry
point rejects custom graphs; use public `Tensor.backward`.

This subset has no non-tensor arguments, multiple outputs, boolean saved tensors,
higher-order gradients, forward-mode AD or custom compiler nodes. Tensor gradient
hooks are supported separately; see TENSOR_HOOKS.md.
A supported compiled module may consume a custom result and propagate its VJP
back into eager custom backward. It does not compile the Python callback.
