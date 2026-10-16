import gc
import weakref
import numpy as np
import pytest
import densecore as d


class Square(d.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert not d.is_grad_enabled()
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad):
        assert not d.is_grad_enabled()
        x, = ctx.saved_tensors
        return grad * x * 2


def test_square_gradcheck_views_and_shared_inputs():
    x = d.tensor([[.2, .3], [.4, .5]], requires_grad=True)
    assert d.autograd.gradcheck(lambda a: Square.apply(a.T), x)
    y = Square.apply(x)
    (y + y).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), x.numpy() * 4)
    with pytest.raises(RuntimeError, match="consumed"):
        y.sum().backward()


def test_saved_parameter_versions_and_no_grad():
    p = d.nn.Parameter([2.])
    y = Square.apply(p)
    (p * p).sum().backward()
    d.optim.SGD([p], lr=.1).step()
    with pytest.raises(RuntimeError, match="modified"):
        y.backward()
    with d.no_grad():
        out = Square.apply(p)
    assert not out.requires_grad and not out._contexts


def test_callback_exception_preserves_gradients_and_allows_retry():
    class F(d.autograd.Function):
        fail = True
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x
        @staticmethod
        def backward(ctx, grad):
            if F.fail:
                raise ValueError("custom sentinel")
            x, = ctx.saved_tensors
            return grad * x * 2
    x, sibling = d.tensor(2., requires_grad=True), d.tensor(3., requires_grad=True)
    sibling.backward()
    out = F.apply(x) + sibling
    with pytest.raises(ValueError, match="custom sentinel") as exc:
        out.backward()
    assert sibling.grad.numpy().item() == 1 and x.grad is None
    F.fail = False
    out.backward()
    assert x.grad.numpy().item() == 4 and sibling.grad.numpy().item() == 2
    assert exc.value.args == ("custom sentinel",)


def test_callback_context_survives_descendants_and_gc_cycle_collects():
    refs = []
    class F(Square):
        @staticmethod
        def forward(ctx, x):
            refs.append(weakref.ref(ctx))
            ctx.save_for_backward(x)
            return x * x
    x = d.tensor(2., requires_grad=True)
    y = F.apply(x)
    z = (y + 1).sum()
    del y
    gc.collect()
    assert refs[-1]() is not None
    z.backward()
    assert x.grad.numpy().item() == 4
    del z
    gc.collect()
    assert refs[-1]() is None
    y = F.apply(x)
    refs[-1]().output = y
    del y
    gc.collect()
    assert refs[-1]() is None


@pytest.mark.parametrize("operation", [lambda x: x.T.reshape(4).sum(),
    lambda x: d.functional.linear(x, d.tensor([[1., 2.]])).sum(),
    lambda x: d.cat([x, x], dim=0).sum(),
    lambda x: d.stack([x, x], dim=0).sum(),
    lambda x: d.compile(d.nn.ReLU())(x).sum()])
def test_owner_propagation_across_public_operations(operation):
    x = d.tensor([[.2, .3], [.4, .5]], requires_grad=True)
    out = operation(Square.apply(x))
    assert out._native.has_custom and out._contexts
    gc.collect()
    out.backward()
    assert x.grad is not None


@pytest.mark.parametrize('bad', ['count', 'shape', 'type', 'bool', 'higher'])
def test_invalid_callback_result_preserves_graph_for_retry(bad):
    class F(Square):
        invalid = True
        @staticmethod
        def backward(ctx, grad):
            if not F.invalid:
                return grad * 2
            return {'count': (), 'shape': d.tensor([1.]), 'type': 1,
                    'bool': d.tensor(True, dtype='bool'),
                    'higher': d.tensor(1., requires_grad=True)}[bad]
    x = d.tensor(2., requires_grad=True)
    y = F.apply(x)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        y.backward()
    assert x.grad is None
    F.invalid = False
    y.backward()
    assert x.grad.numpy().item() == 2


def test_duplicate_inputs_frozen_inputs_and_none_gradients():
    class F(d.autograd.Function):
        @staticmethod
        def forward(ctx, x, y, frozen):
            assert ctx.needs_input_grad == (True, True, False)
            return x + y + frozen
        @staticmethod
        def backward(ctx, grad):
            return grad, grad * 2, None
    x = d.tensor(2., requires_grad=True)
    F.apply(x, x, d.tensor(3.)).backward()
    assert x.grad.numpy().item() == 3


def test_callback_can_read_grad_and_run_independent_backward():
    import os
    import subprocess
    import sys
    script = '''
import densecore as d
class F(d.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        return x * x
    @staticmethod
    def backward(ctx, g):
        assert ctx.x.grad is None
        with d.enable_grad():
            independent = d.tensor(3., requires_grad=True)
            (independent * independent).backward()
            assert independent.grad.numpy().item() == 6
        return g * 4
x = d.tensor(2., requires_grad=True)
F.apply(x).backward()
assert x.grad.numpy().item() == 4
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=15, env=os.environ.copy())


def test_reentrant_custom_backward_rejects_and_retry_succeeds():
    class F(Square):
        recurse = True
        @staticmethod
        def backward(ctx, grad):
            if F.recurse:
                other.backward()
            return grad * 4
    x = d.tensor(2., requires_grad=True)
    other = Square.apply(d.tensor(3., requires_grad=True))
    y = F.apply(x)
    with pytest.raises(RuntimeError, match='reentrant'):
        y.backward()
    assert x.grad is None
    F.recurse = False
    y.backward()
    other.backward()
    assert x.grad.numpy().item() == 4


def test_external_saved_parameter_version_and_context_phase():
    p = d.nn.Parameter(3.)
    contexts = []
    class F(Square):
        @staticmethod
        def forward(ctx, x):
            contexts.append(ctx)
            ctx.save_for_backward(p)
            return x * x
    x = d.tensor(2., requires_grad=True)
    y = F.apply(x)
    with pytest.raises(RuntimeError, match='only available'):
        _ = contexts[0].saved_tensors
    with pytest.raises(RuntimeError, match='only available'):
        contexts[0].save_for_backward(x)
    p.backward()
    d.optim.SGD([p], lr=.1).step()
    with pytest.raises(RuntimeError, match='modified'):
        y.backward()
    assert x.grad is None


def test_strided_gradients_retention_and_detached_owner_release():
    class F(d.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.T
        @staticmethod
        def backward(ctx, grad):
            return grad.T
    x = d.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    y = F.apply(x)
    seed = d.tensor([[1., 2.], [3., 4.]]).T
    y.backward(seed, retain_graph=True)
    y.backward(seed)
    np.testing.assert_array_equal(x.grad.numpy(), [[2., 4.], [6., 8.]])
    assert not y.detach()._contexts
    assert not x.grad._contexts


@pytest.mark.parametrize('kind', ['empty', 'scalar', 'bool', 'multiple'])
def test_function_rejects_unsupported_signature(kind):
    class F(d.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return (x, x) if kind == 'multiple' else x
    with pytest.raises(TypeError):
        if kind == 'empty':
            F.apply()
        elif kind == 'scalar':
            F.apply(1.)
        elif kind == 'bool':
            F.apply(d.tensor(True, dtype='bool'))
        else:
            F.apply(d.tensor(1., requires_grad=True))


@pytest.mark.parametrize('op', [lambda x: x.exp().sum(), lambda x: x.log().sum(),
    lambda x: x.sqrt().sum(), lambda x: x.sin().sum(), lambda x: x.cos().sum(),
    lambda x: (-x).mean(), lambda x: x.softmax(-1).sum(),
    lambda x: x.log_softmax(-1).sum(), lambda x: x[0, :].sum(),
    lambda x: x.T.contiguous().sum(), lambda x: (d.tensor([[1., 0.], [0., 1.]]) @ x).sum(),
    lambda x: d.functional.mse_loss(d.tensor([[1., 2.], [3., 4.]]), x),
    lambda x: d.nn.LayerNorm(2)(x).sum()])
def test_additional_owner_paths(op):
    x = d.tensor([[.2, .3], [.4, .5]], requires_grad=True)
    output = op(Square.apply(x))
    assert output._contexts
    output.backward()
    assert x.grad is not None


def test_forward_input_replacement_cannot_change_captured_parent():
    model = d.nn.Linear(1, 1, bias=False)
    p = model.weight
    class F(d.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            model.load_state_dict(model.state_dict())
            return x * x
        @staticmethod
        def backward(ctx, grad):
            return grad
    with pytest.raises(RuntimeError, match='modified'):
        F.apply(p)
    assert p.grad is None


def test_parameter_update_inside_callback_prevents_leaf_publication():
    p = d.nn.Parameter(2.)
    p.backward()
    optimizer = d.optim.SGD([p], lr=.1)
    x = d.tensor(3., requires_grad=True)
    class F(Square):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(p)
            return value * value
        @staticmethod
        def backward(ctx, grad):
            optimizer.step()
            return grad * 6
    with pytest.raises(RuntimeError, match='modified'):
        F.apply(x).backward()
    assert x.grad is None
    # Callback side effects persist; only this backward's publication is atomic.
    np.testing.assert_allclose(p.numpy(), 1.9)


def test_same_context_concurrent_backward_rejects_without_blocking():
    import os
    import subprocess
    import sys
    script = '''
import threading
import densecore as d
entered, release = threading.Event(), threading.Event()
class F(d.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * x
    @staticmethod
    def backward(ctx, grad):
        entered.set()
        assert release.wait(5)
        return grad * 4
x = d.tensor(2., requires_grad=True)
y = F.apply(x)
errors = []
def first():
    try:
        y.backward()
    except BaseException as exc:
        errors.append(exc)
thread = threading.Thread(target=first)
thread.start()
try:
    assert entered.wait(5)
    try:
        y.backward()
    except RuntimeError as exc:
        assert 'already running' in str(exc)
    else:
        raise AssertionError('concurrent callback was accepted')
finally:
    release.set()
    thread.join(5)
assert not thread.is_alive() and not errors, errors
assert x.grad.numpy().item() == 4
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=15, env=os.environ.copy())
