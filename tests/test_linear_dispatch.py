import numpy as np
import pytest
import densecore as d
from densecore._native import _GraphExecutor


def execute(x, w, b, compiled):
    if not compiled: return d.functional.linear(x, w, b)
    args = [x, w] + ([] if b is None else [b])
    ex = _GraphExecutor(len(args), [("linear", 0, 1, None if b is None else 2)])
    return d.Tensor._wrap(ex.execute([a._native for a in args]))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_strided_linear_full_jacobians(compiled, bias):
    rng = np.random.default_rng(64)
    x = d.tensor(rng.normal(size=(3, 2)).astype(np.float32), requires_grad=True)
    w = d.tensor(rng.normal(size=(3, 2)).astype(np.float32), requires_grad=True)
    b = d.tensor([.2, -.1], requires_grad=True)
    assert d.autograd.gradcheck(lambda a, c, z: execute(a.T, c.T, z if bias else None, compiled), (x, w, b))


@pytest.mark.parametrize("shape,weight_shape", [((0, 3), (4, 3)), ((2, 0), (4, 0)), ((2, 3), (0, 3))])
@pytest.mark.parametrize("compiled", [False, True])
def test_zero_dimensions_and_bias_gradients(shape, weight_shape, compiled):
    x = d.tensor(np.zeros(shape, dtype=np.float32), requires_grad=True)
    w = d.tensor(np.zeros(weight_shape, dtype=np.float32), requires_grad=True)
    b = d.tensor(np.ones(weight_shape[0], dtype=np.float32), requires_grad=True)
    out = execute(x, w, b, compiled)
    np.testing.assert_array_equal(out.numpy(), np.ones((shape[0], weight_shape[0]), dtype=np.float32))
    out.backward(d.tensor(np.ones(out.shape, dtype=np.float32)))
    assert x.grad.shape == x.shape and w.grad.shape == w.shape
    np.testing.assert_array_equal(b.grad.numpy(), np.full(weight_shape[0], shape[0], dtype=np.float32))


@pytest.mark.parametrize("compiled", [False, True])
def test_shared_input_weight_storage(compiled):
    x = d.tensor([[.2, .4], [.7, -.3]], requires_grad=True)
    assert d.autograd.gradcheck(lambda a: execute(a, a.T, None, compiled), x)
