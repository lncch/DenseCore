import numpy as np
import pytest
import densecore as d
from densecore._native import _GraphExecutor


@pytest.mark.parametrize("compiled", [False, True])
def test_relu_strided_values_and_gradients_with_special_values(compiled):
    x = d.tensor([[-2., float("nan")], [3., -4.]], requires_grad=True)
    view = x.T
    if compiled:
        ex = _GraphExecutor(1, [("relu", 0, None, None)])
        y = d.Tensor._wrap(ex.execute([view._native]))
    else: y = view.relu()
    np.testing.assert_allclose(y.numpy(), [[0., 3.], [float("nan"), 0.]], equal_nan=True)
    y.backward(d.tensor([[1., 2.], [3., 4.]]))
    np.testing.assert_array_equal(x.grad.numpy(), [[0., 3.], [2., 0.]])


@pytest.mark.parametrize("logarithmic", [False, True])
def test_softmax_registered_gradcheck_and_empty_layout(logarithmic):
    x = d.tensor([[.2, .4], [.7, -.3], [.1, .6]], requires_grad=True)
    name = "log_softmax" if logarithmic else "softmax"
    assert d.autograd.gradcheck(lambda a: getattr(a.T, name)(axis=1), x)
    empty = d.tensor(np.empty((0, 3), dtype=np.float32), requires_grad=True)
    out = getattr(empty, name)(axis=1)
    out.backward(d.tensor(np.empty((0, 3), dtype=np.float32)))
    assert empty.grad.shape == empty.shape
