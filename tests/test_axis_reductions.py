import numpy as np
import pytest
from densecore import tensor


@pytest.mark.parametrize("shape,axis,keepdims", [
    ((2, 3, 4), 0, False), ((2, 3, 4), -1, True),
    ((2, 3, 4), (0, 2), False), ((2, 3, 4), (2, 0), True),
    ((2, 3, 4), (), False), ((2, 3, 4), None, True),
    ((0, 3), 0, False), ((2, 0, 3), 1, True),
    ((2, 0, 3), 2, False), ((), None, True),
])
def test_axis_forward(shape, axis, keepdims):
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    t = tensor(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_allclose(t.sum(axis=axis, keepdims=keepdims).numpy(),
                                   np.sum(x, axis=axis, keepdims=keepdims), equal_nan=True)
        np.testing.assert_allclose(t.mean(axis=axis, keepdims=keepdims).numpy(),
                                   np.mean(x, axis=axis, keepdims=keepdims), equal_nan=True)


@pytest.mark.parametrize("axis,keepdims", [
    (0, False), (-1, False), ((0, 2), True), ((), False), (None, True),
])
@pytest.mark.parametrize("mean", [False, True])
def test_axis_backward(axis, keepdims, mean):
    shape = (2, 3, 4)
    x = np.arange(24, dtype=np.float32).reshape(shape)
    t = tensor(x, requires_grad=True)
    y = t.mean(axis=axis, keepdims=keepdims) if mean else t.sum(axis=axis, keepdims=keepdims)
    upstream = np.arange(y.numpy().size, dtype=np.float32).reshape(y.shape) + 0.5
    y.backward(tensor(upstream))
    actual = t.grad.numpy()
    axes = tuple(range(len(shape))) if axis is None else ((axis,) if isinstance(axis, int) else axis)
    axes = tuple(a % len(shape) for a in axes)
    expanded = upstream if keepdims else np.reshape(upstream, [1 if i in axes else shape[i] for i in range(len(shape))])
    expected = np.broadcast_to(expanded, shape).copy()
    if mean:
        count = int(np.prod([shape[i] for i in axes]))
        expected /= count
    np.testing.assert_allclose(actual, expected)


def test_axis_rejections():
    t = tensor(np.ones((2, 3), dtype=np.float32))
    for axis in (2, -3, (0, 0)):
        with pytest.raises(ValueError): t.sum(axis=axis)
    for axis in (True, 1.5, (0, False)):
        with pytest.raises(TypeError): t.mean(axis=axis)
    with pytest.raises(TypeError): t.sum(keepdims=1)
    with pytest.raises(ValueError): tensor(3.0).sum(axis=1)


def test_empty_reduction_gradients():
    t = tensor(np.empty((2, 0, 3), dtype=np.float32), requires_grad=True)
    y = t.sum(axis=1)
    np.testing.assert_array_equal(y.numpy(), np.zeros((2, 3), dtype=np.float32))
    y.backward(tensor(np.ones((2, 3), dtype=np.float32)))
    assert t.grad.shape == (2, 0, 3)
