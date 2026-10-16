import numpy as np
import pytest
from densecore import tensor


@pytest.mark.parametrize("shape,axes", [
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4), (-1, 0, 1)),
    ((0, 3, 4), (2, 0, 1)),
    ((2, 0, 4), (2, 1, 0)),
    ((2, 3), (1, 0)),
    ((), ()),
])
def test_permutation_forward_backward(shape, axes):
    arr = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    x = tensor(arr, requires_grad=True)
    y = x.permute(*axes)
    np.testing.assert_array_equal(y.numpy(), arr.transpose(axes))
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones_like(arr))


def test_transpose_axes():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = tensor(arr, requires_grad=True)
    y = x.transpose(0, -1)
    np.testing.assert_array_equal(y.numpy(), np.swapaxes(arr, 0, -1))
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones_like(arr))


def test_reshape_inference_and_backward():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = tensor(arr, requires_grad=True)
    y = x.reshape(2, -1)
    assert y.shape == (2, 12)
    z = y.reshape((-1, 2, 3))
    assert z.shape == (4, 2, 3)
    z.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones_like(arr))


@pytest.mark.parametrize("shape", [(0, -1), (-1, 0), (-1, -1), (7, -1), (-2,), (2.5,)])
def test_invalid_reshape(shape):
    x = tensor(np.arange(6, dtype=np.float32))
    with pytest.raises((TypeError, ValueError)):
        x.reshape(*shape)


@pytest.mark.parametrize("axes", [(0, 0, 2), (0, 1), (0, 1, 4), (True, 1, 2)])
def test_invalid_permute(axes):
    with pytest.raises((TypeError, ValueError)):
        tensor(np.zeros((2, 3, 4), dtype=np.float32)).permute(*axes)


def test_invalid_transpose():
    x = tensor(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(TypeError):
        x.transpose(0)
    with pytest.raises(ValueError):
        x.transpose(0, 3)
