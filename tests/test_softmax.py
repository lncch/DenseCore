import numpy as np
import pytest
from densecore import tensor


def oracle(x, axis):
    z = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=axis, keepdims=True)


@pytest.mark.parametrize("shape,axis", [
    ((2, 3), -1), ((2, 3, 4), 0), ((2, 3, 4), 1),
    ((2, 3, 4), -1), ((1, 5, 2), -2), ((0, 3), 1),
    ((4, 1), 1), ((2, 0, 3), 2),
])
def test_forward(shape, axis):
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 3
    y = tensor(x)
    expected = oracle(x, axis)
    np.testing.assert_allclose(y.softmax(axis).numpy(), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(y.log_softmax(axis).numpy(), np.log(expected), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("log", [False, True])
def test_vjp(axis, log):
    rng = np.random.default_rng(23)
    values = rng.normal(size=(2, 3, 4)).astype('float32')
    seed = rng.normal(size=values.shape).astype('float32')
    x = tensor(values, requires_grad=True)
    y = x.log_softmax(axis) if log else x.softmax(axis)
    y.backward(tensor(seed))
    p = oracle(values, axis)
    expected = seed - p * np.sum(seed, axis=axis, keepdims=True) if log else p * (
        seed - np.sum(seed * p, axis=axis, keepdims=True))
    np.testing.assert_allclose(x.grad.numpy(), expected, atol=2e-6, rtol=2e-6)


def test_numerical_stability():
    arr = np.array([[10000, 10001, 9999], [-10000, -9999, -10001]], dtype=np.float32)
    x = tensor(arr)
    s = x.softmax().numpy()
    ls = x.log_softmax().numpy()
    assert np.isfinite(s).all() and np.isfinite(ls).all()
    np.testing.assert_allclose(s.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.exp(ls), s, atol=1e-6)


@pytest.mark.parametrize("bad_axis", [2, -3, True, 1.5, '1'])
def test_invalid_axis(bad_axis):
    with pytest.raises((TypeError, ValueError)):
        tensor(np.zeros((2, 3), dtype=np.float32)).softmax(bad_axis)


def test_invalid_empty_reduction_axis():
    with pytest.raises(ValueError):
        tensor(np.empty((2, 0), dtype=np.float32)).softmax(1)


def test_scalar_not_supported():
    with pytest.raises(ValueError):
        tensor(1.).softmax()
