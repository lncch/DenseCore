import numpy as np
import pytest
import densecore as d


@pytest.mark.parametrize("name", ["neg", "exp", "log", "sqrt", "sin", "cos"])
def test_registered_unary_values_and_full_gradients(name):
    a = np.arange(12, dtype=np.float32).reshape(3, 4) / 10 + .2
    x = d.tensor(a, requires_grad=True)
    fn = lambda z: -z if name == "neg" else getattr(z, name)()
    expected = -a.T[::2] if name == "neg" else getattr(np, name)(a.T[::2])
    y = fn(x.T[::2])
    np.testing.assert_allclose(y.numpy(), expected, rtol=2e-6, atol=1e-7)
    assert y.dtype == "float32" and y.device == "cpu"
    assert d.autograd.gradcheck(lambda z: fn(z.T[::2]), x)
    assert y.is_contiguous()
    assert not y.shares_storage(x)


@pytest.mark.parametrize("name", ["neg", "exp", "log", "sqrt", "sin", "cos"])
def test_unary_scalar_empty_and_special_values(name):
    values = np.array([0., -1., float("inf"), float("-inf"), float("nan")], dtype=np.float32)
    fn = lambda z: -z if name == "neg" else getattr(z, name)()
    with np.errstate(all="ignore"):
        want = -values if name == "neg" else getattr(np, name)(values)
    np.testing.assert_allclose(fn(d.tensor(values)).numpy(), want, rtol=2e-6, atol=1e-7, equal_nan=True)
    assert fn(d.tensor(.5)).shape == ()
    assert fn(d.tensor(np.empty((0, 3), dtype=np.float32))).shape == (0, 3)


def test_native_dtype_device_and_unknown_operator():
    x = d.tensor([1.])
    assert x._native.dtype == "float32" and x._native.device == "cpu"
    with pytest.raises(ValueError, match="unregistered unary"):
        x._native.unary_math("unknown")
    with pytest.raises(ValueError, match="device"):
        d.tensor([1.], device="cuda")
    with pytest.raises(TypeError, match="dtype"):
        d.tensor([1.], dtype="complex32")


@pytest.mark.parametrize("name", ["neg", "exp", "log", "sqrt", "sin", "cos"])
def test_registered_unary_vjp_matches_torch(name):
    torch = pytest.importorskip("torch")
    a = np.arange(12, dtype=np.float32).reshape(3, 4) / 10 + .2
    x = d.tensor(a, requires_grad=True)
    ref = torch.tensor(a, requires_grad=True)
    view, control = x.T[1:, ::2], ref.T[1:, ::2]
    got = -view if name == "neg" else getattr(view, name)()
    want = -control if name == "neg" else getattr(control, name)()
    seed = np.arange(6, dtype=np.float32).reshape(3, 2) / 7 - .2
    got.backward(d.tensor(seed))
    want.backward(torch.tensor(seed))
    np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(x.grad.numpy(), ref.grad.numpy(), rtol=2e-6, atol=1e-7)


def test_sqrt_backward_preserves_resolvable_subnormal_seed():
    x = d.tensor(2.0 ** -40, requires_grad=True)
    x.sqrt().backward(d.tensor(np.float32(2.0 ** -149)))
    np.testing.assert_array_equal(x.grad.numpy(), np.float32(2.0 ** -130))
