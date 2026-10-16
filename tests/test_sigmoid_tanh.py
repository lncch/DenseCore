import numpy as np
import pytest
import densecore as d


@pytest.mark.parametrize('name', ['sigmoid', 'tanh'])
def test_activation_strided_full_jacobian(name):
    x = d.tensor([[-1.2, -.3], [.2, 1.1]], requires_grad=True)
    assert d.autograd.gradcheck(lambda a: getattr(a.T, name)(), x)
    out = getattr(x.T, name)()
    ref = 1 / (1 + np.exp(-x.numpy().T)) if name == 'sigmoid' else np.tanh(x.numpy().T)
    np.testing.assert_allclose(out.numpy(), ref, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize('name', ['sigmoid', 'tanh'])
def test_activation_saturation_nonfinite_and_torch_gradients(name):
    torch = pytest.importorskip('torch')
    values = np.array([-np.inf, -100., -20., -2., -0., 0., 2., 20., 100., np.inf, np.nan], np.float32)
    x = d.tensor(values, requires_grad=True, dtype="float32")
    tx = torch.tensor(values, requires_grad=True)
    out, ref = getattr(x, name)(), getattr(tx, name)()
    out.sum().backward()
    ref.sum().backward()
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), atol=1e-7, rtol=2e-6, equal_nan=True)
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), atol=1e-7, rtol=2e-6, equal_nan=True)


@pytest.mark.parametrize('name', ['sigmoid', 'tanh'])
def test_activation_empty_scalar_nograd_and_bool_rejection(name):
    x = d.tensor(np.empty((0, 3)), requires_grad=True, dtype="float32")
    getattr(x, name)().sum().backward()
    assert x.grad.shape == x.shape
    with d.no_grad():
        out = getattr(d.tensor(0., requires_grad=True), name)()
    assert not out.requires_grad and out.shape == ()
    assert out.numpy().item() == (.5 if name == 'sigmoid' else 0.)
    with pytest.raises(TypeError, match='float32'):
        getattr(d.tensor(True, dtype='bool'), name)()


@pytest.mark.parametrize('name', ['sigmoid', 'tanh'])
def test_activation_shared_input_custom_owner_and_versions(name):
    class Identity(d.autograd.Function):
        @staticmethod
        def forward(ctx, x): return x
        @staticmethod
        def backward(ctx, g): return g
    x = d.tensor(.2, requires_grad=True)
    y = getattr(Identity.apply(x), name)()
    assert y._contexts
    (y + y).backward(retain_graph=True)
    first = x.grad.numpy().copy()
    y.backward()
    np.testing.assert_allclose(x.grad.numpy(), first * 1.5)
    p = d.nn.Parameter(.2)
    out = getattr(p, name)()
    p.backward()
    d.optim.SGD([p], lr=.1).step()
    with pytest.raises(RuntimeError, match='modified'):
        out.backward()


def test_sigmoid_extreme_inputs_without_optional_torch():
    # Repointed 2026-10-16 (roadmap: PyTorch's bits are the spec; approved by the owner).
    # PyTorch computes 1 / (1 + exp(90)) = 1 / inf = 0 here; the old test wanted the true
    # subnormal 8.2e-40 and a positive gradient.
    x = d.tensor([-np.inf, -90., 0., 90., np.inf, np.nan], requires_grad=True)
    result = x.sigmoid()
    values = result.numpy()
    np.testing.assert_array_equal(values[:5], [0., 0., .5, 1., 1.])
    assert np.isnan(values[5])
    result.sum().backward()
    gradient = x.grad.numpy()
    np.testing.assert_array_equal(gradient[:5], [0., 0., .25, 0., 0.])
    assert np.isnan(gradient[5])


def test_tanh_signed_zero_and_nonfinite_without_optional_torch():
    x = d.tensor([-np.inf, -0., 0., np.inf, np.nan], requires_grad=True)
    result = x.tanh()
    values = result.numpy()
    np.testing.assert_array_equal(values[:4], [-1., -0., 0., 1.])
    assert np.signbit(values[1]) and not np.signbit(values[2])
    assert np.isnan(values[4])
    result.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy()[:4], [0., 1., 1., 0.])
    assert np.isnan(x.grad.numpy()[4])
