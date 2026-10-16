import numpy as np
import pytest
import densecore as d


@pytest.mark.parametrize('kind', ['sub', 'div'])
def test_broadcast_strided_gradients(kind):
    a = d.tensor([[2., 3.], [4., 6.]], requires_grad=True)
    b = d.tensor([[2.], [3.]], requires_grad=True)
    fn = (lambda x, y: x.T - y) if kind == 'sub' else (lambda x, y: x.T / y)
    assert d.autograd.gradcheck(fn, (a, b))
    expected = a.numpy().T - b.numpy() if kind == 'sub' else a.numpy().T / b.numpy()
    np.testing.assert_allclose(fn(a, b).numpy(), expected)


@pytest.mark.parametrize('kind', ['sub', 'div'])
def test_reflected_scalars_numpy_and_empty(kind):
    x = d.tensor([2., 4.], requires_grad=True)
    op = np.subtract if kind == 'sub' else np.true_divide
    result = op(np.float32(8.), x)
    expected = [6., 4.] if kind == 'sub' else [4., 2.]
    np.testing.assert_allclose(result.numpy(), expected)
    result.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [-1., -1.] if kind == 'sub' else [-2., -.5])
    empty = d.tensor(np.empty((0, 2)), requires_grad=True, dtype="float32")
    out = empty - 2. if kind == 'sub' else empty / 2.
    out.sum().backward()
    assert empty.grad.shape == (0, 2)


def test_shared_parents_and_frozen_operand():
    x = d.tensor([2., 4.], requires_grad=True)
    ((x - x) + (x / x)).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [0., 0.])
    assert not (d.tensor(2.) / 2.).requires_grad


def test_invalid_dtype_shapes_and_numpy_arrays():
    x = d.tensor([1., 2.])
    for op in (lambda a, b: a - b, lambda a, b: a / b):
        with pytest.raises((TypeError, ValueError)):
            op(x, d.tensor([True, False], dtype='bool'))
        with pytest.raises(ValueError):
            op(x, d.tensor([1., 2., 3.]))
        with pytest.raises(TypeError):
            op(x, np.array([1., 2.]))


def test_division_ieee_forward_and_nograd():
    x = d.tensor([1., 0., -1.], requires_grad=True)
    with d.no_grad():
        result = x / 0.
    np.testing.assert_array_equal(np.isposinf(result.numpy()), [True, False, False])
    assert np.isnan(result.numpy()[1]) and np.isneginf(result.numpy()[2])
    assert not result.requires_grad


@pytest.mark.parametrize('kind', ['sub', 'div'])
def test_torch_parity_and_custom_owner(kind):
    torch = pytest.importorskip('torch')
    av = np.array([[2., 3., 5.], [4., 6., 8.]], dtype=np.float32)
    bv = np.array([[2.], [3.]], dtype=np.float32)
    a, b = d.tensor(av, requires_grad=True, dtype="float32"), d.tensor(bv, requires_grad=True, dtype="float32")
    ta, tb = torch.tensor(av, requires_grad=True), torch.tensor(bv, requires_grad=True)
    out, ref = (a - b, ta - tb) if kind == 'sub' else (a / b, ta / tb)
    out.sum().backward()
    ref.sum().backward()
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy())
    np.testing.assert_allclose(a.grad.numpy(), ta.grad.numpy(), rtol=1e-6)
    np.testing.assert_allclose(b.grad.numpy(), tb.grad.numpy(), rtol=1e-6)
    class Identity(d.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x
        @staticmethod
        def backward(ctx, g):
            return g
    x = d.tensor(2., requires_grad=True)
    y = Identity.apply(x)
    result = 4. - y if kind == 'sub' else 4. / y
    assert result._contexts
    result.backward()
    assert x.grad.numpy().item() == -1.


def test_division_large_denominator_gradient_avoids_square_overflow():
    x = d.tensor(1e20, requires_grad=True)
    y = d.tensor(1e20, requires_grad=True)
    (x / y).backward()
    np.testing.assert_allclose(x.grad.numpy(), np.float32(1e-20), rtol=1e-6, atol=0.)
    np.testing.assert_allclose(y.grad.numpy(), np.float32(-1e-20), rtol=1e-6, atol=0.)


def test_subtraction_division_parameter_version_checks():
    for op in (lambda x: x - 1., lambda x: 2. / x):
        p = d.nn.Parameter(2.)
        out = op(p)
        p.backward()
        d.optim.SGD([p], lr=.1).step()
        with pytest.raises(RuntimeError, match='modified'):
            out.backward()
