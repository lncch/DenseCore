import numpy as np
import pytest
import densecore as d


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_cat_strided_full_jacobian(dim):
    a = d.tensor([[.1, .2], [.3, .4]], requires_grad=True)
    b = d.tensor([[.5, .6], [.7, .8]], requires_grad=True)
    out = d.cat([a.T, b], dim=dim)
    np.testing.assert_array_equal(out.numpy(), np.concatenate([a.numpy().T, b.numpy()], axis=dim))
    assert d.autograd.gradcheck(lambda x, y: d.cat([x.T, y], dim=dim), (a, b))


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -3])
def test_stack_shared_parent_and_axis(dim):
    a = d.tensor([[.1, .2], [.3, .4]], requires_grad=True)
    np.testing.assert_array_equal(d.stack([a, a], dim).numpy(), np.stack([a.numpy(), a.numpy()], axis=dim))
    assert d.autograd.gradcheck(lambda x: d.stack([x, x.T], dim), a)


def test_stack_scalars_and_cat_empty_backward():
    a = d.tensor(.4, requires_grad=True)
    d.stack([a, a], -1).sum().backward()
    assert a.grad.numpy().item() == 2
    x = d.tensor(np.empty((2, 0), dtype="f"), requires_grad=True)
    y = d.tensor(np.ones((2, 3), dtype="f"), requires_grad=True)
    d.cat([x, y, x], dim=1).sum().backward()
    assert x.grad.shape == (2, 0)
    np.testing.assert_array_equal(y.grad.numpy(), np.ones((2, 3)))


@pytest.mark.parametrize("fn", ["cat", "stack"])
def test_empty_output_and_no_grad(fn):
    x = d.tensor(np.empty((0, 2), dtype="f"), requires_grad=True)
    op = getattr(d, fn)
    op([x, x]).sum().backward()
    assert x.grad.shape == x.shape
    with d.no_grad():
        assert not op([x]).requires_grad


@pytest.mark.parametrize("fn", ["cat", "stack"])
def test_input_and_axis_errors(fn):
    op = getattr(d, fn)
    a = d.tensor([[1., 2.]])
    for tensors in [[], [a, 2], a]:
        with pytest.raises((ValueError, TypeError)):
            op(tensors)
    for dim in [True, .5, "0"]:
        with pytest.raises(TypeError):
            op([a], dim=dim)
    for dim in [-100, 100]:
        with pytest.raises(ValueError):
            op([a], dim=dim)
    with pytest.raises(ValueError):
        op([a, d.tensor([[1., 2., 3.]])], dim=0)


def test_cat_scalar_rejected():
    with pytest.raises(ValueError):
        d.cat([d.tensor(1.)])


def test_cat_stale_parameter_version():
    p = d.nn.Parameter([[.2, .3], [.4, .5]])
    loss = d.cat([p, p.T], dim=1).sum()
    loss.backward(retain_graph=True)
    d.optim.SGD([p], lr=.1).step()
    with pytest.raises(RuntimeError, match="modified"):
        loss.backward()


def test_cat_stack_torch_seeded_vjp():
    torch = pytest.importorskip("torch")
    for fn in ["cat", "stack"]:
        a = d.tensor([[.1, .2], [.3, .4]], requires_grad=True)
        ta = torch.tensor(a.numpy(), requires_grad=True)
        out = getattr(d, fn)([a, a.T], dim=-1)
        ref = getattr(torch, fn)([ta, ta.T], dim=-1)
        seed = np.arange(out.numel(), dtype="f").reshape(out.shape) / 10
        out.backward(d.tensor(seed))
        ref.backward(torch.tensor(seed))
        np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(a.grad.numpy(), ta.grad.numpy(), atol=1e-6)
