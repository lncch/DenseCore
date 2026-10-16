import numpy as np
import pytest
import densecore as d


def test_bmm_strided_values_and_full_jacobians():
    a = d.tensor(np.arange(12, dtype=np.float32).reshape(2, 3, 2) / 10,
                 requires_grad=True, dtype="float32")
    b = d.tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10,
                 requires_grad=True, dtype="float32")
    out = a.bmm(b)
    np.testing.assert_allclose(out.numpy(), np.matmul(a.numpy(), b.numpy()), atol=1e-6)
    assert d.autograd.gradcheck(lambda x, y: d.bmm(x.permute(0, 2, 1), y),
                                (a.permute(0, 2, 1), b))


def test_bmm_shared_parent_gradients():
    a = d.tensor([[[.2, .3], [.4, .5]]], requires_grad=True)
    assert d.autograd.gradcheck(lambda x: d.bmm(x, x), a)
    assert d.autograd.gradcheck(lambda x: d.bmm(x, x.permute(0, 2, 1)), a)


def test_bmm_saved_parameter_version_and_snapshot():
    p = d.nn.Parameter([[[.2, .3], [.4, .5]]])
    view = p.permute(0, 2, 1)
    snapshot = view.numpy().copy()
    out = d.bmm(p, view).sum()
    out.backward(retain_graph=True)
    d.optim.SGD([p], lr=.1).step()
    np.testing.assert_array_equal(view.numpy(), snapshot)
    with pytest.raises(RuntimeError, match="modified"):
        out.backward()


def test_bmm_no_grad_and_frozen_operand():
    a = d.tensor([[[1.]]], requires_grad=True)
    b = d.tensor([[[2.]]])
    d.bmm(a, b).sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), [[[2.]]])
    assert b.grad is None
    with d.no_grad():
        assert not d.bmm(a, b).requires_grad


@pytest.mark.parametrize("shape", [(0, 2, 3, 4), (2, 0, 3, 4),
                                  (2, 3, 0, 4), (2, 3, 4, 0)])
def test_bmm_zero_dimensions_backward(shape):
    batch, m, k, n = shape
    a = d.tensor(np.zeros((batch, m, k), dtype=np.float32), requires_grad=True, dtype="float32")
    b = d.tensor(np.zeros((batch, k, n), dtype=np.float32), requires_grad=True, dtype="float32")
    out = d.bmm(a, b)
    np.testing.assert_array_equal(out.numpy(), np.zeros((batch, m, n)))
    out.sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), np.zeros(a.shape))
    np.testing.assert_array_equal(b.grad.numpy(), np.zeros(b.shape))


@pytest.mark.parametrize("a,b", [((2, 3), (3, 2)), ((2, 2, 3), (1, 3, 2)),
                                 ((2, 2, 3), (2, 4, 2))])
def test_bmm_rejects_incompatible_shapes(a, b):
    with pytest.raises(ValueError):
        d.bmm(d.tensor(np.zeros(a), dtype="float32"), d.tensor(np.zeros(b), dtype="float32"))


def test_bmm_rejects_non_tensor():
    with pytest.raises(TypeError):
        d.tensor([[[1.]]]).bmm(2)


def test_bmm_torch_vjp_strided():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    av, bv = rng.normal(size=(2, 3, 2)).astype("f"), rng.normal(size=(2, 3, 4)).astype("f")
    a, b = d.tensor(av, requires_grad=True, dtype="float32"), d.tensor(bv, requires_grad=True, dtype="float32")
    ta, tb = torch.tensor(av, requires_grad=True), torch.tensor(bv, requires_grad=True)
    out = d.bmm(a.permute(0, 2, 1), b)
    ref = torch.bmm(ta.transpose(1, 2), tb)
    seed = rng.normal(size=out.shape).astype("f")
    out.backward(d.tensor(seed, dtype="float32"))
    ref.backward(torch.tensor(seed))
    for actual, expected in [(out, ref), (a.grad, ta.grad), (b.grad, tb.grad)]:
        np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), rtol=1e-5, atol=1e-6)
