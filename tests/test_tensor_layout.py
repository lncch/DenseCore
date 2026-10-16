import numpy as np
import pytest
import densecore as d


def test_views_share_storage_and_report_layout():
    x = d.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    r = x.reshape(6, 4)
    p = x.permute(2, 0, 1)
    assert x.stride() == (12, 4, 1)
    assert p.stride() == (1, 12, 4)
    assert x.is_contiguous() and not p.is_contiguous()
    assert x.shares_storage(r) and x.shares_storage(p)
    assert x.numel() == p.numel() == 24
    assert p.storage_offset() == 0
    np.testing.assert_array_equal(p.numpy(), np.arange(24).reshape(2, 3, 4).transpose(2, 0, 1))
    # Reading logical values must not replace shared backing storage.
    assert x.shares_storage(p)
    del x
    np.testing.assert_array_equal(r.numpy(), np.arange(24).reshape(6, 4))


def test_noncontiguous_reshape_and_contiguous_materialize():
    x = d.tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
    t = x.T
    assert t.shares_storage(x)
    r = t.reshape(12)
    c = t.contiguous()
    assert not r.shares_storage(x) and not c.shares_storage(x)
    assert r.is_contiguous() and c.is_contiguous()
    assert c.contiguous().shares_storage(c)
    np.testing.assert_array_equal(r.numpy(), np.arange(12).reshape(3, 4).T.reshape(12))
    np.testing.assert_array_equal(c.numpy(), t.numpy())
    assert not x.shares_storage(d.tensor(x.numpy()))
    with pytest.raises(TypeError):
        x.shares_storage(None)


@pytest.mark.parametrize("shape,axes", [((), ()), ((0, 3), (1, 0)), ((2, 1, 3), (1, 0, 2))])
def test_scalar_empty_and_singleton_layouts(shape, axes):
    a = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    x = d.tensor(a)
    y = x.permute(axes)
    assert y.is_contiguous()
    assert x.shares_storage(y)
    np.testing.assert_array_equal(y.numpy(), a.transpose(axes))


def test_strided_operator_gradients_match_torch():
    torch = pytest.importorskip("torch")
    a = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    x = d.tensor(a, requires_grad=True)
    ref = torch.tensor(a, requires_grad=True)
    y = ((x.T * x.T).softmax(0).sum(axis=1) * d.tensor([1., 2., 3., 4.])).sum()
    target = ((ref.T * ref.T).softmax(0).sum(dim=1) * torch.tensor([1., 2., 3., 4.])).sum()
    y.backward()
    target.backward()
    np.testing.assert_allclose(y.numpy(), target.detach().numpy(), rtol=2e-6)
    np.testing.assert_allclose(x.grad.numpy(), ref.grad.numpy(), atol=2e-6)


def test_materialization_keeps_autograd_path():
    x = d.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
    y = x.T.contiguous().reshape(6)
    y.backward(d.tensor([1., 2., 3., 4., 5., 6.]))
    np.testing.assert_array_equal(x.grad.numpy(), np.array([1., 2., 3., 4., 5., 6.]).reshape(3, 2).T)


def test_shared_storage_does_not_merge_gradient_identity():
    x = d.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    (x + x.T).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.full((2, 2), 2.))


def test_view_is_snapshot_and_saved_version_is_checked():
    p = d.nn.Parameter([[1., 2.], [3., 4.]])
    view = p.T
    out = (view * view).sum()
    out.backward(retain_graph=True)
    d.optim.SGD([p], lr=.1).step()
    np.testing.assert_array_equal(view.numpy(), [[1., 3.], [2., 4.]])
    np.testing.assert_allclose(p.numpy(), [[.8, 1.6], [2.4, 3.2]], rtol=1e-6)
    assert not view.shares_storage(p)
    with pytest.raises(RuntimeError, match="modified"):
        out.backward()


def test_compiled_strided_input_forward_and_backward():
    from densecore._native import _GraphExecutor
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = d.tensor(a, requires_grad=True)
    v = x.T
    ex = _GraphExecutor(1, [("mul", 0, 0, None), ("sum", 1, None, None)])
    y = d.Tensor._wrap(ex.execute([v._native]))
    y.backward()
    np.testing.assert_allclose(y.numpy(), (a * a).sum())
    np.testing.assert_array_equal(x.grad.numpy(), 2 * a)


def test_random_permutations_and_finite_difference():
    import itertools
    rng = np.random.default_rng(37)
    for shape in [(2, 3, 4), (1, 4, 2), (0, 2, 3)]:
        a = rng.normal(size=shape).astype(np.float32)
        x = d.tensor(a)
        for axes in itertools.permutations(range(3)):
            y = x.permute(axes)
            np.testing.assert_array_equal(y.numpy(), a.transpose(axes))
            assert x.shares_storage(y)
    a = rng.normal(size=(2, 3)).astype(np.float32)
    x = d.tensor(a, requires_grad=True)
    (x.T * x.T).sum().backward()
    for index in np.ndindex(a.shape):
        plus, minus = a.copy(), a.copy()
        plus[index] += .001
        minus[index] -= .001
        f = lambda z: (d.tensor(z).T * d.tensor(z).T).sum().numpy().item()
        numerical = (f(plus) - f(minus)) / .002
        np.testing.assert_allclose(x.grad.numpy()[index], numerical, atol=.002)


@pytest.mark.parametrize("kind", ["SGD", "Adam"])
def test_optimizer_accepts_strided_gradients_with_decay(kind):
    torch = pytest.importorskip("torch")
    a = np.array([[1., -2., 3.], [4., 5., -6.]], dtype=np.float32)
    p = d.nn.Parameter(a)
    ref = torch.tensor(a, requires_grad=True)
    opt = getattr(d.optim, kind)([p], lr=.01, weight_decay=.03)
    control = getattr(torch.optim, kind)([ref], lr=.01, weight_decay=.03)
    seed = np.arange(6, dtype=np.float32).reshape(3, 2) + 1
    p.T.backward(d.tensor(seed))
    ref.T.backward(torch.tensor(seed))
    assert not p.grad.is_contiguous()
    np.testing.assert_array_equal(p.grad.numpy(), seed.T)
    opt.step()
    control.step()
    np.testing.assert_allclose(p.numpy(), ref.detach().numpy(), rtol=2e-6, atol=1e-6)
