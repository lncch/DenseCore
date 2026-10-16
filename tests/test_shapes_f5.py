import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)
X = rng.normal(size=(3, 4, 5)).astype(np.float32)
SRC = rng.normal(size=(3, 4, 5)).astype(np.float32)


def as_dense(value):
    if isinstance(value, torch.Tensor):
        return d.tensor(value.numpy(), dtype=str(value.dtype).split('.')[1])
    return value


def check(f, values=X, extra=None):
    """f(x, lib, *extras) in both libraries; compare values and gradients of every input."""
    extras = [] if extra is None else [e.copy() for e in extra]
    x, tx = d.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)
    ex = [d.tensor(e, requires_grad=True) for e in extras]
    tex = [torch.tensor(e, requires_grad=True) for e in extras]
    out, ref = f(x, d, *ex), f(tx, torch, *tex)
    outs, refs = (out, ref) if isinstance(ref, (tuple, list)) else ((out,), (ref,))
    assert len(outs) == len(refs)
    for o, r in zip(outs, refs):
        assert o.shape == tuple(r.shape), (o.shape, tuple(r.shape))
        np.testing.assert_allclose(o.float().numpy(), r.float().detach().numpy(), rtol=1e-5, atol=1e-6)
    grads = [(o, r) for o, r in zip(outs, refs) if r.requires_grad]
    if not grads:
        return
    rng_state = np.random.default_rng(1)
    weights = [rng_state.normal(size=tuple(r.shape)).astype(np.float32) for _, r in grads]
    loss = sum((o * d.tensor(w)).sum() for (o, _), w in zip(grads, weights))
    tloss = sum((r * torch.from_numpy(w)).sum() for (_, r), w in zip(grads, weights))
    loss.backward()
    tloss.backward()
    for a, b in zip([x, *ex], [tx, *tex]):
        if b.grad is not None:
            np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=1e-5, atol=1e-6)


IDX = np.array([[2, 0], [1, 1]])
CASES = [
    lambda x, m: x[[0, 2], :, [1, 3]], lambda x, m: x[:, [1, 1, 3]], lambda x, m: x[x > 0],
    lambda x, m: x[0, ..., [2, 4]], lambda x, m: x[None, 1:, [0]], lambda x, m: x[[[0], [2]], [1, 2]],
    lambda x, m: x.flip(1), lambda x, m: x.flip((0, 2)), lambda x, m: x.roll(2, 1), lambda x, m: x.roll((1, -2), (0, 2)),
    lambda x, m: x.roll(3), lambda x, m: x.rot90(1, (1, 2)), lambda x, m: x.rot90(3, (0, 1)),
    lambda x, m: x.narrow(2, 1, 3), lambda x, m: x.select(1, -1), lambda x, m: x.movedim(0, -1),
    lambda x, m: x.movedim((0, 1), (2, 0)), lambda x, m: x.swapaxes(0, 2), lambda x, m: x.unflatten(2, (1, 5)),
    lambda x, m: x.chunk(2, 1), lambda x, m: x.tensor_split(3, 2), lambda x, m: x.tensor_split([1, 4], 2),
    lambda x, m: x.unbind(1), lambda x, m: x.index_select(1, m.tensor([3, 0, 3], dtype=m.int64)), lambda x, m: x.take(m.tensor([[0, 59], [7, 7]], dtype=m.int64)),
    lambda x, m: x.gather(2, m.tensor(np.broadcast_to(IDX[None, :, :1], (3, 2, 1)).copy(), dtype=m.int64)),
    lambda x, m: m.take_along_dim(x, m.tensor(np.argsort(X, 2)[:, :, :2].copy(), dtype=m.int64), 2),
    lambda x, m: x.repeat_interleave(2, 1), lambda x, m: x.repeat_interleave(m.tensor([1, 0, 2], dtype=m.int64), 0),
    lambda x, m: x.tile((2, 1)), lambda x, m: x.diagonal(0, 1, 2), lambda x, m: x.diagonal(1, 0, 2),
    lambda x, m: x[0].diag(), lambda x, m: x[0, 0].diag(1), lambda x, m: x[0].trace(), lambda x, m: x.triu(1),
    lambda x, m: x.tril(-1), lambda x, m: x.diag_embed(), lambda x, m: x.diff(dim=1), lambda x, m: x.diff(2, 2),
    lambda x, m: x[0, 0].outer(x[1, 1]), lambda x, m: x[0, 0].dot(x[1, 1]), lambda x, m: x[0].inner(x[1]),
    lambda x, m: x[0].mm(x[1].t()), lambda x, m: x[0].mv(x[1, 0]), lambda x, m: x[0, :, :4].addmm(x[1], x[2].t()[:, :4], beta=.5, alpha=2.),
    lambda x, m: x[:, :, :4].baddbmm(x, x.transpose(1, 2)[:, :, :4], beta=0.),
    lambda x, m: x[0, :2].kron(x[1, :1, :3]), lambda x, m: m.fill(x[0], 2.5), lambda x, m: m.zeros_like(x),
]


@pytest.mark.parametrize('f', CASES)
def test_shape_and_index_ops_match_torch(f):
    check(f)


SCATTERS = [
    lambda x, m, s: x.scatter(1, m.tensor(np.broadcast_to(IDX[None, :, :1], (3, 2, 1)).copy(), dtype=m.int64), s[:, :2, :1]),
    lambda x, m, s: x.scatter_add(2, m.tensor(np.array([[[0, 0, 4]]] * 3), dtype=m.int64), s[:, :1, :3]),
    lambda x, m, s: x.scatter_reduce(2, m.tensor(np.array([[[1, 1, 3]]] * 3), dtype=m.int64), s[:, :1, :3], 'sum'),
    lambda x, m, s: x.scatter_reduce(2, m.tensor(np.array([[[1, 1, 3]]] * 3), dtype=m.int64), s[:, :1, :3], 'mean', include_self=False),
    lambda x, m, s: x.index_add(1, m.tensor([0, 0, 2, 3], dtype=m.int64), s, alpha=2.),
    lambda x, m, s: x.index_copy(1, m.tensor([3, 0, 2, 1], dtype=m.int64), s),
    lambda x, m, s: x.index_fill(2, m.tensor([0, 4], dtype=m.int64), -1.),
    lambda x, m, s: x.index_put((m.tensor([0, 2, 0], dtype=m.int64), m.tensor([1, 3, 1], dtype=m.int64)), s[0, 0, :1], accumulate=True),
    lambda x, m, s: x.select_scatter(s[:, 0], 1, 2), lambda x, m, s: x.slice_scatter(s[:, :, 1:3], 2, 1, 3),
    lambda x, m, s: x.diagonal_scatter(s[:, :4, 0], 0, 1, 2),
]


@pytest.mark.parametrize('f', SCATTERS)
def test_scatters_match_torch(f):
    check(f, X, [SRC])


def test_setitem_forms_match_torch():
    for key, value in [((slice(None), [0, 2]), 7.), ((X > 0.5), -1.), ((0, ..., [1, 3]), 3.),
                       (([0, 2], [1, 1]), 0.5), ((1, slice(1, 3)), np.ones((2, 5), np.float32))]:
        t = torch.tensor(X).clone()
        ours = d.tensor(X)
        tv = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        t[key] = tv
        ours[key] = d.tensor(value) if isinstance(value, np.ndarray) else value
        np.testing.assert_array_equal(ours.numpy(), t.numpy())
    x, tx = d.tensor(X, requires_grad=True), torch.tensor(X, requires_grad=True)
    v, tv = d.tensor(np.ones(5, np.float32) * 2, requires_grad=True), torch.full((5,), 2., requires_grad=True)
    y, ty = x * 3, tx * 3
    y[1, 2] = v
    ty[1, 2] = tv
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), tx.grad.numpy())
    np.testing.assert_array_equal(v.grad.numpy(), tv.grad.numpy())


def test_constructors_and_module_functions():
    a, b = d.tensor([1., 2.]), d.tensor([3., 4., 5.])
    ta, tb = torch.tensor([1., 2.]), torch.tensor([3., 4., 5.])
    for ours, theirs in [(d.meshgrid(a, b), torch.meshgrid(ta, tb, indexing='ij')),
                         (d.meshgrid(a, b, indexing='xy'), torch.meshgrid(ta, tb, indexing='xy')),
                         ((d.cartesian_prod(a, b),), (torch.cartesian_prod(ta, tb),)),
                         ((d.block_diag(a.reshape(1, 2), b.reshape(3, 1)),), (torch.block_diag(ta.reshape(1, 2), tb.reshape(3, 1)),)),
                         ((d.hstack([a, b]),), (torch.hstack([ta, tb]),)), ((d.vstack([a, a]),), (torch.vstack([ta, ta]),)),
                         ((d.combinations(b, 2),), (torch.combinations(tb, 2),)),
                         (d.broadcast_tensors(a.reshape(2, 1), b), torch.broadcast_tensors(ta.reshape(2, 1), tb))]:
        for o, t in zip(ours, theirs):
            np.testing.assert_array_equal(o.numpy(), t.numpy())
    for steps in (1, 2, 5, 10):
        np.testing.assert_array_equal(d.linspace(-1.3, 7.7, steps).numpy(), torch.linspace(-1.3, 7.7, steps).numpy())
    np.testing.assert_array_equal(d.eye(3, 4).numpy(), torch.eye(3, 4).numpy())
    assert d.full((2,), 3).dtype == 'int64' and d.full((2,), 1.5).dtype == 'float32'
    assert d.broadcast_shapes((2, 1), (3,)) == (2, 3)
    z = d.ones(2, 3)
    assert z.zero_() is z and not z.numpy().any()
    np.testing.assert_array_equal(d.tensor([[0., 1.], [2., 0.]]).nonzero().numpy(), [[0, 1], [1, 0]])
    assert d.nonzero(d.tensor([0., 3.]), as_tuple=True)[0].tolist() == [1]
