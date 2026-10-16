import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)
A = rng.normal(size=(3, 4)).astype(np.float32)
B = rng.normal(size=(4, 5)).astype(np.float32)


def check(f, *arrays, rtol=1e-5):
    ours = [d.tensor(a, requires_grad=True) for a in arrays]
    theirs = [torch.tensor(a, requires_grad=True) for a in arrays]
    o, t = f(d, *ours), f(torch, *theirs)
    o, t = (o[0], t[0]) if isinstance(t, tuple) else (o, t)
    np.testing.assert_allclose(o.numpy(), t.detach().numpy(), rtol=rtol, atol=1e-5)
    if t.requires_grad:
        w = rng.normal(size=tuple(t.shape)).astype(np.float32)
        (o * d.tensor(w)).sum().backward()
        (t * torch.from_numpy(w)).sum().backward()
        for a, b in zip(ours, theirs):
            if b.grad is not None:
                np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('f,arrays', [
    (lambda m, a: a.unfold(1, 2, 1), (A,)), (lambda m, a: a.unfold(0, 2, 2), (A,)),
    (lambda m, a, b: m.tensordot(a, b, 1), (A, B)), (lambda m, a, b: m.tensordot(a, b, ([1], [0])), (A, B)),
    (lambda m, a, b: m.einsum('ij,jk->ik', a, b), (A, B)), (lambda m, a, b: m.einsum('ij,jk->k', a, b), (A, B)),
    (lambda m, a: m.einsum('ii->i', a[:, :3]), (A,)), (lambda m, a, b: m.einsum('ij,kj->ik', a, b.T), (A, B)),
    (lambda m, a, b: m.cdist(a, b.T[:2, :]), (A, B)), (lambda m, a: m.cdist(a, a * 2, p=1), (A,)),
    (lambda m, a: a[:, :3].cross(a[:, 1:], dim=1), (A,)), (lambda m, a: a.cov(), (A,)),
    (lambda m, a: a.corrcoef(), (A,)), (lambda m, a: m.trapezoid(a, dim=1), (A,)),
    (lambda m, a: m.trapezoid(a, dx=0.5, dim=0), (A,)), (lambda m, a: m.cumulative_trapezoid(a, dim=1), (A,)),
    (lambda m, a: a.renorm(2, 0, 1.0), (A,)), (lambda m, a: a.ldexp(m.tensor([1., -2., 3., 0.])), (A,)),
    (lambda m, a: a.frexp(), (A,)), (lambda m, a: a.angle(), (A,)),
    (lambda m, a: a.quantile(0.3), (A,)), (lambda m, a: a.quantile(m.tensor([0.25, 0.9]), dim=1, keepdim=True), (A,)),
    (lambda m, a: a.quantile(0.5, 0, interpolation='midpoint'), (A,)), (lambda m, a: a.quantile(0.4, 1, interpolation='nearest'), (A,)),
    (lambda m, a: a.norm(0, 1), (A,)), (lambda m, a: a.sum(0).sum(0).sum(0), (A,))])
def test_extra_ops_match_torch(f, arrays):
    check(f, *arrays)


@pytest.mark.filterwarnings("ignore:index_reduce\\(\\) is in beta")  # PyTorch's own beta notice
def test_put_index_reduce_and_nanquantile():
    x = np.zeros(6, np.float32)
    idx = np.array([1, 4, 1])
    ours = d.tensor(x).put(d.tensor(idx, dtype='int64'), d.tensor([1., 2., 3.]), accumulate=True)
    theirs = torch.zeros(6).put(torch.from_numpy(idx), torch.tensor([1., 2., 3.]), accumulate=True)
    np.testing.assert_array_equal(ours.numpy(), theirs.numpy())
    src = rng.normal(size=(3, 4)).astype(np.float32)
    check(lambda m, a, s: a.index_reduce(0, m.tensor([0, 2, 0], dtype=m.int64), s, 'mean'), A, src)
    v = A.copy()
    v[0, 1] = np.nan
    np.testing.assert_allclose(d.tensor(v).nanquantile(0.5, 1).numpy(), torch.tensor(v).nanquantile(0.5, 1).numpy(), rtol=1e-6)
    assert np.isnan(d.tensor(v).quantile(0.5, 1).numpy()[0])


def test_searching_sets_and_histograms_match_torch():
    seq = np.array([1., 3., 3., 7.], np.float32)
    vals = np.array([[0., 3., 8.], [3., 5., 7.]], np.float32)
    for right in (False, True):
        np.testing.assert_array_equal(d.searchsorted(d.tensor(seq), d.tensor(vals), right=right).numpy(),
                                      torch.searchsorted(torch.from_numpy(seq), torch.from_numpy(vals), right=right).numpy())
        np.testing.assert_array_equal(d.tensor(vals).bucketize(d.tensor(seq), right=right).numpy(),
                                      torch.bucketize(torch.from_numpy(vals), torch.from_numpy(seq), right=right).numpy())
    np.testing.assert_array_equal(d.isin(d.tensor(vals), d.tensor(seq)).numpy(), torch.isin(torch.from_numpy(vals), torch.from_numpy(seq)).numpy())
    x = np.array([3., 1., 3., 2., 2., 2.], np.float32)
    for ours, theirs in zip(d.tensor(x).unique(return_inverse=True, return_counts=True),
                            torch.from_numpy(x).unique(return_inverse=True, return_counts=True)):
        np.testing.assert_array_equal(ours.numpy(), theirs.numpy())
    for ours, theirs in zip(d.tensor(x).unique_consecutive(return_inverse=True, return_counts=True),
                            torch.from_numpy(x).unique_consecutive(return_inverse=True, return_counts=True)):
        np.testing.assert_array_equal(ours.numpy(), theirs.numpy())
    np.testing.assert_array_equal(d.tensor(x).histc(4, 0, 4).numpy(), torch.from_numpy(x).histc(4, 0, 4).numpy())


def test_random_generators_are_seeded_and_shaped():
    d.manual_seed(7)
    first = [d.randn(2, 3), d.rand(4), d.randperm(5), d.normal(d.zeros(3), 2.), d.randn_like(d.ones(2, 2))]
    d.manual_seed(7)
    second = [d.randn(2, 3), d.rand(4), d.randperm(5), d.normal(d.zeros(3), 2.), d.randn_like(d.ones(2, 2))]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    assert sorted(first[2].tolist()) == list(range(5)) and first[2].dtype == 'int64'
    t = d.zeros(1000)
    t.uniform_(2., 3.)
    assert 2 <= t.numpy().min() and t.numpy().max() < 3
    assert d.ones(50).bernoulli(0.0).numpy().sum() == 0
    with pytest.raises(RuntimeError):
        d.nn.Parameter(np.zeros(2, np.float32)).normal_()


def test_logical_ops_scalar_dims_and_real_identities():
    a, b = np.array([0., 2., -1.], np.float32), np.array([0., 0., 3.], np.float32)
    for name in ('logical_and', 'logical_or', 'logical_xor'):
        np.testing.assert_array_equal(getattr(d, name)(d.tensor(a), d.tensor(b)).numpy(),
                                      getattr(torch, name)(torch.from_numpy(a), torch.from_numpy(b)).numpy())
    np.testing.assert_array_equal(d.tensor(a).logical_not().numpy(), torch.from_numpy(a).logical_not().numpy())
    with pytest.raises(TypeError):
        d.tensor([True], dtype='bool') & d.tensor([1.])
    assert d.tensor(3.).sum(0).item() == 3. and d.tensor(3.).mean(-1).item() == 3.
    x = d.tensor(A)
    assert x.conj() is x and x.resolve_neg() is x and x.real is x  # real is a property, as in PyTorch
