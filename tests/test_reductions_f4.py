import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)
TIES = np.array([[3., 1., 3., -2.], [0., 0., 0., 0.], [5., -1., 2., 5.]], np.float32)


def pair(values):
    return d.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)


def first(result):
    return result[0] if isinstance(result, tuple) else result


def check(f, values, rtol=1e-5, grad=True):
    x, tx = pair(values)
    out, ref = f(x, d), f(tx, torch)
    if isinstance(ref, tuple):
        for a, b in zip(out, ref):
            np.testing.assert_allclose(a.float().numpy(), b.float().detach().numpy(), rtol=rtol, atol=1e-6, equal_nan=True)
    else:
        np.testing.assert_allclose(out.float().numpy(), ref.float().detach().numpy(), rtol=rtol, atol=1e-6, equal_nan=True)
    if grad:
        o, r = first(out), first(ref)
        w = rng.normal(size=tuple(r.shape)).astype(np.float32)
        (o * d.tensor(w)).sum().backward()
        (r * torch.from_numpy(w)).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-4, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize('f', [
    lambda x, m: x.amax(), lambda x, m: x.amax(1), lambda x, m: x.amin((0, 1), keepdim=True),
    lambda x, m: x.max(), lambda x, m: x.max(1), lambda x, m: x.min(0, keepdim=True),
    lambda x, m: m.max(x, 1), lambda x, m: x.sort(1), lambda x, m: x.sort(0, descending=True),
    lambda x, m: x.median(), lambda x, m: x.median(1),
    lambda x, m: x.prod(), lambda x, m: x.prod(1), lambda x, m: x.cumsum(1), lambda x, m: x.cumprod(1),
    lambda x, m: x.logcumsumexp(0), lambda x, m: x.cummax(1), lambda x, m: x.cummin(0),
    lambda x, m: x.var(), lambda x, m: x.var(1, correction=0), lambda x, m: x.std(0, keepdim=True),
    lambda x, m: x.norm(), lambda x, m: x.norm(1, 1), lambda x, m: x.norm(float('inf'), 0),
    lambda x, m: x.norm(3, 1), lambda x, m: x.logsumexp(1), lambda x, m: x.nansum(0),
    lambda x, m: x.nanmean(1), lambda x, m: m.dist(x, x * 0.5)])
def test_reductions_match_torch_with_ties(f):
    check(f, TIES)


def test_index_outputs_and_boolean_reductions():
    x, tx = d.tensor(TIES), torch.tensor(TIES)
    for f in [lambda t: t.argmax(), lambda t: t.argmax(1), lambda t: t.argmin(0, keepdim=True),
              lambda t: t.max(1).indices, lambda t: t.sort(1).indices, lambda t: t.argsort(1, descending=True),
              lambda t: t.cummax(1).indices, lambda t: t.count_nonzero(1)]:
        assert f(x).dtype == 'int64'
        np.testing.assert_array_equal(f(x).numpy(), f(tx).numpy())
    for name in ('all', 'any'):
        for kwargs in ({}, {'dim': 1}, {'dim': 0, 'keepdim': True}):
            np.testing.assert_array_equal(getattr(x, name)(**kwargs).numpy(), getattr(tx, name)(**kwargs).numpy())


def test_kthvalue_without_ties():
    # PyTorch leaves the index of tied values unspecified (kthvalue and median disagree), so
    # indices are compared on tie-free data.
    values = rng.permutation(24).astype(np.float32).reshape(4, 6)
    check(lambda x, m: x.kthvalue(3, 1), values)
    check(lambda x, m: x.kthvalue(2, 0, keepdim=True), values)


def test_nan_handling_matches_torch():
    v = np.array([[1., np.nan, 3.], [4., 5., 6.], [np.nan, np.nan, 0.]], np.float32)
    x, tx = d.tensor(v), torch.tensor(v)
    for f in [lambda t: t.amax(1), lambda t: t.max(1).values, lambda t: t.max(1).indices, lambda t: t.argmax(1),
              lambda t: t.sort(1).values, lambda t: t.sort(1, descending=True).indices, lambda t: t.nansum(1),
              lambda t: t.nanmedian(1).values, lambda t: t.median(1).values, lambda t: t.nanmean(0)]:
        np.testing.assert_array_equal(f(x).numpy(), f(tx).numpy())


def test_zero_handling_and_scalars():
    check(lambda x, m: x.prod(1), np.array([[2., 0., 3.], [0., 0., 4.], [1., 2., 3.]], np.float32))
    check(lambda x, m: x.norm(), np.zeros((2, 2), np.float32))
    check(lambda x, m: x.logsumexp(1), np.array([[-np.inf, -np.inf], [0., 1.]], np.float32))
    s, ts = d.tensor(2.5), torch.tensor(2.5)
    for f in [lambda t: t.amax(0), lambda t: t.var(0, correction=0), lambda t: t.logsumexp(0), lambda t: t.norm(dim=-1)]:
        np.testing.assert_allclose(f(s).numpy(), f(ts).numpy())
    with pytest.raises(RuntimeError):
        d.ones(0, 3).amax(0)
    assert d.arange(1, 5).prod().item() == 24 and d.arange(1, 5).prod().dtype == 'int64'
    assert d.arange(4).cumsum(0).tolist() == [0, 1, 3, 6]
