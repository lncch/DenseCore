import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)
X = rng.normal(size=(3, 5)).astype(np.float32)
MASK = rng.random((3, 5)) > 0.3
MASK[1] = False  # a fully masked row


@pytest.mark.parametrize('name,kwargs', [('hann', {}), ('hamming', {}), ('blackman', {}), ('nuttall', {}),
                                         ('bartlett', {}), ('cosine', {}), ('exponential', {'tau': 2.}),
                                         ('gaussian', {'std': 1.5}), ('kaiser', {'beta': 8.}),
                                         ('general_hamming', {'alpha': 0.6}), ('general_cosine', {'a': [0.4, 0.4, 0.2]})])
@pytest.mark.parametrize('M', [0, 1, 2, 7, 10])
@pytest.mark.parametrize('sym', [True, False])
def test_windows_match_torch(name, kwargs, M, sym):
    ours = getattr(d.signal.windows, name)(M, sym=sym, **kwargs)
    theirs = getattr(torch.signal.windows, name)(M, sym=sym, **kwargs)
    assert ours.shape == tuple(theirs.shape) and ours.dtype == 'float32'
    np.testing.assert_allclose(ours.numpy(), theirs.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('f', [
    lambda m, x, k: m.masked.sum(x, 1, mask=k), lambda m, x, k: m.masked.prod(x, 0, mask=k),
    lambda m, x, k: m.masked.mean(x, 0, mask=k), lambda m, x, k: m.masked.amax(x, 0, mask=k),
    lambda m, x, k: m.masked.amin(x, (0, 1), mask=k), lambda m, x, k: m.masked.cumsum(x, 1, mask=k),
    lambda m, x, k: m.masked.cumprod(x, 1, mask=k), lambda m, x, k: m.masked.softmax(x, 0, mask=k),
    lambda m, x, k: m.masked.log_softmax(x, 0, mask=k), lambda m, x, k: m.masked.softmin(x, 0, mask=k),
    lambda m, x, k: m.masked.logsumexp(x, 0, mask=k), lambda m, x, k: m.masked.norm(x, 2.0, 0, mask=k),
    lambda m, x, k: m.masked.var(x, 0, mask=k), lambda m, x, k: m.masked.std(x, 0, unbiased=False, mask=k),
    lambda m, x, k: m.masked.normalize(x, 2.0, 0, mask=k), lambda m, x, k: m.masked.argmax(x, 0, mask=k)])
def test_masked_reductions_match_torch(f):
    x, tx = d.tensor(X, requires_grad=True), torch.tensor(X, requires_grad=True)
    k, tk = d.tensor(MASK), torch.tensor(MASK)
    out, ref = f(d, x, k), f(torch, tx, tk)
    assert out.dtype == str(ref.dtype).split('.')[1]
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-5, atol=1e-6, equal_nan=True)
    if ref.requires_grad:
        finite = np.isfinite(ref.detach().numpy())
        w = np.where(finite, rng.normal(size=ref.shape), 0).astype(np.float32)
        (out * d.tensor(w)).sum().backward()
        (ref * torch.from_numpy(w)).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-4, atol=1e-5, equal_nan=True)


def test_masked_select_scatter_and_fill_with_tensor_value():
    x, tx = d.tensor(X, requires_grad=True), torch.tensor(X, requires_grad=True)
    out, ref = x.masked_select(d.tensor(MASK)), tx.masked_select(torch.tensor(MASK))
    np.testing.assert_array_equal(out.numpy(), ref.detach().numpy())
    out.sum().backward(); ref.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), tx.grad.numpy())
    src = np.arange(15, dtype=np.float32)
    np.testing.assert_array_equal(d.tensor(X).masked_scatter(d.tensor(MASK), d.tensor(src)).numpy(),
                                  torch.tensor(X).masked_scatter(torch.tensor(MASK), torch.from_numpy(src)).numpy())
    v, tv = d.tensor(2.5, requires_grad=True), torch.tensor(2.5, requires_grad=True)
    y, ty = d.tensor(X, requires_grad=True), torch.tensor(X, requires_grad=True)
    filled, tfilled = y.masked_fill(d.tensor(MASK), v), ty.masked_fill(torch.tensor(MASK), tv)
    np.testing.assert_array_equal(filled.numpy(), tfilled.detach().numpy())
    filled.sum().backward(); tfilled.sum().backward()
    assert v.grad.item() == tv.grad.item()
    np.testing.assert_array_equal(y.grad.numpy(), ty.grad.numpy())
