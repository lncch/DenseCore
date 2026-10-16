import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)

LEFT = np.array([[-3.5, -1., -0., 0., 0.5, 2., 2., np.inf, np.nan, 7.25]], np.float32)
RIGHT = np.array([[2., -1., 0., -0., 0.5, 3., 2., 1., 1., -2.]], np.float32).T  # broadcast to 10x10
POSITIVE = np.abs(LEFT[np.isfinite(LEFT)]).reshape(1, -1) + np.float32(0.25)
OPS = ['maximum', 'minimum', 'fmax', 'fmin', 'atan2', 'hypot', 'fmod', 'remainder', 'copysign',
       'logaddexp', 'logaddexp2', 'xlogy', 'pow']


@pytest.mark.parametrize('name', OPS)
def test_binary_math_values_and_gradients_match_torch(name):
    left = POSITIVE if name in ('pow', 'xlogy') else LEFT
    right = np.abs(RIGHT) + np.float32(0.5) if name in ('xlogy', 'fmod', 'remainder') else RIGHT
    a, ta = d.tensor(left, requires_grad=True), torch.tensor(left, requires_grad=True)
    b, tb = d.tensor(right, requires_grad=True), torch.tensor(right, requires_grad=True)
    out, ref = getattr(a, name)(b), getattr(torch, name)(ta, tb)
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=2e-6, atol=1e-6, equal_nan=True)
    w = rng.normal(size=ref.shape).astype(np.float32)
    finite = np.isfinite(ref.detach().numpy()) & np.isfinite(np.broadcast_to(left, ref.shape)) & np.isfinite(np.broadcast_to(right, ref.shape))
    w = np.where(finite, w, 0).astype(np.float32)
    (out * d.tensor(w)).sum().backward()
    (ref * torch.from_numpy(w)).sum().backward()
    for got, want in ((a.grad, ta.grad), (b.grad, tb.grad)):
        np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-4, atol=1e-5, equal_nan=True)


def test_operators_scalars_and_dtype_promotion():
    x, tx = d.tensor([1., 2., 3.], requires_grad=True), torch.tensor([1., 2., 3.], requires_grad=True)
    out, ref = x ** 2 + 2 ** x + x.pow(0.5), tx ** 2 + 2 ** tx + tx.pow(0.5)
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-6)
    out.sum().backward(); ref.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-5)
    ints, tints = d.arange(1, 6), torch.arange(1, 6)
    for f in [lambda t: t ** 2, lambda t: t.maximum(t * 0 + 3),
              lambda t: t.remainder(3), lambda t: t.pow(0.5), lambda t: t.fmod(-2)]:
        ours, theirs = f(ints), f(tints)
        assert ours.dtype == str(theirs.dtype).split('.')[1]
        np.testing.assert_allclose(ours.float().numpy(), theirs.float().numpy(), rtol=1e-6)
    with pytest.raises(TypeError):
        d.tensor([True], dtype='bool').maximum(d.tensor([False], dtype='bool'))


def test_where_lerp_addc_and_comparisons():
    cond = np.array([[True, False, True], [False, False, True]])
    x, tx = d.tensor(rng.normal(size=(2, 3)).astype(np.float32), requires_grad=True), None
    tx = torch.tensor(x.numpy(), requires_grad=True)
    y, ty = d.tensor(np.array([-0., 5., np.inf], np.float32), requires_grad=True), torch.tensor([-0., 5., np.inf], requires_grad=True)
    out, ref = d.where(d.tensor(cond, dtype='bool'), x, y), torch.where(torch.from_numpy(cond), tx, ty)
    np.testing.assert_array_equal(out.numpy().view(np.uint32), ref.detach().numpy().view(np.uint32))
    out.sum().backward(); ref.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), tx.grad.numpy())
    np.testing.assert_array_equal(y.grad.numpy(), ty.grad.numpy())
    assert d.where(cond, 1., 0.).numpy().tolist() == torch.where(torch.from_numpy(cond), 1., 0.).tolist()
    s, e = d.tensor([0., 10.]), torch.tensor([0., 10.])
    for w in (0.25, 0.75):
        np.testing.assert_array_equal(d.lerp(s, d.tensor([4., -2.]), w).numpy(), torch.lerp(e, torch.tensor([4., -2.]), w).numpy())
    wt = d.tensor([0.1, 0.9])
    np.testing.assert_allclose(s.lerp(d.tensor([4., -2.]), wt).numpy(), e.lerp(torch.tensor([4., -2.]), torch.tensor([0.1, 0.9])).numpy())
    a, b, c = (rng.normal(size=4).astype(np.float32) for _ in range(3))
    np.testing.assert_allclose(d.addcmul(d.tensor(a), d.tensor(b), d.tensor(c), value=0.5).numpy(),
                               torch.addcmul(*(torch.from_numpy(v) for v in (a, b, c)), value=0.5).numpy(), rtol=1e-6)
    np.testing.assert_allclose(d.addcdiv(d.tensor(a), d.tensor(b), d.tensor(c), value=2.).numpy(),
                               torch.addcdiv(*(torch.from_numpy(v) for v in (a, b, c)), value=2.).numpy(), rtol=1e-6)
    p, q = d.tensor([1., 2., np.nan]), d.tensor([1. + 1e-7, 2.1, np.nan])
    assert d.isclose(p, q).numpy().tolist() == [True, False, False]
    assert d.isclose(p, q, equal_nan=True).numpy().tolist()[2] is True
    assert not d.allclose(p, q) and d.equal(d.ones(2), d.ones(2)) and not d.equal(d.ones(2), d.ones(3))


def test_clamp_with_tensor_bounds_matches_torch():
    v = np.array([-3., -1., 0.5, 1., 4.], np.float32)
    lo, hi = np.array([-1., -1., 0., 0., 0.], np.float32), np.array([1., 1., 1., 1., 2.], np.float32)
    x, tx = d.tensor(v, requires_grad=True), torch.tensor(v, requires_grad=True)
    out, ref = x.clamp(d.tensor(lo), d.tensor(hi)), tx.clamp(torch.from_numpy(lo), torch.from_numpy(hi))
    np.testing.assert_array_equal(out.numpy(), ref.detach().numpy())
    out.sum().backward(); ref.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), tx.grad.numpy())
