import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')

# Domain-safe inputs per op, with special values where PyTorch defines them.
GENERAL = np.array([-3.7, -1.5, -0.5, -0., 0., 0.25, 0.5, 1.5, 2.5, 7.9, np.inf, -np.inf, np.nan], np.float32)
UNIT = np.array([-0.99, -0.5, -0., 0., 0.3, 0.75, 0.99], np.float32)
POSITIVE = np.array([1e-6, 0.1, 0.5, 1., 2., 37.5, 1e6, np.inf], np.float32)
CASES = {'abs': GENERAL, 'sign': GENERAL, 'floor': GENERAL, 'ceil': GENERAL, 'trunc': GENERAL,
         'round': np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.49, -0.], np.float32), 'frac': GENERAL[:-3],
         'reciprocal': POSITIVE, 'rsqrt': POSITIVE, 'square': GENERAL, 'exp2': GENERAL, 'expm1': GENERAL,
         'log1p': POSITIVE, 'log2': POSITIVE, 'log10': POSITIVE, 'tan': UNIT, 'asin': UNIT, 'acos': UNIT,
         'atan': GENERAL, 'sinh': UNIT * 5, 'cosh': UNIT * 5, 'asinh': GENERAL, 'acosh': POSITIVE + 1,
         'atanh': UNIT, 'deg2rad': GENERAL, 'rad2deg': GENERAL, 'erfc': GENERAL,
         'erf': np.array([-4., -0.7, -1e-4, 1e-8, 0., 0.3, 0.49, 0.51, 2., 6., np.inf], np.float32),
         'logit': np.array([0.01, 0.2, 0.5, 0.8, 0.99], np.float32), 'sinc': np.array([-2.5, -1., 0., 1e-3, 0.5, 3.25], np.float32)}
NO_GRAD = {'sign', 'floor', 'ceil', 'trunc', 'round'}


@pytest.mark.parametrize('name', CASES)
def test_unary_values_and_gradients_match_torch(name):
    values = CASES[name]
    # Reference: PyTorch in float64 (the accurate answer), rounded to float32. float32 PyTorch
    # can be less accurate (its sinc gradient loses ~1% to cancellation near 0).
    x, tx = d.tensor(values, requires_grad=True), torch.tensor(values.astype(np.float64), requires_grad=True)
    out, ref = getattr(x, name)(), getattr(torch, name)(tx)
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy().astype(np.float32), rtol=2e-6, atol=1e-7, equal_nan=True)
    finite = np.isfinite(values)
    if not finite.any():
        return
    keep = d.tensor(finite.astype(np.float32))
    (out * keep).sum().backward()
    (ref * torch.from_numpy(finite.astype(np.float64))).sum().backward()
    if name in NO_GRAD:
        assert not x.grad.numpy().any()
    got, want = x.grad.numpy()[finite], tx.grad.numpy()[finite]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize('dtype', ['int64', 'int8', 'float64', 'float16', 'bfloat16'])
@pytest.mark.parametrize('name', ['abs', 'sign', 'square', 'floor', 'exp', 'sqrt', 'sin', 'log1p', 'tanh', 'sigmoid'])
def test_unary_on_other_dtypes_follows_torch_promotion(name, dtype):
    values = np.array([0, 1, 2, 3, 9], np.float64)
    ours = getattr(d.tensor(values, dtype=dtype), name)()
    theirs = getattr(torch.tensor(values).to(getattr(torch, dtype)), name)()
    assert ours.dtype == str(theirs.dtype).split('.')[1]
    np.testing.assert_allclose(ours.float().numpy(), theirs.float().numpy(), rtol=1e-6)


def test_clamp_nan_to_num_and_predicates():
    values = np.array([-3., -1., 0.5, 1., 4., np.nan, np.inf, -np.inf], np.float32)
    x, tx = d.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)
    for f in [lambda t: t.clamp(-1., 1.), lambda t: t.clamp(min=0.), lambda t: t.clamp_max(2.),
              lambda t: t.nan_to_num(), lambda t: t.nan_to_num(nan=7., posinf=1., neginf=-1.)]:
        x.zero_grad(); tx.grad = None
        out, ref = f(x), f(tx)
        np.testing.assert_array_equal(out.numpy(), ref.detach().numpy())
        out.sum().backward(); ref.sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), tx.grad.numpy())
    for name in ('isnan', 'isinf', 'isfinite', 'isposinf', 'isneginf', 'signbit'):
        np.testing.assert_array_equal(getattr(d, name)(x).numpy(), getattr(torch, name)(tx).numpy())
    assert d.arange(5).clamp(1, 3).tolist() == [1, 1, 2, 3, 3]
    with pytest.raises(TypeError, match='float32'):
        d.tensor([True], dtype='bool').abs()


def test_round_decimals_logit_eps_and_rounding_div_keep_graphs():
    values = np.array([-1234.5678, -2.555, 0.125, 0.5, 2.5, 987.654], np.float32)
    x, tx = d.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)
    for decimals in (0, 2, -2):
        np.testing.assert_array_equal(x.round(decimals=decimals).numpy(), tx.round(decimals=decimals).detach().numpy())
    out = d.div(x, 3., rounding_mode='floor') + x.round(decimals=1)
    out.sum().backward()
    assert not x.grad.numpy().any() and out.requires_grad
    p = np.array([0.0, 0.05, 0.5, 0.97, 1.0], np.float32)
    y, ty = d.tensor(p, requires_grad=True), torch.tensor(p, requires_grad=True)
    np.testing.assert_allclose(y.logit(eps=0.1).numpy(), ty.logit(eps=0.1).detach().numpy(), rtol=1e-6)
    y.logit(eps=0.1).sum().backward()
    ty.logit(eps=0.1).sum().backward()
    interior = (p > 0.1) & (p < 0.9)
    np.testing.assert_allclose(y.grad.numpy()[interior], ty.grad.numpy()[interior], rtol=1e-5)
    assert not y.grad.numpy()[~interior].any()
