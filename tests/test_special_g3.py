import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
TS = torch.special


def reference(f, values, *extra):
    """PyTorch in float64 (the accurate answer), rounded to float32."""
    args = [torch.tensor(np.asarray(v, np.float64)) for v in (values, *extra)]
    return f(*args).numpy().astype(np.float32)


GAMMA = np.array([-7.9506, -3.5, -2.25, -0.5, 1e-4, 0.1, 0.5, 1., 1.5, 2., 3.7, 10., 55.5, 170.], np.float32)
UNIT = np.array([-0.999, -0.9, -0.5, -1e-3, 0., 1e-6, 0.3, 0.75, 0.99, 0.999999], np.float32)
REAL = np.array([-40., -12., -5., -1.5, -1e-3, 0., 1e-3, 0.5, 2., 7.5, 15., 29.9, 30.1, 80.], np.float32)
POSITIVE = np.array([1e-3, 0.1, 0.5, 1., 2., 5., 9.9, 25., 60.], np.float32)

CASES = [
    ('lgamma', lambda x: d.lgamma(x), lambda x: torch.lgamma(x), GAMMA),
    ('digamma', lambda x: d.digamma(x), lambda x: torch.digamma(x), GAMMA[GAMMA != 170.]),
    ('trigamma', lambda x: d.polygamma(1, x), lambda x: torch.polygamma(1, x), GAMMA),
    ('polygamma3', lambda x: d.polygamma(3, x), lambda x: torch.polygamma(3, x), GAMMA[GAMMA > 0]),
    ('erfinv', lambda x: d.erfinv(x), lambda x: torch.erfinv(x), UNIT),
    ('ndtri', lambda x: d.special.ndtri(x), lambda x: TS.ndtri(x), (UNIT[UNIT > 0] + 1e-7).clip(1e-7, 0.999999)),
    ('ndtr', lambda x: d.special.ndtr(x), lambda x: TS.ndtr(x), REAL),
    ('log_ndtr', lambda x: d.special.log_ndtr(x), lambda x: TS.log_ndtr(x), REAL),
    ('erfcx', lambda x: d.special.erfcx(x), lambda x: TS.erfcx(x), REAL[REAL > -10]),
    ('entr', lambda x: d.special.entr(x), lambda x: TS.entr(x), POSITIVE),
    ('i0', lambda x: d.i0(x), lambda x: torch.i0(x), REAL[np.abs(REAL) < 80]),
    ('i0e', lambda x: d.special.i0e(x), lambda x: TS.i0e(x), REAL),
    ('i1', lambda x: d.special.i1(x), lambda x: TS.i1(x), REAL[np.abs(REAL) < 80]),
    ('i1e', lambda x: d.special.i1e(x), lambda x: TS.i1e(x), REAL),
    ('k0', lambda x: d.special.modified_bessel_k0(x), lambda x: TS.modified_bessel_k0(x), POSITIVE),
    ('k1', lambda x: d.special.modified_bessel_k1(x), lambda x: TS.modified_bessel_k1(x), POSITIVE),
    ('k0e', lambda x: d.special.scaled_modified_bessel_k0(x), lambda x: TS.scaled_modified_bessel_k0(x), POSITIVE),
    ('j0', lambda x: d.special.bessel_j0(x), lambda x: TS.bessel_j0(x), REAL),
    ('j1', lambda x: d.special.bessel_j1(x), lambda x: TS.bessel_j1(x), REAL),
    ('y0', lambda x: d.special.bessel_y0(x), lambda x: TS.bessel_y0(x), POSITIVE),
    ('y1', lambda x: d.special.bessel_y1(x), lambda x: TS.bessel_y1(x), POSITIVE),
    ('airy', lambda x: d.special.airy_ai(x), lambda x: TS.airy_ai(x), np.array([-30., -11., -9., -2., 0., 1.5, 4.9, 5.1, 12.], np.float32)),
    ('sph_j0', lambda x: d.special.spherical_bessel_j0(x), lambda x: TS.spherical_bessel_j0(x), REAL),
]


@pytest.mark.parametrize('name,ours,theirs,values', CASES)
def test_special_values_match_float64_torch(name, ours, theirs, values):
    out = ours(d.tensor(values))
    assert out.dtype == 'float32'
    np.testing.assert_allclose(out.numpy(), reference(theirs, values), rtol=2e-6, atol=1e-30, equal_nan=True)


@pytest.mark.parametrize('name,ours,theirs,values', [c for c in CASES if c[0] in (
    'lgamma', 'digamma', 'trigamma', 'erfinv', 'ndtri', 'ndtr', 'log_ndtr', 'erfcx', 'entr', 'i0', 'i0e', 'i1', 'i1e')])
def test_special_gradients_match_float64_torch(name, ours, theirs, values):
    x = d.tensor(values, requires_grad=True)
    tx = torch.tensor(values.astype(np.float64), requires_grad=True)
    ours(x).sum().backward()
    theirs(tx).sum().backward()
    want = tx.grad.numpy().astype(np.float32)
    finite = np.isfinite(want)
    np.testing.assert_allclose(x.grad.numpy()[finite], want[finite], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('name', ['chebyshev_polynomial_t', 'chebyshev_polynomial_u', 'chebyshev_polynomial_v',
                                  'chebyshev_polynomial_w', 'shifted_chebyshev_polynomial_t', 'shifted_chebyshev_polynomial_w',
                                  'hermite_polynomial_h', 'hermite_polynomial_he', 'laguerre_polynomial_l', 'legendre_polynomial_p'])
def test_polynomials_match_torch(name):
    x = np.linspace(-1.2, 1.2, 9).astype(np.float32)
    for n in range(-1, 10):
        np.testing.assert_allclose(getattr(d.special, name)(d.tensor(x), n).numpy(),
                                   reference(lambda a: getattr(TS, name)(a, n), x), rtol=1e-5, atol=1e-5)


def test_binary_special_functions_and_mvlgamma():
    a = np.array([0.5, 1., 2.5, 7.], np.float32)
    x = np.array([0.1, 1.5, 2.5, 20.], np.float32)
    for f, g in [(d.igamma, torch.igamma), (d.igammac, torch.igammac), (d.special.zeta, TS.zeta),
                 (d.special.xlog1py, TS.xlog1py)]:
        first = a + 1.5 if f is d.special.zeta else a
        np.testing.assert_allclose(f(d.tensor(first), d.tensor(x)).numpy(), reference(g, first, x), rtol=1e-5, atol=1e-7)
    xs, txs = d.tensor(x, requires_grad=True), torch.tensor(x.astype(np.float64), requires_grad=True)
    d.igamma(d.tensor(a), xs).sum().backward()
    torch.igamma(torch.tensor(a.astype(np.float64)), txs).sum().backward()
    np.testing.assert_allclose(xs.grad.numpy(), txs.grad.numpy(), rtol=1e-5)
    m = np.array([2., 3.5, 10.], np.float32)
    np.testing.assert_allclose(d.mvlgamma(d.tensor(m), 3).numpy(), reference(lambda t: torch.mvlgamma(t, 3), m), rtol=1e-6)
    with pytest.raises(RuntimeError):
        d.mvlgamma(d.tensor([0.5]), 3)
