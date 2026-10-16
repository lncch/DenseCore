"""Special functions (torch.special and friends), computed in float64 NumPy.

Differentiable ones attach their local derivative through the native `pointwise` node, so
gradients are exact products grad * slope. Functions PyTorch does not differentiate (Bessel
J/Y/K, Airy, orthogonal polynomials, ...) return plain tensors.

Methods: Lanczos (lgamma), recurrence + asymptotic series (digamma, trigamma as PyTorch),
Euler-Maclaurin (Hurwitz zeta, polygamma), Newton-refined inverses (erfinv, ndtri), power
and asymptotic series (I0, I1, Airy), exponentially convergent quadrature (K0, K1, J, Y),
series and continued fractions (regularized incomplete gamma).
"""
import math
import types

import numpy as np

from .tensor import Tensor
from ._native import _pointwise

_erf = np.vectorize(math.erf, otypes=[np.float64])
_erfc = np.vectorize(math.erfc, otypes=[np.float64])
SQRT2, SQRTPI, SQRT2PI = math.sqrt(2), math.sqrt(math.pi), math.sqrt(2 * math.pi)


def _values(x):
    return x._array().astype(np.float64) if isinstance(x, Tensor) else np.asarray(x, np.float64)


def _result(inputs, value, slopes, dtype="float32"):
    """A float32 result; float32 inputs that need gradients get grad * slope."""
    value = np.asarray(value, np.float64)
    tracked = [(t, s) for t, s in zip(inputs, slopes) if isinstance(t, Tensor) and t.requires_grad and s is not None]
    if dtype != "float32" or not tracked:
        return Tensor(value.astype(np.float32) if dtype == "float32" else value, dtype=dtype)
    shape = value.shape
    tensors = [t.expand(*shape) if t.shape != shape else t for t, _ in tracked]
    data = np.ascontiguousarray(value, "<f4").tobytes()
    slope_bytes = [np.ascontiguousarray(np.broadcast_to(s, shape), "<f4").tobytes() for _, s in tracked]
    native = _pointwise([t._native for t in tensors], data, list(shape), slope_bytes)
    return Tensor._wrap(native, parents=tuple(tensors))


def _out_dtype(*inputs):
    dtypes = [t.dtype for t in inputs if isinstance(t, Tensor)]
    return "float64" if "float64" in dtypes else "float32"


# ---- gamma family ------------------------------------------------------------------------

def _lgamma_scalar(x):
    try:
        return math.lgamma(x)
    except (ValueError, OverflowError):
        return math.inf  # poles at 0, -1, -2, ...


_lgamma_libm = np.vectorize(_lgamma_scalar, otypes=[np.float64])


def _lgamma(x):
    """libm's lgamma (exact zeros at 1 and 2), elementwise."""
    x = np.asarray(x, np.float64)
    return np.where(np.isnan(x), np.nan, _lgamma_libm(np.where(np.isnan(x), 1.0, x)))


def _digamma(x):
    x = np.asarray(x, np.float64).copy()
    out = np.zeros_like(x)
    with np.errstate(all="ignore"):
        pole = (x <= 0) & (x == np.floor(x))
        negative = (x < 0) & ~pole
        # Reflection for negatives: psi(x) = psi(1 - x) - pi / tan(pi x).
        reflection = np.where(negative, -np.pi / np.tan(np.pi * x), 0.0)
        x = np.where(negative, 1 - x, x)
        while True:
            low = (x < 6) & np.isfinite(x)
            if not low.any():
                break
            out -= np.where(low, 1 / np.where(low, x, 1), 0.0)
            x = np.where(low, x + 1, x)
        inv = 1 / x
        inv2 = inv * inv
        series = np.log(x) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))))
        out = out + series + reflection
    out = np.where(pole, np.where(np.asarray(x) == 0, -np.inf, np.nan), out)
    return out


def _trigamma(x):
    """PyTorch's calc_trigamma, in float64."""
    x = np.asarray(x, np.float64).copy()
    sign = np.ones_like(x)
    result = np.zeros_like(x)
    with np.errstate(all="ignore"):
        small = x < 0.5
        sign = np.where(small, -1.0, 1.0)
        sin_pi_x = np.sin(np.pi * x)
        result = np.where(small, -(np.pi * np.pi) / (sin_pi_x * sin_pi_x), 0.0)
        x = np.where(small, 1 - x, x)
        for _ in range(6):
            result = result + 1 / (x * x)
            x = x + 1
        ixx = 1 / (x * x)
        result = result + (1 + 1 / (2 * x) + ixx * (1 / 6 - ixx * (1 / 30 - ixx * (1 / 42)))) / x
    return sign * result


_ZETA_A = [12.0, -720.0, 30240.0, -1209600.0, 47900160.0, -1.8924375803183791606e9, 7.47242496e10,
           -2.950130727918164224e12, 1.1646782814350067249e14, -4.5979787224074726105e15,
           1.8152105401943546773e17, -7.1661652561756670113e18]


def _zeta(s, q):
    """Hurwitz zeta (Cephes' Euler-Maclaurin algorithm), elementwise in float64."""
    s, q = np.broadcast_arrays(np.asarray(s, np.float64), np.asarray(q, np.float64))
    out = np.empty(s.shape)
    for index in np.ndindex(s.shape):
        out[index] = _zeta_scalar(float(s[index]), float(q[index]))
    return out


def _zeta_scalar(x, q):
    if x == 1:
        return math.inf
    if x < 1:
        return math.nan
    if q <= 0:
        if q == math.floor(q):
            return math.inf
        if x != math.floor(x):
            return math.nan
    total = q ** -x
    a, i, b = q, 0, 0.0
    while i < 9 or a <= 9:
        i += 1
        a += 1
        b = a ** -x
        total += b
        if abs(b) < abs(total) * 1e-17:
            return total
    w = a
    total += b * w / (x - 1) - 0.5 * b
    a_term, k = 1.0, 0.0
    for j in range(12):
        a_term *= x + k
        b /= w
        t = a_term * b / _ZETA_A[j]
        total += t
        if abs(t) < abs(total) * 1e-17:
            break
        k += 1
        a_term *= x + k
        b /= w
        k += 1
    return total


def _polygamma(n, x):
    n = int(n)
    if n == 0:
        return _digamma(x)
    if n == 1:
        return _trigamma(x)
    return (-1) ** (n + 1) * math.factorial(n) * _zeta(n + 1, x)


def lgamma(input):
    x = _values(input)
    return _result([input], _lgamma(x), [_digamma(x)], _out_dtype(input))


def digamma(input):
    x = _values(input)
    return _result([input], _digamma(x), [_trigamma(x)], _out_dtype(input))


def polygamma(n, input):
    x = _values(input)
    return _result([input], _polygamma(n, x), [_polygamma(n + 1, x)], _out_dtype(input))


def mvlgamma(input, p):
    x = _values(input)
    if np.any(x <= (p - 1) / 2):
        raise RuntimeError(f"all elements must be greater than (p-1)/2 = {(p - 1) / 2}")
    value = p * (p - 1) / 4 * math.log(math.pi) + sum(_lgamma(x - i / 2) for i in range(p))
    slope = sum(_digamma(x - i / 2) for i in range(p))
    return _result([input], value, [slope], _out_dtype(input))


def zeta(input, other):
    s, q = _values(input), _values(other)
    s_b, q_b = np.broadcast_arrays(s, q)
    value = _zeta(s_b, q_b)
    slope_q = -s_b * _zeta(s_b + 1, q_b)  # PyTorch differentiates with respect to q only
    return _result([input, other], value, [None, slope_q], _out_dtype(input, other))


def _igamma_series(a, x):
    total = term = 1.0 / a
    n = a
    for _ in range(500):
        n += 1
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-16:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _igammac_fraction(a, x):
    # Lentz's continued fraction for Q(a, x).
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b if b != 0 else 1 / tiny
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _igamma_pair(a, x):
    a, x = np.broadcast_arrays(np.asarray(a, np.float64), np.asarray(x, np.float64))
    p = np.empty(a.shape)
    for index in np.ndindex(a.shape):
        ai, xi = float(a[index]), float(x[index])
        if np.isnan(ai) or np.isnan(xi) or ai < 0 or xi < 0:
            p[index] = math.nan
        elif ai == 0:
            p[index] = 1.0 if xi > 0 else math.nan
        elif xi == 0:
            p[index] = 0.0
        elif math.isinf(xi):
            p[index] = 1.0
        elif math.isinf(ai):
            p[index] = 0.0
        elif xi < ai + 1:
            p[index] = _igamma_series(ai, xi)
        else:
            p[index] = 1.0 - _igammac_fraction(ai, xi)
    return a, x, p


def _igamma_slope(a, x):
    with np.errstate(all="ignore"):
        return np.exp(-x + (a - 1) * np.log(x) - _lgamma(a))


def igamma(input, other):
    a, x, p = _igamma_pair(_values(input), _values(other))
    return _result([input, other], p, [None, _igamma_slope(a, x)], _out_dtype(input, other))


def igammac(input, other):
    a, x, p = _igamma_pair(_values(input), _values(other))
    return _result([input, other], 1.0 - p, [None, -_igamma_slope(a, x)], _out_dtype(input, other))


# ---- error function family ---------------------------------------------------------------

def _erfinv(y):
    y = np.asarray(y, np.float64)
    with np.errstate(all="ignore"):
        w = -np.log((1 - y) * (1 + y))
        central = w < 5
        wc = w - 2.5
        pc = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503,
                  -0.00417768164, 0.246640727, 1.50140941):
            pc = c + pc * wc
        wt = np.sqrt(w) - 3
        pt = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613,
                  0.00943887047, 1.00167406, 2.83297682):
            pt = c + pt * wt
        x = np.where(central, pc, pt) * y
        # Newton steps on erf (erfc near |y| = 1 for precision).
        for _ in range(3):
            near_one = np.abs(y) > 0.5
            residual = np.where(near_one, -(_erfc(np.abs(x)) - (1 - np.abs(y))) * np.sign(y), _erf(x) - y)
            x = x - residual / (2 / SQRTPI * np.exp(-x * x))
    x = np.where(y == 1, np.inf, np.where(y == -1, -np.inf, x))
    return np.where(np.abs(y) > 1, np.nan, x)


def erfinv(input):
    y = _values(input)
    x = _erfinv(y)
    with np.errstate(all="ignore"):
        slope = SQRTPI / 2 * np.exp(x * x)
    return _result([input], x, [slope], _out_dtype(input))


def ndtr(input):
    x = _values(input)
    return _result([input], 0.5 * _erfc(-x / SQRT2), [np.exp(-x * x / 2) / SQRT2PI], _out_dtype(input))


def ndtri(input):
    p = _values(input)
    x = SQRT2 * _erfinv(2 * p - 1)
    with np.errstate(all="ignore"):
        # Refine against ndtr directly (keeps tail precision).
        for _ in range(2):
            finite = np.isfinite(x)
            f = 0.5 * _erfc(-np.where(finite, x, 0) / SQRT2) - p
            x = np.where(finite, x - f / (np.exp(-x * x / 2) / SQRT2PI), x)
        slope = SQRT2PI * np.exp(x * x / 2)
    return _result([input], x, [slope], _out_dtype(input))


def log_ndtr(input):
    x = _values(input)
    with np.errstate(all="ignore"):
        tail = x < -20
        value = np.log(0.5 * _erfc(-x / SQRT2))
        # Asymptotic tail: log(phi(x)/|x|) + log(1 - 1/x^2 + 3/x^4 - 15/x^6).
        x2 = x * x
        asym = -x2 / 2 - np.log(-x) - 0.5 * math.log(2 * math.pi) + np.log1p(-1 / x2 + 3 / x2 ** 2 - 15 / x2 ** 3)
        value = np.where(tail, asym, value)
        upper = x > 5
        value = np.where(upper, np.log1p(-0.5 * _erfc(x / SQRT2)), value)
        slope = np.exp(-x2 / 2 - value) / SQRT2PI
    return _result([input], value, [slope], _out_dtype(input))


def _erfcx(x):
    x = np.asarray(x, np.float64)
    with np.errstate(all="ignore"):
        direct = np.exp(x * x) * _erfc(x)
        big = x > 25
        inv2 = 1 / (2 * x * x)
        asym = 1 / (x * SQRTPI) * (1 - inv2 + 3 * inv2 ** 2 - 15 * inv2 ** 3 + 105 * inv2 ** 4)
        return np.where(big, asym, direct)


def erfcx(input):
    x = _values(input)
    value = _erfcx(x)
    return _result([input], value, [2 * x * value - 2 / SQRTPI], _out_dtype(input))


def entr(input):
    x = _values(input)
    with np.errstate(all="ignore"):
        value = np.where(x > 0, -x * np.log(x), np.where(x == 0, 0.0, -np.inf))
        value = np.where(np.isnan(x), np.nan, value)
        slope = np.where(x > 0, -(np.log(x) + 1), np.where(x == 0, -np.inf, np.nan))
    return _result([input], value, [slope], _out_dtype(input))


def xlog1py(input, other):
    x, y = np.broadcast_arrays(_values(input), _values(other))
    with np.errstate(all="ignore"):
        value = np.where((x == 0) & ~np.isnan(y), 0.0, x * np.log1p(y))
        dx = np.where(x != 0, np.log1p(y), np.where(np.isnan(y), np.nan, 0.0))
        dy = x / (1 + y)
    return _result([input, other], value, [dx, dy], _out_dtype(input, other))


# ---- Bessel functions --------------------------------------------------------------------

def _bessel_i(nu, x):
    """I_0 or I_1 (unscaled) by power series below 30 and the asymptotic series above."""
    x = np.asarray(x, np.float64)
    ax = np.abs(x)
    out = np.empty_like(ax)
    small = ax < 30
    xs = ax[small]
    term = (xs / 2) ** nu / math.factorial(nu)
    total = term.copy()
    for k in range(1, 120):
        term = term * (xs * xs / 4) / (k * (k + nu))
        total += term
    out[small] = total
    out[~small] = np.exp(ax[~small]) * _bessel_i_scaled_asym(nu, ax[~small])
    return np.where(x < 0, out * (-1) ** nu, out)


def _bessel_i_scaled_asym(nu, ax):
    mu = 4.0 * nu * nu
    term = np.ones_like(ax)
    total = term.copy()
    for k in range(1, 30):
        term = -term * (mu - (2 * k - 1) ** 2) / (k * 8 * ax)
        total += term
    return total / np.sqrt(2 * np.pi * ax)


def _bessel_i_scaled(nu, x):
    x = np.asarray(x, np.float64)
    ax = np.abs(x)
    with np.errstate(all="ignore"):
        value = np.where(ax < 30, _bessel_i(nu, ax) * np.exp(-ax), _bessel_i_scaled_asym(nu, np.where(ax < 30, 30, ax)))
    return np.where(x < 0, value * (-1) ** nu, value)


def _bessel_k(nu, x, scaled=False):
    """K_nu(x) = integral_0^inf exp(-x cosh t) cosh(nu t) dt by trapezoid (exponentially accurate)."""
    x = np.asarray(x, np.float64)
    out = np.full(x.shape, np.nan)
    positive = x > 0
    xs = x[positive][:, None]
    if xs.size:
        top = np.arccosh(np.maximum(750 / np.maximum(xs.min(), 1e-300), 2.0))
        t = np.linspace(0, top, int(max(200, 40 * top)))[None, :]
        h = t[0, 1] - t[0, 0]
        with np.errstate(all="ignore"):
            integrand = np.exp(-xs * (np.cosh(t) - (1 if scaled else 0))) * np.cosh(nu * t)
        weights = np.full(t.shape[1], h)
        weights[0] = h / 2
        out[positive] = integrand @ weights
    out = np.where(x == 0, np.inf, out)
    return out


def _bessel_jy(nu, x, kind):
    """J_nu (kind 'j') or Y_nu (kind 'y') for nu in {0, 1} via Gauss-Legendre quadrature of
    Bessel's integral (and its decaying tail for Y)."""
    x = np.asarray(x, np.float64)
    flat = x.ravel()
    out = np.empty_like(flat)
    for i, value in enumerate(flat):
        out[i] = _bessel_jy_scalar(nu, float(value), kind)
    return out.reshape(x.shape)


def _bessel_jy_scalar(nu, x, kind):
    if math.isnan(x):
        return math.nan
    if kind == "j":
        if math.isinf(x):
            return 0.0
        if x == 0:
            return 1.0 if nu == 0 else 0.0  # J1 is odd: exactly 0 at the origin
        sign = -1.0 if (x < 0 and nu == 1) else 1.0
        x = abs(x)
        n = int(max(64, 2 * x + 64))
        nodes, weights = np.polynomial.legendre.leggauss(n)
        tau = (nodes + 1) * np.pi / 2
        return sign * float(np.dot(weights, np.cos(nu * tau - x * np.sin(tau))) / 2)
    if x < 0:
        return math.nan
    if x == 0:
        return -math.inf
    if math.isinf(x):
        return 0.0
    n = int(max(64, 2 * x + 64))
    nodes, weights = np.polynomial.legendre.leggauss(n)
    tau = (nodes + 1) * np.pi / 2
    first = float(np.dot(weights, np.sin(x * np.sin(tau) - nu * tau)) / 2)
    top = math.asinh(max(750 / x, 1.0)) + 1
    tail_nodes, tail_weights = np.polynomial.legendre.leggauss(400)
    t = (tail_nodes + 1) * top / 2
    integrand = (np.exp(nu * t) + (-1) ** nu * np.exp(-nu * t)) * np.exp(-x * np.sinh(t))
    second = float(np.dot(tail_weights, integrand) * top / 2 / np.pi)
    return first - second


def _airy_ai(x):
    x = np.asarray(x, np.float64)
    out = np.empty_like(x)
    c1, c2 = 0.355028053887817239, 0.258819403792806798
    series = (x >= -10) & (x <= 5)
    xs = x[series]
    f, g = np.ones_like(xs), xs.copy()
    tf, tg = np.ones_like(xs), xs.copy()
    for k in range(1, 80):
        tf = tf * xs ** 3 / ((3 * k - 1) * (3 * k))
        tg = tg * xs ** 3 / ((3 * k) * (3 * k + 1))
        f += tf
        g += tg
    out[series] = c1 * f - c2 * g
    big = x > 5
    zeta = 2 / 3 * x[big] ** 1.5
    out[big] = np.exp(-zeta) / (2 * SQRTPI * x[big] ** 0.25) * _airy_u_sum(zeta, alternate=True)
    neg = x < -10
    ax = -x[neg]
    zeta = 2 / 3 * ax ** 1.5
    even, odd = _airy_u_even_odd(zeta)
    out[neg] = (np.sin(zeta + np.pi / 4) * even - np.cos(zeta + np.pi / 4) * odd) / (SQRTPI * ax ** 0.25)
    return out


def _airy_u(k):
    u = 1.0
    for j in range(1, k + 1):
        u *= (6 * j - 5) * (6 * j - 3) * (6 * j - 1) / ((2 * j - 1) * 216 * j)
    return u


def _airy_u_sum(zeta, alternate):
    total = np.ones_like(zeta)
    for k in range(1, 12):
        total += ((-1) ** k if alternate else 1) * _airy_u(k) / zeta ** k
    return total


def _airy_u_even_odd(zeta):
    even, odd = np.ones_like(zeta), np.zeros_like(zeta)
    for k in range(1, 12):
        term = _airy_u(k) / zeta ** k
        if k % 2 == 0:
            even += (-1) ** (k // 2) * term
        else:
            odd += (-1) ** (k // 2) * term
    return even, odd


def _plain(function):
    """Wrap a float64 NumPy function as a no-gradient special function."""
    def wrapper(input):
        return Tensor(function(_values(input)).astype(np.float32), dtype="float32") if _out_dtype(input) == "float32" \
            else Tensor(function(_values(input)), dtype="float64")
    return wrapper


def i0(input):
    x = _values(input)
    return _result([input], _bessel_i(0, x), [_bessel_i(1, x)], _out_dtype(input))


def i1(input):
    x = _values(input)
    v0, v1 = _bessel_i(0, x), _bessel_i(1, x)
    with np.errstate(all="ignore"):
        slope = np.where(x == 0, 0.5, v0 - v1 / x)
    return _result([input], v1, [slope], _out_dtype(input))


def i0e(input):
    x = _values(input)
    e0, e1 = _bessel_i_scaled(0, x), _bessel_i_scaled(1, x)
    return _result([input], e0, [e1 - np.sign(x) * e0], _out_dtype(input))


def i1e(input):
    x = _values(input)
    e0, e1 = _bessel_i_scaled(0, x), _bessel_i_scaled(1, x)
    with np.errstate(all="ignore"):
        slope = np.where(x == 0, 0.5, e0 - e1 * (np.sign(x) + 1 / x))
    return _result([input], e1, [slope], _out_dtype(input))


def spherical_bessel_j0(input):
    x = _values(input)
    with np.errstate(all="ignore"):
        value = np.where(x == 0, 1.0, np.sin(x) / x)
        value = np.where(np.isinf(x), 0.0, value)
    return Tensor(value.astype(np.float32) if _out_dtype(input) == "float32" else value, dtype=_out_dtype(input))


# ---- orthogonal polynomials (no gradients, as in PyTorch) ----------------------------------

def _poly(kind, x, n):
    x, n = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(n, np.float64))
    n_int = np.where(np.isnan(n), -1, n).astype(np.int64)
    top = int(max(n_int.max(initial=0), 1))
    shifted = kind.startswith("shifted_")
    base = kind.replace("shifted_", "")
    y = 2 * x - 1 if shifted else x
    if base in ("t", "u", "v", "w"):
        p0 = np.ones_like(y)
        p1 = {"t": y, "u": 2 * y, "v": 2 * y - 1, "w": 2 * y + 1}[base]
        step = lambda a, b, k: 2 * y * b - a
    elif base == "h":
        p0, p1 = np.ones_like(y), 2 * y
        step = lambda a, b, k: 2 * y * b - 2 * k * a
    elif base == "he":
        p0, p1 = np.ones_like(y), y.copy()
        step = lambda a, b, k: y * b - k * a
    elif base == "l":
        p0, p1 = np.ones_like(y), 1 - y
        step = lambda a, b, k: ((2 * k + 1 - y) * b - k * a) / (k + 1)
    elif base == "p":
        p0, p1 = np.ones_like(y), y.copy()
        step = lambda a, b, k: ((2 * k + 1) * y * b - k * a) / (k + 1)
    out = np.where(n_int == 0, p0, np.where(n_int == 1, p1, 0.0))
    a, b = p0, p1
    with np.errstate(all="ignore"):
        for k in range(1, top):
            a, b = b, step(a, b, k)
            out = np.where(n_int == k + 1, b, out)
    return np.where(n_int < 0, 0.0, out)


def _poly_function(kind):
    def function(input, n):
        value = _poly(kind, _values(input), _values(n))
        return Tensor(value.astype(np.float32), dtype="float32") if _out_dtype(input) == "float32" else Tensor(value, dtype="float64")
    function.__name__ = kind
    return function


special = types.SimpleNamespace(
    airy_ai=_plain(_airy_ai),
    bessel_j0=_plain(lambda v: _bessel_jy(0, v, "j")), bessel_j1=_plain(lambda v: _bessel_jy(1, v, "j")),
    bessel_y0=_plain(lambda v: _bessel_jy(0, v, "y")), bessel_y1=_plain(lambda v: _bessel_jy(1, v, "y")),
    modified_bessel_i0=_plain(lambda v: _bessel_i(0, v)), modified_bessel_i1=_plain(lambda v: _bessel_i(1, v)),
    modified_bessel_k0=_plain(lambda v: _bessel_k(0, v)), modified_bessel_k1=_plain(lambda v: _bessel_k(1, v)),
    scaled_modified_bessel_k0=_plain(lambda v: _bessel_k(0, v, scaled=True)),
    scaled_modified_bessel_k1=_plain(lambda v: _bessel_k(1, v, scaled=True)),
    spherical_bessel_j0=spherical_bessel_j0, i0=i0, i0e=i0e, i1=i1, i1e=i1e,
    entr=entr, erfcx=erfcx, ndtr=ndtr, ndtri=ndtri, log_ndtr=log_ndtr, erfinv=erfinv,
    xlog1py=xlog1py, zeta=zeta, polygamma=polygamma, digamma=digamma, psi=digamma, gammaln=lgamma,
    multigammaln=mvlgamma, gammainc=igamma, gammaincc=igammac,
    **{f"chebyshev_polynomial_{letter}": _poly_function(letter) for letter in "tuvw"},
    **{f"shifted_chebyshev_polynomial_{letter}": _poly_function(f"shifted_{letter}") for letter in "tuvw"},
    hermite_polynomial_h=_poly_function("h"), hermite_polynomial_he=_poly_function("he"),
    laguerre_polynomial_l=_poly_function("l"), legendre_polynomial_p=_poly_function("p"),
    # Aliases of ops DenseCore already has.
    erf=lambda x: x.erf(), erfc=lambda x: x.erfc(), expit=lambda x: x.sigmoid(), exp2=lambda x: x.exp2(),
    expm1=lambda x: x.expm1(), log1p=lambda x: x.log1p(), sinc=lambda x: x.sinc(), round=lambda x, **k: x.round(**k),
    logit=lambda x, eps=None: x.logit(eps), xlogy=lambda x, y: x.xlogy(y) if isinstance(x, Tensor) else Tensor(float(x)).xlogy(y),
    logsumexp=lambda x, dim, keepdim=False: x.logsumexp(dim, keepdim),
    softmax=lambda x, dim, dtype=None: x.softmax(dim, dtype), log_softmax=lambda x, dim, dtype=None: x.log_softmax(dim, dtype))
