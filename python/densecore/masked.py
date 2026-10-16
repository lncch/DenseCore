"""torch.masked (strided masks) and torch.signal.windows, with PyTorch's formulas.

Masked reductions treat masked-out elements as the operation's identity (0 for sums,
1 for products, -inf for maxima and softmax inputs), as torch.masked does.
"""
import math
import types

import numpy as np

from .tensor import Tensor, where


def _mask(input, mask):
    if mask is None:
        return np.ones(input.shape, bool)
    return np.broadcast_to(mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask, bool), input.shape)


def _dims(input, dim):
    if dim is None or (isinstance(dim, (tuple, list)) and not dim):
        return tuple(range(len(input.shape)))
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    return tuple(d % max(1, len(input.shape)) for d in dims) if input.shape else ()


def _filled(input, mask, value):
    """input where mask, else value; masked-out spots get no gradient."""
    return input.masked_fill(~_mask(input, mask), value)


def sum(input, dim=None, *, keepdim=False, dtype=None, mask=None):
    return _filled(input, mask, 0.0).sum(_dims(input, dim), keepdim=keepdim)


def prod(input, dim=None, *, keepdim=False, dtype=None, mask=None):
    filled = _filled(input, mask, 1.0)
    dims = _dims(input, dim)
    if not dims:
        return filled
    out = filled
    for d in sorted(dims, reverse=True):
        out = out.prod(d, keepdim=True)
    return out if keepdim else out.squeeze(dims)


def _count(input, dim, keepdim, mask):
    present = Tensor(_mask(input, mask).astype(np.float32))
    return present.sum(_dims(input, dim), keepdim=keepdim)


def mean(input, dim=None, *, keepdim=False, dtype=None, mask=None):
    return sum(input, dim, keepdim=keepdim, mask=mask) / _count(input, dim, keepdim, mask)


def amax(input, dim=None, *, keepdim=False, dtype=None, mask=None):
    return _filled(input, mask, -np.inf).amax(_dims(input, dim), keepdim)


def amin(input, dim=None, *, keepdim=False, dtype=None, mask=None):
    return _filled(input, mask, np.inf).amin(_dims(input, dim), keepdim)


def argmax(input, dim=None, *, keepdim=False, dtype=None, mask=None):
    """PyTorch's masked argmax returns the input's dtype unless dtype is given."""
    return _filled(input, mask, -np.inf).argmax(dim, keepdim).to(dtype or input.dtype)


def argmin(input, dim=None, *, keepdim=False, dtype=None, mask=None):
    return _filled(input, mask, np.inf).argmin(dim, keepdim).to(dtype or input.dtype)


def cumsum(input, dim, *, dtype=None, mask=None):
    return _filled(input, mask, 0.0).cumsum(dim)


def cumprod(input, dim, *, dtype=None, mask=None):
    return _filled(input, mask, 1.0).cumprod(dim)


def logsumexp(input, dim=None, *, keepdim=False, dtype=None, mask=None):
    return _filled(input, mask, -np.inf).logsumexp(_dims(input, dim), keepdim)


def softmax(input, dim, *, dtype=None, mask=None):
    return _filled(input, mask, -np.inf).softmax(dim)


def log_softmax(input, dim, *, dtype=None, mask=None):
    return _filled(input, mask, -np.inf).log_softmax(dim)


def softmin(input, dim, *, dtype=None, mask=None):
    return (-_filled(input, mask, np.inf)).softmax(dim)


def norm(input, ord=2.0, dim=None, *, keepdim=False, dtype=None, mask=None):
    fill = -np.inf if ord in (-np.inf, float("-inf")) else 0.0
    if ord in (-np.inf, float("-inf")):
        return _filled(input.abs(), mask, np.inf).amin(_dims(input, dim), keepdim)
    return _filled(input, mask, fill).norm(ord, _dims(input, dim), keepdim)


def normalize(input, ord, dim, *, eps=1e-12, dtype=None, mask=None):
    norms = norm(input, ord, dim, keepdim=True, mask=mask)
    return _filled(input, mask, 0.0) / norms.clamp_min(eps)


def var(input, dim=None, unbiased=None, *, correction=None, keepdim=False, dtype=None, mask=None):
    if correction is None:
        correction = 0 if unbiased is False else 1
    count = _count(input, dim, True, mask)
    centered = _filled(input, mask, 0.0) - sum(input, dim, keepdim=True, mask=mask) / count
    squares = _filled(centered * centered, mask, 0.0).sum(_dims(input, dim), keepdim=keepdim)
    denominator = _count(input, dim, keepdim, mask) - float(correction)
    return squares / denominator.clamp_min(0.0)


def std(input, dim=None, unbiased=None, *, correction=None, keepdim=False, dtype=None, mask=None):
    return var(input, dim, unbiased, correction=correction, keepdim=keepdim, mask=mask).sqrt()


def median(input, dim=-1, *, keepdim=False, dtype=None, mask=None):
    """Lower median of the unmasked values (NaN where a slice is fully masked)."""
    filled = _filled(input, mask, np.nan)
    return filled.nanmedian(dim, keepdim).values if input.shape else filled


def logaddexp(input, other, *, dtype=None, input_mask=None, other_mask=None):
    return _filled(input, input_mask, -np.inf).logaddexp(_filled(other, other_mask, -np.inf))


# ---- core masked ops -------------------------------------------------------------------

def masked_select(input, mask):
    from .shapes import flat_gather
    shape = np.broadcast_shapes(input.shape, np.shape(mask.numpy() if isinstance(mask, Tensor) else mask))
    full = input.expand(*shape)
    selected = np.broadcast_to(mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask), shape)
    return flat_gather(full, np.flatnonzero(selected))


def masked_scatter(input, mask, source):
    """Copy source's leading elements, in order, into input where mask is True."""
    from .shapes import overwrite_flat
    values = mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask)
    shape = np.broadcast_shapes(input.shape, values.shape)
    selected = np.flatnonzero(np.broadcast_to(values, shape))
    return overwrite_flat(input.expand(*shape), selected, source.reshape(-1)[:len(selected)])


# ---- signal windows --------------------------------------------------------------------

def _window(M, values, dtype):
    return Tensor(np.asarray(values, np.float64), dtype=dtype or "float32")


def _positions(M, sym):
    return np.arange(M, dtype=np.float64), (M - 1 if sym else M)


def general_cosine(M, *, a, sym=True, dtype=None, **_):
    if M <= 1:
        return _window(M, np.ones(M), dtype)
    n, denom = _positions(M, sym)
    k = 2 * math.pi * n / denom
    values = np.zeros(M)
    for i, coefficient in enumerate(a):
        values += (-1) ** i * coefficient * np.cos(i * k)
    return _window(M, values, dtype)


def general_hamming(M, *, alpha=0.54, sym=True, dtype=None, **_):
    return general_cosine(M, a=[alpha, 1.0 - alpha], sym=sym, dtype=dtype)


def hann(M, *, sym=True, dtype=None, **_):
    return general_hamming(M, alpha=0.5, sym=sym, dtype=dtype)


def hamming(M, *, sym=True, dtype=None, **_):
    return general_hamming(M, alpha=0.54, sym=sym, dtype=dtype)


def blackman(M, *, sym=True, dtype=None, **_):
    return general_cosine(M, a=[0.42, 0.5, 0.08], sym=sym, dtype=dtype)


def nuttall(M, *, sym=True, dtype=None, **_):
    return general_cosine(M, a=[0.3635819, 0.4891775, 0.1365995, 0.0106411], sym=sym, dtype=dtype)


def bartlett(M, *, sym=True, dtype=None, **_):
    if M <= 1:
        return _window(M, np.ones(M), dtype)
    n, denom = _positions(M, sym)
    return _window(M, 1 - np.abs(2 * n / denom - 1), dtype)


def cosine(M, *, sym=True, dtype=None, **_):
    if M <= 1:
        return _window(M, np.ones(M), dtype)
    n, _ = _positions(M, sym)
    return _window(M, np.sin(math.pi * (n + 0.5) / (M if sym else M + 1)), dtype)


def exponential(M, *, center=None, tau=1.0, sym=True, dtype=None, **_):
    if M <= 1:
        return _window(M, np.ones(M), dtype)
    n, _ = _positions(M, sym)
    if center is None:
        center = (M if not sym and M > 1 else M - 1) / 2.0
    return _window(M, np.exp(-np.abs(n - center) / tau), dtype)


def gaussian(M, *, std=1.0, sym=True, dtype=None, **_):
    if M <= 1:
        return _window(M, np.ones(M), dtype)
    n, _ = _positions(M, sym)
    center = (M - 1 if sym else M) / 2.0
    return _window(M, np.exp(-((n - center) / std) ** 2 / 2), dtype)


def kaiser(M, *, beta=12.0, sym=True, dtype=None, **_):
    if M <= 1:
        return _window(M, np.ones(M), dtype)
    n, denom = _positions(M, sym)
    ratio = np.clip(1 - (2 * n / denom - 1) ** 2, 0, None)
    return _window(M, np.i0(beta * np.sqrt(ratio)) / np.i0(beta), dtype)


windows = types.SimpleNamespace(**{name: globals()[name] for name in (
    "general_cosine", "general_hamming", "hann", "hamming", "blackman", "nuttall", "bartlett",
    "cosine", "exponential", "gaussian", "kaiser")})
