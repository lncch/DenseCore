"""Smaller PyTorch ops: random generators, contractions, statistics, searching, set ops.

Random draws come from PyTorch's generator (see _random.py): same seed, same values for the
ported samplers.
Ops PyTorch does not differentiate (searchsorted, unique, histc, ...) return plain tensors.
"""
import builtins
import math
import string

import numpy as np

from .tensor import Tensor, INTEGER_DTYPES, where
from .shapes import flat_gather, scatter_add_flat, overwrite_flat, _axis, _index_array

_FLOATS = ("float32", "float64", "float16", "bfloat16")


def _rng(generator=None):
    from ._random import rng
    return rng(generator)


def _gen(generator=None):
    from ._random import _gen
    return _gen(generator)


def _size(size):
    return tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)


# ---- random -----------------------------------------------------------------------------

def randn(*size, dtype=None, device=None, requires_grad=False, generator=None):
    dtype = dtype or "float32"
    values = _gen(generator)._normal(_size(size), dtype=dtype)
    return Tensor(values, dtype=dtype, requires_grad=requires_grad)


def rand(*size, dtype=None, device=None, requires_grad=False, generator=None):
    dtype = dtype or "float32"
    values = _gen(generator)._uniform(_size(size), dtype=dtype)
    return Tensor(values, dtype=dtype, requires_grad=requires_grad)


def randn_like(self, *, dtype=None, generator=None, **_):
    return randn(*self.shape, dtype=dtype or self.dtype, generator=generator)


def rand_like(self, *, dtype=None, generator=None, **_):
    return rand(*self.shape, dtype=dtype or self.dtype, generator=generator)


def randint_like(self, low=0, high=None, *, dtype=None, generator=None, **_):
    low, high = (0, low) if high is None else (low, high)
    return Tensor(_gen(generator)._randint(self.shape, low, high), dtype=dtype or self.dtype)


def randperm(n, *, dtype="int64", device=None, requires_grad=False, generator=None):
    return Tensor(_gen(generator)._randperm(n), dtype=dtype)


def normal(mean=0.0, std=1.0, size=None, *, generator=None, **_):
    """normal(mean, std) with tensors or scalars; normal(mean, std, size) for scalars."""
    m = mean._array() if isinstance(mean, Tensor) else mean
    s = std._array() if isinstance(std, Tensor) else std
    shape = tuple(size) if size is not None else np.broadcast_shapes(np.shape(m), np.shape(s))
    like = mean if isinstance(mean, Tensor) else std if isinstance(std, Tensor) else None
    dtype = like.dtype if like is not None else "float32"
    if np.ndim(m) or np.ndim(s):
        # torch: normal_(0, 1), then mul_(std).add_(mean), each op rounded.
        z = _gen(generator)._normal(shape, dtype=dtype)
        return Tensor(z * np.asarray(s, z.dtype) + np.asarray(m, z.dtype), dtype=dtype)
    return Tensor(_gen(generator)._normal(shape, m, s, dtype=dtype), dtype=dtype)


def _refill(self, values):
    """Replace this tensor's values in place (the rebinding used by fill_ and friends)."""
    from .shapes import setitem
    setitem(self, ..., Tensor(np.asarray(values), dtype=self.dtype))
    return self


def normal_(self, mean=0.0, std=1.0, *, generator=None):
    return _refill(self, _gen(generator)._normal(self.shape, mean, std, dtype=self.dtype))


def uniform_(self, a=0.0, b=1.0, *, generator=None):
    return _refill(self, _gen(generator)._uniform(self.shape, a, b, dtype=self.dtype))


_UNSET = object()


def random_(self, from_=_UNSET, to=None, *, generator=None, **kwargs):
    """Tensor.random_(): with no bounds, [0, 2**mantissa] for floats or [0, max] for integers;
    random_(to) is [0, to); random_(from, to) is [from, to). PyTorch's draws."""
    if "from" in kwargs:
        from_ = kwargs.pop("from")
    if kwargs:
        raise TypeError(f"random_() got unexpected keyword arguments {sorted(kwargs)}")
    g = _gen(generator)
    if from_ is _UNSET and to is None:
        n = int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1
        if self.dtype in ("int64", "float64"):
            limit = 2 ** 63 if self.dtype == "int64" else 2 ** 53 + 1
            values = [g._native.random64() % limit for _ in range(n)]
        else:
            limit = {"float32": 2 ** 24 + 1, "int32": 2 ** 31, "int16": 2 ** 15, "int8": 2 ** 7,
                     "uint8": 2 ** 8, "bool": 2}[self.dtype]
            values = [g._native.random() % limit for _ in range(n)]
        return _refill(self, np.array(values).reshape(self.shape))
    from_ = 0 if from_ is _UNSET else from_
    low, high = (0, from_) if to is None else (from_, to)
    return _refill(self, g._randint(self.shape, low, high))


def exponential_(self, lambd=1.0, *, generator=None):
    return _refill(self, _gen(generator)._sample("exponential", self.shape, lambd, dtype=self.dtype))


def geometric_(self, p, *, generator=None):
    return _refill(self, _gen(generator)._sample("geometric", self.shape, p, dtype=self.dtype))


def cauchy_(self, median=0.0, sigma=1.0, *, generator=None):
    return _refill(self, _gen(generator)._sample("cauchy", self.shape, median, sigma, dtype=self.dtype))


def log_normal_(self, mean=1.0, std=2.0, *, generator=None):
    return _refill(self, _gen(generator)._sample("log_normal", self.shape, mean, std, dtype=self.dtype))


def _bernoulli_tensor(shape, probability, generator):
    """PyTorch's tensor-p kernel: one uniform per element, float64 if p is float64."""
    probability = np.broadcast_to(probability, shape)
    dtype = "float64" if probability.dtype == np.float64 else "float32"
    return _gen(generator)._uniform(shape, dtype=dtype) < probability


def bernoulli(self, p=None, *, generator=None):
    if p is None or isinstance(p, Tensor):
        probability = (self if p is None else p)._array()
        return Tensor(_bernoulli_tensor(self.shape, probability, generator), dtype=self.dtype)
    return Tensor(_gen(generator)._bernoulli(self.shape, p), dtype=self.dtype)


def bernoulli_(self, p=0.5, *, generator=None):
    if isinstance(p, Tensor):
        return _refill(self, _bernoulli_tensor(self.shape, p._array(), generator))
    return _refill(self, _gen(generator)._bernoulli(self.shape, p))


# ---- windows, contractions, geometry -------------------------------------------------------

def unfold(self, dimension, size, step):
    """Sliding windows of `size` along `dimension`, moved to a new last dimension."""
    from .functional import gather
    if not self.shape:
        return self.reshape(1)
    dim = _axis(self, dimension)
    count = (self.shape[dim] - size) // step + 1
    index = (np.arange(count)[:, None] * step + np.arange(size)[None, :])
    picked = gather(self, dim, index)          # shape: ..., count, size, ... (at dim)
    order = list(range(len(picked.shape)))
    order.append(order.pop(dim + 1))           # the window axis goes last, as in PyTorch
    return picked.permute(order)


def tensordot(a, b, dims=2):
    if isinstance(dims, int):
        dims_a, dims_b = list(range(len(a.shape) - dims, len(a.shape))), list(range(dims))
    else:
        dims_a, dims_b = ([d] if isinstance(d, int) else list(d) for d in dims)
    dims_a = [d % len(a.shape) for d in dims_a]
    dims_b = [d % len(b.shape) for d in dims_b]
    # PyTorch broadcasts a contracted size-1 dimension against the other side.
    sizes = [max(a.shape[i], b.shape[j]) for i, j in zip(dims_a, dims_b)]
    if any(a.shape[i] != n for i, n in zip(dims_a, sizes)):
        shape = list(a.shape)
        for i, n in zip(dims_a, sizes):
            shape[i] = n
        a = a.expand(*shape)
    if any(b.shape[j] != n for j, n in zip(dims_b, sizes)):
        shape = list(b.shape)
        for j, n in zip(dims_b, sizes):
            shape[j] = n
        b = b.expand(*shape)
    free_a = [i for i in range(len(a.shape)) if i not in dims_a]
    free_b = [i for i in range(len(b.shape)) if i not in dims_b]
    k = int(np.prod([a.shape[i] for i in dims_a])) if dims_a else 1
    left = a.permute(free_a + dims_a).reshape(-1 if free_a else 1, k)
    right = b.permute(dims_b + free_b).reshape(k, -1 if free_b else 1)
    return (left @ right).reshape(*[a.shape[i] for i in free_a], *[b.shape[i] for i in free_b])


def einsum(equation, *operands):
    """General einsum by aligning every operand to one axis order, multiplying with
    broadcasting and summing the contracted axes. shortcut: builds the full product."""
    operands = operands[0] if len(operands) == 1 and isinstance(operands[0], (list, tuple)) else operands
    equation = equation.replace(" ", "")
    inputs, _, output = equation.partition("->")
    terms = inputs.split(",")
    if "..." in equation:
        raise NotImplementedError("einsum with '...' is unsupported")
    if not _:
        counts = {}
        for letter in inputs.replace(",", ""):
            counts[letter] = counts.get(letter, 0) + 1
        output = "".join(sorted(letter for letter, n in counts.items() if n == 1))
    letters = list(dict.fromkeys(output + "".join(terms)))
    sizes = {}
    product = None
    for term, operand in zip(terms, operands):
        x = operand
        # Repeated letters in one term mean a diagonal.
        while len(set(term)) < len(term):
            letter = next(l for l in term if term.count(l) > 1)
            first, second = term.index(letter), term.index(letter, term.index(letter) + 1)
            x = x.diagonal(0, first, second)
            term = term[:first] + term[first + 1:second] + term[second + 1:] + letter
        for letter, n in zip(term, x.shape):
            sizes[letter] = n
        order = sorted(range(len(term)), key=lambda i: letters.index(term[i]))
        aligned = x.permute(order) if order else x
        shape = [sizes[l] if l in term else 1 for l in letters]
        aligned = aligned.reshape(shape)
        product = aligned if product is None else product * aligned
    contracted = tuple(i for i, l in enumerate(letters) if l not in output)
    result = product.sum(contracted) if contracted else product
    kept = [l for l in letters if l in output]
    return result.permute([kept.index(l) for l in output]) if output else result


def cdist(x1, x2, p=2.0, compute_mode=None):
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    return diff.norm(p, -1)


def cross(self, other, dim=None):
    if dim is None:
        dim = next(i for i, n in enumerate(self.shape) if n == 3)
    dim = _axis(self, dim)
    a = [self.select(dim, i) for i in range(3)]
    b = [other.select(dim, i) for i in range(3)]
    from . import stack
    return stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], dim)


def cov(self, *, correction=1, fweights=None, aweights=None):
    """PyTorch's formula: weights w = fweights * aweights; a one-row result is a scalar."""
    x = self if len(self.shape) == 2 else self.reshape(1, -1)
    weights = None
    for given in (fweights, aweights):
        if given is not None:
            given = given if given.dtype == "float32" else Tensor(given._array().astype(np.float32))
            weights = given if weights is None else weights * given
    if weights is None:
        centered = x - x.mean(1, keepdim=True)
        norm = float(max(x.shape[1] - correction, 0))
    else:
        total = weights.sum()
        centered = x - (x * weights).sum(1, keepdim=True) / total
        norm = total - correction if aweights is None else total - correction * (weights * aweights).sum() / total
        norm = norm.clamp(min=0.0)
        result = ((centered * weights) @ centered.t()) / norm
        return result.reshape(()) if x.shape[0] == 1 else result
    result = (centered @ centered.t()) / norm
    return result.reshape(()) if x.shape[0] == 1 else result


def corrcoef(self):
    c = cov(self)
    if not c.shape:
        return c / c
    std = c.diagonal().sqrt()
    return (c / std.unsqueeze(1) / std.unsqueeze(0)).clamp(-1.0, 1.0)


def _trapezoid_parts(y, x, dx, dim):
    dim = _axis(y, dim)
    n = y.shape[dim]
    left, right = y.narrow(dim, 0, n - 1), y.narrow(dim, 1, n - 1)
    if x is None:
        return (left + right) * (0.5 * dx), dim
    spacing = x.diff(dim=dim if len(x.shape) == len(y.shape) else -1)
    if len(x.shape) == 1:
        shape = [1] * len(y.shape)
        shape[dim] = n - 1
        spacing = spacing.reshape(shape)
    return (left + right) * spacing * 0.5, dim


def trapezoid(y, x=None, *, dx=None, dim=-1):
    if y.shape and y.shape[_axis(y, dim)] == 0:
        return y.sum(dim)
    parts, dim = _trapezoid_parts(y, x, 1.0 if dx is None else dx, dim)
    return parts.sum(dim)


def cumulative_trapezoid(y, x=None, *, dx=None, dim=-1):
    parts, dim = _trapezoid_parts(y, x, 1.0 if dx is None else dx, dim)
    return parts.cumsum(dim)


# ---- searching, sets, histograms (no gradients, as in PyTorch) ----------------------------

def searchsorted(sorted_sequence, input, *, out_int32=False, right=False, side=None, sorter=None):
    side = side or ("right" if right else "left")
    seq = sorted_sequence._array()
    values = input._array() if isinstance(input, Tensor) else np.asarray(input)
    if sorter is not None:
        seq = np.take_along_axis(seq, _index_array(sorter), -1)
    if seq.ndim == 1:
        result = np.searchsorted(seq, values, side=side)
    else:
        rows_seq, rows_val = seq.reshape(-1, seq.shape[-1]), values.reshape(-1, values.shape[-1])
        result = np.stack([np.searchsorted(s, v, side=side) for s, v in zip(rows_seq, rows_val)]).reshape(values.shape)
    return Tensor(result, dtype="int32" if out_int32 else "int64")


def bucketize(self, boundaries, *, out_int32=False, right=False):
    return searchsorted(boundaries, self, out_int32=out_int32, side="left" if not right else "right")


def isin(elements, test_elements, *, assume_unique=False, invert=False):
    values = elements._array() if isinstance(elements, Tensor) else np.asarray(elements)
    tests = test_elements._array() if isinstance(test_elements, Tensor) else np.asarray(test_elements)
    return Tensor(np.isin(values, tests, invert=invert), dtype="bool")


def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
    values = self._array()
    if dim is None:
        found, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(values.shape)
    else:
        found, inverse, counts = np.unique(values, axis=dim, return_inverse=True, return_counts=True)
    out = [Tensor(found, dtype=self.dtype)]
    if return_inverse:
        out.append(Tensor(inverse, dtype="int64"))
    if return_counts:
        out.append(Tensor(counts, dtype="int64"))
    return out[0] if len(out) == 1 else tuple(out)


def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
    values = self._array()
    if dim is None:
        flat = values.ravel()
        starts = np.concatenate([[True], flat[1:] != flat[:-1]]) if flat.size else np.zeros(0, bool)
        found = flat[starts]
        inverse = (np.cumsum(starts) - 1).reshape(values.shape)
        counts = np.diff(np.append(np.flatnonzero(starts), flat.size))
    else:
        dim = _axis(self, dim)
        moved = np.moveaxis(values, dim, 0)
        change = np.concatenate([[True], [not np.array_equal(moved[i], moved[i - 1]) for i in range(1, len(moved))]])
        found = np.moveaxis(moved[change], 0, dim)
        inverse = np.cumsum(change) - 1
        counts = np.diff(np.append(np.flatnonzero(change), len(moved)))
    out = [Tensor(found, dtype=self.dtype)]
    if return_inverse:
        out.append(Tensor(inverse, dtype="int64"))
    if return_counts:
        out.append(Tensor(counts, dtype="int64"))
    return out[0] if len(out) == 1 else tuple(out)


def histc(self, bins=100, min=0, max=0):
    values = self._array().astype(np.float64).ravel()
    low, high = (values.min(), values.max()) if min == max and values.size else (min, max)
    if low == high:
        low, high = low - 1, high + 1
    counts, _ = np.histogram(values, bins=bins, range=(low, high))
    return Tensor(counts.astype(np.float32), dtype=self.dtype)


# ---- in-place style scatters and small arithmetic ----------------------------------------

def renorm(self, p, dim, maxnorm):
    """Rescale each slice along dim whose p-norm exceeds maxnorm (PyTorch's 1e-7 guard).
    Gradients flow through the norm as well, as in PyTorch."""
    dim = _axis(self, dim)
    others = tuple(i for i in range(len(self.shape)) if i != dim)
    norms = self.norm(p, others, keepdim=True)
    ones = norms.masked_fill(np.ones(norms.shape, bool), 1.0)
    factor = where(norms.numpy() > maxnorm, (norms + 1e-7).reciprocal() * maxnorm, ones)
    return self * factor


def put(self, index, source, accumulate=False):
    positions = np.where(_index_array(index) < 0, _index_array(index) + self.numel(), _index_array(index))
    return scatter_add_flat(self, positions, source) if accumulate else overwrite_flat(self, positions, source)


def index_reduce(self, dim, index, source, reduce, *, include_self=True):
    if not self.shape:
        return index_reduce(self.reshape(1), 0, index, source.reshape(1), reduce, include_self=include_self).reshape(())
    dim = _axis(self, dim) if self.shape else 0
    index = _index_array(index).reshape(-1)
    shape = source.shape if source.shape else (1,)
    full = np.broadcast_to(index.reshape([-1 if i == dim else 1 for i in range(len(shape))]), shape)
    return self.scatter_reduce(dim, Tensor(full.copy(), dtype="int64"), source, reduce, include_self=include_self)


def ldexp(self, other):
    if isinstance(other, Tensor) and other.dtype == "float32":
        return self * (other * math.log(2.0)).exp()  # keeps the exponent's gradient
    exponent = other._array() if isinstance(other, Tensor) else other
    return self * Tensor(np.power(2.0, exponent).astype(np.float32))


def frexp(self):
    mantissa, exponent = np.frexp(self._array())
    scale = Tensor(np.power(2.0, -exponent.astype(np.float64)).astype(np.float32))
    return self * scale, Tensor(exponent, dtype="int32")


def angle(self):
    """For real tensors: pi where negative, 0 otherwise, NaN stays NaN; zero gradient.
    For complex tensors: the argument, as a real tensor."""
    values = self._array()
    if values.dtype.kind == "c":
        return Tensor(np.angle(values), dtype="float32" if self.dtype == "complex64" else "float64")
    result = np.where(values < 0, np.pi, 0.0).astype(np.float32)
    result = np.where(np.isnan(values), np.nan, result)
    if self.dtype == "float32":
        return self.masked_fill(np.ones(self.shape, bool), 0.0) + Tensor(result)
    return Tensor(result, dtype="float32" if self.dtype in INTEGER_DTYPES else self.dtype)


def _quantile(self, q, dim, keepdim, interpolation, ignore_nan):
    """Sort, then interpolate between neighbouring order statistics (PyTorch's methods)."""
    qs = q._array() if isinstance(q, Tensor) else np.asarray(q, np.float64)
    scalar_q = qs.ndim == 0
    qs = np.atleast_1d(qs).astype(np.float64)
    x = self.reshape(-1) if dim is None else self
    axis = 0 if dim is None else _axis(self, dim)
    ordered = x.sort(axis).values
    values = ordered._array()
    moved = np.moveaxis(values, axis, -1)
    count = (~np.isnan(moved)).sum(-1, keepdims=True) if ignore_nan else np.full(moved.shape[:-1] + (1,), moved.shape[-1])
    results = []
    # PyTorch differentiates linear interpolation with respect to q as well.
    q_graph = isinstance(q, Tensor) and q.requires_grad and q.dtype == "float32" and interpolation == "linear"
    for i, quantile in enumerate(qs):
        rank = quantile * (count - 1)
        low = np.floor(rank).astype(np.int64)
        high = np.ceil(rank).astype(np.int64)
        if interpolation == "lower":
            high = low
        elif interpolation == "higher":
            low = high
        elif interpolation == "nearest":
            low = high = np.round(rank).astype(np.int64)
        frac = (rank - low).astype(np.float32) if interpolation == "linear" else np.zeros(rank.shape, np.float32)
        if interpolation == "midpoint":
            frac = np.where(high > low, 0.5, 0.0).astype(np.float32)
        low, high = np.clip(low, 0, moved.shape[-1] - 1), np.clip(high, 0, moved.shape[-1] - 1)
        pick = lambda idx: ordered.gather(axis, np.moveaxis(idx, -1, axis))
        a, b = pick(low), pick(high)
        weight = Tensor(np.moveaxis(frac, -1, axis))
        if q_graph:
            q_i = q if scalar_q else q[i]
            weight = q_i * Tensor(np.moveaxis(count - 1, -1, axis).astype(np.float32)) \
                - Tensor(np.moveaxis(low, -1, axis).astype(np.float32))
        out = a + (b - a) * weight
        if not ignore_nan:
            has_nan = np.isnan(moved).any(-1, keepdims=True)
            out = out.masked_fill(np.moveaxis(has_nan, -1, axis), np.nan)
        results.append(out if keepdim and dim is not None else out.squeeze(axis))
    from . import stack
    return results[0] if scalar_q else stack(results, 0)


def quantile(self, q, dim=None, keepdim=False, *, interpolation="linear"):
    return _quantile(self if self.shape else self.reshape(1), q, dim if self.shape else None, keepdim, interpolation,
                     ignore_nan=False)


def nanquantile(self, q, dim=None, keepdim=False, *, interpolation="linear"):
    return _quantile(self if self.shape else self.reshape(1), q, dim if self.shape else None, keepdim, interpolation,
                     ignore_nan=True)


# ---- numerical gradient and histograms -------------------------------------------------------

def gradient(input, *, spacing=1, dim=None, edge_order=1):
    """Central differences inside, one-sided (edge_order 1 or 2) at the ends, NumPy's
    formulas for uneven spacing. Differentiable: the result is a weighted sum of slices."""
    dims = list(range(len(input.shape))) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    dims = [_axis(input, k) for k in dims]
    spacing = 1 if spacing is None else spacing
    spacings = list(spacing) if isinstance(spacing, (list, tuple)) else [spacing] * len(dims)
    if len(spacings) == 1 and len(dims) > 1:
        spacings = spacings * len(dims)
    results = []
    for axis, h in zip(dims, spacings):
        n = input.shape[axis]
        if n < edge_order + 1:
            raise RuntimeError("torch.gradient expected each dimension size to be at least edge_order+1")
        coords = h._array().astype(np.float64) if isinstance(h, Tensor) else np.arange(n) * float(h)
        dx = np.diff(coords)
        shape = [1] * len(input.shape)

        def part(start, length, weights):
            shape[axis] = length
            return input.narrow(axis, start, length) * Tensor(np.asarray(weights, np.float32).reshape(shape))

        pieces = []
        d1, d2 = dx[0], dx[1] if n > 2 else dx[0]
        if edge_order == 1:
            pieces.append(part(1, 1, [1 / d1]) - part(0, 1, [1 / d1]))
        else:
            pieces.append(part(0, 1, [-(2 * d1 + d2) / (d1 * (d1 + d2))]) + part(1, 1, [(d1 + d2) / (d1 * d2)])
                          + part(2, 1, [-d1 / (d2 * (d1 + d2))]))
        if n > 2:
            a, c = dx[:-1], dx[1:]
            pieces.append(part(0, n - 2, -c / (a * (a + c))) + part(1, n - 2, (c - a) / (a * c))
                          + part(2, n - 2, a / (c * (a + c))))
        d1, d2 = (dx[-2], dx[-1]) if n > 2 else (dx[-1], dx[-1])
        if edge_order == 1:
            pieces.append(part(n - 1, 1, [1 / d2]) - part(n - 2, 1, [1 / d2]))
        else:
            pieces.append(part(n - 3, 1, [d2 / (d1 * (d1 + d2))]) + part(n - 2, 1, [-(d1 + d2) / (d1 * d2)])
                          + part(n - 1, 1, [(2 * d2 + d1) / (d2 * (d1 + d2))]))
        from . import cat
        results.append(cat(pieces, axis))
    return tuple(results)


def histogram(input, bins=100, *, range=None, weight=None, density=False):
    counts, edges = histogramdd(input.reshape(-1, 1), [bins], range=range, weight=weight, density=density)
    return counts, edges[0]


def histogramdd(input, bins=10, *, range=None, weight=None, density=False):
    """NumPy's histogramdd. As in PyTorch, an edge tensor that is not increasing catches nothing
    (counts 0, density NaN) and is returned as given."""
    points = input._array().reshape(-1, input.shape[-1])
    bins = [b._array() if isinstance(b, Tensor) else b for b in bins] if isinstance(bins, (list, tuple)) else bins
    given = bins if isinstance(bins, list) else None
    broken = given is not None and any(np.ndim(b) and np.any(np.diff(b) <= 0) for b in given)
    if broken:
        bins = [np.sort(b) if np.ndim(b) else b for b in given]
    limits = None if range is None else [tuple(range[2 * i:2 * i + 2]) for i in builtins.range(len(range) // 2)]
    counts, edges = np.histogramdd(points, bins, limits, density=density and not broken,
                                   weights=None if weight is None else weight._array().reshape(-1))
    if broken:
        counts = np.full(counts.shape, np.nan if density else 0.0)
        edges = [g if np.ndim(g) else e for g, e in zip(given, edges)]
    return Tensor(counts.astype(np.float32)), [Tensor(np.asarray(e).astype(np.float32)) for e in edges]


# ---- strided views, resizing, static nonzero -------------------------------------------------

def _strided_positions(size, stride, storage_offset):
    positions = np.full(tuple(size), storage_offset or 0, np.int64)
    for axis, (n, step) in enumerate(zip(size, stride)):
        shape = [1] * len(size)
        shape[axis] = n
        positions = positions + (np.arange(n) * step).reshape(shape)
    return positions


def as_strided(self, size, stride, storage_offset=None):
    """shortcut: indexes self's row-major layout and returns a copy, not a view sharing storage;
    upgrade if code writes through the result."""
    return flat_gather(self, _strided_positions(size, stride, storage_offset))


def as_strided_scatter(self, src, size, stride, storage_offset=None):
    positions = _strided_positions(size, stride, storage_offset)
    return overwrite_flat(self.reshape(-1), positions, src.reshape(positions.shape)).reshape(self.shape)


def resize_(self, *size):
    """Keep the first elements in row-major order; new elements are zero (PyTorch leaves them
    uninitialized)."""
    size = _size(size)
    count = int(np.prod(size))
    values = self._array().reshape(-1)
    values = np.concatenate([values, np.zeros(max(count - values.size, 0), values.dtype)])[:count]
    self._native = Tensor(values.reshape(size), dtype=self.dtype)._native
    return self


def nonzero_static(self, *, size, fill_value=-1):
    found = self.nonzero()._array()
    rows = np.full((size, found.shape[1]), fill_value, np.int64)
    rows[:min(size, len(found))] = found[:size]
    return Tensor(rows, dtype="int64")


def slice_(input, dim=0, start=None, end=None, step=1):
    """aten.slice: input[..., start:end:step, ...] along dim."""
    return input[(slice(None),) * _axis(input, dim) + (slice(start, end, step),)]


TENSOR_METHODS = {
    "as_strided": as_strided, "as_strided_copy": as_strided, "as_strided_scatter": as_strided_scatter,
    "resize_": resize_, "resize_as_": lambda self, other: resize_(self, other.shape),
    "nonzero_static": nonzero_static, "alias_copy": lambda self: self.clone(), "histogram": histogram,
    "randn_like": randn_like, "rand_like": rand_like, "randint_like": randint_like, "normal_": normal_,
    "random_": random_,
    "uniform_": uniform_, "exponential_": exponential_, "geometric_": geometric_, "cauchy_": cauchy_,
    "log_normal_": log_normal_, "bernoulli": bernoulli, "bernoulli_": bernoulli_, "unfold": unfold,
    "tensordot": tensordot, "cross": cross, "cov": cov, "corrcoef": corrcoef, "isin": isin, "unique": unique,
    "unique_consecutive": unique_consecutive, "histc": histc, "renorm": renorm, "put": put,
    "index_reduce": index_reduce, "ldexp": ldexp, "frexp": frexp, "angle": angle, "quantile": quantile,
    "nanquantile": nanquantile, "bucketize": bucketize, "trapezoid": trapezoid, "trapz": trapezoid,
    "cumulative_trapezoid": cumulative_trapezoid, "resolve_conj": lambda self: self,
    "resolve_neg": lambda self: self,
    "unsafe_chunk": lambda self, chunks, dim=0: self.chunk(chunks, dim),
    "unsafe_split": lambda self, size, dim=0: self.split(size, dim),
    "new_empty_strided": lambda self, size, stride, **kw: self.new_zeros(*size, **{k: v for k, v in kw.items() if k == "dtype"}),
}
def complex_(real, imag):
    if real.dtype != imag.dtype or real.dtype not in ("float32", "float64"):
        raise RuntimeError("complex expects two float32 or two float64 tensors of the same dtype")
    return Tensor(real._array() + 1j * imag._array(), dtype="complex64" if real.dtype == "float32" else "complex128")


def polar(abs, angle):
    return complex_(abs * angle.cos(), abs * angle.sin())


def view_as_real(input):
    if input.dtype not in ("complex64", "complex128"):
        raise RuntimeError("view_as_real is only supported for complex tensors")
    values = input._array()
    return Tensor(np.stack([values.real, values.imag], -1), dtype="float32" if input.dtype == "complex64" else "float64")


def view_as_complex(input):
    if not input.shape or input.shape[-1] != 2:
        raise RuntimeError("view_as_complex needs a last dimension of size 2")
    values = input._array()
    return Tensor(values[..., 0] + 1j * values[..., 1], dtype="complex64" if input.dtype == "float32" else "complex128")


MODULE_FUNCTIONS = {
    "complex": complex_, "polar": polar, "view_as_real": view_as_real, "view_as_complex": view_as_complex,
    "real": lambda input: input.real, "imag": lambda input: input.imag,
    "is_complex": lambda input: input.is_complex(), "is_floating_point": lambda input: input.is_floating_point(),
    "randn": randn, "rand": rand, "randperm": randperm, "normal": normal, "einsum": einsum, "cdist": cdist,
    "searchsorted": searchsorted, "empty": lambda *size, dtype=None, **_: Tensor(np.zeros(_size(size)), dtype=dtype or "float32"),
    "empty_strided": lambda size, stride, dtype=None, **_: Tensor(np.zeros(tuple(size)), dtype=dtype or "float32"),
    "empty_permuted": lambda size, physical_layout, dtype=None, **_: Tensor(np.zeros(tuple(size)), dtype=dtype or "float32"),
    "constant_pad_nd": lambda input, pad, value=0: _constant_pad_nd(input, pad, value),
    "slice": slice_,  # shadows the builtin only as densecore.slice
    "gradient": gradient, "histogramdd": histogramdd,
}


def _constant_pad_nd(input, pad, value=0):
    from .functional import pad as pad_function
    return pad_function(input, pad, "constant", value)
