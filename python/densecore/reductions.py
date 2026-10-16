"""Reductions with PyTorch semantics, attached to Tensor at import.

Selection reductions (max/min with dim, sort, median, kthvalue, mode, cummax/cummin) pick
indices with NumPy and gather the values natively, so values are exact and gradients go to
the chosen elements. amax/amin are native and split gradients across ties, as PyTorch does.
"""
import collections
import operator

import numpy as np

from .tensor import Tensor, INTEGER_DTYPES

_types = {name: collections.namedtuple(name, fields) for name, fields in [
    ("max", "values indices"), ("min", "values indices"), ("sort", "values indices"),
    ("kthvalue", "values indices"), ("median", "values indices"), ("mode", "values indices"),
    ("cummax", "values indices"), ("cummin", "values indices"), ("aminmax", "min max")]}


def _axis(self, dim):
    rank = max(1, len(self.shape))
    if isinstance(dim, (bool, np.bool_)):
        raise TypeError("dim must be an integer")
    dim = operator.index(dim)
    if not -rank <= dim < rank:
        raise IndexError("dimension out of range")
    return dim % rank


def _axes(self, dim):
    """None or () means every axis; otherwise a sorted list of distinct axes. A 0-d tensor
    accepts dim 0 or -1 and has nothing to reduce, as in PyTorch."""
    if dim is None or (isinstance(dim, (tuple, list)) and not dim):
        return list(range(len(self.shape)))
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    if not self.shape:
        for d in dims:
            _axis(self, d)
        return []
    axes = sorted({_axis(self, d) for d in dims})
    if len(axes) != len(dims):
        raise ValueError("duplicate reduction dimension")
    return axes


def _take_along(x, dim, index):
    """x values at `index` (same rank as x) along dim; float32 gradients scatter back."""
    from .functional import gather
    index = np.asarray(index, np.int64)
    if not x.shape:
        return x.reshape(index.shape) if index.size else Tensor(np.zeros(index.shape, np.float32), dtype=x.dtype)
    coords = list(np.indices(index.shape))
    coords[dim] = index
    flat = np.ravel_multi_index(coords, x.shape) if index.size else np.zeros(index.shape, np.int64)
    return gather(x.reshape(-1), 0, flat)


def _evenly(self, value):
    """A full reduction equal to `value`, with the gradient split evenly over the elements
    equal to it (PyTorch's evenly_distribute_backward)."""
    exact = Tensor(value, dtype=self.dtype)
    if self.dtype != "float32" or not self.requires_grad:
        return exact
    mask = self.numpy() == value
    count = max(int(mask.sum()), 1)
    share = self.masked_fill(~mask, 0.0).sum() / float(count)
    return exact + (share - share.detach())


def _check_nonempty(self, axes, name):
    if any(self.shape[a] == 0 for a in axes) or (not self.shape and self.numel() == 0):
        raise RuntimeError(f"{name}(): cannot reduce over a zero-size dimension")


def amax(self, dim=None, keepdim=False):
    axes = _axes(self, dim)
    _check_nonempty(self, axes, "amax")
    if self.dtype == "float32":
        return Tensor._wrap(self._native.extreme(axes, keepdim, True), parents=(self,))
    return Tensor(np.max(self._array(), axis=tuple(axes), keepdims=keepdim), dtype=self.dtype)


def amin(self, dim=None, keepdim=False):
    axes = _axes(self, dim)
    _check_nonempty(self, axes, "amin")
    if self.dtype == "float32":
        return Tensor._wrap(self._native.extreme(axes, keepdim, False), parents=(self,))
    return Tensor(np.min(self._array(), axis=tuple(axes), keepdims=keepdim), dtype=self.dtype)


def _arg(self, dim, keepdim, largest):
    values = self._array()
    if dim is None:
        flat = int(np.argmax(values) if largest else np.argmin(values)) if values.size else 0
        shape = (1,) * len(self.shape) if keepdim else ()
        return Tensor(np.full(shape, flat), dtype="int64")
    dim = _axis(self, dim)
    picked = np.argmax(values, axis=dim, keepdims=True) if largest else np.argmin(values, axis=dim, keepdims=True)
    return Tensor(picked if keepdim else np.squeeze(picked, dim), dtype="int64")


def argmax(self, dim=None, keepdim=False):
    """Index of the largest value (first occurrence; NaN counts as largest)."""
    return _arg(self, dim, keepdim, True)


def argmin(self, dim=None, keepdim=False):
    return _arg(self, dim, keepdim, False)


def _reduce_extreme(self, dim=None, keepdim=False, *, largest):
    if dim is None:
        return amax(self) if largest else amin(self)
    dim = _axis(self, dim)
    _check_nonempty(self, [dim] if self.shape else [], "max" if largest else "min")
    index = _arg(self, dim, True, largest).numpy()
    values = _take_along(self, dim, index)
    if not keepdim and self.shape:
        values, index = values.squeeze(dim), np.squeeze(index, dim)
    return _types["max" if largest else "min"](values, Tensor(index, dtype="int64"))


def max_(self, dim=None, keepdim=False):
    """max over everything, max along dim (values, indices), or elementwise max(other)."""
    if isinstance(dim, Tensor):
        return self.maximum(dim)
    return _reduce_extreme(self, dim, keepdim, largest=True)


def min_(self, dim=None, keepdim=False):
    if isinstance(dim, Tensor):
        return self.minimum(dim)
    return _reduce_extreme(self, dim, keepdim, largest=False)


def aminmax(self, *, dim=None, keepdim=False):
    return _types["aminmax"](amin(self, dim, keepdim), amax(self, dim, keepdim))


def _order(values, dim, descending):
    """Stable sort order along dim; NaN sorts as the largest value, as in PyTorch."""
    moved = np.moveaxis(values, dim, -1)
    nan = np.isnan(moved) if moved.dtype.kind == "f" else np.zeros(moved.shape, bool)
    key = np.where(nan, 0, moved)
    if key.dtype.kind in "iub":
        key = key.astype(np.int64)
    order = np.lexsort((-key, ~nan) if descending else (key, nan), axis=-1)
    return np.moveaxis(order, -1, dim)


def sort(self, dim=-1, descending=False, stable=False):
    if not self.shape:
        return _types["sort"](self, Tensor(np.zeros((), np.int64), dtype="int64"))
    dim = _axis(self, dim)
    order = _order(self._array(), dim, descending)
    return _types["sort"](_take_along(self, dim, order), Tensor(order, dtype="int64"))


def argsort(self, dim=-1, descending=False, stable=False):
    return sort(self, dim, descending).indices


def msort(self):
    return sort(self, 0).values


def kthvalue(self, k, dim=-1, keepdim=False):
    dim = _axis(self, dim) if self.shape else 0
    if not self.shape:
        return _types["kthvalue"](self, Tensor(np.zeros((), np.int64), dtype="int64"))
    if not 1 <= k <= self.shape[dim]:
        raise IndexError("k out of range")
    order = _order(self._array(), dim, False)
    index = np.take(order, [k - 1], axis=dim)
    values = _take_along(self, dim, index)
    if not keepdim:
        values, index = values.squeeze(dim), np.squeeze(index, dim)
    return _types["kthvalue"](values, Tensor(index, dtype="int64"))


def _median(self, dim, keepdim, ignore_nan):
    values = self._array()
    if dim is None:
        flat = values.ravel()
        if not ignore_nan and np.isnan(flat).any():
            return _evenly(self, np.float32(np.nan) if self.dtype == "float32" else np.nan)
        kept = np.sort(flat[~np.isnan(flat)] if flat.dtype.kind == "f" else flat)
        if not kept.size:
            return Tensor(np.nan, dtype=self.dtype)
        return _evenly(self, kept[(kept.size - 1) // 2])
    dim = _axis(self, dim) if self.shape else 0
    if not self.shape:
        return _types["median"](self, Tensor(np.zeros((), np.int64), dtype="int64"))
    order = _order(values, dim, False)
    moved = np.moveaxis(values, dim, -1)
    nan_count = np.isnan(moved).sum(-1, keepdims=True) if moved.dtype.kind == "f" else np.zeros(moved.shape[:-1] + (1,), int)
    length = moved.shape[-1]
    if ignore_nan:
        position = np.maximum((length - nan_count - 1) // 2, 0)
    else:
        # A row with NaN reports its first NaN (NaN sorts last); otherwise the lower median.
        position = np.where(nan_count > 0, length - nan_count, (length - 1) // 2)
    index = np.take_along_axis(np.moveaxis(order, dim, -1), position, -1)
    index = np.moveaxis(index, -1, dim)
    out = _take_along(self, dim, index)
    if not keepdim:
        out, index = out.squeeze(dim), np.squeeze(index, dim)
    return _types["median"](out, Tensor(index, dtype="int64"))


def median(self, dim=None, keepdim=False):
    return _median(self, dim, keepdim, ignore_nan=False)


def nanmedian(self, dim=None, keepdim=False):
    return _median(self, dim, keepdim, ignore_nan=True)


def mode(self, dim=-1, keepdim=False):
    """Most frequent value along dim (smallest among ties) and its last index.
    shortcut: loops over rows in Python; fine for moderate sizes."""
    if not self.shape:
        return _types["mode"](self, Tensor(np.zeros((), np.int64), dtype="int64"))
    dim = _axis(self, dim)
    moved = np.moveaxis(self._array(), dim, -1)
    rows = moved.reshape(-1, moved.shape[-1])
    picks = np.empty(len(rows), np.int64)
    for r, row in enumerate(rows):
        values, counts = np.unique(row, return_counts=True)
        best = values[np.argmax(counts)]
        picks[r] = np.flatnonzero(row == best)[-1] if not np.isnan(best) else np.flatnonzero(np.isnan(row))[-1]
    index = np.moveaxis(picks.reshape(moved.shape[:-1] + (1,)), -1, dim)
    out = _take_along(self, dim, index)
    if not keepdim:
        out, index = out.squeeze(dim), np.squeeze(index, dim)
    return _types["mode"](out, Tensor(index, dtype="int64"))


def prod(self, dim=None, keepdim=False, *, dtype=None):
    """Product by pairwise halving, so autograd's product rule handles zeros exactly."""
    x = self if dtype in (None, self.dtype) else self.to(dtype)
    if x.dtype != "float32":
        result = np.prod(x._array(), axis=None if dim is None else _axis(x, dim), keepdims=keepdim)
        out_dtype = "int64" if x.dtype in INTEGER_DTYPES or x.dtype == "bool" else x.dtype
        return Tensor(result, dtype=out_dtype)
    if dim is None:
        flat = x.reshape(1, -1)
        out = _pairwise_product(flat).reshape(())
        return out.reshape((1,) * len(x.shape)) if keepdim else out
    dim = _axis(x, dim)
    order = [i for i in range(len(x.shape)) if i != dim] + [dim]
    moved = x.permute(order)
    rows = moved.reshape(-1, x.shape[dim])
    out = _pairwise_product(rows).reshape(moved.shape[:-1])
    return out.unsqueeze(dim) if keepdim else out


def _pairwise_product(rows):
    from . import cat
    while rows.shape[1] > 1:
        n = rows.shape[1]
        half = (n // 2) * 2
        pairs = rows[:, :half].reshape(rows.shape[0], n // 2, 2)
        product = pairs[:, :, 0] * pairs[:, :, 1]
        rows = cat([product, rows[:, half:]], 1) if n % 2 else product
    if rows.shape[1] == 0:
        return Tensor(np.ones((rows.shape[0], 1), np.float32))
    return rows


def _moved_last(x, dim):
    order = [i for i in range(len(x.shape)) if i != dim] + [dim]
    inverse = list(np.argsort(order))
    return x.permute(order), inverse


def cumsum(self, dim, *, dtype=None):
    """Running sum along dim. float32 multiplies by an upper-triangular ones matrix.
    shortcut: O(n^2) work along dim; fine for moderate lengths, upgrade to a native scan."""
    x = self if dtype in (None, self.dtype) else self.to(dtype)
    if not x.shape:
        return x.clone()
    dim = _axis(x, dim)
    if x.dtype != "float32":
        out_dtype = "int64" if x.dtype in INTEGER_DTYPES or x.dtype == "bool" else x.dtype
        return Tensor(np.cumsum(x._array(), axis=dim), dtype=out_dtype)
    moved, inverse = _moved_last(x, dim)
    n = x.shape[dim]
    out = moved @ Tensor(np.triu(np.ones((n, n), np.float32)))
    return out.permute(inverse)


def cumprod(self, dim, *, dtype=None):
    """Running product along dim, one slice at a time (exact product-rule gradients).
    shortcut: one Python step per position along dim."""
    from . import cat
    x = self if dtype in (None, self.dtype) else self.to(dtype)
    if not x.shape:
        return x.clone()
    dim = _axis(x, dim)
    if x.dtype != "float32":
        out_dtype = "int64" if x.dtype in INTEGER_DTYPES or x.dtype == "bool" else x.dtype
        return Tensor(np.cumprod(x._array(), axis=dim), dtype=out_dtype)
    columns = x.split(1, dim)
    running, outputs = columns[0], [columns[0]]
    for column in columns[1:]:
        running = running * column
        outputs.append(running)
    return cat(outputs, dim) if outputs else x


def logcumsumexp(self, dim):
    if not self.shape:
        _axis(self, dim)
        return self.clone()
    dim = _axis(self, dim)
    shift = amax(self, dim, True).detach()
    shift = shift.masked_fill(~np.isfinite(shift.numpy()), 0.0)
    return cumsum((self - shift).exp(), dim).log() + shift


def _cumulative_extreme(self, dim, largest):
    if not self.shape:
        return _types["cummax" if largest else "cummin"](self, Tensor(np.zeros((), np.int64), dtype="int64"))
    dim = _axis(self, dim)
    moved = np.moveaxis(self._array(), dim, -1)
    index = np.zeros(moved.shape, np.int64)
    best = moved[..., 0].copy()
    best_index = np.zeros(moved.shape[:-1], np.int64)
    for i in range(1, moved.shape[-1]):
        current = moved[..., i]
        with np.errstate(invalid="ignore"):
            # Ties move to the later index; NaN takes over and stays (PyTorch).
            better = (current >= best) if largest else (current <= best)
        if moved.dtype.kind == "f":
            better = (better | np.isnan(current)) & ~np.isnan(best)
        best = np.where(better, current, best)
        best_index = np.where(better, i, best_index)
        index[..., i] = best_index
    index = np.moveaxis(index, -1, dim)
    return _types["cummax" if largest else "cummin"](_take_along(self, dim, index), Tensor(index, dtype="int64"))


def cummax(self, dim):
    return _cumulative_extreme(self, dim, True)


def cummin(self, dim):
    return _cumulative_extreme(self, dim, False)


def _bool_reduce(self, dim, keepdim, function):
    values = self._array().astype(bool)
    axes = None if dim is None else tuple(_axes(self, dim))
    return Tensor(function(values, axis=axes, keepdims=keepdim), dtype="bool")


def all_(self, dim=None, keepdim=False):
    return _bool_reduce(self, dim, keepdim, np.all)


def any_(self, dim=None, keepdim=False):
    return _bool_reduce(self, dim, keepdim, np.any)


def count_nonzero(self, dim=None):
    axes = None if dim is None else tuple(_axes(self, dim))
    return Tensor(np.count_nonzero(self._array(), axis=axes), dtype="int64")


def _reduced_count(self, axes):
    return int(np.prod([self.shape[a] for a in axes]))


def _legacy(dim, unbiased):
    """var(x, True) is the old var(x, unbiased=True)."""
    return (None, dim) if isinstance(dim, bool) else (dim, unbiased)


def var(self, dim=None, unbiased=None, keepdim=False, *, correction=None):
    dim, unbiased = _legacy(dim, unbiased)
    if correction is None:
        correction = 0 if unbiased is False else 1
    axes = tuple(_axes(self, dim))
    mean = self.mean(axes, keepdim=True)
    diff = self - mean
    total = (diff * diff).sum(axes, keepdim=keepdim)
    with np.errstate(divide="ignore", invalid="ignore"):
        return total / float(max(0, _reduced_count(self, axes) - correction))


def std(self, dim=None, unbiased=None, keepdim=False, *, correction=None):
    return var(self, dim, unbiased, keepdim, correction=correction).sqrt()


def var_mean(self, dim=None, unbiased=None, keepdim=False, *, correction=None):
    dim, unbiased = _legacy(dim, unbiased)
    axes = tuple(_axes(self, dim))
    return var(self, axes, unbiased, keepdim, correction=correction), self.mean(axes, keepdim=keepdim)


def std_mean(self, dim=None, unbiased=None, keepdim=False, *, correction=None):
    dim, unbiased = _legacy(dim, unbiased)
    axes = tuple(_axes(self, dim))
    return std(self, axes, unbiased, keepdim, correction=correction), self.mean(axes, keepdim=keepdim)


def norm(self, p="fro", dim=None, keepdim=False, dtype=None):
    """Vector norms over dim (all elements by default); the 2-norm has zero gradient at 0."""
    x = self if dtype in (None, self.dtype) else self.to(dtype)
    if p == "nuc":
        from .linalg import matrix_norm
        return matrix_norm(x, "nuc", (-2, -1) if dim is None else dim, keepdim)
    dim = tuple(_axes(x, dim))
    if p in ("fro", 2, 2.0):
        squares = (x * x).sum(dim, keepdim=keepdim)
        zero = squares.numpy() == 0
        return squares.masked_fill(zero, 1.0).sqrt().masked_fill(zero, 0.0)
    if p in (float("inf"), "inf"):
        return amax(x.abs(), dim, keepdim)
    if p in (float("-inf"), "-inf"):
        return amin(x.abs(), dim, keepdim)
    if p == 0:
        if x.dtype != "float32":
            return count_nonzero(x, dim)
        # The count, still on x's graph with a zero gradient (PyTorch).
        zero = x.masked_fill(np.ones(x.shape, bool), 0.0).sum(dim, keepdim=keepdim)
        return zero + (x != 0).float().sum(dim, keepdim=keepdim)
    if p == 1:
        return x.abs().sum(dim, keepdim=keepdim)
    from .special import _result
    values = x._array().astype(np.float64)
    with np.errstate(all="ignore"):
        powered = np.abs(values) ** float(p)
        # PyTorch: entries that are exactly 0 get gradient 0 (p * |x|^(p-1) would blow up).
        slope = np.where(values == 0, 0.0, float(p) * np.sign(values) * np.abs(values) ** (float(p) - 1))
    total = _result([x], powered, [slope]).sum(dim, keepdim=keepdim)
    zero = total.numpy() == 0  # PyTorch: zero gradient where the norm is 0
    return (total.masked_fill(zero, 1.0) ** (1.0 / float(p))).masked_fill(zero, 0.0)


def nansum(self, dim=None, keepdim=False, *, dtype=None):
    dim = tuple(_axes(self, dim))
    clean = self.masked_fill(np.isnan(self.numpy()), 0.0) if self.dtype == "float32" else self
    return clean.sum(dim, keepdim=keepdim, dtype=dtype)


def nanmean(self, dim=None, keepdim=False, *, dtype=None):
    dim = tuple(_axes(self, dim))
    present = ~np.isnan(self._array()) if self.dtype not in INTEGER_DTYPES else np.ones(self.shape, bool)
    count = Tensor(present.astype(np.float32)).sum(dim, keepdim=keepdim)
    return nansum(self, dim, keepdim, dtype=dtype) / count


def logsumexp(self, dim=None, keepdim=False):
    dim = tuple(_axes(self, dim))
    shift = amax(self, dim, True).detach()
    shift = shift.masked_fill(~np.isfinite(shift.numpy()), 0.0)
    out = (self - shift).exp().sum(dim, keepdim=True).log() + shift
    if keepdim:
        return out
    axes = _axes(self, dim)
    return out.squeeze(tuple(axes)) if axes else out


def dist(self, other, p=2):
    return norm(self - other, p)


for _name, _function in {
        "amax": amax, "amin": amin, "max": max_, "min": min_, "argmax": argmax, "argmin": argmin, "aminmax": aminmax,
        "sort": sort, "argsort": argsort, "msort": msort, "kthvalue": kthvalue, "median": median,
        "nanmedian": nanmedian, "mode": mode, "prod": prod, "cumsum": cumsum, "cumprod": cumprod,
        "logcumsumexp": logcumsumexp, "cummax": cummax, "cummin": cummin, "all": all_, "any": any_,
        "count_nonzero": count_nonzero, "var": var, "std": std, "var_mean": var_mean,
        "std_mean": std_mean, "norm": norm, "nansum": nansum, "nanmean": nanmean,
        "logsumexp": logsumexp, "dist": dist}.items():
    setattr(Tensor, _name, _function)
