"""Shape, indexing and scatter ops with PyTorch semantics, attached to Tensor at import.

Two native primitives carry everything: a flat gather (the embedding kernel) and a flat
scatter_add. Index arithmetic happens in NumPy on position arrays, so values stay exact
and float32 gradients flow through gather/scatter.
"""
import itertools
import math
import operator

import numpy as np

from .tensor import Tensor, INTEGER_DTYPES, where

_FLOATS = ("float32", "float64", "float16", "bfloat16")


def _axis(x, dim, extra=0):
    rank = max(1, len(x.shape) + extra)
    if isinstance(dim, (bool, np.bool_)):
        raise TypeError("dim must be an integer")
    dim = operator.index(dim)
    if not -rank <= dim < rank:
        raise IndexError("dimension out of range")
    return dim % rank


def _index_array(value):
    if isinstance(value, Tensor):
        return value.numpy()
    return np.asarray(value)


def flat_gather(x, positions):
    """x.reshape(-1)[positions] with positions of any shape; exact, differentiable."""
    from .functional import gather
    positions = np.asarray(positions, np.int64)
    if positions.size == 0:
        # An empty result still hangs off x's graph, as in PyTorch.
        empty = x.reshape(x.numel())[0:0]
        return empty.reshape(positions.shape)
    return gather(x.reshape(x.numel()), 0, positions)


def scatter_add_flat(base, positions, src):
    """base with src added at flat positions (src has positions' shape); duplicates add up."""
    positions = np.asarray(positions, np.int64)
    if base.dtype == "float32" and (not isinstance(src, Tensor) or src.dtype != "float64"):
        src = src if isinstance(src, Tensor) else Tensor(np.asarray(src, np.float32))
        src = (src if src.dtype == "float32" else src.float()).expand(positions.shape).reshape(-1) \
            if positions.size else Tensor(np.zeros(0, np.float32))
        data = np.ascontiguousarray(positions.ravel(), "<i8").tobytes()
        return Tensor._wrap(base._native.scatter_add(src._native, data), parents=(base, src))
    values = base._array().copy().reshape(-1)
    np.add.at(values, positions.ravel(), np.broadcast_to(_index_array(src), positions.shape).ravel())
    return Tensor(values.reshape(base.shape), dtype=base.dtype)


def overwrite_flat(base, positions, src):
    """base with src written at flat positions (the last write wins for duplicates)."""
    positions = np.asarray(positions, np.int64)
    flat = positions.ravel()
    if not flat.size:
        return base
    _, last_from_end = np.unique(flat[::-1], return_index=True)
    keep = np.sort(flat.size - 1 - last_from_end)
    if base.dtype not in ("float32",):
        values = base._array().copy().reshape(-1)
        values[flat[keep]] = np.broadcast_to(_index_array(src), positions.shape).ravel()[keep]
        return Tensor(values.reshape(base.shape), dtype=base.dtype)
    mask = np.zeros(base.numel(), bool)
    mask[flat] = True
    cleared = base.masked_fill(mask.reshape(base.shape), 0.0)
    if isinstance(src, Tensor):
        chosen = flat_gather(src.float().expand(positions.shape) if src.dtype != "float32" else src.expand(positions.shape), keep)
    else:
        chosen = Tensor(np.broadcast_to(np.asarray(src, np.float32), positions.shape).ravel()[keep])
    return scatter_add_flat(cleared, flat[keep], chosen)


def positions_of(x, key):
    """Flat positions selected by a PyTorch-style index key.

    PyTorch applies plain integers (and None) as basic indexing first, then the advanced
    parts with NumPy rules; plain NumPy would treat integers as advanced too and could move
    dimensions (x[0, ..., [2]] is (6, 1) in PyTorch but (1, 6) in NumPy)."""
    key = _as_tuple_key(key)
    keys = key if isinstance(key, tuple) else (key,)
    keys = tuple(_index_array(k) if isinstance(k, (Tensor, list)) else k for k in keys)
    advanced = lambda k: isinstance(k, np.ndarray)
    grid = np.arange(x.numel()).reshape(x.shape)
    first = tuple(slice(None) if advanced(k) else k for k in keys)
    grid = grid[first]
    if not any(advanced(k) for k in keys):
        return grid
    second = tuple(k if advanced(k) or k is Ellipsis else slice(None) for k in keys
                   if not (isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_))))
    return grid[second]


def _as_tuple_key(key):
    """PyTorch 2.14 reads a list holding sequences, slices, None or Ellipsis as a tuple of
    indices. PyTorch has deprecated this and plans to read it as one tensor index instead."""
    if isinstance(key, list) and any(isinstance(k, (list, tuple, slice, Tensor, np.ndarray)) or k is None
                                     or k is Ellipsis for k in key):
        return tuple(key)
    return key


def _is_basic(key):
    keys = key if isinstance(key, tuple) else (key,)
    return all(k is None or k is Ellipsis or isinstance(k, slice)
               or (isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_))) for k in keys)


_basic_getitem = Tensor.__getitem__


def getitem(self, key):
    """Basic keys return native views; any advanced key (lists, arrays, tensors, masks,
    several of them) gathers exact values through flat positions, NumPy semantics."""
    if _is_basic(key):
        return _basic_getitem(self, key)
    if isinstance(key, Tensor) and key.dtype not in INTEGER_DTYPES + ("bool",):
        raise IndexError("tensors used as indices must be integer or bool tensors")
    return flat_gather(self, positions_of(self, key))


def setitem(self, key, value):
    """tensor[key] = value for any key. The tensor is rebound to the result: storage is
    immutable, so other views keep the old values (unlike PyTorch)."""
    from .grad_mode import is_grad_enabled
    if self.requires_grad and self.is_leaf and is_grad_enabled():
        raise RuntimeError("a leaf tensor that requires grad cannot be modified in place")
    if not isinstance(value, Tensor):
        value = np.asarray(value)
        if value.dtype.kind not in "biuf":
            raise TypeError("assigned value must be numeric")
    result = overwrite_flat(self, positions_of(self, key), value)
    self._native, self._hook_slot, self._contexts = result._native, result._hook_slot, result._contexts


# ---- views and reshaping -------------------------------------------------------------

def flip(self, dims):
    from .functional import gather
    out = self
    for dim in (dims if isinstance(dims, (tuple, list)) else (dims,)):
        dim = _axis(self, dim)
        out = gather(out, dim, np.arange(self.shape[dim])[::-1].copy())
    return out


def roll(self, shifts, dims=None):
    from .functional import gather
    if dims is None:
        return roll(self.reshape(-1), shifts, 0).reshape(self.shape)
    shifts = shifts if isinstance(shifts, (tuple, list)) else (shifts,)
    dims = dims if isinstance(dims, (tuple, list)) else (dims,)
    out = self
    for shift, dim in zip(shifts, dims):
        dim = _axis(self, dim)
        n = self.shape[dim]
        if n:
            out = gather(out, dim, (np.arange(n) - shift) % n)
    return out


def rot90(self, k=1, dims=(0, 1)):
    k %= 4
    a, b = dims
    if k == 1:
        return flip(self, b).transpose(a, b)
    if k == 2:
        return flip(self, (a, b))
    if k == 3:
        return flip(self.transpose(a, b), b)
    return self


def narrow(self, dim, start, length):
    dim = _axis(self, dim)
    start = start + self.shape[dim] if start < 0 else start
    return self[(slice(None),) * dim + (slice(start, start + length),)]


def select(self, dim, index):
    dim = _axis(self, dim)
    return self[(slice(None),) * dim + (index,)]


def movedim(self, source, destination):
    source = [_axis(self, s) for s in (source if isinstance(source, (tuple, list)) else (source,))]
    destination = [_axis(self, s) for s in (destination if isinstance(destination, (tuple, list)) else (destination,))]
    order = [i for i in range(len(self.shape)) if i not in source]
    for dst, src in sorted(zip(destination, source)):
        order.insert(dst, src)
    return self.permute(order)


def swapaxes(self, axis0, axis1):
    return self.transpose(axis0, axis1)


def unflatten(self, dim, sizes):
    dim = _axis(self, dim)
    return self.reshape(*self.shape[:dim], *sizes, *self.shape[dim + 1:])


def ravel(self):
    return self.reshape(-1)


def broadcast_to(self, shape):
    return self.expand(*shape)


def tile(self, dims):
    dims = tuple(dims)
    dims = (1,) * (len(self.shape) - len(dims)) + dims
    return self.repeat(*dims)


def repeat_interleave(self, repeats, dim=None, *, output_size=None):
    from .functional import gather
    if dim is None:
        return repeat_interleave(self.reshape(-1), repeats, 0)
    dim = _axis(self, dim)
    counts = _index_array(repeats)
    index = np.repeat(np.arange(self.shape[dim]), counts)
    return gather(self, dim, index)


def chunk(self, chunks, dim=0):
    dim = _axis(self, dim)
    size = math.ceil(self.shape[dim] / chunks) if self.shape[dim] else 1
    return self.split(max(size, 1), dim)


def tensor_split(self, indices_or_sections, dim=0):
    dim = _axis(self, dim)
    n = self.shape[dim]
    if isinstance(indices_or_sections, Tensor) and not indices_or_sections.shape:
        indices_or_sections = indices_or_sections.item()
    if isinstance(indices_or_sections, (int, np.integer)):
        sections = indices_or_sections
        base, extra = divmod(n, sections)
        sizes = [base + 1] * extra + [base] * (sections - extra)
        bounds = np.cumsum([0] + sizes)
    else:
        bounds = [0, *[min(max(int(i), 0), n) for i in _index_array(indices_or_sections)], n]
    return tuple(narrow(self, dim, int(a), max(int(b) - int(a), 0)) for a, b in zip(bounds[:-1], bounds[1:]))


def split_with_sizes(self, split_sizes, dim=0):
    return self.split(list(split_sizes), dim)


def hsplit(self, sections):
    return tensor_split(self, sections, 1 if len(self.shape) > 1 else 0)


def vsplit(self, sections):
    return tensor_split(self, sections, 0)


def dsplit(self, sections):
    return tensor_split(self, sections, 2)


def unbind(self, dim=0):
    dim = _axis(self, dim)
    return tuple(select(self, dim, i) for i in range(self.shape[dim]))


# ---- gathers and scatters --------------------------------------------------------------

def index_select(self, dim, index):
    from .functional import gather
    index = _index_array(index)
    if not self.shape:
        _axis(self, dim)
        return flat_gather(self, index.reshape(()) if index.size == 1 else index.reshape(-1))
    return gather(self, _axis(self, dim), index.reshape(-1))


def take(self, index):
    return flat_gather(self, _index_array(index))


def _dim_positions(shape, dim, index):
    """Flat positions in an array of `shape` for torch.gather/scatter-style indices."""
    index = np.asarray(index, np.int64)
    if not len(shape):
        return np.zeros(index.shape, np.int64)
    size = shape[dim]
    if index.size and (index.min() < -size or index.max() >= size):
        raise IndexError("index out of range")
    coords = list(np.indices(index.shape))
    coords[dim] = np.where(index < 0, index + size, index)
    return np.ravel_multi_index(coords, shape) if index.size else np.zeros(index.shape, np.int64)


def gather(self, dim, index):
    dim = _axis(self, dim) if self.shape else 0
    return flat_gather(self, _dim_positions(self.shape, dim, _index_array(index)))


def take_along_dim(self, indices, dim=None):
    if dim is None:
        return flat_gather(self, _index_array(indices).reshape(-1))
    dim = _axis(self, dim)
    index = _index_array(indices)
    shape = list(np.broadcast_shapes(tuple(n if i != dim else 1 for i, n in enumerate(self.shape)),
                                     tuple(n if i != dim else 1 for i, n in enumerate(index.shape))))
    target = shape.copy()
    target[dim] = index.shape[dim]
    expanded = self.expand(*[n if i != dim else self.shape[dim] for i, n in enumerate(shape)])
    return gather(expanded, dim, np.broadcast_to(index, target))


def scatter_add(self, dim, index, src):
    dim = _axis(self, dim) if self.shape else 0
    index = _index_array(index)
    source = src[tuple(slice(0, n) for n in index.shape)] if isinstance(src, Tensor) else src
    return scatter_add_flat(self, _dim_positions(self.shape, dim, index), source)


def scatter(self, dim, index, src=None, *, value=None, reduce=None):
    if src is None:
        src = value
    if reduce == "add":
        return scatter_add(self, dim, index, src)
    if reduce == "multiply":
        raise NotImplementedError("scatter(reduce='multiply') is unsupported")
    dim = _axis(self, dim) if self.shape else 0
    index = _index_array(index)
    source = src[tuple(slice(0, n) for n in index.shape)] if isinstance(src, Tensor) else src
    return overwrite_flat(self, _dim_positions(self.shape, dim, index), source)


def scatter_reduce(self, dim, index, src, reduce, *, include_self=True):
    """sum and mean use the native graph; prod/amax/amin run in NumPy with a custom backward."""
    dim = _axis(self, dim) if self.shape else 0
    index = _index_array(index)
    source = src[tuple(slice(0, n) for n in index.shape)]
    positions = _dim_positions(self.shape, dim, index)
    if reduce in ("sum", "mean"):
        base = self if include_self else overwrite_flat(self, positions, 0.0)
        total = scatter_add_flat(base, positions, source)
        if reduce == "sum":
            return total
        counts = np.zeros(self.numel(), np.float32)
        np.add.at(counts, positions.ravel(), 1.0)
        counts = counts.reshape(self.shape) + (1.0 if include_self else 0.0)
        counts = np.where(counts == 0, 1.0, counts)
        return total / Tensor(counts.astype(np.float32)) if self.dtype == "float32" else Tensor(
            total._array() / counts, dtype=self.dtype)
    function = {"prod": np.multiply, "amax": np.maximum, "amin": np.minimum}[reduce]
    values = self._array().copy().reshape(-1)
    if not include_self:
        initial = {"prod": 1, "amax": -np.inf, "amin": np.inf}[reduce]
        touched = np.unique(positions.ravel())
        values[touched] = initial
    function.at(values, positions.ravel(), source._array().ravel())
    if self.dtype != "float32" or source.dtype != "float32":
        return Tensor(values.reshape(self.shape), dtype=self.dtype)
    from .linalg import _custom
    base, flat_positions, flat_source = self._array().reshape(-1), positions.ravel(), source._array().ravel()
    touched = set(flat_positions.tolist())

    def vjp(g):
        g = g.reshape(-1)
        self_grad, source_grad = np.zeros(base.shape), np.zeros(flat_source.shape)
        # PyTorch's rule: amax/amin split the gradient evenly between tied winners; prod passes
        # the product of the other contributors. shortcut: Python loop per position, fine for small tensors.
        for p in range(base.size):
            members = np.flatnonzero(flat_positions == p)
            keep_self = include_self or p not in touched
            values = np.concatenate([[base[p]] if keep_self else [], flat_source[members]])
            if reduce == "prod":
                parts = np.array([np.prod(np.delete(values, i)) for i in range(values.size)])
            else:
                winners = values == values.max() if reduce == "amax" else values == values.min()
                parts = winners / winners.sum()
            if keep_self:
                self_grad[p], parts = g[p] * parts[0], parts[1:]
            source_grad[members] = g[p] * parts
        return [self_grad.reshape(self.shape), source_grad.reshape(source.shape)]
    return _custom([self, source], values.reshape(self.shape), vjp)


def index_add(self, dim, index, source, *, alpha=1):
    dim = _axis(self, dim) if self.shape else 0
    index = _index_array(index).reshape(-1)
    shape = list(source.shape) if source.shape else [1]
    full = np.broadcast_to(index.reshape([-1 if i == dim else 1 for i in range(len(shape))]), shape) if self.shape else index
    scaled = source * alpha if alpha != 1 else source
    return scatter_add_flat(self, _dim_positions(self.shape, dim, full), scaled.reshape(full.shape) if self.shape else scaled)


def index_copy(self, dim, index, source):
    dim = _axis(self, dim) if self.shape else 0
    index = _index_array(index).reshape(-1)
    if not self.shape:
        return overwrite_flat(self, np.zeros(index.shape, np.int64), source.reshape(index.shape))
    full = np.broadcast_to(index.reshape([-1 if i == dim else 1 for i in range(len(source.shape))]), source.shape)
    return overwrite_flat(self, _dim_positions(self.shape, dim, full), source)


def index_fill(self, dim, index, value):
    dim = _axis(self, dim) if self.shape else 0
    keys = [slice(None)] * len(self.shape)
    if not self.shape:
        return overwrite_flat(self, np.zeros(_index_array(index).size, np.int64), value if not isinstance(value, Tensor) else value.reshape(-1).expand(_index_array(index).size))
    keys[dim] = _index_array(index).reshape(-1)
    positions = positions_of(self, tuple(keys))
    return overwrite_flat(self, positions, value)


def index_put(self, indices, values, accumulate=False):
    positions = positions_of(self, tuple(indices))
    return scatter_add_flat(self, positions, values) if accumulate else overwrite_flat(self, positions, values)


def _region_scatter(self, src, key):
    positions = positions_of(self, key)
    return overwrite_flat(self, positions, src)


def select_scatter(self, src, dim, index):
    dim = _axis(self, dim)
    return _region_scatter(self, src, (slice(None),) * dim + (index,))


def slice_scatter(self, src, dim=0, start=None, end=None, step=1):
    dim = _axis(self, dim)
    return _region_scatter(self, src, (slice(None),) * dim + (slice(start, end, step),))


def diagonal_positions(shape, offset, dim1, dim2):
    grid = np.arange(int(np.prod(shape))).reshape(shape)
    return np.diagonal(grid, offset, dim1, dim2)


def diagonal(self, offset=0, dim1=0, dim2=1):
    return flat_gather(self, diagonal_positions(self.shape, offset, _axis(self, dim1), _axis(self, dim2)))


def diagonal_scatter(self, src, offset=0, dim1=0, dim2=1):
    positions = diagonal_positions(self.shape, offset, _axis(self, dim1), _axis(self, dim2))
    return overwrite_flat(self, positions, src)


def diag_embed(self, offset=0, dim1=-2, dim2=-1):
    n = self.shape[-1] + abs(offset)
    out_rank = len(self.shape) + 1
    shape = list(self.shape[:-1])
    d1, d2 = sorted((dim1 % out_rank, dim2 % out_rank))
    shape.insert(d1, n)
    shape.insert(d2, n)
    base = Tensor(np.zeros(shape, np.float32), dtype=self.dtype) if self.dtype == "float32" else Tensor(np.zeros(shape), dtype=self.dtype)
    positions = diagonal_positions(tuple(shape), offset, dim1 % out_rank, dim2 % out_rank)
    return scatter_add_flat(base, positions, self)


def diag(self, diagonal=0):
    if len(self.shape) == 1:
        n = self.shape[0] + abs(diagonal)
        base = Tensor(np.zeros((n, n)), dtype=self.dtype)
        return scatter_add_flat(base, diagonal_positions((n, n), diagonal, 0, 1), self)
    return globals()["diagonal"](self, diagonal)


def diagflat(self, offset=0):
    return diag(self.reshape(-1), offset)


def trace(self):
    return globals()["diagonal"](self).sum()


def triu(self, diagonal=0):
    keep = np.triu(np.ones(self.shape[-2:], bool), diagonal)
    if self.dtype == "float32":
        return self.masked_fill(~keep, 0.0)
    return Tensor(np.where(keep, self._array(), 0), dtype=self.dtype)


def tril(self, diagonal=0):
    from ._compat import tril as _tril
    if self.dtype in ("float32", "bool"):
        return _tril(self, diagonal)
    return Tensor(np.where(np.tril(np.ones(self.shape[-2:], bool), diagonal), self._array(), 0), dtype=self.dtype)


def nonzero(self, *, as_tuple=False):
    if not self.shape:
        count = int(bool(self.item()))
        if as_tuple:
            return (Tensor(np.zeros(count, np.int64), dtype="int64"),)
        return Tensor(np.zeros((count, 0), np.int64), dtype="int64")
    found = np.nonzero(self._array())
    if as_tuple:
        return tuple(Tensor(i, dtype="int64") for i in found) if self.shape else (Tensor(np.zeros(int(bool(self.item())), np.int64), dtype="int64"),)
    return Tensor(np.stack(found, 1) if found else np.zeros((int(bool(self.item())), 0), np.int64), dtype="int64")


def argwhere(self):
    return nonzero(self)


# ---- products --------------------------------------------------------------------------

def outer(self, vec2):
    return self.reshape(-1, 1) * vec2.reshape(1, -1)


def dot(self, other):
    if len(self.shape) != 1 or len(other.shape) != 1:
        raise RuntimeError("dot expects 1-D tensors")
    return self._dot(other) if self.dtype == other.dtype == "float32" else self @ other


def inner(self, other):
    """Sum over the last dimension of both; shape self.shape[:-1] + other.shape[:-1]."""
    if not self.shape or not other.shape:
        return self * other
    k = self.shape[-1]
    product = self.reshape(-1, k) @ other.reshape(-1, k).t()
    return product.reshape(*self.shape[:-1], *other.shape[:-1])


def mm(self, mat2):
    if len(self.shape) != 2 or len(mat2.shape) != 2:
        raise RuntimeError("mm expects 2-D tensors")
    return self._mm(mat2) if self.dtype == mat2.dtype == "float32" else self @ mat2


def mv(self, vec):
    if len(self.shape) != 2 or len(vec.shape) != 1:
        raise RuntimeError("mv expects a matrix and a vector")
    return self._mv(vec) if self.dtype == vec.dtype == "float32" else self @ vec


def _scaled_sum(self, product, beta, alpha):
    left = self if beta == 1 else (self * 0.0 if beta == 0 else self * beta)
    right = product if alpha == 1 else product * alpha
    return left + right


def addmm(self, mat1, mat2, *, beta=1, alpha=1):
    return _scaled_sum(self, mat1 @ mat2, beta, alpha)


def addmv(self, mat, vec, *, beta=1, alpha=1):
    return _scaled_sum(self, mat @ vec, beta, alpha)


def addr(self, vec1, vec2, *, beta=1, alpha=1):
    return _scaled_sum(self, outer(vec1, vec2), beta, alpha)


def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
    return _scaled_sum(self, batch1.bmm(batch2), beta, alpha)


def addbmm(self, batch1, batch2, *, beta=1, alpha=1):
    return _scaled_sum(self, batch1.bmm(batch2).sum(0), beta, alpha)


def kron(self, other):
    a, b = self, other
    rank = max(len(a.shape), len(b.shape))
    a = a.reshape((1,) * (rank - len(a.shape)) + a.shape)
    b = b.reshape((1,) * (rank - len(b.shape)) + b.shape)
    left = a.reshape([d for n in a.shape for d in (n, 1)])
    right = b.reshape([d for n in b.shape for d in (1, n)])
    return (left * right).reshape([x * y for x, y in zip(a.shape, b.shape)])


def diff(self, n=1, dim=-1, prepend=None, append=None):
    from . import cat
    dim = _axis(self, dim)
    x = self
    if prepend is not None or append is not None:
        parts = [p for p in (prepend, x, append) if p is not None]
        parts = [p if isinstance(p, Tensor) else Tensor(p, dtype=x.dtype) for p in parts]
        x = cat([p.expand(*[s if i != dim else p.shape[dim] if p.shape else 1 for i, s in enumerate(x.shape)]) if p.shape != x.shape and p is not x else p for p in parts], dim)
    for _ in range(n):
        length = x.shape[dim]
        x = narrow(x, dim, 1, length - 1) - narrow(x, dim, 0, length - 1) if x.dtype != "bool" else \
            Tensor(np.diff(x._array(), axis=dim), dtype="bool")
    return x


# ---- like-constructors and fills -------------------------------------------------------

def zeros_like(self, *, dtype=None, requires_grad=False, **_):
    return Tensor(np.zeros(self.shape), dtype=dtype or self.dtype, requires_grad=requires_grad)


def ones_like(self, *, dtype=None, requires_grad=False, **_):
    return Tensor(np.ones(self.shape), dtype=dtype or self.dtype, requires_grad=requires_grad)


def full_like(self, fill_value, *, dtype=None, requires_grad=False, **_):
    return Tensor(np.full(self.shape, fill_value), dtype=dtype or self.dtype, requires_grad=requires_grad)


def empty_like(self, *, dtype=None, requires_grad=False, **_):
    # shortcut: zero-filled; PyTorch leaves "empty" memory uninitialized.
    return zeros_like(self, dtype=dtype, requires_grad=requires_grad)


def fill(self, value):
    """torch.fill: this shape filled with value; self gets a zero gradient, a tensor value
    gets the summed gradient."""
    if self.dtype != "float32":
        return full_like(self, value.item() if isinstance(value, Tensor) else value)
    everything = np.ones(self.shape, bool)
    if isinstance(value, Tensor):
        return self.masked_fill(everything, 0.0) + value.float().expand(*self.shape)
    return self.masked_fill(everything, float(value))


def fill_(self, value):
    setitem(self, ..., value)
    return self


def zero_(self):
    return fill_(self, 0)


def combinations(self, r=2, with_replacement=False):
    picks = (itertools.combinations_with_replacement if with_replacement else itertools.combinations)(range(self.shape[0]), r)
    index = np.array(list(picks), np.int64).reshape(-1, r)
    return flat_gather(self, index) if index.size else self.reshape(-1)[:0].reshape(0, r)


for _name, _function in {
        "__getitem__": getitem, "__setitem__": setitem, "flip": flip, "fliplr": lambda self: flip(self, 1),
        "flipud": lambda self: flip(self, 0), "roll": roll, "rot90": rot90, "narrow": narrow,
        "select": select, "movedim": movedim, "moveaxis": movedim, "swapaxes": swapaxes,
        "swapdims": swapaxes, "unflatten": unflatten, "ravel": ravel, "broadcast_to": broadcast_to,
        "tile": tile, "repeat_interleave": repeat_interleave, "chunk": chunk,
        "tensor_split": tensor_split, "split_with_sizes": split_with_sizes, "hsplit": hsplit,
        "vsplit": vsplit, "dsplit": dsplit, "unbind": unbind, "index_select": index_select,
        "take": take, "gather": gather, "take_along_dim": take_along_dim, "scatter": scatter,
        "scatter_add": scatter_add, "scatter_reduce": scatter_reduce, "index_add": index_add,
        "index_copy": index_copy, "index_fill": index_fill, "index_put": index_put,
        "select_scatter": select_scatter, "slice_scatter": slice_scatter, "diagonal": diagonal,
        "diagonal_scatter": diagonal_scatter, "diag_embed": diag_embed, "diag": diag,
        "diagflat": diagflat, "trace": trace, "triu": triu, "tril": tril, "nonzero": nonzero,
        "argwhere": argwhere, "outer": outer, "ger": outer, "dot": dot, "vdot": dot, "inner": inner,
        "mm": mm, "mv": mv, "addmm": addmm, "addmv": addmv, "addr": addr, "baddbmm": baddbmm,
        "addbmm": addbmm, "kron": kron, "diff": diff, "zeros_like": zeros_like,
        "ones_like": ones_like, "full_like": full_like, "empty_like": empty_like, "fill": fill,
        "fill_": fill_, "zero_": zero_, "combinations": combinations}.items():
    setattr(Tensor, _name, _function)
def _copy_variant(name):
    """x.<name>_copy(...): the same result, never sharing storage with x."""
    def method(self, *args, **kwargs):
        result = getattr(self, name)(*args, **kwargs)
        return tuple(t.clone() for t in result) if isinstance(result, tuple) else result.clone()
    return method


for _name in ("permute", "squeeze", "unsqueeze", "transpose", "t", "view", "expand", "unbind",
              "diagonal", "narrow", "detach", "split_with_sizes", "unfold"):
    setattr(Tensor, _name + "_copy", _copy_variant(_name))


# ---- module-level constructors (no tensor receiver) ----------------------------------

def broadcast_shapes(*shapes):
    return tuple(np.broadcast_shapes(*[tuple(s) if not isinstance(s, int) else (s,) for s in shapes]))


def broadcast_tensors(*tensors):
    shape = broadcast_shapes(*[t.shape for t in tensors])
    return tuple(t.expand(*shape) for t in tensors)


def meshgrid(*tensors, indexing="ij"):
    tensors = tensors[0] if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)) else tensors
    if indexing not in ("ij", "xy"):
        raise ValueError('indexing must be "ij" or "xy"')
    sizes = [t.numel() for t in tensors]
    if indexing == "xy" and len(tensors) >= 2:
        sizes[0], sizes[1] = sizes[1], sizes[0]
    grids = []
    for i, t in enumerate(tensors):
        axis = i if indexing == "ij" or len(tensors) < 2 else {0: 1, 1: 0}.get(i, i)
        shape = [1] * len(tensors)
        shape[axis] = t.numel()
        grids.append(t.reshape(shape).expand(*sizes))
    return tuple(grids)


def block_diag(*tensors):
    from . import cat
    mats = [t.reshape(1, -1) if len(t.shape) < 2 and t.shape else (t.reshape(1, 1) if not t.shape else t) for t in tensors]
    if not mats:
        return Tensor(np.zeros((1, 0), np.float32))
    width = sum(m.shape[1] for m in mats)
    rows, left = [], 0
    for m in mats:
        parts = []
        if left:
            parts.append(Tensor(np.zeros((m.shape[0], left)), dtype=m.dtype))
        parts.append(m)
        right = width - left - m.shape[1]
        if right:
            parts.append(Tensor(np.zeros((m.shape[0], right)), dtype=m.dtype))
        rows.append(cat(parts, 1) if len(parts) > 1 else m)
        left += m.shape[1]
    return cat(rows, 0)


def cartesian_prod(*tensors):
    if len(tensors) == 1:
        return tensors[0]
    from . import stack
    return stack([g.reshape(-1) for g in meshgrid(*tensors, indexing="ij")], 1)


def atleast_1d(*tensors):
    out = tuple(t.reshape(1) if not t.shape else t for t in (tensors[0] if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)) else tensors))
    return out[0] if len(out) == 1 else out


def atleast_2d(*tensors):
    items = tensors[0] if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)) else tensors
    out = tuple(t.reshape(1, 1) if not t.shape else t.reshape(1, -1) if len(t.shape) == 1 else t for t in items)
    return out[0] if len(out) == 1 else out


def atleast_3d(*tensors):
    items = tensors[0] if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)) else tensors
    def lift(t):
        if not t.shape:
            return t.reshape(1, 1, 1)
        if len(t.shape) == 1:
            return t.reshape(1, -1, 1)
        if len(t.shape) == 2:
            return t.unsqueeze(-1)
        return t
    out = tuple(lift(t) for t in items)
    return out[0] if len(out) == 1 else out


def hstack(tensors):
    from . import cat
    items = atleast_1d(*tensors)
    items = items if isinstance(items, tuple) else (items,)
    return cat(items, 0 if len(items[0].shape) == 1 else 1)


def vstack(tensors):
    from . import cat
    items = atleast_2d(*tensors)
    return cat(items if isinstance(items, tuple) else (items,), 0)


def dstack(tensors):
    from . import cat
    items = atleast_3d(*tensors)
    return cat(items if isinstance(items, tuple) else (items,), 2)


def column_stack(tensors):
    from . import cat
    return cat([t.reshape(-1, 1) if len(t.shape) <= 1 else t for t in tensors], 1)


def eye(n, m=None, *, dtype=None, device=None, requires_grad=False):
    return Tensor(np.eye(n, n if m is None else m), dtype=dtype or "float32", requires_grad=requires_grad)


def full(size, fill_value, *, dtype=None, device=None, requires_grad=False):
    size = (size,) if isinstance(size, int) else tuple(size)
    if dtype is None:
        dtype = "bool" if isinstance(fill_value, bool) else "int64" if isinstance(fill_value, int) else "float32"
    return Tensor(np.full(size, fill_value), dtype=dtype, requires_grad=requires_grad)


def linspace(start, end, steps, *, dtype=None, device=None, requires_grad=False):
    start, end = (float(v.item()) if isinstance(v, Tensor) else float(v) for v in (start, end))
    kind = np.float64 if dtype == "float64" else np.float32
    start, end = kind(start), kind(end)
    if steps == 1:
        values = np.array([start], kind)
    else:
        # PyTorch, in the output precision: the first half steps up from start, the second
        # half steps back from end.
        step = (end - start) / kind(steps - 1)
        i = np.arange(steps).astype(kind)
        values = np.where(i < steps // 2, start + step * i, end - step * (kind(steps - 1) - i)).astype(kind)
    return Tensor(values, dtype=dtype or "float32", requires_grad=requires_grad)


def logspace(start, end, steps, base=10.0, *, dtype=None, device=None, requires_grad=False):
    exponents = linspace(start, end, steps).numpy().astype(np.float64)
    return Tensor(np.power(base, exponents).astype(np.float32), dtype=dtype or "float32", requires_grad=requires_grad)


def scalar_tensor(value, *, dtype=None, device=None, requires_grad=False):
    return Tensor(value, dtype=dtype or "float32", requires_grad=requires_grad)


MODULE_FUNCTIONS = {name: globals()[name] for name in (
    "broadcast_shapes", "broadcast_tensors", "meshgrid", "block_diag", "cartesian_prod",
    "atleast_1d", "atleast_2d", "atleast_3d", "hstack", "vstack", "dstack", "column_stack",
    "eye", "full", "linspace", "logspace", "scalar_tensor")}
