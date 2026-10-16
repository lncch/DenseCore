"""PyTorch-named entry points so CPU training scripts written for torch run unchanged."""
import collections
import contextlib
import math
import pickle
import types
import warnings
import numpy as np
from .tensor import Tensor, INTEGER_DTYPES

# Dtype names resolve to the strings Tensor already uses; unsupported ones fail in Tensor().
DTYPES = {"float32": "float32", "float": "float32", "bool": "bool",
          "float16": "float16", "half": "float16", "bfloat16": "bfloat16",
          "float64": "float64", "double": "float64", "uint8": "uint8", "int8": "int8",
          "int16": "int16", "short": "int16", "int32": "int32", "int": "int32",
          "int64": "int64", "long": "int64", "complex64": "complex64", "cfloat": "complex64",
          "complex128": "complex128", "cdouble": "complex128"}


def _size(size):
    return tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else size


def _filled(array, dtype, device, requires_grad):
    return Tensor(array, dtype=dtype or "float32", device=device or "cpu", requires_grad=requires_grad)


def zeros(*size, dtype=None, device=None, requires_grad=False):
    return _filled(np.zeros(_size(size)), dtype, device, requires_grad)


def ones(*size, dtype=None, device=None, requires_grad=False):
    return _filled(np.ones(_size(size)), dtype, device, requires_grad)


def arange(start, end=None, step=1, *, dtype=None, device=None, requires_grad=False):
    """Range like PyTorch: int64 when every argument is an integer, float32 otherwise.

    Float values are start + i*step computed in float64 and rounded once; PyTorch's
    float32 kernel rounds partway, so its values can differ by a few ulp.
    """
    if end is None:
        start, end = 0, start
    if step == 0:
        raise ValueError("arange step must be nonzero")
    if (end - start) * step < 0:
        raise ValueError("arange bounds are inconsistent with the step sign")
    integral = all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in (start, end, step))
    if dtype is None and integral:
        dtype = "int64"
    if dtype in INTEGER_DTYPES:
        if not integral:
            raise ValueError("integer arange requires integer start, end and step")
        return _filled(np.arange(start, end, step, dtype=np.int64), dtype, device, requires_grad)
    count = max(0, math.ceil((end - start) / step))
    return _filled(start + np.arange(count, dtype=np.float64) * step, dtype, device, requires_grad)


def tril(input, diagonal=0):
    """Keep the lower triangle of the last two dimensions; the rest becomes exactly 0."""
    if not isinstance(input, Tensor) or len(input.shape) < 2:
        raise ValueError("tril requires a Tensor with at least 2 dimensions")
    keep = np.tril(np.ones(input.shape[-2:], bool), diagonal)
    if input.dtype == "bool":
        return Tensor(input.numpy() & keep, dtype="bool")
    return input.masked_fill(~keep, 0.0)


def from_numpy(array):
    """Copy a NumPy array into a tensor of the same dtype (PyTorch shares memory instead)."""
    if not isinstance(array, np.ndarray):
        raise TypeError("from_numpy expects a NumPy array")
    if array.dtype.name not in DTYPES:
        raise TypeError(f"from_numpy does not support {array.dtype} arrays")
    return Tensor(array, dtype=array.dtype.name)


def randint(*args, low=None, high=None, size=None, dtype=None, device=None, generator=None,
            requires_grad=False, out=None, layout=None):
    """randint(high, size) or randint(low, high, size) (also by keyword): int64 uniform on
    [low, high), PyTorch's values for the same seed."""
    from ._random import _gen
    args = list(args)
    if size is None:
        if not args:
            raise TypeError("randint() missing required argument 'size'")
        size = args.pop()
    if high is None:
        if not args:
            raise TypeError("randint() missing required argument 'high'")
        high = args.pop()
    if low is None:
        low = args.pop() if args else 0
    if args:
        raise TypeError("randint() takes at most 3 positional arguments")
    return _filled(_gen(generator)._randint(_size((size,)), low, high), dtype or "int64", device, requires_grad)


def multinomial(input, num_samples, replacement=False, *, generator=None):
    """PyTorch's CPU multinomial: without replacement (or one sample), argmax/topk of
    weights / Exp(1) noise; with replacement, a binary search in the normalized cumulative
    sum for each double uniform. Same values as PyTorch for the same seed."""
    from ._random import _gen
    if not isinstance(input, Tensor) or input.dtype not in ("float32", "float64") or len(input.shape) not in (1, 2):
        raise TypeError("multinomial expects a rank-1 or rank-2 float32/float64 Tensor of weights")
    num_samples = int(num_samples)
    weights = np.atleast_2d(input.numpy())
    categories = weights.shape[1]
    if num_samples <= 0:
        raise RuntimeError("cannot sample n_sample <= 0 samples")
    if not replacement and num_samples > categories:
        raise RuntimeError("cannot sample n_sample > prob_dist.size(-1) samples without replacement")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")
    if (weights.sum(1) <= 0).any():
        raise RuntimeError("invalid multinomial distribution (sum of probabilities <= 0)")
    g = _gen(generator)
    if not replacement or num_samples == 1:
        noise = g._sample("exponential", weights.shape, 1.0, dtype=weights.dtype.name)
        q = weights / noise
        if num_samples == 1:
            picked = q.argmax(-1)[:, None]
        else:
            picked = np.argsort(-q, axis=-1, kind="stable")[:, :num_samples]
    else:
        picked = np.empty((len(weights), num_samples), np.int64)
        for row, w in enumerate(weights):
            cum, total = np.empty_like(w), w.dtype.type(0)
            for j, value in enumerate(w):
                total = w.dtype.type(total + value)
                cum[j] = total
            cum = cum / total
            cum[-1] = 1
            u = g._uniform_doubles(num_samples)
            picked[row] = np.searchsorted(cum.astype(np.float64), u, side="left")
    return Tensor(picked if len(input.shape) == 2 else picked[0], dtype="int64")


topk_result = collections.namedtuple("topk", ["values", "indices"])


def topk(input, k, dim=-1, largest=True, sorted=True):
    """The k largest (or smallest) entries along dim with their int64 indices. NaN counts
    as largest, as in PyTorch; ties keep the lower index first. Values carry gradients."""
    from .functional import gather
    if not isinstance(input, Tensor):
        raise TypeError("topk input must be a Tensor")
    rank = max(1, len(input.shape))
    if not -rank <= dim < rank:
        raise IndexError("dimension out of range")
    dim %= rank
    values = np.moveaxis(input._array(), dim, -1)
    if not 0 <= k <= values.shape[-1]:
        raise ValueError("k must lie within the dimension length")
    nan = np.isnan(values)
    key = np.where(nan, 0, values)
    if key.dtype.kind in "iu":
        key = key.astype(np.int64)  # so negation cannot wrap
    # lexsort's last key is primary: NaN first (largest) or last (smallest), then by value.
    order = np.lexsort((-key, ~nan) if largest else (key, nan), axis=-1)[..., :k]
    indices = np.moveaxis(order, -1, dim)
    # Gather values through a flattened view so gradients reach the chosen entries.
    moved = input.permute([i for i in range(rank) if i != dim] + [dim])
    flat = moved.reshape(-1)
    rows = int(np.prod(order.shape[:-1]))
    offsets = (np.arange(rows) * values.shape[-1]).reshape(order.shape[:-1] + (1,))
    picked = gather(flat, 0, order + offsets)
    picked_order = list(range(rank - 1))
    picked_order.insert(dim, rank - 1)
    return topk_result(picked.permute(picked_order), Tensor(indices, dtype="int64"))


class autocast:
    """Mixed precision is unsupported. Only a disabled autocast context is accepted."""
    def __init__(self, device_type="cpu", dtype=None, enabled=True, cache_enabled=None):
        if enabled:
            raise NotImplementedError("autocast is unsupported; DenseCore computes in float32")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class GradScaler:
    """Loss scaling only matters for CUDA float16. Without CUDA, PyTorch disables the
    scaler with a warning; this does the same, so every method passes through."""
    def __init__(self, *args, enabled=True, **kwargs):
        if enabled:
            warnings.warn("GradScaler is enabled, but CUDA is not available. Disabling.")

    def is_enabled(self):
        return False

    def scale(self, outputs):
        return outputs

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


amp = types.SimpleNamespace(autocast=autocast, GradScaler=GradScaler)
# Seeding CUDA is a silent no-op on CPU-only PyTorch too.
cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False,
                             device_count=lambda: 0, amp=amp,
                             manual_seed=lambda seed: None, manual_seed_all=lambda seed: None)
backends = types.SimpleNamespace(
    cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
    cudnn=types.SimpleNamespace(allow_tf32=False))


_LOADABLE = {("densecore.tensor", "_rebuild"), ("collections", "OrderedDict")}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _LOADABLE:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}: "
                                         "only tensors and plain Python containers are allowed")
        return super().find_class(module, name)


def _opened(f, mode):
    return contextlib.nullcontext(f) if hasattr(f, "read" if "r" in mode else "write") else open(f, mode)


def save(obj, f):
    """Pickle obj to a path or binary file. Tensors are stored as raw float32/bool bytes."""
    with _opened(f, "wb") as handle:
        pickle.dump(obj, handle, protocol=4)


def load(f, map_location=None, weights_only=True):
    """Load what save() wrote. Only tensors and plain containers are rebuilt; any other
    pickled class is refused, whatever weights_only says, so loading never runs code."""
    if map_location not in (None, "cpu"):
        raise ValueError("map_location must be None or 'cpu'")
    with _opened(f, "rb") as handle:
        return _Unpickler(handle).load()


def as_tensor(data, dtype=None, device=None):
    """torch.as_tensor: a Tensor of the right dtype is returned as is; anything else is
    converted like torch.tensor (DenseCore copies NumPy data instead of sharing it)."""
    from .tensor import tensor
    if isinstance(data, Tensor):
        return data if dtype is None or data.dtype == dtype else data.to(dtype)
    return tensor(data, dtype=dtype)


def is_tensor(obj):
    return isinstance(obj, Tensor)

