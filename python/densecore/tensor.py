"""Shared immutable float32/bool CPU storage with copying NumPy interoperability."""
import operator
import numbers
import numpy as np
from ._native import _Tensor, _BoolTensor, _TypedTensor

# Storage-only dtypes (views and conversion, no math yet). bfloat16 has no NumPy type.
_STORAGE = {"float64": np.float64, "float16": np.float16, "bfloat16": None, "int8": np.int8,
            "int16": np.int16, "int32": np.int32, "int64": np.int64, "uint8": np.uint8,
            "complex64": np.complex64, "complex128": np.complex128}
COMPLEX_DTYPES = ("complex64", "complex128")
_REAL_OF = {"complex64": "float32", "complex128": "float64"}
DTYPES = ("float32", "bool", *_STORAGE)
INTEGER_DTYPES = ("int8", "int16", "int32", "int64", "uint8")


def _bfloat16_bits(values):
    """Round to bfloat16 bits, nearest-even, as PyTorch does. Going through float32 first
    could round twice, so float32 is made round-to-odd (truncate, then set a sticky bit)."""
    x = np.asarray(values, np.float64)
    with np.errstate(over="ignore", invalid="ignore"):  # values past float32 range become inf on purpose
        f = np.array(x, np.float32, order="C")  # keeps 0-d shape (ascontiguousarray makes it 1-d)
        bits = f.view(np.uint32).copy()
        inexact = (f.astype(np.float64) != x) & np.isfinite(x)
        over = inexact & (np.abs(f.astype(np.float64)) > np.abs(x))
    bits[over] -= 1
    bits[inexact] |= 1
    wide = bits.astype(np.uint64)
    out = np.asarray((wide + 0x7FFF + ((wide >> 16) & 1)) >> 16).astype(np.uint16)  # 0-d stays an array
    out[np.isnan(x)] = 0x7FC0
    return out


def _bfloat16_values(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)

_BITWISE_UFUNCS = {np.bitwise_and: "logical_and", np.bitwise_or: "logical_or",
                   np.bitwise_xor: "logical_xor"}
_COMPARISON_UFUNCS = {np.equal: "eq", np.not_equal: "ne", np.less: "lt",
                      np.less_equal: "le", np.greater: "gt", np.greater_equal: "ge"}

class Tensor:
    """A Rust-owned tensor. Construction and numpy() both copy data."""
    __slots__ = ("_native", "_contexts", "_hook_slot")

    def __init__(self, data, *, dtype="float32", device="cpu", requires_grad=False):
        self._contexts = ()
        self._hook_slot = None
        if dtype not in DTYPES:
            raise TypeError(f"unsupported dtype {dtype!r}; supported: {', '.join(DTYPES)}")
        if device != "cpu":
            raise ValueError("supported device is cpu")
        if type(requires_grad) is not bool:
            raise TypeError("requires_grad must be a bool")
        if requires_grad and dtype != "float32":
            raise ValueError("gradients are supported for float32 tensors only")
        if dtype in _STORAGE:
            source = np.asarray(data)
            if source.dtype.kind not in "biufc":
                raise TypeError("tensor data must be boolean or numeric")
            if (dtype in INTEGER_DTYPES and not isinstance(data, np.ndarray) and source.size
                    and source.dtype.kind in "iu"):
                # Python integers out of range raise (as in PyTorch); array casts wrap like .to().
                info = np.iinfo(_STORAGE[dtype])
                if source.min() < info.min or source.max() > info.max:
                    raise OverflowError(f"value cannot be converted to {dtype} without overflow")
            if dtype == "bfloat16":
                array = _bfloat16_bits(source)
            else:
                array = np.array(source, dtype=np.dtype(_STORAGE[dtype]).newbyteorder("<"), order="C")
            self._native = _TypedTensor(array.tobytes(), list(array.shape), dtype)
        elif dtype == "bool":
            if requires_grad:
                raise ValueError("bool tensors cannot require gradients")
            source = np.asarray(data)
            if source.dtype.kind not in "biufc":
                raise TypeError("bool tensor data must be boolean or numeric")
            array = np.array(source, dtype=np.bool_, order="C", copy=True)
            self._native = _BoolTensor(array.ravel().tolist(), list(array.shape))
        else:
            array = np.asarray(data, dtype=np.dtype("<f4"))
            self._native = _Tensor.from_bytes(array.tobytes(), list(array.shape), requires_grad)
        if requires_grad:
            from .hooks import _HookSlot
            self._hook_slot = _HookSlot(self._native)
            self._contexts = (self._hook_slot,)


    @property
    def shape(self):
        return tuple(self._native.shape)

    @property
    def dtype(self):
        return self._native.dtype

    @property
    def device(self):
        return self._native.device

    def numpy(self):
        if self.dtype in _STORAGE:
            if self.dtype == "bfloat16":
                raise TypeError("NumPy has no bfloat16; convert with .float() first")
            raw = np.frombuffer(self._native.to_bytes(), np.dtype(_STORAGE[self.dtype]).newbyteorder("<"))
            return raw.astype(_STORAGE[self.dtype]).reshape(self.shape)
        if self.dtype == "float32":
            return np.frombuffer(self._native.to_bytes(), "<f4").astype(np.float32).reshape(self.shape)
        return np.array(self._native.values, dtype=self.dtype).reshape(self.shape)

    def tolist(self):
        return self._array().tolist()

    def _flat(self):
        """Logical values as a 1-D NumPy array (works at any rank); bfloat16 widens to float32."""
        if self.dtype == "bfloat16":
            return _bfloat16_values(np.frombuffer(self._native.to_bytes(), "<u2"))
        if self.dtype in _STORAGE:
            return np.frombuffer(self._native.to_bytes(), np.dtype(_STORAGE[self.dtype]).newbyteorder("<"))
        if self.dtype == "float32":
            return np.frombuffer(self._native.to_bytes(), "<f4")
        return np.array(self._native.values, dtype=self.dtype)

    def _array(self):
        """Values as NumPy; bfloat16 widens exactly to float32."""
        return self._flat().astype(np.float32 if self.dtype == "bfloat16" else self.dtype).reshape(self.shape)

    def _cast(self, dtype):
        """Convert to another dtype (a detached copy; casts are not differentiable yet)."""
        if dtype not in DTYPES:
            raise TypeError(f"unsupported dtype {dtype!r}")
        return self if dtype == self.dtype else Tensor(self._array(), dtype=dtype)

    def float(self, memory_format=None):
        return self._cast("float32")

    def double(self, memory_format=None):
        return self._cast("float64")

    def half(self, memory_format=None):
        return self._cast("float16")

    def bfloat16(self, memory_format=None):
        return self._cast("bfloat16")

    def long(self, memory_format=None):
        return self._cast("int64")

    def int(self, memory_format=None):
        return self._cast("int32")

    def bool(self, memory_format=None):
        return self._cast("bool")

    def byte(self, memory_format=None):
        return self._cast("uint8")

    def char(self, memory_format=None):
        return self._cast("int8")

    def short(self, memory_format=None):
        return self._cast("int16")

    def clone(self, memory_format=None):
        """A separate tensor with the same values; gradients flow back to self."""
        if self.dtype != "float32":
            return self.reshape(self.shape)
        return self._wrap(self._native.clone_contiguous(), parents=(self,))

    def view_as(self, other):
        return self.view(other.shape)

    def reshape_as(self, other):
        return self.reshape(other.shape)

    def expand(self, *sizes):
        """Broadcast to a larger shape (-1 keeps a dimension). Returns a copy, not a view;
        float32 gradients sum back over the broadcast dimensions."""
        sizes = tuple(sizes[0]) if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)) else sizes
        lead = len(sizes) - len(self.shape)
        if lead < 0:
            raise RuntimeError("expand needs at least as many sizes as tensor dimensions")
        target = []
        for i, size in enumerate(sizes):
            size = operator.index(size)
            if i < lead:
                if size < 0:
                    raise RuntimeError("new leading expand dimensions cannot be -1")
                target.append(size)
                continue
            current = self.shape[i - lead]
            size = current if size == -1 else size
            if current != size and current != 1:
                raise RuntimeError(f"cannot expand dimension of size {current} to {size}")
            target.append(size)
        target = tuple(target)
        if target == self.shape:
            return self
        if self.dtype == "float32":
            return self * Tensor(np.ones(target, np.float32))
        return Tensor(np.broadcast_to(self._array(), target), dtype=self.dtype)

    def expand_as(self, other):
        return self.expand(other.shape)

    def repeat(self, *sizes):
        """Tile the tensor; sizes may add leading dimensions."""
        sizes = tuple(sizes[0]) if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)) else sizes
        if len(sizes) < len(self.shape):
            raise RuntimeError("repeat needs at least as many sizes as tensor dimensions")
        shape = (1,) * (len(sizes) - len(self.shape)) + self.shape
        interleaved = self.reshape([d for n in shape for d in (1, n)])
        tiled = interleaved.expand([d for r, n in zip(sizes, shape) for d in (r, n)])
        return tiled.reshape([r * n for r, n in zip(sizes, shape)])

    def sum_to_size(self, *size):
        size = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
        lead = len(self.shape) - len(size)
        x = self.sum(dim=tuple(range(lead))) if lead > 0 else self
        dims = tuple(i for i, n in enumerate(size) if n == 1 and x.shape[i] != 1)
        return x.sum(dim=dims, keepdim=True) if dims else x

    def new_zeros(self, *size, dtype=None, device=None, requires_grad=False):
        size = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
        return Tensor(np.zeros(size), dtype=dtype or self.dtype, requires_grad=requires_grad)

    def new_ones(self, *size, dtype=None, device=None, requires_grad=False):
        size = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
        return Tensor(np.ones(size), dtype=dtype or self.dtype, requires_grad=requires_grad)

    def new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad=False):
        return Tensor(np.full(tuple(size), fill_value), dtype=dtype or self.dtype, requires_grad=requires_grad)

    def new_empty(self, *size, dtype=None, device=None, requires_grad=False):
        # shortcut: "empty" memory is zero-filled; PyTorch leaves it uninitialized.
        return self.new_zeros(*size, dtype=dtype, requires_grad=requires_grad)

    @property
    def mT(self):
        return self.transpose(-2, -1)

    mH = mT  # real dtypes only, so the conjugate is a no-op

    @property
    def H(self):
        if len(self.shape) > 2:
            raise RuntimeError("H is only defined for tensors with at most 2 dimensions")
        return self.t()

    def add(self, other, *, alpha=1):
        return self + (other * alpha if alpha != 1 else other)

    def sub(self, other, *, alpha=1):
        return self - (other * alpha if alpha != 1 else other)

    def mul(self, other):
        return self * other

    def div(self, other, *, rounding_mode=None):
        """True division, or rounded with "trunc"/"floor" (rounded results have no gradient)."""
        if rounding_mode is None:
            return self / other
        if rounding_mode not in ("trunc", "floor"):
            raise ValueError('rounding_mode must be None, "trunc" or "floor"')
        quotient = self / other
        if quotient.dtype not in ("float32", "float64", "float16", "bfloat16"):
            return self // other if rounding_mode == "floor" else quotient
        # Native floor/trunc keep the graph with a zero gradient, as in PyTorch.
        return quotient.floor() if rounding_mode == "floor" else quotient.trunc()

    multiply = mul
    subtract = sub
    divide = div
    true_divide = div

    def __getitem__(self, key):
        """Basic keys: integers, positive-step slices, None and Ellipsis give native views.
        Advanced keys are handled in shapes.getitem, which replaces this method."""
        keys = key if isinstance(key, tuple) else (key,)
        if any(item is None for item in keys):
            # Index without the Nones, then insert each new axis at its output position.
            plain = tuple(item for item in keys if item is not None)
            result = self[plain]
            consumed = sum(1 for item in plain if item is not Ellipsis)
            position = 0
            for item in keys:
                if item is None:
                    result = result.unsqueeze(position)
                    position += 1
                elif item is Ellipsis:
                    position += len(self.shape) - consumed
                elif isinstance(item, slice):
                    position += 1
                elif isinstance(item, (list, np.ndarray, Tensor)):
                    position += np.ndim(item.numpy() if isinstance(item, Tensor) else item)
            return result
        rank = len(self.shape)
        ellipses = sum(item is Ellipsis for item in keys)
        if ellipses > 1:
            raise IndexError("only one ellipsis is permitted")
        if len(keys) - ellipses > rank:
            raise IndexError("too many indices for tensor")
        if ellipses:
            expanded = []
            for item in keys:
                expanded.extend([slice(None)] * (rank - len(keys) + 1) if item is Ellipsis else [item])
            keys = tuple(expanded)
        else:
            keys += (slice(None),) * (rank - len(keys))
        specs = []
        limit = np.iinfo(np.uintp).max
        for dim, item in zip(self.shape, keys):
            if isinstance(item, slice):
                for bound in (item.start, item.stop, item.step):
                    if isinstance(bound, (bool, np.bool_)):
                        raise TypeError("slice bounds must be integers, not booleans")
                start, stop, step = item.indices(dim)
                if step <= 0:
                    raise ValueError("only positive slice steps are supported")
                if step > limit:
                    raise ValueError("slice step exceeds platform size limit")
                length = max(0, (stop - start + step - 1) // step)
                specs.append((start, step, length, False))
            else:
                if isinstance(item, (bool, np.bool_)):
                    raise TypeError("boolean indexing is unsupported")
                try:
                    index = operator.index(item)
                except TypeError:
                    raise TypeError("indices must be integers, slices or ellipsis") from None
                if index < 0:
                    index += dim
                if not 0 <= index < dim:
                    raise IndexError("tensor index out of bounds")
                specs.append((index, 1, 1, True))
        return self._wrap(self._native.slice(specs), parents=(self,))

    def numel(self):
        return self._native.numel()

    def size(self, dim=None):
        """PyTorch-style shape access: the shape tuple, or the length of one dimension."""
        if dim is None:
            return self.shape
        if isinstance(dim, (bool, np.bool_)):
            raise TypeError("dim must be an integer")
        dim = operator.index(dim)
        rank = len(self.shape)
        if not -rank <= dim < rank:
            raise IndexError("dimension out of range")
        return self.shape[dim]

    def dim(self):
        return len(self.shape)

    ndim = property(dim)

    def view(self, *shape):
        """Alias of reshape. Storage is immutable, so where PyTorch would refuse a
        non-contiguous view this copies instead; values are the same either way."""
        return self.reshape(*shape)

    def t(self):
        """Transpose a rank-2 tensor; rank 0 and 1 are returned unchanged."""
        if len(self.shape) > 2:
            raise ValueError("t() expects a tensor with at most 2 dimensions")
        return self if len(self.shape) < 2 else self.transpose()

    def to(self, *args, **kwargs):
        """Move to cpu (a no-op) and/or convert to another dtype."""
        unknown = set(kwargs) - {"device", "dtype", "non_blocking", "copy", "memory_format"}
        if unknown:
            raise TypeError(f"unexpected to() arguments: {sorted(unknown)}")
        result = self
        for target in (*args, kwargs.get("device"), kwargs.get("dtype")):
            if type(target).__name__ == "memory_format":
                continue
            if isinstance(target, Tensor):
                target = target.dtype
            if target is None or isinstance(target, bool) or target == "cpu":
                continue
            if target not in DTYPES:
                raise ValueError(f"unsupported to() target {target!r}: only the cpu device exists")
            result = result._cast(target)
        return result

    def item(self):
        if self.numel() != 1:
            raise RuntimeError(f"a Tensor with {self.numel()} elements cannot be converted to a scalar")
        value = self._flat()[0].item()
        if self.dtype in INTEGER_DTYPES:
            return int(value)
        return bool(value) if self.dtype == "bool" else float(value)

    def __reduce__(self):
        # Pickle as raw bytes so loading needs only _rebuild (see densecore.load).
        data = self._native.to_bytes() if self.dtype in _STORAGE else self.numpy().tobytes()
        return (_rebuild, (data, self.shape, self.dtype, self.requires_grad and self.is_leaf))

    def stride(self):
        """Return element strides in backing storage."""
        return tuple(self._native.strides)

    def storage_offset(self):
        return self._native.storage_offset

    def is_contiguous(self):
        return self._native.is_contiguous()

    def shares_storage(self, other):
        """Whether tensors reference the same immutable backing allocation."""
        if not isinstance(other, Tensor):
            raise TypeError("other must be a Tensor")
        if self.dtype != other.dtype:
            return False
        return self._native.shares_storage(other._native)

    def contiguous(self):
        """Materialize strided values; already contiguous tensors share storage."""
        return self if self.is_contiguous() else self.clone()

    def detach(self):
        return self._wrap(self._native.detach(), parents=(self,))

    def unsqueeze(self, dim):
        """Insert a singleton axis using existing reshape storage semantics."""
        if isinstance(dim, (bool, np.bool_)):
            raise TypeError("unsqueeze dimension must be an integer, not boolean")
        dim = operator.index(dim)
        rank = len(self.shape) + 1
        if not -rank <= dim < rank:
            raise IndexError("unsqueeze dimension out of range")
        dim %= rank
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    def squeeze(self, dim=None):
        """Remove singleton axes; an optional integer selects one axis."""
        shape = self.shape
        if isinstance(dim, (tuple, list)):
            rank = max(1, len(shape))
            dims = {operator.index(d) % rank for d in dim}
            result = tuple(n for i, n in enumerate(shape) if not (i in dims and n == 1))
        elif dim is None:
            result = tuple(size for size in shape if size != 1)
        else:
            if isinstance(dim, (bool, np.bool_)):
                raise TypeError("squeeze dimension must be an integer, not boolean")
            dim = operator.index(dim)
            rank = max(1, len(shape))
            if not -rank <= dim < rank:
                raise IndexError("squeeze dimension out of range")
            dim %= rank
            if not shape or shape[dim] != 1:
                return self
            result = shape[:dim] + shape[dim + 1:]
        return self if result == shape else self.reshape(result)

    def flatten(self, start_dim=0, end_dim=-1):
        """Collapse an inclusive dimension range using checked reshape semantics."""
        axes = []
        rank = max(1, len(self.shape))
        for axis in (start_dim, end_dim):
            if isinstance(axis, (bool, np.bool_)):
                raise TypeError("flatten dimensions must be integers, not booleans")
            axis = operator.index(axis)
            if not -rank <= axis < rank:
                raise IndexError("flatten dimension out of range")
            axes.append(axis % rank)
        start, end = axes
        if start > end:
            raise ValueError("flatten start_dim must not exceed end_dim")
        if not self.shape:
            return self.reshape(1)
        if start == end:
            return self
        import math
        shape = self.shape
        return self.reshape(shape[:start] + (math.prod(shape[start:end + 1]),) + shape[end + 1:])

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        dims = []
        for dim in shape:
            if isinstance(dim, (bool, np.bool_)):
                raise TypeError("shape dimensions must be integers, not booleans")
            try:
                dim = operator.index(dim)
            except TypeError:
                raise TypeError("shape dimensions must be integers") from None
            if dim < -1:
                raise ValueError("shape dimensions must be nonnegative or -1")
            if dim > np.iinfo(np.uintp).max:
                raise ValueError("shape dimension exceeds platform size limit")
            dims.append(dim)
        if dims.count(-1) > 1:
            raise ValueError("only one inferred dimension is permitted")
        if -1 in dims:
            known = 1
            for dim in dims:
                if dim != -1:
                    known *= dim
            count = self.numel()
            if known == 0 or count % known:
                raise ValueError("cannot infer reshape dimension")
            inferred = count // known
            if inferred > np.iinfo(np.uintp).max:
                raise ValueError("inferred dimension exceeds platform size limit")
            dims[dims.index(-1)] = inferred
        return self._wrap(self._native.reshape(dims), parents=(self,))

    @property
    def requires_grad(self):
        return self._native.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self.requires_grad_(value)

    def requires_grad_(self, requires_grad=True):
        """PyTorch's rules: only leaves may change, and only float32 tensors can require grad.
        shortcut: the leaf is rebuilt from its values, so views taken earlier stop sharing it."""
        requires_grad = bool(requires_grad)
        if requires_grad == self.requires_grad:
            return self
        if not self.is_leaf:
            raise RuntimeError("you can only change requires_grad flags of leaf variables.")
        if requires_grad and self.dtype != "float32":
            raise RuntimeError("only Tensors of floating point dtype can require gradients")
        self._native = Tensor(self._array(), dtype=self.dtype, requires_grad=requires_grad)._native
        return self

    @property
    def is_leaf(self):
        return self._native.is_leaf

    @property
    def grad(self):
        value = self._native.grad
        return None if value is None else self._wrap(value, parents=(self,))

    @grad.setter
    def grad(self, value):
        if self.dtype != "float32":
            raise RuntimeError(f"{self.dtype} tensors do not have gradients")
        if value is not None and (not isinstance(value, Tensor) or value.dtype != "float32"):
            raise TypeError("grad must be a float32 Tensor or None")
        self._native.set_grad(None if value is None else value._native)

    def split(self, split_size_or_sections, dim=0):
        """Views of consecutive chunks along dim: equal chunks of a size, or listed sizes."""
        if isinstance(dim, (bool, np.bool_)):
            raise TypeError("dim must be an integer")
        dim = operator.index(dim)
        rank = len(self.shape)
        if not -rank <= dim < rank:
            raise IndexError("dimension out of range")
        dim %= rank
        length = self.shape[dim]
        if isinstance(split_size_or_sections, (list, tuple)):
            sizes = [operator.index(n) for n in split_size_or_sections]
            if any(n < 0 for n in sizes) or sum(sizes) != length:
                raise ValueError("split sizes must be nonnegative and sum to the dimension length")
        else:
            size = operator.index(split_size_or_sections)
            if size <= 0 and length:
                raise ValueError("split size must be positive")
            sizes = [min(size, length - start) for start in range(0, length, size)] if length else [0]
        chunks, start = [], 0
        for n in sizes:
            key = (slice(None),) * dim + (slice(start, start + n),)
            chunks.append(self[key])
            start += n
        return tuple(chunks)

    def copy_(self, src, non_blocking=False):
        """Copy src's values into this tensor, cast to its dtype and broadcast to its shape."""
        if not isinstance(src, Tensor):
            raise TypeError("copy_ source must be a Tensor")
        values = np.broadcast_to(src._array(), self.shape)
        if self.requires_grad:
            from .grad_mode import is_grad_enabled
            if is_grad_enabled() or not self.is_leaf:
                raise RuntimeError("copy_ into a tensor that requires grad needs a leaf under no_grad()")
            from .nn import _initialize
            return _initialize(self, values)
        self._native = Tensor(values, dtype=self.dtype)._native
        return self

    def zero_grad(self):
        if self.dtype != "float32":
            raise RuntimeError(f"{self.dtype} tensors do not have gradients")
        self._native.zero_grad()

    def backward(self, gradient=None, *, retain_graph=False):
        if self.dtype != "float32":
            raise RuntimeError(f"{self.dtype} tensors do not require gradients")
        if gradient is not None and not isinstance(gradient, Tensor):
            raise TypeError("gradient must be a Tensor or None")
        if type(retain_graph) is not bool:
            raise TypeError("retain_graph must be a bool")
        from .function import _callback_state
        if getattr(_callback_state, "hook_active", False):
            raise RuntimeError("reentrant backward from tensor hooks is unsupported")
        from .hooks import _active_slots
        if self._native.has_custom or _active_slots:
            self._sync_hook_slot()
            from .function import _backward
            return _backward(self, gradient, retain_graph)
        self._native.backward(None if gradient is None else gradient._native,
                              retain_graph=retain_graph)

    def register_hook(self, hook):
        """Register an ordered first-order gradient hook; return a removable handle."""
        from .hooks import register
        return register(self, hook)

    def _sync_hook_slot(self):
        if self._hook_slot is not None:
            self._hook_slot.sync(self._native)

    @staticmethod
    def _wrap(native, *, parents=(), contexts=None):
        result = object.__new__(Tensor)
        result._native = native
        result._hook_slot = None
        if native.requires_grad:
            from .hooks import _HookSlot
            owners = Tensor._parent_contexts(parents) if contexts is None else contexts
            if native.has_custom and not owners:
                raise RuntimeError("custom autograd context ownership was not propagated")
            result._hook_slot = _HookSlot(native)
            result._contexts = (result._hook_slot, owners) if owners else (result._hook_slot,)
        else:
            result._contexts = ()
        return result

    @staticmethod
    def _parent_contexts(parents):
        owners = []
        for parent in parents:
            if parent is not None:
                parent._sync_hook_slot()
            if parent is not None and parent._contexts and all(parent._contexts is not old for old in owners):
                owners.append(parent._contexts)
        if len(owners) == 1:
            return owners[0]
        return tuple(owners)

    def _float_native(self):
        if self.dtype != "float32":
            raise TypeError("operation requires float32 tensors")
        return self._native

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __iter__(self):
        if not self.shape:
            raise TypeError("iteration over a 0-d tensor")
        return (self[i] for i in range(self.shape[0]))

    def __index__(self):
        if self.dtype not in INTEGER_DTYPES or self.numel() != 1:
            raise TypeError("only single-element integer tensors can be used as an index")
        return self.item()

    def __int__(self):
        return int(self.item())

    def __float__(self):
        return float(self.item())

    def __format__(self, spec):
        return format(self.item(), spec) if self.numel() == 1 else object.__format__(self, spec)

    def __bool__(self):
        if self.numel() != 1:
            raise RuntimeError("Boolean value of empty or multi-element Tensor is ambiguous")
        return bool(self.item())

    def _truth(self):
        """Nonzero as a bool tensor (PyTorch's logical ops accept any dtype)."""
        return self if self.dtype == "bool" else Tensor(self._array() != 0, dtype="bool")

    def logical_not(self):
        truth = self._truth()
        return truth._wrap(truth._native.logical_not(), parents=(truth,))

    def _compare(self, other, operation):
        if self.dtype == "bool":
            if isinstance(other, (bool, np.bool_)):
                other = Tensor(other, dtype="bool")
            if not isinstance(other, Tensor) or other.dtype != "bool":
                raise TypeError("bool comparisons require bool tensors or boolean scalars")
            return self._wrap(self._native.compare(other._native, operation), parents=(self,))
        if isinstance(other, np.bool_):
            other = bool(other)
        dtypes = {self.dtype} | ({other.dtype} if isinstance(other, Tensor) else set())
        if dtypes & set(_STORAGE) and "bool" not in dtypes and isinstance(other, (Tensor, numbers.Real)):
            right = other._array() if isinstance(other, Tensor) else other
            return Tensor(_COMPARE[operation](self._array(), right), dtype="bool")
        other = self._operand(other)
        if other is NotImplemented:
            raise TypeError("comparison operand must be a float32 Tensor or real scalar")
        return self._wrap(self._float_native().compare(other._float_native(), operation), parents=(self,))

    def eq(self, other):
        return self._compare(other, "eq")

    def ne(self, other):
        return self._compare(other, "ne")

    def lt(self, other):
        return self._compare(other, "lt")

    def le(self, other):
        return self._compare(other, "le")

    def gt(self, other):
        return self._compare(other, "gt")

    def ge(self, other):
        return self._compare(other, "ge")

    # Equality returns a mask; Parameter containers retain identity-based hashing.
    __hash__ = object.__hash__

    def _comparison_operator(self, other, operation):
        if not isinstance(other, (Tensor, numbers.Real, np.bool_)):
            return NotImplemented
        return self._compare(other, operation)

    def __eq__(self, other):
        return self._comparison_operator(other, "eq")

    def __ne__(self, other):
        return self._comparison_operator(other, "ne")

    def __lt__(self, other):
        return self._comparison_operator(other, "lt")

    def __le__(self, other):
        return self._comparison_operator(other, "le")

    def __gt__(self, other):
        return self._comparison_operator(other, "gt")

    def __ge__(self, other):
        return self._comparison_operator(other, "ge")

    def _logical(self, other, operation, strict=False):
        """logical_and/or/xor: tensors of any dtype count nonzero as True (PyTorch). The &, |,
        ^ operators are strict: bool tensors and bool scalars only, since for PyTorch they
        are bitwise ops that reject floats."""
        if isinstance(other, (bool, np.bool_)):
            other = Tensor(other, dtype="bool")
        if not isinstance(other, Tensor) or (strict and (self.dtype != "bool" or other.dtype != "bool")):
            raise TypeError("logical operations need a Tensor (or a bool scalar); &, |, ^ need bool tensors")
        left, right = self._truth(), other._truth()
        return left._wrap(left._native.logical(right._native, operation), parents=(left,))

    def logical_and(self, other):
        return self._logical(other, "logical_and")

    def logical_or(self, other):
        return self._logical(other, "logical_or")

    def logical_xor(self, other):
        return self._logical(other, "logical_xor")

    __and__ = __rand__ = lambda self, other: self._logical(other, "logical_and", strict=True)
    __or__ = __ror__ = lambda self, other: self._logical(other, "logical_or", strict=True)
    __xor__ = __rxor__ = lambda self, other: self._logical(other, "logical_xor", strict=True)
    __invert__ = logical_not

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__" and not kwargs and len(inputs) == 2:
            if ufunc in _COMPARISON_UFUNCS or ufunc in _BITWISE_UFUNCS:
                kinds = "b" if ufunc in _BITWISE_UFUNCS else "biuf"
                inputs = tuple(x.item() if isinstance(x, np.ndarray) and x.ndim == 0
                               and x.dtype.kind in kinds else x for x in inputs)
            left, right = inputs
            if ufunc in _BITWISE_UFUNCS:
                if isinstance(left, Tensor):
                    return left._logical(right, _BITWISE_UFUNCS[ufunc])
                if isinstance(right, Tensor):
                    return right._logical(left, _BITWISE_UFUNCS[ufunc])
            if (ufunc in _COMPARISON_UFUNCS
                    and all(isinstance(x, (Tensor, numbers.Real, np.bool_)) for x in inputs)):
                operation = _COMPARISON_UFUNCS[ufunc]
                if isinstance(left, Tensor):
                    return left._compare(right, operation)
                if isinstance(right, Tensor):
                    reflected = {"eq": "eq", "ne": "ne", "lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}
                    return right._compare(left, reflected[operation])
            if (ufunc in (np.add, np.multiply, np.subtract, np.true_divide)
                    and all(isinstance(x, (Tensor, numbers.Real)) for x in inputs)):
                left, right = (self._operand(x) for x in inputs)
                if ufunc is np.add:
                    return left + right
                if ufunc is np.multiply:
                    return left * right
                return left - right if ufunc is np.subtract else left / right
        raise TypeError("NumPy array arithmetic is unsupported; convert operands with tensor()")

    def _mixed(self, other, op, reflected=False):
        """Arithmetic that involves a storage dtype (int64, float64, ...), with PyTorch's
        promotion. Returns None when only float32/bool take part, so the native path (and
        its errors) applies. float32 mixed with another dtype runs natively in float32."""
        if not isinstance(other, (Tensor, numbers.Number)) or isinstance(other, (bool, np.bool_)):
            return None
        dtypes = {self.dtype} | ({other.dtype} if isinstance(other, Tensor) else set())
        complex_scalar = isinstance(other, numbers.Complex) and not isinstance(other, numbers.Real)
        if (not dtypes & set(_STORAGE) and not complex_scalar) or "bool" in dtypes:
            return None
        function = _ARITHMETIC[op]
        if dtypes & set(COMPLEX_DTYPES) or complex_scalar:
            return _numpy_binary(self, other, function, reflected)
        if "float32" in dtypes and "float64" in dtypes:
            from .grad_mode import is_grad_enabled
            if is_grad_enabled() and any(isinstance(t, Tensor) and t.requires_grad for t in (self, other)):
                raise RuntimeError("float32 with float64 promotes to float64, which has no gradients yet")
        elif "float32" in dtypes:
            if op in ("floordiv", "mod"):
                raise TypeError(f"{op} is unsupported for float32 tensors")
            left, right = (t if t.dtype == "float32" else t.float() for t in (self, other))
            return function(right, left) if reflected else function(left, right)
        return _numpy_binary(self, other, function, reflected)

    @staticmethod
    def _operand(other):
        if isinstance(other, Tensor):
            return other
        if isinstance(other, numbers.Real):
            return Tensor(other)
        return NotImplemented

    def __add__(self, other):
        mixed = self._mixed(other, "add")
        if mixed is not None:
            return mixed
        other = self._operand(other)
        if other is NotImplemented:
            return NotImplemented
        return self._wrap(self._float_native().add(other._native), parents=(self, other))

    __radd__ = __add__

    def __mul__(self, other):
        mixed = self._mixed(other, "mul")
        if mixed is not None:
            return mixed
        other = self._operand(other)
        if other is NotImplemented:
            return NotImplemented
        return self._wrap(self._float_native().mul(other._native), parents=(self, other))

    __rmul__ = __mul__

    def __sub__(self, other):
        mixed = self._mixed(other, "sub")
        if mixed is not None:
            return mixed
        other = self._operand(other)
        if other is NotImplemented:
            return NotImplemented
        return self._wrap(self._float_native().sub(other._native), parents=(self, other))

    def __rsub__(self, other):
        mixed = self._mixed(other, "sub", reflected=True)
        if mixed is not None:
            return mixed
        other = self._operand(other)
        if other is NotImplemented:
            return NotImplemented
        return other.__sub__(self)

    def __truediv__(self, other):
        mixed = self._mixed(other, "truediv")
        if mixed is not None:
            return mixed
        other = self._operand(other)
        if other is NotImplemented:
            return NotImplemented
        return self._wrap(self._float_native().div(other._native), parents=(self, other))

    def __rtruediv__(self, other):
        mixed = self._mixed(other, "truediv", reflected=True)
        if mixed is not None:
            return mixed
        other = self._operand(other)
        if other is NotImplemented:
            return NotImplemented
        return other.__truediv__(self)

    def __floordiv__(self, other):
        mixed = self._mixed(other, "floordiv")
        if mixed is None:
            return self.floor_divide(other)  # float32, as torch: x // y == floor(x / y)
        return mixed

    def __mod__(self, other):
        mixed = self._mixed(other, "mod")
        if mixed is None:
            return self.remainder(other)  # float32, as torch: Python-style sign
        return mixed

    def __rfloordiv__(self, other):
        if isinstance(other, Tensor):
            return other.floor_divide(self)
        mixed = self._mixed(other, "floordiv", reflected=True)
        return mixed if mixed is not None else Tensor(np.asarray(other, np.float32)).floor_divide(self)

    def __rmod__(self, other):
        if isinstance(other, Tensor):
            return other.remainder(self)
        mixed = self._mixed(other, "mod", reflected=True)
        return mixed if mixed is not None else Tensor(np.asarray(other, np.float32)).remainder(self)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __rmatmul__(self, other):
        return other.__matmul__(self) if isinstance(other, Tensor) else NotImplemented

    def __matmul__(self, other):
        """torch.matmul: 1-D operands get a temporary dimension, batch dimensions broadcast."""
        if not isinstance(other, Tensor):
            return NotImplemented
        if not self.shape or not other.shape:
            raise RuntimeError("both arguments to matmul need to be at least 1-D")
        if self.dtype in _STORAGE or other.dtype in _STORAGE:
            if self.dtype != other.dtype:
                raise RuntimeError(f"matmul needs matching dtypes, got {self.dtype} and {other.dtype}")
            return _numpy_binary(self, other, np.matmul)
        return _matmul_impl(self, other)

    def _mm(self, other):
        """Native 2-D product (PyTorch's addmm_impl_cpu_ / sgemm_ path)."""
        return self._wrap(self._float_native().matmul(other._native), parents=(self, other))

    def _mv(self, vec):
        return self._wrap(self._float_native().mv(vec._float_native()), parents=(self, vec))

    def _dot(self, other):
        return self._wrap(self._float_native().dot(other._float_native()), parents=(self, other))

    def _mse_loss(self, target):
        if not isinstance(target, Tensor):
            raise TypeError("MSE target must be a Tensor")
        return self._wrap(self._float_native().mse_loss(target._native), parents=(self, target))

    def bmm(self, other):
        """Multiply equal-batch rank-3 tensors; no broadcasting."""
        if not isinstance(other, Tensor):
            raise TypeError("bmm operand must be a Tensor")
        return self._wrap(self._float_native().bmm(other._native), parents=(self, other))

    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("matmul operand must be a Tensor")
        return self @ other

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        rank = len(self.shape)
        if len(dims) != rank:
            raise ValueError("permutation rank mismatch")
        normalized = []
        for dim in dims:
            if isinstance(dim, (bool, np.bool_)):
                raise TypeError("permutation axes must be integers")
            try:
                dim = operator.index(dim)
            except TypeError:
                raise TypeError("permutation axes must be integers") from None
            if not -rank <= dim < rank:
                raise ValueError("permutation axis out of bounds")
            normalized.append(dim % rank)
        if len(set(normalized)) != rank:
            raise ValueError("duplicate permutation axes")
        return self._wrap(self._native.permute(normalized), parents=(self,))

    def transpose(self, dim0=None, dim1=None):
        if dim0 is None and dim1 is None:
            if len(self.shape) != 2:
                raise ValueError("transpose without axes requires rank 2")
            return self._wrap(self._native.transpose(), parents=(self,))
        if dim0 is None or dim1 is None:
            raise TypeError("transpose requires both axes")
        rank = len(self.shape)
        normalized = []
        for axis in (dim0, dim1):
            if isinstance(axis, (bool, np.bool_)):
                raise TypeError("transpose axes must be integers")
            try:
                axis = operator.index(axis)
            except TypeError:
                raise TypeError("transpose axes must be integers") from None
            if not -rank <= axis < rank:
                raise ValueError("transpose axis out of bounds")
            normalized.append(axis % rank)
        axes = list(range(rank))
        a, b = normalized
        axes[a], axes[b] = axes[b], axes[a]
        return self.permute(axes)

    @property
    def T(self):
        return self.transpose()

    def _softmax_axis(self, axis):
        if isinstance(axis, (bool, np.bool_)):
            raise TypeError("axis must be an integer")
        try:
            axis = operator.index(axis)
        except TypeError:
            raise TypeError("axis must be an integer") from None
        rank = len(self.shape)
        if not -rank <= axis < rank:
            raise ValueError("softmax axis out of range")
        return axis % rank

    def softmax(self, dim=-1, dtype=None, *, axis=None):
        source = self if dtype in (None, self.dtype) else self.to(dtype)
        native = source._float_native().softmax(source._softmax_axis(dim if axis is None else axis))
        return source._wrap(native, parents=(source,))

    def log_softmax(self, dim=-1, dtype=None, *, axis=None):
        source = self if dtype in (None, self.dtype) else self.to(dtype)
        native = source._float_native().log_softmax(source._softmax_axis(dim if axis is None else axis))
        return source._wrap(native, parents=(source,))

    def __neg__(self):
        if self.dtype in _STORAGE:
            return Tensor(-self._array(), dtype=self.dtype)
        return self._wrap(self._float_native().unary_math("neg"), parents=(self,))

    def exp(self):
        return self._wrap(self._float_native().unary_math("exp"), parents=(self,))

    def log(self):
        return self._wrap(self._float_native().unary_math("log"), parents=(self,))

    def sqrt(self):
        return self._wrap(self._float_native().unary_math("sqrt"), parents=(self,))

    def sin(self):
        return self._wrap(self._float_native().unary_math("sin"), parents=(self,))

    def cos(self):
        return self._wrap(self._float_native().unary_math("cos"), parents=(self,))

    def sigmoid(self):
        """Stable elementwise CPU float32 sigmoid with first-order gradients."""
        return self._wrap(self._float_native().unary_math("sigmoid"), parents=(self,))

    def tanh(self):
        """Elementwise CPU float32 tanh with first-order gradients."""
        return self._wrap(self._float_native().unary_math("tanh"), parents=(self,))

    def masked_fill(self, mask, value):
        """Return a copy with elements where mask is True set to value.

        mask is a bool Tensor or NumPy/Python bools broadcastable to this shape.
        Masked elements get exactly value and zero gradient; others pass through.
        """
        import numbers
        if isinstance(value, Tensor) and value.requires_grad:
            # A differentiable fill value gets the summed gradient of the filled spots.
            filled = np.broadcast_to(mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask), self.shape)
            return where(filled, value, self)
        if isinstance(value, Tensor) and value.numel() == 1 and value.dtype != "bool":
            value = value.item()
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise TypeError("masked_fill value must be a real number")
        if isinstance(mask, Tensor):
            if mask.dtype != "bool":
                raise TypeError("masked_fill mask must be boolean")
            mask = mask.numpy()
        mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            raise TypeError("masked_fill mask must be boolean")
        try:
            mask = np.broadcast_to(mask, self.shape)
        except ValueError:
            raise ValueError("masked_fill mask is not broadcastable to the input shape") from None
        native = self._float_native().masked_fill(np.ascontiguousarray(mask, np.uint8).tobytes(),
                                                  float(np.float32(value)))
        return self._wrap(native, parents=(self,))

    def gelu(self, approximate="none"):
        """Elementwise CPU float32 GELU; approximate is "none" (erf) or "tanh"."""
        if approximate not in ("none", "tanh"):
            raise ValueError('approximate must be "none" or "tanh"')
        name = "gelu" if approximate == "none" else "gelu_tanh"
        return self._wrap(self._float_native().unary_math(name), parents=(self,))

    def relu(self):
        return self._wrap(self._float_native().relu(), parents=(self,))

    def _reduce(self, axis, keepdims, mean):
        if type(keepdims) is not bool:
            raise TypeError("keepdims must be a bool")
        if axis is None:
            if not keepdims:
                return self._wrap(self._float_native().mean() if mean else self._float_native().sum(), parents=(self,))
            axes = list(range(len(self.shape)))
        else:
            requested = axis if isinstance(axis, tuple) else (axis,)
            axes = []
            for item in requested:
                if isinstance(item, (bool, np.bool_)):
                    raise TypeError("axis must be an integer or tuple of integers")
                try:
                    value = operator.index(item)
                except TypeError:
                    raise TypeError("axis must be an integer or tuple of integers") from None
                rank = len(self.shape)
                if rank == 0 and value in (-1, 0):
                    continue  # a 0-d tensor accepts dim 0/-1 and has nothing to reduce (PyTorch)
                if not -rank <= value < rank:
                    raise ValueError("axis out of bounds")
                value %= rank
                if value in axes:
                    raise ValueError("duplicate reduction axis")
                axes.append(value)
        return self._wrap(self._float_native().reduce_axes(axes, keepdims, mean), parents=(self,))

    def sum(self, dim=None, keepdim=False, *, dtype=None, axis=None, keepdims=None):
        """PyTorch names (dim, keepdim); NumPy names (axis, keepdims) still work."""
        source = self if dtype in (None, self.dtype) else self.to(dtype)
        keep = keepdim if keepdims is None else keepdims
        if source.dtype in _STORAGE:
            return source._numpy_reduce(_axes(dim, axis), keep, mean=False)
        return source._reduce(_axes(dim, axis), keep, False)

    def mean(self, dim=None, keepdim=False, *, dtype=None, axis=None, keepdims=None):
        source = self if dtype in (None, self.dtype) else self.to(dtype)
        keep = keepdim if keepdims is None else keepdims
        if source.dtype in _STORAGE:
            if source.dtype in INTEGER_DTYPES:
                raise RuntimeError("mean(): input dtype must be floating point or complex")
            return source._numpy_reduce(_axes(dim, axis), keep, mean=True)
        return source._reduce(_axes(dim, axis), keep, True)

    def _numpy_reduce(self, axes, keepdim, mean):
        """Sum/mean for non-float32 dtypes (no gradients): integers sum to int64, half types
        accumulate in float32, as PyTorch does."""
        values = self._array()
        if values.dtype.kind == "f" and values.itemsize < 4:
            values = values.astype(np.float32)
        if axes is None or not self.shape:
            axis = None if self.shape else None
        else:
            axis = tuple(a % len(self.shape) for a in (axes if isinstance(axes, tuple) else (axes,)))
        result = (np.mean if mean else np.sum)(values, axis=axis, keepdims=keepdim and bool(self.shape))
        dtype = "int64" if self.dtype in INTEGER_DTYPES else self.dtype
        return Tensor(np.asarray(result), dtype=dtype)

def _numpy_binary(a, b, function, reflected=False):
    """Binary math on NumPy values with PyTorch's result dtype (no gradients). Half types
    compute in float32 and round once."""
    def values(t):
        if not isinstance(t, Tensor):
            return t
        array = t._array()
        return array.astype(np.float32) if t.dtype in ("float16", "bfloat16") else array
    left, right = values(a), values(b)
    if reflected:
        left, right = right, left
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.asarray(function(left, right))
    floats = {t.dtype for t in (a, b) if isinstance(t, Tensor) and t.dtype in ("float64", "float32", "float16", "bfloat16")}
    if result.dtype.kind == "f":
        # Integers with true division or a Python float give PyTorch's default float32.
        dtype = "float64" if "float64" in floats else floats.pop() if len(floats) == 1 else "float32"
    elif result.dtype.kind == "b":
        dtype = "bool"
    else:
        dtype = result.dtype.name
    return Tensor(result, dtype=dtype)


def _axes(dim, axis):
    """Merge PyTorch's dim and NumPy's axis; lists become tuples."""
    chosen = dim if axis is None else axis
    return tuple(chosen) if isinstance(chosen, list) else chosen


_ARITHMETIC = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
               "truediv": operator.truediv, "floordiv": operator.floordiv, "mod": operator.mod}
_COMPARE = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
            "gt": operator.gt, "ge": operator.ge}


def _rebuild(data, shape, dtype, requires_grad):
    if dtype not in DTYPES:
        raise ValueError("unsupported pickled tensor dtype")
    if dtype in _STORAGE:
        return Tensor._wrap(_TypedTensor(data, list(shape), dtype))
    array = np.frombuffer(data, dtype=np.bool_ if dtype == "bool" else np.float32)
    return Tensor(array.reshape(shape), dtype=dtype, requires_grad=requires_grad)

# Elementwise unary ops. float32 runs the native kernel (with gradients); other numeric
# dtypes use NumPy without gradients. Float-valued functions of integers return float32,
# as in PyTorch; the rest keep the input dtype.
_SHAPE_KEEPING = {"abs": np.abs, "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
                  "trunc": np.trunc, "round": np.round, "square": np.square}
_FLOATING = {"frac": lambda v: v - np.trunc(v), "reciprocal": lambda v: 1 / v,
             "rsqrt": lambda v: 1 / np.sqrt(v), "exp2": np.exp2, "expm1": np.expm1,
             "log1p": np.log1p, "log2": np.log2, "log10": np.log10, "tan": np.tan,
             "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "sinh": np.sinh,
             "cosh": np.cosh, "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
             "deg2rad": np.deg2rad, "rad2deg": np.rad2deg, "logit": lambda v: np.log(v / (1 - v)),
             "sinc": np.sinc, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "sin": np.sin,
             "cos": np.cos, "tanh": np.tanh, "sigmoid": lambda v: 1 / (1 + np.exp(-v))}
_NATIVE_ONLY = ("erf", "erfc")


def _unary(name):
    def method(self):
        if self.dtype == "float32":
            return self._wrap(self._native.unary_math(name), parents=(self,))
        if self.dtype not in _STORAGE or name in _NATIVE_ONLY:
            raise TypeError(f"{name} requires a float32 or other numeric tensor, not {self.dtype}")
        floating = name in _FLOATING
        result_dtype = "float32" if floating and self.dtype in INTEGER_DTYPES else self.dtype
        if name == "abs" and self.dtype in COMPLEX_DTYPES:
            return Tensor(np.abs(self._array()), dtype=_REAL_OF[self.dtype])
        if name == "sign" and self.dtype in COMPLEX_DTYPES:
            raise RuntimeError("sign is not defined for complex tensors; use sgn")
        # Half types compute in float32 and round once at the end, as PyTorch does.
        widen = result_dtype in ("float32", "float16", "bfloat16")
        values = self._array().astype(np.float32) if widen else self._array()
        with np.errstate(all="ignore"):
            out = (_FLOATING if floating else _SHAPE_KEEPING)[name](values)
        return Tensor(np.asarray(out), dtype=result_dtype)
    method.__name__ = name
    return method


for _name in (*_SHAPE_KEEPING, *_FLOATING, *_NATIVE_ONLY):
    setattr(Tensor, _name, _unary(_name))
_round, _logit = Tensor.round, Tensor.logit


def _round_decimals(self, decimals=0):
    """Round half to even; with decimals, PyTorch's nearbyint(x * 10**d) / 10**d."""
    if not decimals:
        return _round(self)
    scale = float(10 ** abs(decimals))
    return _round(self * scale) / scale if decimals > 0 else _round(self / scale) * scale


def _logit_eps(self, eps=None):
    return _logit(self if eps is None else self.clamp(eps, 1 - eps))


Tensor.round, Tensor.logit = _round_decimals, _logit_eps
for _alias, _name in {"absolute": "abs", "fix": "trunc", "arcsin": "asin", "arccos": "acos",
                      "arctan": "atan", "arcsinh": "asinh", "arccosh": "acosh", "arctanh": "atanh",
                      "sgn": "sign", "negative": "__neg__", "neg": "__neg__"}.items():
    setattr(Tensor, _alias, getattr(Tensor, _name))


def _clamp(self, min=None, max=None):
    """Limit values to [min, max] (scalars). As in PyTorch 2.14, gradients pass only where
    min < x < max strictly; NaN stays NaN with zero gradient."""
    if min is None and max is None:
        raise ValueError("clamp needs at least one of min and max")
    if self.dtype != "float32":
        values = self._array()
        values = values if min is None else np.maximum(values, min)
        return Tensor(values if max is None else np.minimum(values, max), dtype=self.dtype)
    values = self.numpy()
    out = self
    with np.errstate(invalid="ignore"):
        if min is not None:
            out = out.masked_fill(values <= min, min)
        if max is not None:
            out = out.masked_fill(values >= max, max)
    return out.masked_fill(np.isnan(values), np.nan)


# Binary math. float32 operands (or float32 with a scalar or an integer/half tensor) run the
# native kernel with gradients; other dtype mixes use NumPy with PyTorch's promotion.
_BINARY_NUMPY = {"pow": np.power, "maximum": np.maximum, "minimum": np.minimum, "fmax": np.fmax,
                 "fmin": np.fmin, "atan2": np.arctan2, "hypot": np.hypot, "fmod": np.fmod,
                 "remainder": np.remainder, "copysign": np.copysign, "logaddexp": np.logaddexp,
                 "logaddexp2": np.logaddexp2,
                 "xlogy": lambda l, r: np.where(np.asarray(l) == 0, np.where(np.isnan(r), np.nan, 0.0), np.asarray(l) * np.log(r))}


def _binary(name):
    def method(self, other):
        if not isinstance(other, (Tensor, numbers.Real)) or isinstance(other, (bool, np.bool_)):
            raise TypeError(f"{name} expects a Tensor or real scalar")
        dtypes = {self.dtype} | ({other.dtype} if isinstance(other, Tensor) else set())
        if "bool" in dtypes:
            raise TypeError(f"{name} requires float32 or other numeric tensors, not bool")
        if "float32" in dtypes and "float64" not in dtypes:
            left = self if self.dtype == "float32" else self.float()
            right = other if isinstance(other, Tensor) else Tensor(float(other))
            right = right if right.dtype == "float32" else right.float()
            return Tensor._wrap(left._native.binary_math(right._native, name), parents=(left, right))
        if not isinstance(other, Tensor) and self.dtype == "float32":
            return method(self, Tensor(float(other)))
        return _numpy_binary(self, other, _BINARY_NUMPY[name])
    method.__name__ = name
    return method


for _name in _BINARY_NUMPY:
    setattr(Tensor, _name, _binary(_name))
Tensor.arctan2, Tensor.float_power = Tensor.atan2, lambda self, other: _numpy_binary(
    self.double(), other.double() if isinstance(other, Tensor) else other, np.power)
Tensor.__pow__ = Tensor.pow
def _rpow(self, other):
    if isinstance(other, Tensor):
        return other.pow(self)
    if self.dtype == "float32":
        return Tensor(float(other)).pow(self)
    return _numpy_binary(self, other, np.power, reflected=True)


Tensor.__rpow__ = _rpow
Tensor.heaviside = lambda self, values: _numpy_binary(self, values, np.heaviside)
Tensor.nextafter = lambda self, other: _numpy_binary(self, other, np.nextafter)
Tensor.floor_divide = lambda self, other: self.div(other, rounding_mode="floor")
Tensor.rsub = lambda self, other, alpha=1: (other - self * alpha) if alpha != 1 else other - self


def _lerp(self, end, weight):
    """PyTorch's two-branch form: start + w*(end-start) for w < 0.5, else end - (end-start)*(1-w)."""
    diff = end - self
    if isinstance(weight, Tensor):
        low = self + weight * diff
        high = end - diff * (1 - weight)
        return where(weight.numpy() < 0.5, low, high)
    return self + diff * weight if weight < 0.5 else end - diff * (1 - weight)


Tensor.lerp = _lerp
Tensor.addcmul = lambda self, t1, t2, *, value=1: self + ((t1 * value) * t2 if value != 1 else t1 * t2)
Tensor.addcdiv = lambda self, t1, t2, *, value=1: self + ((t1 * value) / t2 if value != 1 else t1 / t2)
Tensor.isclose = lambda self, other, rtol=1e-05, atol=1e-08, equal_nan=False: Tensor(
    np.isclose(self._array(), other._array() if isinstance(other, Tensor) else other, rtol=rtol, atol=atol,
               equal_nan=equal_nan), dtype="bool")
Tensor.allclose = lambda self, other, rtol=1e-05, atol=1e-08, equal_nan=False: bool(
    self.isclose(other, rtol, atol, equal_nan).numpy().all())
Tensor.equal = lambda self, other: self.shape == other.shape and bool(np.array_equal(self._array(), other._array()))


def where(condition, input=None, other=None):
    """Pick input where condition is True, else other (broadcast). float32 results gather
    exact values and send gradients to both sides."""
    if input is None and other is None:
        raise NotImplementedError("where(condition) without values is unsupported; use nonzero")
    cond = condition.numpy() if isinstance(condition, Tensor) else np.asarray(condition)
    if cond.dtype != np.bool_:
        raise TypeError("where condition must be boolean")
    as_tensor = lambda v, like: v if isinstance(v, Tensor) else Tensor(v, dtype=like.dtype if
                                                                       isinstance(like, Tensor) and like.dtype != "bool" else "float32")
    x, y = as_tensor(input, other), as_tensor(other, input)
    shape = np.broadcast_shapes(cond.shape, x.shape, y.shape)
    if x.dtype == y.dtype == "float32":
        from .functional import gather
        from . import cat
        count = int(np.prod(shape))
        if count == 0:
            return x.expand(*shape)  # empty, but still on the graph
        both = cat([y.expand(shape).reshape(-1), x.expand(shape).reshape(-1)])
        index = np.arange(count) + count * np.broadcast_to(cond, shape).ravel()
        return gather(both, 0, index).reshape(shape)
    return _numpy_binary(x, y, lambda l, r: np.where(cond, l, r))


def _clamp_any(self, min=None, max=None):
    """clamp with scalar bounds (PyTorch's strict scalar gradient rule) or tensor bounds
    (PyTorch: minimum(maximum(x, min), max), so ties split the gradient)."""
    if isinstance(min, Tensor) or isinstance(max, Tensor):
        out = self
        if min is not None:
            out = out.maximum(min)
        if max is not None:
            out = out.minimum(max)
        return out
    return _clamp(self, min, max)


Tensor.clamp = Tensor.clip = _clamp_any
Tensor.clamp_min = lambda self, min: _clamp_any(self, min=min)
Tensor.clamp_max = lambda self, max: _clamp_any(self, max=max)



def _nan_to_num(self, nan=0.0, posinf=None, neginf=None):
    """Replace NaN/inf (default: 0, float32 max, float32 min); replaced spots get zero gradient."""
    if self.dtype != "float32":
        return Tensor(np.nan_to_num(self._array(), nan=nan, posinf=posinf, neginf=neginf), dtype=self.dtype)
    values = self.numpy()
    info = np.finfo(np.float32)
    out = self.masked_fill(np.isnan(values), nan)
    out = out.masked_fill(np.isposinf(values), info.max if posinf is None else posinf)
    return out.masked_fill(np.isneginf(values), info.min if neginf is None else neginf)


Tensor.nan_to_num = _nan_to_num
for _name, _test in {"isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite,
                     "isposinf": np.isposinf, "isneginf": np.isneginf, "signbit": np.signbit,
                     "isreal": lambda v: np.ones(v.shape, bool)}.items():
    setattr(Tensor, _name, (lambda test: lambda self: Tensor(test(self._array()), dtype="bool"))(_test))
Tensor.positive = lambda self: self


def _infer_dtype(data):
    """torch.tensor's rule: tensors and NumPy data keep their dtype; Python bools -> bool,
    Python ints -> int64, anything with a Python float -> float32 (the default dtype)."""
    if isinstance(data, Tensor):
        return data.dtype
    if isinstance(data, (np.ndarray, np.generic)):
        name = np.asarray(data).dtype.name
        if name not in DTYPES:
            raise TypeError(f"unsupported NumPy dtype {name}")
        return name
    if isinstance(data, (list, tuple)) and any(isinstance(v, Tensor) for v in np.ravel(np.asarray(data, dtype=object))):
        raise TypeError("lists of Tensors are unsupported; use stack")
    kind = np.asarray(data).dtype.kind
    return {"b": "bool", "i": "int64", "u": "int64", "c": "complex64"}.get(kind, "float32")


# ---- complex helpers ------------------------------------------------------------------

def _real(self):
    return Tensor(self._array().real, dtype=_REAL_OF[self.dtype]) if self.dtype in COMPLEX_DTYPES else self


def _imag(self):
    if self.dtype not in COMPLEX_DTYPES:
        raise RuntimeError("imag is not implemented for tensors with non-complex dtypes")
    return Tensor(self._array().imag, dtype=_REAL_OF[self.dtype])


def _sgn(self):
    if self.dtype not in COMPLEX_DTYPES:
        return self.sign()
    values = self._array()
    with np.errstate(all="ignore"):
        return Tensor(np.where(values == 0, 0, values / np.abs(values)), dtype=self.dtype)


Tensor.real = property(_real)
Tensor.imag = property(_imag)
Tensor.sgn = _sgn
Tensor.conj = lambda self: Tensor(np.conj(self._array()), dtype=self.dtype) if self.dtype in COMPLEX_DTYPES else self
Tensor.conj_physical = lambda self: Tensor(np.conj(self._array()), dtype=self.dtype) if self.dtype in COMPLEX_DTYPES else self.clone()
Tensor.is_complex = lambda self: self.dtype in COMPLEX_DTYPES
Tensor.is_floating_point = lambda self: self.dtype in ("float32", "float64", "float16", "bfloat16")
Tensor.cfloat = lambda self, memory_format=None: self._cast("complex64")
Tensor.cdouble = lambda self, memory_format=None: self._cast("complex128")


def _should_fold(tensor1, tensor2):
    """PyTorch's should_fold (LinearAlgebra.cpp): fold the larger operand's leading dims into
    one mm/mv when its reshape is a view (or the other side needs a gradient)."""
    d1, d2 = len(tensor1.shape), len(tensor2.shape)
    larger = d1 >= d2
    t1 = tensor1 if larger else tensor2.transpose(-2, -1)
    dim_t2 = d2 if larger else d1
    if not (len(t1.shape) >= 3 and dim_t2 <= 2):
        return False
    if (tensor2 if larger else tensor1).requires_grad:
        return True
    if d1 == 2:
        return False
    if t1.numel() == 0:
        return True
    folded = t1.reshape(int(np.prod(t1.shape[:-1])), t1.shape[-1])
    return folded.shares_storage(t1)


def _matmul_impl(tensor1, tensor2):
    """torch.matmul, routed as PyTorch's _matmul_impl routes it, so each case runs the same
    kernel (dot, mv, mm, folded mm/mv, or bmm) with the same strides."""
    d1, d2 = len(tensor1.shape), len(tensor2.shape)
    if d1 == 1 and d2 == 1:
        return tensor1._dot(tensor2)
    if d1 == 2 and d2 == 1:
        return tensor1._mv(tensor2)
    if d1 == 1 and d2 == 2:
        return tensor1.unsqueeze(0)._mm(tensor2).squeeze(0)
    if d1 == 2 and d2 == 2:
        return tensor1._mm(tensor2)
    if _should_fold(tensor1, tensor2):
        transpose = d2 > d1
        t1 = tensor2.transpose(-2, -1) if transpose else tensor1
        t2 = tensor2 if not transpose else (tensor1.t() if d1 == 2 else tensor1)
        output_shape = list(t1.shape[:-1])
        folded = t1.reshape(int(np.prod(output_shape)), t1.shape[-1])
        if len(t2.shape) == 2:
            output = folded._mm(t2).reshape(*output_shape, t2.shape[1])
            return output.transpose(-2, -1).contiguous() if transpose else output
        return folded._mv(t2).reshape(*output_shape)
    s1, s2 = tensor1.shape, tensor2.shape
    n = s1[-2] if d1 > 1 else 1
    m1 = s1[-1]
    m2 = s2[-2] if d2 > 1 else s2[0]
    p = s2[-1] if d2 > 1 else 1
    batch1, batch2 = s1[:max(d1 - 2, 0)], s2[:max(d2 - 2, 0)]
    if d1 == 3 and d2 == 3 and batch1[0] != batch2[0]:
        if batch1[0] == 1 and tensor1.requires_grad:
            return _matmul_impl(tensor1.squeeze(0), tensor2)
        if batch2[0] == 1 and tensor2.requires_grad:
            return _matmul_impl(tensor1, tensor2.squeeze(0))
    batch = tuple(np.broadcast_shapes(batch1, batch2))
    count = int(np.prod(batch))
    t1e = tensor1.expand(*batch, n, m1).reshape(count, n, m1)
    if d2 == 1:
        t2e = tensor2.expand(*batch, m2).reshape(count, m2).unsqueeze(2)
    else:
        t2e = tensor2.expand(*batch, m2, p).reshape(count, m2, p)
    output_shape = list(batch) + ([n] if d1 > 1 else []) + ([p] if d2 > 1 else [])
    return t1e.bmm(t2e).reshape(*output_shape)


def tensor(data, *, dtype=None, device="cpu", requires_grad=False):
    """Infers the dtype like torch.tensor when dtype is not given (Tensor(...) stays float32,
    like the legacy torch.Tensor constructor)."""
    if isinstance(data, Tensor):
        data = data._array() if data.dtype != "bfloat16" else data._array()
    return Tensor(data, dtype=dtype or (_infer_dtype(data) if not isinstance(data, Tensor) else data.dtype),
                  device=device, requires_grad=requires_grad)
