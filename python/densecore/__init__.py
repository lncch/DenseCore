from .tensor import Tensor, tensor, where
from ._native import cpu_backend, get_num_threads, set_num_threads
from . import nn, functional, optim, checkpoints, autograd, reductions, shapes
from .shapes import MODULE_FUNCTIONS as _shape_functions
globals().update(_shape_functions)
from . import extras as _extras
for _name, _method in _extras.TENSOR_METHODS.items():
    setattr(Tensor, _name, _method)
globals().update(_extras.MODULE_FUNCTIONS)
from . import masked
import types as _types
signal = _types.SimpleNamespace(windows=masked.windows)
Tensor.masked_select = masked.masked_select
from . import special as _special_module
special = _special_module.special
for _name in ("lgamma", "digamma", "erfinv", "i0", "mvlgamma", "igamma", "igammac"):
    setattr(Tensor, _name, getattr(_special_module, _name))
Tensor.polygamma = lambda self, n: _special_module.polygamma(n, self)
polygamma = _special_module.polygamma
from . import linalg, fft
stft, istft = fft.stft, fft.istft
Tensor.stft = fft.stft
Tensor.istft = fft.istft
_linalg_functions = {
    "inverse": linalg.inv, "det": linalg.det, "slogdet": linalg.slogdet, "logdet": linalg.logdet,
    "cholesky_inverse": linalg.cholesky_inverse, "geqrf": linalg.geqrf,
    "svd": linalg.legacy_svd, "lu_unpack": linalg.lu_unpack,
    "cholesky": lambda A, upper=False: linalg.cholesky(A, upper=upper),
    "qr": lambda A, some=True: linalg.qr(A, "reduced" if some else "complete"),
    "lu": lambda A, pivot=True, get_infos=False: (linalg.lu_factor_ex(A, pivot=pivot) if get_infos
                                                   else linalg.lu_factor(A, pivot=pivot)),
    "lu_solve": linalg.lu_solve_legacy, "triangular_solve": linalg.triangular_solve,
    "cholesky_solve": linalg.cholesky_solve, "pinverse": linalg.pinverse, "matrix_exp": linalg.matrix_exp,
    "matrix_power": linalg.matrix_power, "ormqr": linalg.ormqr, "orgqr": linalg.householder_product}
globals().update(_linalg_functions)
for _name, _function in _linalg_functions.items():
    if _name not in ("lu_unpack",):
        setattr(Tensor, _name, _function)
Tensor.masked_scatter = masked.masked_scatter
from .nn import manual_seed
from ._random import Generator, default_generator, initial_seed, seed, get_rng_state, set_rng_state
from . import _random as random  # torch.random: manual_seed, seed, initial_seed, get/set_rng_state
from .convolution import grid_sampler_2d, grid_sampler_3d
from .compilation import compile
from .grad_mode import no_grad, enable_grad, is_grad_enabled
from ._compat import (zeros, ones, arange, tril, from_numpy, save, load, autocast,
                      amp, cuda, backends, randint, multinomial, topk, as_tensor, is_tensor)

# `from torch.nn import functional as F` and `import densecore.nn.functional` both work.
import sys as _sys
nn.functional = functional
_sys.modules.setdefault(__name__ + ".nn.functional", functional)


_NOT_FUNCTIONS = {"numpy", "tolist", "item", "backward", "register_hook", "zero_grad", "to",
                  "shares_storage", "stride", "storage_offset", "copy_", "size", "dim", "numel"}


def __getattr__(name):
    # Dtype names live here, not as globals, so `bool`/`float`/`int` builtins stay intact.
    from ._compat import DTYPES
    if name in DTYPES:
        return DTYPES[name]
    # torch.add(x, y) and friends: forward to the Tensor method of the same name.
    method = getattr(Tensor, name, None)
    if callable(method) and not name.startswith("_") and name not in _NOT_FUNCTIONS:
        def function(input, *args, **kwargs):
            if not isinstance(input, Tensor):
                raise TypeError(f"{name}() expects a Tensor as its first argument")
            return getattr(input, name)(*args, **kwargs)
        function.__name__ = name
        return function
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or", "logical_xor", "logical_not", "cat", "stack", "bmm", "autograd", "Tensor", "tensor", "nn", "manual_seed", "Generator", "default_generator", "initial_seed", "seed", "get_rng_state", "set_rng_state", "random", "as_tensor", "is_tensor", "utils", "functional", "optim", "checkpoints", "no_grad", "enable_grad", "is_grad_enabled", "cpu_backend", "compile", "compiled_train_step",
           "zeros", "ones", "arange", "tril", "from_numpy", "save", "load", "autocast", "amp", "cuda", "backends",
           "randint", "multinomial", "topk", "where"]

from .native_training import compiled_train_step


def bmm(input, other):
    """Multiply rank-3 tensors with equal batch and inner dimensions."""
    if not isinstance(input, Tensor):
        raise TypeError("bmm input must be a Tensor")
    return input.bmm(other)


def _joining_inputs(tensors, dim, *, stacking):
    import operator
    if isinstance(tensors, Tensor):
        raise TypeError("joining requires a sequence of Tensors, not a Tensor")
    inputs = list(tensors)
    if not inputs:
        raise ValueError("joining requires at least one tensor")
    if any(not isinstance(t, Tensor) for t in inputs):
        raise TypeError("joining inputs must be Tensors")
    if isinstance(dim, bool):
        raise TypeError("dim must be an integer")
    dim = operator.index(dim)
    rank = len(inputs[0].shape) + int(stacking)
    if dim < -rank or dim >= rank:
        raise ValueError("dim out of range")
    return inputs, dim % rank


def cat(tensors, dim=0):
    """Concatenate tensors along an existing axis; eager CPU float32 only."""
    from ._native import _Tensor
    inputs, dim = _joining_inputs(tensors, dim, stacking=False)
    dtypes = {t.dtype for t in inputs}
    if len(dtypes) > 1:
        raise TypeError("cat inputs must share one dtype; convert them first")
    from .tensor import _STORAGE
    if dtypes <= set(_STORAGE):
        # Storage dtypes have no gradients, so a NumPy copy is the whole operation.
        # Bool stays rejected by the native path (pinned by test_boolean_tensor.py).
        import numpy as np
        return Tensor(np.concatenate([t._array() for t in inputs], dim), dtype=inputs[0].dtype)
    return Tensor._wrap(_Tensor.cat([t._native for t in inputs], dim), parents=inputs)


def stack(tensors, dim=0):
    """Join equal-shaped tensors along a new axis; gradients are preserved."""
    inputs, dim = _joining_inputs(tensors, dim, stacking=True)
    shape = inputs[0].shape
    if any(t.shape != shape for t in inputs):
        raise ValueError("stack input shapes must match")
    expanded = list(shape)
    expanded.insert(dim, 1)
    return cat([t.reshape(expanded) for t in inputs], dim)


def logical_not(input):
    """Invert a boolean tensor; returns a nondifferentiable boolean tensor."""
    if not isinstance(input, Tensor):
        raise TypeError("logical_not input must be a Tensor")
    return input.logical_not()


def _predicate(input, other, operation):
    if not isinstance(input, Tensor):
        raise TypeError("predicate input must be a Tensor")
    return getattr(input, operation)(other)


def eq(input, other):
    return _predicate(input, other, "eq")


def ne(input, other):
    return _predicate(input, other, "ne")


def lt(input, other):
    return _predicate(input, other, "lt")


def le(input, other):
    return _predicate(input, other, "le")


def gt(input, other):
    return _predicate(input, other, "gt")


def ge(input, other):
    return _predicate(input, other, "ge")


def logical_and(input, other):
    return _predicate(input, other, "logical_and")


def logical_or(input, other):
    return _predicate(input, other, "logical_or")


def logical_xor(input, other):
    return _predicate(input, other, "logical_xor")

from . import utils  # last: utils.data imports densecore's factories
