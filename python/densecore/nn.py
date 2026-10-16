"""Registered dense modules for the supported CPU float32 API."""
import math
import operator
import types
import numpy as np
from .tensor import Tensor

from ._random import manual_seed, rng as _rng_for

# Draws in PyTorch's order from the default generator (see _random.py).
_rng = _rng_for()

class Parameter(Tensor):
    """A registered, independently owned leaf tensor."""
    def __init__(self, data, requires_grad=True):
        if isinstance(data, Tensor) and data.dtype != "float32":
            raise TypeError("Parameters require float32 tensors")
        super().__init__(data.numpy() if isinstance(data, Tensor) else data,
                         requires_grad=requires_grad)

class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Module)):
            if "_parameters" not in self.__dict__:
                raise RuntimeError("call Module.__init__ before registering members")
            if not name or "." in name or name in {"_parameters", "_modules", "training"}:
                raise ValueError("registered names must be nonempty, contain no dots and avoid internal names")
            if isinstance(value, Module) and any(m is self for m in value._walk()):
                raise ValueError("module registration would create a cycle")
        if "_parameters" in self.__dict__:
            if name in self._parameters and value is not None and not isinstance(value, Parameter):
                raise TypeError("registered parameter replacement must be Parameter or None")
            if name in self._modules and value is not None and not isinstance(value, Module):
                raise TypeError("registered module replacement must be Module or None")
            if not isinstance(value, Parameter):
                self._parameters.pop(name, None)
            if not isinstance(value, Module):
                self._modules.pop(name, None)
            if isinstance(value, Parameter):
                self._parameters[name] = value
            elif isinstance(value, Module):
                self._modules[name] = value
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        object.__delattr__(self, name)

    def _walk(self):
        visited = set()
        stack = [self]
        while stack:
            m = stack.pop()
            if id(m) not in visited:
                visited.add(id(m))
                yield m
                stack.extend(reversed(list(m._modules.values())))

    def named_parameters(self, *, remove_duplicate=True):
        seen = set()
        stack = [("", self)]
        while stack:
            prefix, m = stack.pop()
            for name, p in m._parameters.items():
                if not remove_duplicate or id(p) not in seen:
                    seen.add(id(p))
                    yield prefix + name, p
            stack.extend((prefix + name + ".", child)
                         for name, child in reversed(list(m._modules.items())))

    def parameters(self):
        return (p for _, p in self.named_parameters())

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self):
        return {name: Tensor(p.numpy()) for name, p in
                self.named_parameters(remove_duplicate=False)}

    def _prepare_state_dict(self, state):
        if not isinstance(state, dict):
            raise TypeError("model state must be a dictionary")
        named = dict(self.named_parameters(remove_duplicate=False))
        if set(state) != set(named):
            raise ValueError("model state parameter names must match exactly")
        params, values = [], []
        staged = {}
        for name, p in named.items():
            value = state[name]
            if not isinstance(value, Tensor) or value.shape != p.shape:
                raise ValueError("model state values must be Tensors with matching shapes")
            if value.dtype != "float32":
                raise TypeError("model state values must be float32 tensors")
            array = value.numpy()
            if id(p) in staged:
                if not np.array_equal(staged[id(p)], array, equal_nan=True):
                    raise ValueError("shared parameter aliases contain conflicting values")
            else:
                staged[id(p)] = array
                params.append(p)
                values.append(Tensor(array))
        return params, values

    def load_state_dict(self, state):
        from ._native import _replace_leaves
        params, values = self._prepare_state_dict(state)
        updated = _replace_leaves([p._native for p in params], [v._native for v in values])
        for p, native in zip(params, updated):
            p._native = native
            if p._hook_slot is not None and p._hook_slot.callbacks:
                p._sync_hook_slot()
        return None

    def apply(self, fn):
        """Call fn on every submodule (children first, as PyTorch does), then on self."""
        for child in self._modules.values():
            child.apply(fn)
        fn(self)
        return self

    def register_buffer(self, name, tensor, persistent=True):
        """Attach a non-trainable tensor as an attribute.

        shortcut: buffers are not part of state_dict or checkpoints yet; add them when a
        model needs its buffers saved.
        """
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("buffer must be a Tensor or None")
        if not isinstance(name, str) or not name or "." in name or name in self._parameters or name in self._modules:
            raise ValueError("buffer name must be a new nonempty string without dots")
        self.__dict__.setdefault("_buffers", {})[name] = tensor
        object.__setattr__(self, name, tensor)

    def to(self, *args, **kwargs):
        """Accept the cpu device and float32; parameters already live there."""
        for target in (*args, *kwargs.values()):
            if target not in (None, "cpu", "float32") and not isinstance(target, bool):
                raise ValueError(f"unsupported Module.to() target {target!r}: modules are float32 on cpu")
        return self

    def train(self, mode=True):
        if type(mode) is not bool:
            raise TypeError("training mode must be a bool")
        for m in self._walk():
            object.__setattr__(m, "training", mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("implement Module.forward")

    def compile(self):
        """Enable supported graph compilation in place; return None."""
        from .compilation import compile
        compile(self)
        return None

    def __call__(self, *args, **kwargs):
        runner = self.__dict__.get("_compiled_runner")
        if runner is not None:
            return runner(self, args, kwargs)
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        dims = []
        for n in (in_features, out_features):
            if isinstance(n, (bool, np.bool_)):
                raise TypeError("feature counts must be integers")
            n = operator.index(n)
            if n <= 0:
                raise ValueError("feature counts must be positive")
            dims.append(n)
        if type(bias) is not bool:
            raise TypeError("bias must be a bool")
        self.in_features, self.out_features = dims
        # torch's reset_parameters: kaiming_uniform_(a=sqrt(5)), then bias in +-1/sqrt(fan_in).
        self.weight = Parameter(_uniform((self.out_features, self.in_features), _kaiming_a5_bound(self.in_features)))
        self.bias = Parameter(_uniform(self.out_features, 1 / math.sqrt(self.in_features))) if bias else None

    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("Linear input must be a Tensor")
        if len(x.shape) > 2:
            from .functional import _flatten_leading
            return _flatten_leading(x, self.forward)
        return Tensor._wrap(x._float_native().linear(self.weight._native,
                            None if self.bias is None else self.bias._native), parents=(x, self.weight, self.bias))

class Embedding(Module):
    """Lookup table of num_embeddings rows; weight starts from a standard normal."""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        dims = []
        for n in (num_embeddings, embedding_dim):
            if isinstance(n, (bool, np.bool_)):
                raise TypeError("embedding sizes must be integers")
            n = operator.index(n)
            if n <= 0:
                raise ValueError("embedding sizes must be positive")
            dims.append(n)
        self.num_embeddings, self.embedding_dim = dims
        self.weight = Parameter(_rng.standard_normal(dims).astype(np.float32))

    def forward(self, indices):
        from .functional import embedding
        return embedding(indices, self.weight)

class Dropout(Module):
    """Inverted dropout; active only in training mode (see Module.train/eval)."""
    def __init__(self, p=0.5):
        super().__init__()
        from .functional import _dropout_probability
        self.p = _dropout_probability(p)

    def forward(self, x):
        from .functional import dropout
        return dropout(x, self.p, self.training)

class LayerNorm(Module):
    """Normalize the feature dimension of a rank2 float32 batch."""
    def __init__(self,normalized_shape,eps=1e-5,elementwise_affine=True,bias=True):
        super().__init__()
        from .functional import _normalization_shape,_normalization_eps
        self.normalized_shape=_normalization_shape(normalized_shape)
        self.eps=_normalization_eps(eps)
        if type(elementwise_affine) is not bool or type(bias) is not bool:
            raise TypeError("affine and bias flags must be bools")
        self.elementwise_affine=elementwise_affine
        self.weight=Parameter(np.ones(self.normalized_shape,dtype=np.float32)) if elementwise_affine else None
        self.bias=Parameter(np.zeros(self.normalized_shape,dtype=np.float32)) if elementwise_affine and bias else None
    def forward(self,x):
        from .functional import layer_norm
        return layer_norm(x,self.normalized_shape,self.weight,self.bias,self.eps)

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sigmoid(Module):
    """Elementwise CPU float32 sigmoid; no trainable parameters."""
    def forward(self, x):
        from .functional import sigmoid
        return sigmoid(x)

class Tanh(Module):
    """Elementwise CPU float32 tanh; no trainable parameters."""
    def forward(self, x):
        from .functional import tanh
        return tanh(x)

class GELU(Module):
    """Elementwise CPU float32 GELU; approximate is "none" (erf) or "tanh"."""
    def __init__(self, approximate="none"):
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError('approximate must be "none" or "tanh"')
        self.approximate = approximate

    def forward(self, x):
        from .functional import gelu
        return gelu(x, self.approximate)

class Flatten(Module):
    """Collapse dimensions start_dim through end_dim, preserving the batch by default."""
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        import operator
        for axis in (start_dim, end_dim):
            if isinstance(axis, (bool, np.bool_)):
                raise TypeError("flatten dimensions must be integers, not booleans")
        self.start_dim = operator.index(start_dim)
        self.end_dim = operator.index(end_dim)

    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("flatten input must be a Tensor")
        return x.flatten(self.start_dim, self.end_dim)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError("Sequential members must be Modules")
            setattr(self, str(i), module)

    def __getitem__(self, index):
        return list(self._modules.values())[index]

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

class ModuleList(Module):
    """An indexable list of registered submodules, named "0", "1", ..."""
    def __init__(self, modules=None):
        super().__init__()
        self.extend(modules or ())

    def append(self, module):
        if not isinstance(module, Module):
            raise TypeError("ModuleList members must be Modules")
        setattr(self, str(len(self._modules)), module)
        return self

    def extend(self, modules):
        for module in modules:
            self.append(module)
        return self

    def __getitem__(self, index):
        items = list(self._modules.values())
        return ModuleList(items[index]) if isinstance(index, slice) else items[index]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(list(self._modules.values()))

class ModuleDict(Module):
    """A name-indexed mapping of registered submodules, also reachable as attributes."""
    def __init__(self, modules=None):
        super().__init__()
        for name, module in dict(modules or {}).items():
            self[name] = module

    def __setitem__(self, name, module):
        if not isinstance(module, Module):
            raise TypeError("ModuleDict values must be Modules")
        setattr(self, name, module)

    def __getitem__(self, name):
        return self._modules[name]

    def __contains__(self, name):
        return name in self._modules

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(list(self._modules))

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

def _initialize(tensor, array):
    """Replace a leaf tensor's values in place, the way optimizers and load_state_dict do."""
    from ._native import _replace_leaves
    if not isinstance(tensor, Tensor) or tensor.dtype != "float32":
        raise TypeError("init functions require a float32 Tensor")
    (native,) = _replace_leaves([tensor._native], [Tensor(array.astype(np.float32))._native])
    tensor._native = native
    if tensor._hook_slot is not None and tensor._hook_slot.callbacks:
        tensor._sync_hook_slot()
    return tensor

def _normal_(tensor, mean=0.0, std=1.0):
    """Fill from N(mean, std**2) using the generator seeded by manual_seed."""
    return _initialize(tensor, _rng.normal(mean, std, tensor.shape))

def _zeros_(tensor):
    return _initialize(tensor, np.zeros(tensor.shape))

def calculate_gain(nonlinearity, param=None):
    if nonlinearity in ("linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
                        "conv_transpose3d", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1 + slope ** 2))
    if nonlinearity == "selu":
        return 3.0 / 4
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _fans(tensor):
    if len(tensor.shape) < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    receptive = int(np.prod(tensor.shape[2:]))
    return tensor.shape[1] * receptive, tensor.shape[0] * receptive


def _xavier_std(tensor, gain):
    fan_in, fan_out = _fans(tensor)
    return gain * math.sqrt(2.0 / float(fan_in + fan_out))


def _kaiming_std(tensor, a, mode, nonlinearity):
    fan_in, fan_out = _fans(tensor)
    return calculate_gain(nonlinearity, a) / math.sqrt(fan_in if mode == "fan_in" else fan_out)


def _trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
    """PyTorch's algorithm: one uniform draw per element in CDF space, erfinv, scale, clamp.
    The draws match PyTorch's; erfinv is DenseCore's, so values can differ in the last bits."""
    from .special import erfinv
    def cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    lo, hi = cdf((a - mean) / std), cdf((b - mean) / std)
    u = _rng_for(generator).uniform(2 * lo - 1, 2 * hi - 1, tensor.shape)
    values = erfinv(Tensor(u)).numpy() * np.float32(std * math.sqrt(2.0)) + np.float32(mean)
    return _initialize(tensor, np.clip(values, np.float32(a), np.float32(b)))


def _orthogonal_(tensor, gain=1.0, generator=None):
    rows = tensor.shape[0]
    flat = _rng.normal(0.0, 1.0, (rows, tensor.numel() // rows))
    q, r = np.linalg.qr(flat.T if rows < flat.shape[1] else flat)
    q = q * np.sign(np.diag(r))
    return _initialize(tensor, gain * (q.T if rows < flat.shape[1] else q).reshape(tensor.shape))


def _eye_(tensor):
    if len(tensor.shape) != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    return _initialize(tensor, np.eye(*tensor.shape))


init = types.SimpleNamespace(
    normal_=_normal_, zeros_=_zeros_, calculate_gain=calculate_gain, eye_=_eye_,
    ones_=lambda tensor: _initialize(tensor, np.ones(tensor.shape)),
    constant_=lambda tensor, val: _initialize(tensor, np.full(tensor.shape, val)),
    uniform_=lambda tensor, a=0.0, b=1.0, generator=None: _initialize(tensor, _rng.uniform(a, b, tensor.shape)),
    xavier_uniform_=lambda tensor, gain=1.0, generator=None: _initialize(tensor, _rng.uniform(
        -_xavier_std(tensor, gain) * math.sqrt(3.0), _xavier_std(tensor, gain) * math.sqrt(3.0), tensor.shape)),
    xavier_normal_=lambda tensor, gain=1.0, generator=None: _initialize(tensor, _rng.normal(
        0, _xavier_std(tensor, gain), tensor.shape)),
    kaiming_uniform_=lambda tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None: _initialize(
        tensor, _rng.uniform(-math.sqrt(3.0) * _kaiming_std(tensor, a, mode, nonlinearity),
                             math.sqrt(3.0) * _kaiming_std(tensor, a, mode, nonlinearity), tensor.shape)),
    kaiming_normal_=lambda tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None: _initialize(
        tensor, _rng.normal(0, _kaiming_std(tensor, a, mode, nonlinearity), tensor.shape)),
    trunc_normal_=_trunc_normal_, orthogonal_=_orthogonal_)

def _clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """Scale gradients so their combined norm is at most max_norm; return the norm before.

    norm_type 2 follows PyTorch bit for bit: a float32 norm per tensor, the norm of those
    norms, then min(1, (1 / (norm + 1e-6)) * max_norm). Other norm types are summed in float64
    and can differ from PyTorch in the last bits.
    """
    params = [parameters] if isinstance(parameters, Tensor) else list(parameters)
    grads = [(p, p.grad) for p in params if p.grad is not None]
    if not grads:
        return Tensor(0.0)
    arrays = [g.numpy() for _, g in grads]
    norm_type = float(norm_type)
    if norm_type == 2.0:
        from ._native import _norm2
        norms = np.array([_norm2(g._native) for _, g in grads], np.float32)
        total = np.float32(_norm2(Tensor(norms)._native))
    else:
        flat = np.concatenate([a.ravel().astype(np.float64) for a in arrays])
        if norm_type == math.inf:
            total = np.abs(flat).max(initial=0.0)
        else:
            total = (np.abs(flat) ** norm_type).sum() ** (1.0 / norm_type)
        total = np.float32(total)
    if error_if_nonfinite and not np.isfinite(total):
        raise RuntimeError("the total gradient norm is not finite")
    # PyTorch: max_norm / (total + 1e-6) is Tensor.__rtruediv__, reciprocal() * max_norm.
    coef = (np.float32(1.0) / (total + np.float32(1e-6))) * np.float32(max_norm)
    scale = np.minimum(coef, np.float32(1.0))
    for (p, _), array in zip(grads, arrays):
        p.grad = Tensor(array * scale)
    return Tensor(total)

utils = types.SimpleNamespace(clip_grad_norm_=_clip_grad_norm_)

class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        from .functional import _reduction
        self.reduction = _reduction(reduction)

    def forward(self, input, target):
        from .functional import mse_loss
        return mse_loss(input, target, reduction=self.reduction)

class CrossEntropyLoss(Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean",
                 label_smoothing=0.0):
        super().__init__()
        from .functional import _reduction
        self.weight, self.ignore_index = weight, ignore_index
        self.reduction, self.label_smoothing = _reduction(reduction), label_smoothing

    def forward(self, input, target):
        from .functional import cross_entropy
        return cross_entropy(input, target, self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, label_smoothing=self.label_smoothing)


def _functional_module(name, function_name, parameters=()):
    """A Module whose forward calls functional.<function_name> with stored keyword options.
    `parameters` lists (name, default) pairs accepted positionally, as in PyTorch."""
    def __init__(self, *args, **kwargs):
        Module.__init__(self)
        options = dict(parameters)
        for (key, _), value in zip(parameters, args):
            options[key] = value
        unknown = set(kwargs) - set(options) - {"inplace"}
        if unknown:
            raise TypeError(f"{name}() got unexpected arguments {sorted(unknown)}")
        options.update({k: v for k, v in kwargs.items() if k != "inplace"})
        object.__setattr__(self, "_options", options)

    def forward(self, *inputs):
        from . import functional
        return getattr(functional, function_name)(*inputs, **self._options)
    return type(name, (Module,), {"__init__": __init__, "forward": forward, "__doc__": f"Module form of functional.{function_name}."})


for _name, _function, _parameters in [
        ("ReLU6", "relu6", ()), ("LeakyReLU", "leaky_relu", (("negative_slope", 0.01),)),
        ("ELU", "elu", (("alpha", 1.0),)), ("CELU", "celu", (("alpha", 1.0),)), ("SELU", "selu", ()),
        ("SiLU", "silu", ()), ("Mish", "mish", ()), ("Softplus", "softplus", (("beta", 1.0), ("threshold", 20.0))),
        ("Softsign", "softsign", ()), ("Softshrink", "softshrink", (("lambd", 0.5),)),
        ("Hardshrink", "hardshrink", (("lambd", 0.5),)), ("Hardtanh", "hardtanh", (("min_val", -1.0), ("max_val", 1.0))),
        ("Hardsigmoid", "hardsigmoid", ()), ("Hardswish", "hardswish", ()), ("LogSigmoid", "logsigmoid", ()),
        ("Tanhshrink", "tanhshrink", ()), ("Threshold", "threshold", (("threshold", 0.0), ("value", 0.0))),
        ("GLU", "glu", (("dim", -1),)), ("Softmax", "softmax", (("dim", -1),)),
        ("LogSoftmax", "log_softmax", (("dim", -1),)), ("Softmin", "softmin", (("dim", -1),)),
        ("PixelShuffle", "pixel_shuffle", (("upscale_factor", 1),)),
        ("PixelUnshuffle", "pixel_unshuffle", (("downscale_factor", 1),)),
        ("ChannelShuffle", "channel_shuffle", (("groups", 1),)),
        ("L1Loss", "l1_loss", (("reduction", "mean"),)),
        ("SmoothL1Loss", "smooth_l1_loss", (("reduction", "mean"), ("beta", 1.0))),
        ("HuberLoss", "huber_loss", (("reduction", "mean"), ("delta", 1.0))),
        ("BCELoss", "binary_cross_entropy", (("weight", None), ("reduction", "mean"))),
        ("BCEWithLogitsLoss", "binary_cross_entropy_with_logits", (("weight", None), ("reduction", "mean"), ("pos_weight", None))),
        ("NLLLoss", "nll_loss", (("weight", None), ("ignore_index", -100), ("reduction", "mean"))),
        ("KLDivLoss", "kl_div", (("reduction", "mean"), ("log_target", False))),
        ("PoissonNLLLoss", "poisson_nll_loss", (("log_input", True), ("full", False), ("eps", 1e-8), ("reduction", "mean"))),
        ("GaussianNLLLoss", "gaussian_nll_loss", (("full", False), ("eps", 1e-6), ("reduction", "mean"))),
        ("MarginRankingLoss", "margin_ranking_loss", (("margin", 0.0), ("reduction", "mean"))),
        ("HingeEmbeddingLoss", "hinge_embedding_loss", (("margin", 1.0), ("reduction", "mean"))),
        ("SoftMarginLoss", "soft_margin_loss", (("reduction", "mean"),)),
        ("CosineEmbeddingLoss", "cosine_embedding_loss", (("margin", 0.0), ("reduction", "mean"))),
        ("MultiLabelSoftMarginLoss", "multilabel_soft_margin_loss", (("weight", None), ("reduction", "mean"))),
        ("MultiMarginLoss", "multi_margin_loss", (("p", 1), ("margin", 1.0), ("weight", None), ("reduction", "mean"))),
        ("MultiLabelMarginLoss", "multilabel_margin_loss", (("size_average", None), ("reduce", None), ("reduction", "mean"))),
        ("CTCLoss", "ctc_loss", (("blank", 0), ("reduction", "mean"), ("zero_infinity", False))),
        ("TripletMarginLoss", "triplet_margin_loss", (("margin", 1.0), ("p", 2.0), ("eps", 1e-6), ("swap", False), ("reduction", "mean"))),
        ("CosineSimilarity", "cosine_similarity", (("dim", 1), ("eps", 1e-8))),
        ("PairwiseDistance", "pairwise_distance", (("p", 2.0), ("eps", 1e-6), ("keepdim", False)))]:
    globals()[_name] = _functional_module(_name, _function, _parameters)


class EmbeddingBag(Module):
    """torch.nn.EmbeddingBag: N(0, 1) weight (padding row zero), forward(input, offsets, per_sample_weights)."""
    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 mode="mean", sparse=False, _weight=None, include_last_offset=False, padding_idx=None,
                 device=None, dtype=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim, self.mode = num_embeddings, embedding_dim, mode
        self.max_norm, self.include_last_offset, self.padding_idx = max_norm, include_last_offset, padding_idx
        values = _rng.normal(size=(num_embeddings, embedding_dim)) if _weight is None else _weight.numpy()
        if padding_idx is not None:
            values[padding_idx] = 0
        self.weight = Parameter(values.astype(np.float32))

    def forward(self, input, offsets=None, per_sample_weights=None):
        from .functional import embedding_bag
        return embedding_bag(input, self.weight, offsets, self.max_norm, mode=self.mode,
                             per_sample_weights=per_sample_weights, include_last_offset=self.include_last_offset,
                             padding_idx=self.padding_idx)


class MultiheadAttention(Module):
    """torch.nn.MultiheadAttention: same parameters, initialization and forward."""
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be greater than 0")
        if embed_dim % num_heads:
            raise AssertionError("embed_dim must be divisible by num_heads")
        self.embed_dim, self.num_heads, self.dropout, self.batch_first = embed_dim, num_heads, dropout, batch_first
        self.kdim, self.vdim = kdim or embed_dim, vdim or embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.head_dim, self.add_zero_attn = embed_dim // num_heads, add_zero_attn
        empty = lambda *shape: Parameter(np.zeros(shape, np.float32))
        if self._qkv_same_embed_dim:
            self.in_proj_weight = empty(3 * embed_dim, embed_dim)
            self.q_proj_weight = self.k_proj_weight = self.v_proj_weight = None
        else:
            self.q_proj_weight, self.k_proj_weight = empty(embed_dim, embed_dim), empty(embed_dim, self.kdim)
            self.v_proj_weight, self.in_proj_weight = empty(embed_dim, self.vdim), None
        self.in_proj_bias = empty(3 * embed_dim) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.bias_k, self.bias_v = (empty(1, 1, embed_dim), empty(1, 1, embed_dim)) if add_bias_kv else (None, None)
        for weight in (self.in_proj_weight, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight):
            if weight is not None:
                init.xavier_uniform_(weight)
        if bias:
            init.zeros_(self.out_proj.bias)
        for extra in (self.bias_k, self.bias_v):
            if extra is not None:
                init.xavier_normal_(extra)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        from .functional import multi_head_attention_forward
        batched = len(query.shape) == 3
        if self.batch_first and batched:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        out, weights = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=not self._qkv_same_embed_dim,
            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight,
            average_attn_weights=average_attn_weights, is_causal=is_causal)
        return (out.transpose(1, 0) if self.batch_first and batched else out), weights


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = Parameter(np.full(num_parameters, init, np.float32))

    def forward(self, input):
        from .functional import prelu
        return prelu(input, self.weight)


def _uniform(shape, bound):
    return _rng.uniform(-bound, bound, shape).astype(np.float32)


def _kaiming_a5_bound(fan_in):
    """init.kaiming_uniform_(w, a=math.sqrt(5))'s bound, computed in torch's order."""
    return math.sqrt(3.0) * (calculate_gain("leaky_relu", math.sqrt(5)) / math.sqrt(fan_in)) if fan_in else 0.0


class _ConvNd(Module):
    """PyTorch's default init: weight and bias uniform in +-1/sqrt(fan_in)."""
    _nd, _transposed = 2, False

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode="zeros", output_padding=0):
        super().__init__()
        from .convolution import _tuple
        n = self._nd
        self.in_channels, self.out_channels, self.groups = in_channels, out_channels, groups
        self.kernel_size, self.stride, self.dilation = _tuple(kernel_size, n), _tuple(stride, n), _tuple(dilation, n)
        self.padding = padding if isinstance(padding, str) else _tuple(padding, n)
        self.output_padding = _tuple(output_padding, n)
        if padding_mode != "zeros":
            raise NotImplementedError("only padding_mode='zeros' is supported")
        if self._transposed:
            shape = (in_channels, out_channels // groups, *self.kernel_size)
        else:
            shape = (out_channels, in_channels // groups, *self.kernel_size)
        fan_in = shape[1] * int(np.prod(self.kernel_size))
        self.weight = Parameter(_uniform(shape, _kaiming_a5_bound(fan_in)))
        self.bias = Parameter(_uniform(out_channels, 1 / math.sqrt(fan_in) if fan_in else 0.0)) if bias else None

    def forward(self, input):
        from . import functional
        if self._transposed:
            fn = getattr(functional, f"conv_transpose{self._nd}d")
            return fn(input, self.weight, self.bias, self.stride, self.padding, self.output_padding,
                      self.groups, self.dilation)
        fn = getattr(functional, f"conv{self._nd}d")
        return fn(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class _BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.affine, self.track_running_stats = affine, track_running_stats
        self.weight = Parameter(np.ones(num_features, np.float32)) if affine else None
        self.bias = Parameter(np.zeros(num_features, np.float32)) if affine else None
        if track_running_stats:
            self.register_buffer("running_mean", Tensor(np.zeros(num_features, np.float32)))
            self.register_buffer("running_var", Tensor(np.ones(num_features, np.float32)))
        else:
            self.running_mean = self.running_var = None

    def forward(self, input):
        from .functional import batch_norm
        use_batch = self.training or not self.track_running_stats
        return batch_norm(input, self.running_mean if self.track_running_stats else None,
                          self.running_var if self.track_running_stats else None, self.weight, self.bias,
                          use_batch, self.momentum, self.eps)


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups, self.eps = num_groups, eps
        self.weight = Parameter(np.ones(num_channels, np.float32)) if affine else None
        self.bias = Parameter(np.zeros(num_channels, np.float32)) if affine else None

    def forward(self, input):
        from .functional import group_norm
        return group_norm(input, self.num_groups, self.weight, self.bias, self.eps)


class _InstanceNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones(num_features, np.float32)) if affine else None
        self.bias = Parameter(np.zeros(num_features, np.float32)) if affine else None

    def forward(self, input):
        from .functional import instance_norm
        return instance_norm(input, weight=self.weight, bias=self.bias, eps=self.eps)


for _nd in (1, 2, 3):
    globals()[f"Conv{_nd}d"] = type(f"Conv{_nd}d", (_ConvNd,), {"_nd": _nd})
    globals()[f"ConvTranspose{_nd}d"] = type(f"ConvTranspose{_nd}d", (_ConvNd,), {"_nd": _nd, "_transposed": True})
    globals()[f"BatchNorm{_nd}d"] = type(f"BatchNorm{_nd}d", (_BatchNorm,), {})
    globals()[f"InstanceNorm{_nd}d"] = type(f"InstanceNorm{_nd}d", (_InstanceNorm,), {})
    globals()[f"MaxPool{_nd}d"] = _functional_module(f"MaxPool{_nd}d", f"max_pool{_nd}d", (
        ("kernel_size", 1), ("stride", None), ("padding", 0), ("dilation", 1), ("return_indices", False), ("ceil_mode", False)))
    globals()[f"AvgPool{_nd}d"] = _functional_module(f"AvgPool{_nd}d", f"avg_pool{_nd}d", (
        ("kernel_size", 1), ("stride", None), ("padding", 0), ("ceil_mode", False), ("count_include_pad", True)))
    globals()[f"AdaptiveAvgPool{_nd}d"] = _functional_module(f"AdaptiveAvgPool{_nd}d", f"adaptive_avg_pool{_nd}d", (("output_size", 1),))
    globals()[f"AdaptiveMaxPool{_nd}d"] = _functional_module(f"AdaptiveMaxPool{_nd}d", f"adaptive_max_pool{_nd}d", (
        ("output_size", 1), ("return_indices", False)))
for _name, _function, _parameters in [
        ("LocalResponseNorm", "local_response_norm", (("size", 1), ("alpha", 1e-4), ("beta", 0.75), ("k", 1.0))),
        ("Upsample", "interpolate", (("size", None), ("scale_factor", None), ("mode", "nearest"), ("align_corners", None))),
        ("Unfold", "unfold", (("kernel_size", 1), ("dilation", 1), ("padding", 0), ("stride", 1))),
        ("Fold", "fold", (("output_size", 1), ("kernel_size", 1), ("dilation", 1), ("padding", 0), ("stride", 1)))]:
    globals()[_name] = _functional_module(_name, _function, _parameters)


def _pad_module(name, mode, dims):
    def __init__(self, padding, value=0.0):
        Module.__init__(self)
        object.__setattr__(self, "padding", tuple(padding) if isinstance(padding, (tuple, list)) else (padding,) * (2 * dims))
        object.__setattr__(self, "value", value)

    def forward(self, input):
        from .functional import pad
        return pad(input, self.padding, mode, self.value if mode == "constant" else None)
    return type(name, (Module,), {"__init__": __init__, "forward": forward})


for _dims in (1, 2, 3):
    globals()[f"ZeroPad{_dims}d"] = _pad_module(f"ZeroPad{_dims}d", "constant", _dims)
    globals()[f"ConstantPad{_dims}d"] = _pad_module(f"ConstantPad{_dims}d", "constant", _dims)
    globals()[f"ReflectionPad{_dims}d"] = _pad_module(f"ReflectionPad{_dims}d", "reflect", _dims)
    globals()[f"ReplicationPad{_dims}d"] = _pad_module(f"ReplicationPad{_dims}d", "replicate", _dims)
    globals()[f"CircularPad{_dims}d"] = _pad_module(f"CircularPad{_dims}d", "circular", _dims)
