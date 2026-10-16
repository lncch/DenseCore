"""Supported dense operations and mean-reduction loss functions."""
import math
import numpy as np
from .tensor import Tensor

def _integer_indices(value, name):
    """Integer tensors become NumPy indices; other tensors are rejected."""
    if not isinstance(value, Tensor):
        return value
    from .tensor import INTEGER_DTYPES
    if value.dtype not in INTEGER_DTYPES:
        raise TypeError(f"{name} must be an integer tensor or integer Python/NumPy data")
    return value.numpy()

def _mean_only(reduction):
    if reduction != "mean":
        raise ValueError("only mean reduction is supported")

def _reduction(reduction):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError('reduction must be "mean", "sum" or "none"')
    return reduction

def mse_loss(input, target, *, reduction="mean"):
    _reduction(reduction)
    if not isinstance(input, Tensor) or not isinstance(target, Tensor):
        raise TypeError("MSE input and target must be Tensors")
    if input.shape != target.shape:
        raise ValueError("MSE input and target shapes must match")
    if reduction == "mean":
        return input._mse_loss(target)
    diff = input - target
    squared = diff * diff
    return squared.sum() if reduction == "sum" else squared

def _cross_entropy_native(input, target, *, ignore_index=-100, reduction="mean"):
    """Mean cross entropy over rows whose label is not ignore_index (PyTorch default -100)."""
    _mean_only(reduction)
    if not isinstance(input, Tensor):
        raise TypeError("cross_entropy input must be a Tensor")
    target = _integer_indices(target, "class labels")
    if isinstance(target, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in target):
        raise TypeError("class labels must be integers, not booleans")
    labels = np.asarray(target)
    if labels.ndim != 1:
        raise ValueError("class labels must have shape (batch,)")
    if labels.size and labels.dtype.kind not in "iu":
        raise TypeError("class labels must be integers, not floats or booleans")
    if len(input.shape) != 2 or input.shape[1] == 0:
        raise ValueError("cross_entropy requires (batch, classes) logits")
    if len(labels) != input.shape[0]:
        raise ValueError("class labels must match batch and lie within class range")
    kept = labels != ignore_index
    if np.any(labels[kept] < 0) or np.any(labels[kept] >= input.shape[1]):
        raise ValueError("class labels must match batch and lie within class range")
    # Ignored rows stay in place (PyTorch's log_softmax and nll cascade see every row), marked
    # with the native sentinel usize::MAX.
    native_labels = [int(v) if k else 2**64 - 1 for v, k in zip(labels.tolist(), kept.tolist())]
    return Tensor._wrap(input._float_native().cross_entropy(native_labels), parents=(input,))

def softmax(input, dim, dtype=None):
    if not isinstance(input, Tensor):
        raise TypeError("softmax input must be a Tensor")
    if dtype not in (None, "float32"):
        raise TypeError("softmax computes in float32 only")
    return input.softmax(dim)

def log_softmax(input, dim, dtype=None):
    if not isinstance(input, Tensor):
        raise TypeError("log_softmax input must be a Tensor")
    if dtype not in (None, "float32"):
        raise TypeError("log_softmax computes in float32 only")
    return input.log_softmax(dim)

def _flatten_leading(input, apply):
    """Run a rank-2 operation on rank>2 input by flattening its leading dimensions."""
    lead = input.shape[:-1]
    out = apply(input.reshape(-1, input.shape[-1]))
    return out.reshape(*lead, out.shape[-1])


def linear(input,weight,bias=None):
    if not isinstance(input,Tensor) or not isinstance(weight,Tensor) or (bias is not None and not isinstance(bias,Tensor)):
        raise TypeError("linear input, weight and optional bias must be Tensors")
    if len(input.shape) > 2:
        return _flatten_leading(input, lambda flat: linear(flat, weight, bias))
    if len(input.shape) == 1:
        return linear(input.unsqueeze(0), weight, bias).squeeze(0)
    return Tensor._wrap(input._float_native().linear(weight._native,None if bias is None else bias._native), parents=(input, weight, bias))


def _normalization_shape(normalized_shape):
    import operator
    if isinstance(normalized_shape, (tuple,list)):
        if len(normalized_shape)!=1:
            raise ValueError("Module LayerNorm supports one feature dimension; use functional.layer_norm")
        normalized_shape=normalized_shape[0]
    if isinstance(normalized_shape,(bool,np.bool_)):
        raise TypeError("normalized_shape must be an integer")
    features=operator.index(normalized_shape)
    if features<=0:
        raise ValueError("normalized_shape must be positive")
    return (features,)


def _normalization_eps(eps):
    import numbers
    if isinstance(eps,(bool,np.bool_)) or not isinstance(eps,numbers.Real):
        raise TypeError("eps must be real")
    eps=float(eps)
    if not np.isfinite(eps) or eps<=0 or eps>np.finfo(np.float32).max or np.float32(eps)==0:
        raise ValueError("eps must be positive finite float32")
    return eps


def layer_norm(input,normalized_shape,weight=None,bias=None,eps=1e-5):
    if isinstance(normalized_shape, (tuple, list)) and len(normalized_shape) > 1 and isinstance(input, Tensor):
        # Normalize over several trailing dimensions by flattening them into one.
        shape = tuple(normalized_shape)
        if input.shape[-len(shape):] != shape:
            raise ValueError("input's trailing dimensions must match normalized_shape")
        size = int(np.prod(shape))
        flat = lambda t: None if t is None else t.reshape(size)
        out = layer_norm(input.reshape(*input.shape[:-len(shape)], size), size, flat(weight), flat(bias), eps)
        return out.reshape(input.shape)
    if isinstance(input, Tensor) and len(input.shape) == 1:
        return layer_norm(input.unsqueeze(0), normalized_shape, weight, bias, eps).squeeze(0)
    features=_normalization_shape(normalized_shape)
    try:
        eps=_normalization_eps(eps)
    except ValueError:
        # PyTorch accepts any eps here (zero, negative, inf); the native kernel does not, so compose.
        mean = input.mean(-1, keepdim=True)
        out = (input - mean) / ((input - mean).pow(2).mean(-1, keepdim=True) + float(eps)).sqrt()
        out = out if weight is None else out * weight
        return out if bias is None else out + bias
    if isinstance(input,Tensor) and len(input.shape)>2 and input.shape[-1:]==features:
        return _flatten_leading(input, lambda flat: layer_norm(flat,normalized_shape,weight,bias,eps))
    if not isinstance(input,Tensor) or len(input.shape)!=2 or input.shape[1:]!=features:
        raise ValueError("layer_norm requires rank>=2 input whose last dimension matches normalized_shape")
    if (weight is not None and not isinstance(weight,Tensor)) or (bias is not None and not isinstance(bias,Tensor)):
        raise TypeError("layer_norm weight and bias must be Tensors")
    return Tensor._wrap(input._float_native().layer_norm(None if weight is None else weight._native,None if bias is None else bias._native,eps), parents=(input, weight, bias))


def sigmoid(input):
    """Apply elementwise CPU float32 sigmoid."""
    if not isinstance(input, Tensor):
        raise TypeError("sigmoid input must be a Tensor")
    return input.sigmoid()


def tanh(input):
    """Apply elementwise CPU float32 tanh."""
    if not isinstance(input, Tensor):
        raise TypeError("tanh input must be a Tensor")
    return input.tanh()


def gelu(input, approximate="none"):
    """Apply elementwise CPU float32 GELU; approximate is "none" (erf) or "tanh"."""
    if not isinstance(input, Tensor):
        raise TypeError("gelu input must be a Tensor")
    return input.gelu(approximate)


def embedding(input, weight):
    """Gather rows of a (num_embeddings, embedding_dim) weight by integer indices.

    Indices are integer Python/NumPy metadata of any shape; the result has shape
    indices.shape + (embedding_dim,). Gradients scatter-add into weight rows.
    """
    input = _integer_indices(input, "embedding indices")
    if not isinstance(weight, Tensor):
        raise TypeError("embedding weight must be a Tensor")
    indices = np.asarray(input)
    if indices.size and indices.dtype.kind not in "iu":
        raise TypeError("embedding indices must be integers, not floats or booleans")
    if len(weight.shape) != 2:
        raise ValueError("embedding weight must have shape (num_embeddings, embedding_dim)")
    if np.any(indices < 0) or np.any(indices >= weight.shape[0]):
        raise ValueError("embedding indices must lie within [0, num_embeddings)")
    native = weight._float_native().embedding(indices.astype(np.int64).ravel().tolist(), list(indices.shape))
    return Tensor._wrap(native, parents=(weight,))


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode="mean",
                  sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None):
    """Sum, mean or max of embedding rows per bag. A 2-D input holds one bag per row; a 1-D
    input is cut into bags at `offsets`. Entries equal to padding_idx are left out."""
    if max_norm is not None:
        raise NotImplementedError("embedding_bag max_norm is unsupported")
    indices = np.asarray(_integer_indices(input, "embedding indices"))
    if indices.ndim == 2:
        starts = np.arange(indices.shape[0]) * indices.shape[1]
        indices = indices.reshape(-1)
    else:
        starts = np.asarray(_integer_indices(offsets, "offsets")).reshape(-1)
        if include_last_offset:
            starts = starts[:-1]
    ends = np.append(starts[1:], indices.size)
    rows = embedding(indices, weight)
    if per_sample_weights is not None:
        rows = rows * per_sample_weights.reshape(-1, 1)
    keep = np.ones(indices.size, bool) if padding_idx is None else indices != padding_idx % weight.shape[0]
    if mode == "max":
        bags = [rows[Tensor(np.flatnonzero(keep[a:b]) + a, dtype="int64")].amax(0) if keep[a:b].any()
                else weight.new_zeros(weight.shape[1]) for a, b in zip(starts, ends)]
        from . import stack
        return stack(bags) if bags else weight.new_zeros(0, weight.shape[1])
    members = np.zeros((len(starts), indices.size), np.float32)
    for bag, (a, b) in enumerate(zip(starts, ends)):
        members[bag, a:b] = keep[a:b]
    if mode == "mean":
        members /= np.maximum(members.sum(1, keepdims=True), 1)
    return Tensor(members) @ rows


def _dropout_probability(p):
    import numbers
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, numbers.Real):
        raise TypeError("dropout probability must be real")
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError("dropout probability must be in [0, 1]")
    return p


def dropout(input, p=0.5, training=True):
    """Zero elements with probability p and scale survivors by 1/(1-p) while training.

    Masks come from the generator seeded by densecore.manual_seed. With training=False
    or p=0 the input is returned unchanged; p=1 returns zeros.
    """
    if not isinstance(input, Tensor):
        raise TypeError("dropout input must be a Tensor")
    p = _dropout_probability(p)
    if type(training) is not bool:
        raise TypeError("training must be a bool")
    if not training or p == 0.0:
        return input
    if p == 1.0:
        return input * Tensor(np.zeros((), np.float32))
    return input * Tensor(_dropout_noise(input.shape, p))


def gather(input, axis, indices):
    """Pick entries along one axis by integer indices of any shape (like input[..., indices, ...]).

    The result replaces that axis with indices.shape. Float32 gradients scatter-add back
    through the native embedding gather; other dtypes have no gradients and use NumPy.
    """
    indices = np.asarray(indices)
    if indices.size and indices.dtype.kind not in "iu":
        raise TypeError("gather indices must be integers")
    size = input.shape[axis]
    indices = indices.astype(np.int64)
    if np.any(indices < -size) or np.any(indices >= size):
        raise IndexError("index out of range")
    indices = np.where(indices < 0, indices + size, indices)
    rank = len(input.shape)
    rest = tuple(i for i in range(rank) if i != axis)
    moved = input.permute((axis,) + rest)
    if input.dtype == "float32":
        flat = moved.reshape(size, int(np.prod(moved.shape[1:])))
        picked = embedding(indices.ravel(), flat) if flat.shape[1] else flat.reshape(-1).reshape(indices.size, 0)
        picked = picked.reshape(*indices.shape, *moved.shape[1:])
    else:
        picked = Tensor(np.take(moved._array(), indices, axis=0), dtype=input.dtype)
    # Move the index dimensions back to where the axis was.
    count = indices.ndim
    order = list(range(count, count + axis)) + list(range(count)) + list(range(count + axis, count + rank - 1))
    return picked.permute(order) if order != list(range(len(order))) else picked


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    """softmax(query @ key^T * scale + mask) @ value for rank-3 (N, L, E) or rank-4 (N, H, L, E).

    attn_mask follows PyTorch: a boolean mask (Tensor or NumPy) is True where attention is
    allowed; a float Tensor/array is added to the scores. is_causal applies a lower-triangular
    boolean mask and cannot be combined with attn_mask. scale defaults to 1/sqrt(E).
    dropout_p > 0 drops attention weights (seeded by manual_seed) whatever the module mode,
    as in PyTorch. Composed from bmm, softmax and masked_fill, so gradients are the existing ones.
    """
    import math
    import numbers
    for name, t in (("query", query), ("key", key), ("value", value)):
        if not isinstance(t, Tensor):
            raise TypeError(f"attention {name} must be a Tensor")
    rank = len(query.shape)
    if enable_gqa and rank == 4 and key.shape[1] != query.shape[1]:
        # Grouped-query attention: each key/value head serves query.shape[1] // key.shape[1] heads.
        groups = query.shape[1] // key.shape[1]
        widen = lambda t: t.unsqueeze(2).expand(t.shape[0], t.shape[1], groups, *t.shape[2:]).reshape(
            t.shape[0], t.shape[1] * groups, *t.shape[2:])
        key, value = widen(key), widen(value)
    if rank not in (3, 4) or len(key.shape) != rank or len(value.shape) != rank:
        raise ValueError("attention query, key and value must all be rank 3 or all rank 4")
    lead, (length, embed) = query.shape[:-2], query.shape[-2:]
    source = key.shape[-2]
    if key.shape[:-2] != lead or value.shape[:-2] != lead or key.shape[-1] != embed or value.shape[-2] != source:
        raise ValueError("attention shapes must be (..., L, E), (..., S, E), (..., S, Ev)")
    if type(is_causal) is not bool:
        raise TypeError("is_causal must be a bool")
    if is_causal and attn_mask is not None:
        raise ValueError("attn_mask and is_causal cannot both be set")
    if scale is None:
        scale = 1.0 / math.sqrt(embed) if embed else 1.0
    elif isinstance(scale, (bool, np.bool_)) or not isinstance(scale, numbers.Real):
        raise TypeError("scale must be real")
    if (rank == 4 and not dropout_p and key.shape[1] == query.shape[1]
            and value.shape[-1] == embed and length and source and embed
            and all(t.dtype == "float32" for t in (query, key, value))):
        flash_mask = _flash_mask(attn_mask, query.shape, source)
        if attn_mask is None or flash_mask is not None:
            return _flash_attention(query, key, value, is_causal, None if scale is None else float(scale), flash_mask)
    # PyTorch's math path (_scaled_dot_product_attention_math), op by op: q and k^T are each
    # scaled by sqrt(scale) before the matmul, masks become an added -inf/0 float tensor.
    factor = math.sqrt(abs(scale))
    q = query * float(-factor if scale < 0 else factor)
    # torch's k^T * factor keeps the transposed layout, which picks its small-bmm kernel.
    attn = q.matmul((key * float(factor)).transpose(-2, -1))
    if is_causal:
        attn_mask = np.tril(np.ones((length, source), bool))
    if attn_mask is not None:
        dtype = attn_mask.dtype if isinstance(attn_mask, Tensor) else np.asarray(attn_mask).dtype
        if dtype in ("bool", np.bool_):
            allowed = attn_mask.numpy() if isinstance(attn_mask, Tensor) else np.asarray(attn_mask)
            additive = Tensor(np.where(allowed, np.float32(0.0), np.float32(-np.inf)))
        else:
            additive = attn_mask if isinstance(attn_mask, Tensor) else Tensor(np.asarray(attn_mask, np.float32))
        attn = attn + additive
    empty = np.isneginf(attn.numpy()).all(-1, keepdims=True) if attn_mask is not None else None
    if empty is not None and empty.any():
        # _safe_softmax: rows that are all -inf give 0 (and a zero gradient), not NaN.
        weights = attn.masked_fill(empty, 0.0).softmax(-1).masked_fill(empty, 0.0)
    else:
        weights = attn.softmax(-1)
    return dropout(weights, dropout_p, True).matmul(value)


def _flash_mask(attn_mask, qshape, source):
    """The (B, H, L, S) float32 additive mask PyTorch's CPU flash kernel would read, or None
    when PyTorch would not take the flash path for this mask (it then uses the math path)."""
    if attn_mask is None:
        return None
    if isinstance(attn_mask, Tensor):
        if attn_mask.requires_grad or attn_mask.dtype not in ("bool", "float32"):
            return None
        values = attn_mask.numpy()
    else:
        values = np.asarray(attn_mask)
        if values.dtype not in (np.bool_, np.float32):
            return None
    batch, heads, length = qshape[:3]
    if values.ndim not in (2, 4) or values.shape[-2] not in (1, length) or values.shape[-1] not in (1, source):
        return None
    if values.ndim == 4 and (values.shape[0] not in (1, batch) or values.shape[1] not in (1, heads)):
        return None
    if values.dtype == np.bool_:
        values = np.where(values, np.float32(0.0), np.float32(-np.inf))
    full = np.broadcast_to(values, (batch, heads, length, source))
    return Tensor(np.ascontiguousarray(full, np.float32))


def _flash_attention(query, key, value, causal, scale, mask=None):
    """PyTorch's CPU fused path (_scaled_dot_product_flash_attention_for_cpu), bit for bit."""
    from ._native import _flash_attention as forward_kernel, _flash_attention_backward as backward_kernel
    from .function import Function

    class FlashAttention(Function):
        @staticmethod
        def forward(ctx, q, k, v):
            out, lse = forward_kernel(q._native, k._native, v._native, causal, scale,
                                      None if mask is None else mask._native)
            out, lse = Tensor._wrap(out), Tensor._wrap(lse)
            ctx.save_for_backward(q, k, v, out, lse)
            return out

        @staticmethod
        def backward(ctx, grad):
            q, k, v, out, lse = ctx.saved_tensors
            grads = backward_kernel(grad._native, q._native, k._native, v._native, out._native,
                                    lse._native, causal, scale, None if mask is None else mask._native)
            return tuple(Tensor._wrap(g) for g in grads)

    return FlashAttention.apply(query, key, value)


def _float_mask(mask):
    """MHA masks: a bool mask is True where attention is NOT allowed (-inf); a float mask adds."""
    if mask is None:
        return None
    if mask.dtype == "bool":
        return Tensor(np.where(mask.numpy(), -np.inf, 0.0).astype(np.float32))
    return mask


def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                                 bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias,
                                 training=True, key_padding_mask=None, need_weights=True, attn_mask=None,
                                 use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
                                 v_proj_weight=None, static_k=None, static_v=None, average_attn_weights=True,
                                 is_causal=False):
    """PyTorch's reference algorithm. Inputs are (L, N, E), or (L, E) unbatched. is_causal is
    only a hint, as in PyTorch: attn_mask must still be given and is the one applied."""
    if is_causal and attn_mask is None:
        raise RuntimeError("Need attn_mask if specifying the is_causal hint")
    batched = len(query.shape) == 3
    if not batched:
        query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)
    tgt_len, bsz, embed_dim = query.shape
    if embed_dim != embed_dim_to_check:
        raise ValueError(f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}")
    head_dim = embed_dim // num_heads
    biases = (None, None, None) if in_proj_bias is None else in_proj_bias.chunk(3)
    weights = (q_proj_weight, k_proj_weight, v_proj_weight) if use_separate_proj_weight else in_proj_weight.chunk(3)
    q, k, v = (linear(x, w, b) for x, w, b in zip((query, key, value), weights, biases))
    attn_mask = _float_mask(attn_mask)
    if attn_mask is not None and len(attn_mask.shape) == 2:
        attn_mask = attn_mask.unsqueeze(0)
    key_padding_mask = _float_mask(key_padding_mask)
    if bias_k is not None and bias_v is not None:
        from . import cat
        k = cat([k, bias_k.expand(1, bsz, embed_dim)])
        v = cat([v, bias_v.expand(1, bsz, embed_dim)])
        attn_mask = None if attn_mask is None else pad(attn_mask, (0, 1))
        key_padding_mask = None if key_padding_mask is None else pad(key_padding_mask, (0, 1))
    q = q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = static_k if static_k is not None else k.reshape(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = static_v if static_v is not None else v.reshape(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    if add_zero_attn:
        from . import cat
        k = cat([k, k.new_zeros(bsz * num_heads, 1, head_dim)], 1)
        v = cat([v, v.new_zeros(bsz * num_heads, 1, head_dim)], 1)
        attn_mask = None if attn_mask is None else pad(attn_mask, (0, 1))
        key_padding_mask = None if key_padding_mask is None else pad(key_padding_mask, (0, 1))
    src_len = k.shape[1]
    if key_padding_mask is not None:
        padding = key_padding_mask.reshape(bsz, 1, 1, src_len).expand(bsz, num_heads, 1, src_len)
        padding = padding.reshape(bsz * num_heads, 1, src_len)
        attn_mask = padding if attn_mask is None else attn_mask + padding
    scores = (q * math.sqrt(1.0 / head_dim)).bmm(k.transpose(1, 2))
    if attn_mask is not None:
        scores = scores + attn_mask
    probabilities = scores.softmax(-1)
    if training and dropout_p > 0:
        probabilities = dropout(probabilities, dropout_p, True)
    out = probabilities.bmm(v).transpose(0, 1).reshape(tgt_len * bsz, embed_dim)
    out = linear(out, out_proj_weight, out_proj_bias).reshape(tgt_len, bsz, -1)
    if not batched:
        out = out.squeeze(1)
    if not need_weights:
        return out, None
    probabilities = probabilities.reshape(bsz, num_heads, tgt_len, src_len)
    if average_attn_weights:
        probabilities = probabilities.mean(1)
    return out, probabilities if batched else probabilities.squeeze(0)


# ---- activations (PyTorch formulas; kinks follow PyTorch's backward rules) ------------

def _where(condition, a, b):
    from .tensor import where
    return where(condition, a, b)


def relu(input, inplace=False):
    return input.relu()


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    return input.clamp(min_val, max_val)


def relu6(input, inplace=False):
    return hardtanh(input, 0.0, 6.0)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    return _where((input > 0).numpy(), input, input * negative_slope)


def elu(input, alpha=1.0, inplace=False, *, scale=1.0, input_scale=1.0):
    negative = (input * input_scale).expm1() * (alpha * scale)
    return _where((input > 0).numpy(), input * scale if scale != 1 else input, negative)


def celu(input, alpha=1.0, inplace=False):
    return elu(input, alpha, input_scale=1.0 / alpha)


def selu(input, inplace=False):
    return elu(input, 1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946)


def silu(input, inplace=False):
    return input * input.sigmoid()


def softplus(input, beta=1.0, threshold=20.0):
    scaled = input * beta if beta != 1 else input
    smooth = scaled.exp().log1p() / beta if beta != 1 else scaled.exp().log1p()
    return _where((scaled > threshold).numpy(), input, smooth)


def mish(input, inplace=False):
    return input * softplus(input).tanh()


def softsign(input):
    return input / (input.abs() + 1.0)


def softshrink(input, lambd=0.5):
    values = input.numpy()
    zero = input.masked_fill(np.ones(input.shape, bool), 0.0)
    return _where(values > lambd, input - lambd, _where(values < -lambd, input + lambd, zero))


def hardshrink(input, lambd=0.5):
    return input.masked_fill(np.abs(input.numpy()) <= lambd, 0.0)


def hardsigmoid(input, inplace=False):
    return (input / 6.0 + 0.5).clamp(0.0, 1.0)


def hardswish(input, inplace=False):
    values = input.numpy()
    zero = input.masked_fill(np.ones(input.shape, bool), 0.0)
    # PyTorch 2.14's backward: 0 at x <= -3, 1 at x >= 3.
    return _where(values <= -3, zero, _where(values >= 3, input, input * (input + 3.0) / 6.0))


def logsigmoid(input):
    return -softplus(-input)


def tanhshrink(input):
    return input - input.tanh()


def threshold(input, threshold, value, inplace=False):
    return input.masked_fill(~(input.numpy() > threshold), float(value))


def glu(input, dim=-1):
    a, b = input.chunk(2, dim)
    return a * b.sigmoid()


def prelu(input, weight):
    shape = [1] * len(input.shape)
    if weight.numel() > 1:
        shape[1] = weight.numel()
    slope = weight.reshape(shape) if input.shape else weight.reshape(())
    return _where((input > 0).numpy(), input, input * slope)


def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    if not training:
        return leaky_relu(input, (lower + upper) / 2)
    from . import nn
    slopes = nn._rng.uniform(lower, upper, input.shape).astype(np.float32)
    return _where((input >= 0).numpy(), input, input * Tensor(slopes))


def softmin(input, dim=None, _stacklevel=3, dtype=None):
    return softmax(-input, -1 if dim is None else dim, dtype)


def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    norms = input.norm(p, dim, keepdim=True).clamp_min(eps)
    return input / norms


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """PyTorch: each input is divided by max(its norm, eps), then the products are summed."""
    n1 = (x1 * x1).sum(dim, keepdim=True).sqrt().clamp_min(eps)
    n2 = (x2 * x2).sum(dim, keepdim=True).sqrt().clamp_min(eps)
    return ((x1 / n1) * (x2 / n2)).sum(dim)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    return (x1 - x2 + eps).norm(p, -1, keepdim=keepdim)


def pdist(input, p=2.0):
    n = input.shape[0]
    i, j = np.triu_indices(n, 1)
    if not len(i):
        return input.reshape(-1)[0:0]
    return (input[i] - input[j]).norm(p, -1)


def pixel_shuffle(input, upscale_factor):
    r = upscale_factor
    *lead, c, h, w = input.shape
    out = input.reshape(*lead, c // (r * r), r, r, h, w)
    k = len(lead)
    out = out.permute(*range(k), k, k + 3, k + 1, k + 4, k + 2)
    return out.reshape(*lead, c // (r * r), h * r, w * r)


def pixel_unshuffle(input, downscale_factor):
    r = downscale_factor
    *lead, c, h, w = input.shape
    out = input.reshape(*lead, c, h // r, r, w // r, r)
    k = len(lead)
    out = out.permute(*range(k), k, k + 2, k + 4, k + 1, k + 3)
    return out.reshape(*lead, c * r * r, h // r, w // r)


def channel_shuffle(input, groups):
    n, c, *rest = input.shape
    return input.reshape(n, groups, c // groups, *rest).transpose(1, 2).reshape(input.shape)


def bilinear(input1, input2, weight, bias=None):
    """y_k = input1 . W_k . input2 + b_k."""
    left = (input1.unsqueeze(-2).unsqueeze(-2) @ weight).squeeze(-2)
    out = (left * input2.unsqueeze(-2)).sum(-1)
    return out + bias if bias is not None else out


def rms_norm(input, normalized_shape, weight=None, eps=None):
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    dims = tuple(range(-len(shape), 0))
    eps = np.finfo(np.float32).eps if eps is None else eps
    out = input * ((input * input).mean(dims, keepdim=True) + eps).rsqrt()
    return out * weight if weight is not None else out


def _dropout_noise(shape, p):
    """PyTorch's _dropout_impl: noise.bernoulli_(1 - p), then noise.div_(1 - p) in float32."""
    from ._random import default_generator
    return default_generator._bernoulli(shape, 1 - p) / np.float32(1 - p)


def _channel_dropout(input, p, training, spatial_dims):
    if not training or p == 0:
        return input
    keep_shape = input.shape[:len(input.shape) - spatial_dims] + (1,) * spatial_dims
    if p == 1:
        return input * Tensor(np.zeros((), np.float32))
    return input * Tensor(_dropout_noise(keep_shape, p))


def dropout1d(input, p=0.5, training=True, inplace=False):
    return _channel_dropout(input, p, training, 1)


def dropout2d(input, p=0.5, training=True, inplace=False):
    return _channel_dropout(input, p, training, 2)


def dropout3d(input, p=0.5, training=True, inplace=False):
    return _channel_dropout(input, p, training, 3)


def alpha_dropout(input, p=0.5, training=False, inplace=False):
    """SELU-preserving dropout: dropped units take selu's negative saturation, then an affine
    correction keeps mean and variance."""
    if not training or p == 0:
        return input
    if p == 1:
        return input * Tensor(np.zeros((), np.float32))
    from ._random import default_generator
    # PyTorch's _dropout_impl with alpha: b = (noise - 1) * (alpha * a) + alpha * a * p,
    # noise *= a, output = input * noise + b, every op rounded in float32.
    alpha = 1.7580993408473766
    a = 1. / math.sqrt((alpha * alpha * p + 1) * (1 - p))
    noise = default_generator._bernoulli(input.shape, 1 - p)
    b = (noise - np.float32(1)) * np.float32(alpha * a) + np.float32(alpha * a * p)
    return input * Tensor(noise * np.float32(a)) + Tensor(b)


def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    return alpha_dropout(input, p, training)


# ---- losses ----------------------------------------------------------------------------

def _reduce(loss, reduction):
    _reduction(reduction)
    if reduction == "mean":
        return loss.mean()
    return loss.sum() if reduction == "sum" else loss


def l1_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    return _reduce((input - target).abs(), reduction)


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction="mean", beta=1.0):
    diff = input - target
    if beta == 0:
        return _reduce(diff.abs(), reduction)
    absolute = diff.abs()
    return _reduce(_where((absolute < beta).numpy(), diff * diff * (0.5 / beta), absolute - 0.5 * beta), reduction)


def huber_loss(input, target, reduction="mean", delta=1.0, weight=None):
    diff = input - target
    absolute = diff.abs()
    loss = _where((absolute < delta).numpy(), diff * diff * 0.5, (absolute - 0.5 * delta) * delta)
    return _reduce(loss * weight if weight is not None else loss, reduction)


def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction="mean"):
    """PyTorch clamps each log term at -100 so 0 and 1 stay finite."""
    log_p = input.log().clamp_min(-100.0)
    log_q = (1.0 - input).log().clamp_min(-100.0)
    loss = -(target * log_p + (1.0 - target) * log_q)
    return _reduce(loss * weight if weight is not None else loss, reduction)


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None,
                                     reduction="mean", pos_weight=None):
    """PyTorch's stable form: (1 - y) x + (1 + (pw - 1) y) softplus(-x)."""
    scale = 1.0 + (pos_weight - 1.0) * target if pos_weight is not None else None
    tail = softplus(-input)
    loss = (1.0 - target) * input + (scale * tail if scale is not None else tail)
    return _reduce(loss * weight if weight is not None else loss, reduction)


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
    """Negative log likelihood over class dim 1; mean divides by the summed target weights."""
    _reduction(reduction)
    labels = _integer_indices(target, "class labels") if isinstance(target, Tensor) else np.asarray(target)
    labels = np.asarray(labels, np.int64)
    expected = () if len(input.shape) == 1 else input.shape[:1] + input.shape[2:]
    if labels.shape != expected:
        raise ValueError(f"expected target shape {expected}, got {labels.shape}")
    unbatched = len(input.shape) == 1
    if unbatched:
        input, labels = input.unsqueeze(0), labels.reshape(1)
    classes = input.shape[1]
    ignored = labels == ignore_index
    safe = np.where(ignored, 0, labels)
    if np.any(safe < 0) or np.any(safe >= classes):
        raise IndexError("target out of bounds")
    moved = input.movedim(1, -1)
    picked = moved.gather(-1, safe[..., None]).squeeze(-1)
    w = np.ones(classes, np.float32) if weight is None else weight.numpy()
    target_weight = np.where(ignored, 0.0, w[safe]).astype(np.float32)
    loss = -picked * Tensor(target_weight)
    if reduction == "none":
        return loss.reshape(()) if unbatched else loss
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / float(target_weight.sum())


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None,
                  reduction="mean", label_smoothing=0.0):
    """Class indices or class probabilities, N-D inputs (class dim 1), weights, ignore_index,
    label smoothing and every reduction, following PyTorch's formulas. The plain 2-D,
    index-target mean case runs the native kernel."""
    _reduction(reduction)
    probabilities = isinstance(target, Tensor) and target.dtype in ("float32", "float64", "float16", "bfloat16")
    if (not probabilities and weight is None and label_smoothing == 0 and reduction == "mean"
            and isinstance(input, Tensor) and len(input.shape) == 2):
        return _cross_entropy_native(input, target, ignore_index=ignore_index)
    if not isinstance(input, Tensor):
        raise TypeError("cross_entropy input must be a Tensor")
    if probabilities and target.shape != input.shape:
        raise TypeError("class-probability targets must match the input shape")
    class_dim = 1 if len(input.shape) > 1 else 0
    classes = input.shape[class_dim]
    logp = log_softmax(input, class_dim)
    if probabilities:
        t = target if target.dtype == "float32" else target.float()
        if label_smoothing:
            t = t * (1.0 - label_smoothing) + label_smoothing / classes
        weighted = logp * t
        if weight is not None:
            shape = [1] * len(input.shape)
            shape[class_dim] = classes
            weighted = weighted * weight.reshape(shape)
        per_item = -weighted.sum(class_dim)
        if reduction == "none":
            return per_item
        total = per_item.sum()
        return total if reduction == "sum" else total / float(max(input.numel() // classes, 1))
    nll = nll_loss(logp, target, weight, ignore_index=ignore_index, reduction=reduction)
    if not label_smoothing:
        return nll
    labels = np.asarray(_integer_indices(target, "class labels") if isinstance(target, Tensor) else target)
    ignored = labels == ignore_index
    if weight is not None:
        shape = [1] * len(input.shape)
        shape[class_dim] = classes
        smooth = -(logp * weight.reshape(shape)).sum(class_dim)
    else:
        smooth = -logp.sum(class_dim)
    smooth = smooth.masked_fill(ignored, 0.0)
    if reduction == "mean":
        if weight is not None:
            denominator = float(weight.numpy()[labels[~ignored]].sum())
        else:
            denominator = float((~ignored).sum())
        smooth = smooth.sum() / denominator
    elif reduction == "sum":
        smooth = smooth.sum()
    return nll * (1.0 - label_smoothing) + smooth * (label_smoothing / classes)


def kl_div(input, target, size_average=None, reduce=None, reduction="mean", log_target=False):
    if log_target:
        loss = target.exp() * (target - input)
    else:
        loss = target.xlogy(target) - target * input
    if reduction == "batchmean":
        return loss.sum() / float(input.shape[0] if input.shape else 1)
    return _reduce(loss, reduction)


def poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-8, reduce=None,
                     reduction="mean"):
    loss = input.exp() - target * input if log_input else input - target * (input + eps).log()
    if full:
        # Stirling term where target > 1, with gradients to target like PyTorch.
        stirling = target * target.log() - target + (target * (2 * np.pi)).log() * 0.5
        zero = target.masked_fill(np.ones(target.shape, bool), 0.0)
        loss = loss + _where(target.numpy() > 1, stirling, zero)
    return _reduce(loss, reduction)


def gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction="mean"):
    """var is clamped at eps for the value only, as PyTorch's no_grad clamp_ does."""
    if isinstance(var, (int, float)):
        var = Tensor(np.full(input.shape, var, np.float32))
    if var.shape != input.shape and var.shape == input.shape[:-1]:
        var = var.unsqueeze(-1)
    clamped = var + (var.clamp_min(eps) - var).detach()
    loss = (clamped.log() + (input - target) ** 2 / clamped) * 0.5
    if full:
        loss = loss + 0.5 * float(np.log(2 * np.pi))
    return _reduce(loss, reduction)


def margin_ranking_loss(input1, input2, target, margin=0.0, size_average=None, reduce=None, reduction="mean"):
    return _reduce((-target * (input1 - input2) + margin).clamp_min(0.0), reduction)


def hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction="mean"):
    y = target.numpy()
    zero = input.masked_fill(np.ones(input.shape, bool), 0.0)
    margin_part = _where(y != 1, (margin - input).clamp_min(0.0), zero)
    self_part = _where(y != -1, input, zero)
    return _reduce(margin_part + self_part, reduction)


def soft_margin_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    return _reduce((-input * target).exp().log1p(), reduction)


def cosine_embedding_loss(input1, input2, target, margin=0.0, size_average=None, reduce=None, reduction="mean"):
    """PyTorch: cos = <x1, x2> / sqrt((|x1|^2 + 1e-12) (|x2|^2 + 1e-12))."""
    dim = len(input1.shape) - 1
    product = (input1 * input2).sum(dim)
    magnitude = ((input1 * input1).sum(dim) + 1e-12) * ((input2 * input2).sum(dim) + 1e-12)
    cos = product / magnitude.sqrt()
    y = target.numpy()
    zero = cos.masked_fill(np.ones(cos.shape, bool), 0.0)
    loss = _where(y == 1, 1.0 - cos, zero) + _where(y == -1, (cos - margin).clamp_min(0.0), zero)
    return _reduce(loss, reduction)


def multilabel_soft_margin_loss(input, target, weight=None, size_average=None, reduce=None, reduction="mean"):
    loss = -(target * logsigmoid(input) + (1.0 - target) * logsigmoid(-input))
    if weight is not None:
        loss = loss * weight
    loss = loss.mean(len(input.shape) - 1)
    return _reduce(loss, reduction)


def multi_margin_loss(input, target, p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction="mean"):
    if p not in (1, 2):
        raise ValueError("only p == 1 and p == 2 are supported")
    x = input.reshape(1, -1) if len(input.shape) < 2 else input
    labels = np.asarray(_integer_indices(target, "class labels") if isinstance(target, Tensor) else target).reshape(-1)
    picked = x.gather(1, labels[:, None])
    margins = (x - picked + margin).clamp_min(0.0)
    if p == 2:
        margins = margins * margins
    if weight is not None:
        margins = margins * Tensor(weight.numpy()[labels][:, None])
    own = np.zeros(x.shape, bool)
    own[np.arange(len(labels)), labels] = True
    loss = margins.masked_fill(own, 0.0).sum(1) / float(x.shape[1])
    loss = loss.reshape(()) if len(input.shape) < 2 else loss
    return _reduce(loss, reduction)


def _ctc_one(lp, labels, blank):
    """Alpha-beta recursion in float64 for one sequence: (nll, gradient w.r.t. lp) with PyTorch's
    gradient formula exp(lp) - exp(log(alpha*beta summed per class) + nll - lp)."""
    steps, classes = lp.shape
    ext = np.full(2 * len(labels) + 1, blank, np.int64)
    ext[1::2] = labels
    size = ext.size
    skip = np.zeros(size, bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    def step(prev, shift_ok):
        out = prev.copy()
        out[1:] = np.logaddexp(out[1:], prev[:-1])
        out[2:] = np.where(shift_ok[2:], np.logaddexp(out[2:], prev[:-2]), out[2:])
        return out
    alpha = np.full((steps, size), -np.inf)
    alpha[0, 0] = lp[0, blank]
    if size > 1:
        alpha[0, 1] = lp[0, ext[1]]
    for t in range(1, steps):
        alpha[t] = step(alpha[t - 1], skip) + lp[t, ext]
    nll = -np.logaddexp(alpha[-1, -1], alpha[-1, -2] if size > 1 else -np.inf)
    beta = np.full((steps, size), -np.inf)
    beta[-1, -1] = lp[-1, blank]
    if size > 1:
        beta[-1, -2] = lp[-1, ext[-2]]
    skip_back = np.zeros(size, bool)
    skip_back[:-2] = skip[2:]
    for t in range(steps - 2, -1, -1):
        reversed_step = step(beta[t + 1][::-1], skip_back[::-1])[::-1]
        beta[t] = reversed_step + lp[t, ext]
    joint = np.full((steps, classes), -np.inf)
    for c in np.unique(ext):
        joint[:, c] = np.logaddexp.reduce(alpha[:, ext == c] + beta[:, ext == c], axis=1)
    with np.errstate(invalid="ignore", over="ignore"):
        grad = np.exp(lp) - np.exp(joint + nll - lp)
    return nll, grad


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False):
    """Connectionist temporal classification loss on (T, N, C) or (T, C) log-probabilities;
    targets are padded (N, S) or concatenated 1-D. 'mean' divides each loss by its target length."""
    batched = len(log_probs.shape) == 3
    lp = log_probs._array().astype(np.float64)
    lp = lp if batched else lp[:, None]
    lengths = np.asarray(input_lengths._array() if isinstance(input_lengths, Tensor) else input_lengths).reshape(-1)
    target_lengths = np.asarray(target_lengths._array() if isinstance(target_lengths, Tensor)
                                else target_lengths).reshape(-1)
    labels = np.asarray(_integer_indices(targets, "targets"))
    starts = np.concatenate([[0], np.cumsum(target_lengths)[:-1]])
    losses, grads = np.zeros(lp.shape[1]), np.zeros(lp.shape)
    for b in range(lp.shape[1]):
        sequence = labels[b, :target_lengths[b]] if labels.ndim == 2 and batched else \
            labels.reshape(-1)[starts[b]:starts[b] + target_lengths[b]]
        nll, grad = _ctc_one(lp[:lengths[b], b], sequence, blank)
        if zero_infinity and np.isinf(nll):
            nll, grad = 0.0, np.zeros_like(grad)
        losses[b], grads[:lengths[b], b] = nll, grad
    if reduction == "mean":
        scale = 1.0 / np.maximum(target_lengths, 1) / lp.shape[1]
        value = (losses * scale).sum()
    elif reduction == "sum":
        scale, value = np.ones(lp.shape[1]), losses.sum()
    else:
        scale, value = None, losses
    if not batched:
        value = value if reduction != "none" else value.reshape(())
    from .linalg import _custom

    def vjp(g):
        per_item = np.broadcast_to(g, losses.shape) if scale is None else g * scale
        out = grads * per_item.reshape(1, -1, 1)
        return [out if batched else out[:, 0]]
    return _custom([log_probs], value, vjp)


def multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    """Each sample's targets are the class ids before the first -1; every (target j, non-target i)
    pair adds max(0, 1 - x[j] + x[i]), divided by the number of classes."""
    x = input.reshape(1, -1) if len(input.shape) < 2 else input
    labels = np.asarray(_integer_indices(target, "class labels")).reshape(x.shape)
    counts, is_target = np.zeros(x.shape, np.float32), np.zeros(x.shape, bool)
    for n, row in enumerate(labels):
        for j in row:
            if j < 0:
                break
            counts[n, j] += 1
            is_target[n, j] = True
    pairs = (1.0 - x.unsqueeze(2) + x.unsqueeze(1)).clamp_min(0.0)
    weights = counts[:, :, None] * (~is_target)[:, None, :]
    loss = (pairs * Tensor(weights.astype(np.float32))).sum((1, 2)) / float(x.shape[1])
    return _reduce(loss.reshape(()) if len(input.shape) < 2 else loss, reduction)


def triplet_margin_with_distance_loss(anchor, positive, negative, *, distance_function=None, margin=1.0,
                                      swap=False, reduction="mean"):
    distance = distance_function or pairwise_distance
    d_pos, d_neg = distance(anchor, positive), distance(anchor, negative)
    if swap:
        d_neg = d_neg.minimum(distance(positive, negative))
    return _reduce((d_pos - d_neg + margin).clamp_min(0.0), reduction)


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6, swap=False,
                        size_average=None, reduce=None, reduction="mean"):
    return triplet_margin_with_distance_loss(
        anchor, positive, negative, distance_function=lambda a, b: pairwise_distance(a, b, p, eps),
        margin=margin, swap=swap, reduction=reduction)


def linear_cross_entropy(input, linear_weight, target, *, linear_bias=None, weight=None, reduction="mean",
                         ignore_index=None, label_smoothing=0.0, options=None):
    """PyTorch's reference path: cross_entropy(linear(input, linear_weight), target). A weight of
    shape (C, d1, ..., D) gives K-dimensional logits (..., C, d1, ...). `options` (chunking) is ignored."""
    out_shape = tuple(linear_weight.shape[:-1])
    flat_bias = None if linear_bias is None else linear_bias.reshape(-1)
    logits = linear(input, linear_weight.reshape(-1, linear_weight.shape[-1]), flat_bias)
    logits = logits.reshape(*input.shape[:-1], *out_shape)
    return cross_entropy(logits, target, weight=weight, reduction=reduction,
                         ignore_index=-100 if ignore_index is None else ignore_index, label_smoothing=label_smoothing)


# Convolution, pooling, padding, normalization and interpolation live in convolution.py.
from . import convolution as _convolution
for _name in _convolution.EXPORTS:
    globals()[_name] = getattr(_convolution, _name)
