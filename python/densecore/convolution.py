"""Convolution, pooling, padding, normalization and interpolation (nn.functional).

Every windowed op is built from one idea: NumPy computes, for each output window, the flat
input positions it reads (padding points at an appended fill element), and a native gather
returns those values exactly. matmul (conv) and native scatter_add (transposed conv) do the
rest, so gradients follow from those primitives.

shortcut: patch matrices are materialized (im2col), so memory grows with kernel size times
output size; fine for small models, upgrade to native conv kernels for speed.
"""
import math

import numpy as np

from .tensor import Tensor, where
from .shapes import flat_gather, scatter_add_flat


def _tuple(value, n):
    """An int, or a one-element sequence, repeats to n entries, as PyTorch does."""
    if isinstance(value, (tuple, list)):
        return tuple(value) * n if len(value) == 1 else tuple(value)
    return (value,) * n


def _batched(x, nd):
    """Add a batch dim to unbatched input; returns (x, was_unbatched)."""
    if len(x.shape) == nd + 1:
        return x.unsqueeze(0), True
    if len(x.shape) != nd + 2:
        raise ValueError(f"expected {nd + 1}-D or {nd + 2}-D input, got shape {x.shape}")
    return x, False


def _output_size(size, kernel, stride, dilation, left, right, ceil_mode=False):
    """PyTorch's pooling_output_shape: with ceil_mode, round up but drop a last window that
    would start beyond the input plus the left padding."""
    span = size + left + right - dilation * (kernel - 1) - 1 + (stride - 1 if ceil_mode else 0)
    out = span // stride + 1
    if ceil_mode and (out - 1) * stride >= size + left:
        out -= 1
    return max(out, 0)


def _window_positions(shape, kernel, stride, dilation, pads, ceil_mode=False):
    """Flat positions (N, C, *out, *kernel) into x.reshape(-1); padding -> x.numel()."""
    n, c, *spatial = shape
    outs, coords = [], []
    for size, k, s, d, (left, right) in zip(spatial, kernel, stride, dilation, pads):
        out = _output_size(size, k, s, d, left, right, ceil_mode)
        outs.append(out)
        coords.append(np.arange(out)[:, None] * s + np.arange(k)[None, :] * d - left)  # (out, k)
    nd = len(spatial)
    flat = np.zeros([1] * nd * 2, np.int64)
    valid = np.ones([1] * nd * 2, bool)
    stride_flat = 1
    for i in reversed(range(nd)):
        c_i = coords[i]
        shape_i = [1] * (2 * nd)
        shape_i[i], shape_i[nd + i] = c_i.shape
        c_i = c_i.reshape(shape_i)
        valid = valid & (c_i >= 0) & (c_i < spatial[i])
        flat = flat + np.clip(c_i, 0, spatial[i] - 1) * stride_flat
        stride_flat *= spatial[i]
    plane = int(np.prod(spatial)) if spatial else 1
    base = (np.arange(n)[:, None] * c + np.arange(c)[None, :]) * plane
    base = base.reshape([n, c] + [1] * (2 * nd))
    positions = np.where(valid[None, None], flat[None, None] + base, n * c * plane)
    return positions, tuple(outs)


def _gather_windows(x, positions, fill):
    """Gather from x's values plus one trailing fill element (where padding positions point)."""
    from . import cat
    if x.dtype == "float32":
        extended = cat([x.reshape(x.numel()), Tensor(np.array([fill], np.float32))])
    else:
        extended = Tensor(np.append(x._array().ravel(), fill), dtype=x.dtype)
    return flat_gather(extended, positions)


def _pads(padding, kernel, dilation, nd):
    if padding == "valid":
        return [(0, 0)] * nd
    if padding == "same":
        totals = [d * (k - 1) for k, d in zip(kernel, dilation)]
        return [(t // 2, t - t // 2) for t in totals]
    return [(p, p) for p in _tuple(padding, nd)]


def _conv(input, weight, bias, stride, padding, dilation, groups, nd):
    x, unbatched = _batched(input, nd)
    out_channels, group_in, *kernel = weight.shape
    stride, dilation = _tuple(stride, nd), _tuple(dilation, nd)
    pads = _pads(padding, kernel, dilation, nd)
    if x.shape[1] != group_in * groups:
        raise RuntimeError(f"expected input with {group_in * groups} channels, got {x.shape[1]}")
    positions, outs = _window_positions(x.shape, kernel, stride, dilation, pads)
    patches = _gather_windows(x, positions, 0.0)                 # (N, C, *out, *k)
    n = x.shape[0]
    locations = int(np.prod(outs)) if outs else 1
    k = int(np.prod(kernel)) if kernel else 1
    out_group = out_channels // groups
    # (N, G, Cg, *out, *k) -> (N, G, L, Cg*K)
    grouped = patches.reshape(n, groups, group_in, *outs, *kernel)
    order = [0, 1, *range(3, 3 + nd), 2, *range(3 + nd, 3 + 2 * nd)]
    columns = grouped.permute(order).reshape(n, groups, locations, group_in * k)
    kernels = weight.reshape(groups, out_group, group_in * k).transpose(1, 2)    # (G, Cg*K, Og)
    result = (columns @ kernels).permute(0, 1, 3, 2).reshape(n, out_channels, *outs)
    if bias is not None:
        result = result + bias.reshape(1, out_channels, *([1] * nd))
    return result.squeeze(0) if unbatched else result


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(input, weight, bias, stride, padding, dilation, groups, 1)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(input, weight, bias, stride, padding, dilation, groups, 2)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(input, weight, bias, stride, padding, dilation, groups, 3)


def _conv_transpose(input, weight, bias, stride, padding, output_padding, groups, dilation, nd):
    x, unbatched = _batched(input, nd)
    in_channels, out_group, *kernel = weight.shape
    stride, dilation = _tuple(stride, nd), _tuple(dilation, nd)
    padding, output_padding = _tuple(padding, nd), _tuple(output_padding, nd)
    n, _, *spatial = x.shape
    group_in = in_channels // groups
    out_channels = out_group * groups
    full = [(i - 1) * s + d * (k - 1) + 1 + op for i, s, d, k, op in zip(spatial, stride, dilation, kernel, output_padding)]
    outs = [f - 2 * p for f, p in zip(full, padding)]
    locations = int(np.prod(spatial)) if spatial else 1
    k = int(np.prod(kernel)) if kernel else 1
    # contributions (N, G, L, Og*K): each input location times its kernel.
    rows = x.reshape(n, groups, group_in, locations).transpose(2, 3)               # (N, G, L, Cg)
    kernels = weight.reshape(groups, group_in, out_group * k)                    # (G, Cg, Og*K)
    contributions = (rows @ kernels).reshape(n, groups, locations, out_group, *kernel)
    # Output coordinate per (input location, kernel offset), after cropping the padding.
    grids = np.meshgrid(*[np.arange(i) for i in spatial], indexing="ij")
    offsets = np.meshgrid(*[np.arange(kk) for kk in kernel], indexing="ij")
    flat = np.zeros((locations, k), np.int64)
    valid = np.ones((locations, k), bool)
    stride_flat = 1
    for d_i in reversed(range(nd)):
        coord = grids[d_i].reshape(-1, 1) * stride[d_i] + offsets[d_i].reshape(1, -1) * dilation[d_i] - padding[d_i]
        valid &= (coord >= 0) & (coord < outs[d_i])
        flat += np.clip(coord, 0, max(outs[d_i] - 1, 0)) * stride_flat
        stride_flat *= outs[d_i]
    plane = stride_flat
    channel = (np.arange(groups)[:, None] * out_group + np.arange(out_group)[None, :])          # (G, Og)
    batch = np.arange(n)
    positions = (batch[:, None, None, None, None] * out_channels + channel[None, :, None, :, None]) * plane \
        + flat[None, None, :, None, :]
    positions = np.where(valid[None, None, :, None, :], positions, n * out_channels * plane)
    total = n * out_channels * plane
    base = Tensor(np.zeros(total + 1, np.float32))
    summed = scatter_add_flat(base, positions, contributions.reshape(positions.shape))
    result = summed[:total].reshape(n, out_channels, *outs)
    if bias is not None:
        result = result + bias.reshape(1, out_channels, *([1] * nd))
    return result.squeeze(0) if unbatched else result


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    return _conv_transpose(input, weight, bias, stride, padding, output_padding, groups, dilation, 1)


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    return _conv_transpose(input, weight, bias, stride, padding, output_padding, groups, dilation, 2)


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    return _conv_transpose(input, weight, bias, stride, padding, output_padding, groups, dilation, 3)


# ---- pooling ---------------------------------------------------------------------------

def _max_pool(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices, nd):
    x, unbatched = _batched(input, nd)
    kernel = _tuple(kernel_size, nd)
    stride = _tuple(stride if stride not in (None, []) else kernel_size, nd)
    dilation = _tuple(dilation, nd)
    pads = [(p, p) for p in _tuple(padding, nd)]
    positions, outs = _window_positions(x.shape, kernel, stride, dilation, pads, ceil_mode)
    n, c = x.shape[:2]
    windows = _gather_windows(x, positions, -np.inf).reshape(n, c, *outs, -1)
    picked = windows.max(-1)  # first maximum, PyTorch's rule; its gradient goes there
    values = picked.values
    if return_indices:
        chosen = np.take_along_axis(positions.reshape(n, c, *outs, -1), picked.indices.numpy()[..., None], -1)[..., 0]
        plane = int(np.prod(x.shape[2:]))
        indices = Tensor(chosen % plane, dtype="int64")
        if unbatched:
            return values.squeeze(0), indices.squeeze(0)
        return values, indices
    return values.squeeze(0) if unbatched else values


def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return _max_pool(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices, 1)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return _max_pool(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices, 2)


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return _max_pool(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices, 3)


def _avg_pool(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, nd):
    x, unbatched = _batched(input, nd)
    kernel = _tuple(kernel_size, nd)
    stride = _tuple(stride if stride not in (None, []) else kernel_size, nd)
    pad = _tuple(padding, nd)
    positions, outs = _window_positions(x.shape, kernel, stride, (1,) * nd, [(p, p) for p in pad], ceil_mode)
    n, c, *spatial = x.shape
    total = _gather_windows(x, positions, 0.0).reshape(n, c, *outs, -1).sum(-1)
    if divisor_override:
        divisor = np.full(outs, float(divisor_override))
    else:
        # PyTorch: windows are clipped to input + padding; padding counts only if asked.
        divisor = np.ones(outs)
        for d_i, (size, k, s, p) in enumerate(zip(spatial, kernel, stride, pad)):
            start = np.arange(outs[d_i]) * s - p
            end = np.minimum(start + k, size + p)
            length = end - start if count_include_pad else np.minimum(end, size) - np.maximum(start, 0)
            shape = [1] * nd
            shape[d_i] = outs[d_i]
            divisor = divisor * length.reshape(shape)
    result = total / Tensor(divisor.astype(np.float32))
    return result.squeeze(0) if unbatched else result


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return _avg_pool(input, kernel_size, stride, padding, ceil_mode, count_include_pad, None, 1)


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return _avg_pool(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, 2)


def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return _avg_pool(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, 3)


def _adaptive(input, output_size, nd, kind):
    x, unbatched = _batched(input, nd)
    n, c, *spatial = x.shape
    outs = [s if o is None else o for s, o in zip(spatial, _tuple(output_size, nd))]
    starts = [np.floor(np.arange(o) * s / o).astype(int) for s, o in zip(spatial, outs)]
    ends = [np.ceil((np.arange(o) + 1) * s / o).astype(int) for s, o in zip(spatial, outs)]
    widths = [int((e - b).max()) if o else 0 for b, e, o in zip(starts, ends, outs)]
    flat = np.zeros([1] * (2 * nd), np.int64)
    valid = np.ones([1] * (2 * nd), bool)
    stride_flat = 1
    for i in reversed(range(nd)):
        coord = starts[i][:, None] + np.arange(widths[i])[None, :]
        shape = [1] * (2 * nd)
        shape[i], shape[nd + i] = coord.shape
        ok = (coord < ends[i][:, None]).reshape(shape)
        valid = valid & ok
        flat = flat + np.clip(coord, 0, spatial[i] - 1).reshape(shape) * stride_flat
        stride_flat *= spatial[i]
    plane = stride_flat
    base = ((np.arange(n)[:, None] * c + np.arange(c)[None, :]) * plane).reshape([n, c] + [1] * (2 * nd))
    positions = np.where(valid[None, None], flat[None, None] + base, n * c * plane)
    fill = -np.inf if kind == "max" else 0.0
    windows = _gather_windows(x, positions, fill).reshape(n, c, *outs, int(np.prod(widths)))
    if kind == "max":
        picked = windows.max(-1)
        result = picked.values
        indices = np.take_along_axis(positions.reshape(n, c, *outs, -1), picked.indices.numpy()[..., None], -1)[..., 0] % plane
    else:
        counts = np.ones(outs)
        for i in range(nd):
            shape = [1] * nd
            shape[i] = outs[i]
            counts = counts * (ends[i] - starts[i]).reshape(shape)
        result = windows.sum(-1) / Tensor(counts.astype(np.float32))
        indices = None
    if unbatched:
        result = result.squeeze(0)
        indices = None if indices is None else indices[0]
    return result, indices


def adaptive_avg_pool1d(input, output_size):
    return _adaptive(input, output_size, 1, "avg")[0]


def adaptive_avg_pool2d(input, output_size):
    return _adaptive(input, output_size, 2, "avg")[0]


def adaptive_avg_pool3d(input, output_size):
    return _adaptive(input, output_size, 3, "avg")[0]


def _adaptive_max(input, output_size, return_indices, nd):
    values, indices = _adaptive(input, output_size, nd, "max")
    return (values, Tensor(indices, dtype="int64")) if return_indices else values


def adaptive_max_pool1d(input, output_size, return_indices=False):
    return _adaptive_max(input, output_size, return_indices, 1)


def adaptive_max_pool2d(input, output_size, return_indices=False):
    return _adaptive_max(input, output_size, return_indices, 2)


def adaptive_max_pool3d(input, output_size, return_indices=False):
    return _adaptive_max(input, output_size, return_indices, 3)


def _max_unpool(input, indices, kernel_size, stride, padding, output_size, nd):
    x, unbatched = _batched(input, nd)
    kernel, stride = _tuple(kernel_size, nd), _tuple(stride if stride else kernel_size, nd)
    padding = _tuple(padding, nd)
    n, c, *spatial = x.shape
    if output_size is None:
        outs = [(i - 1) * s - 2 * p + k for i, s, p, k in zip(spatial, stride, padding, kernel)]
    else:
        outs = list(output_size)[-nd:]
    plane = int(np.prod(outs))
    index = indices.numpy() if not unbatched else indices.numpy()[None]
    base = ((np.arange(n)[:, None] * c + np.arange(c)[None, :]) * plane).reshape([n, c] + [1] * nd)
    positions = index + base
    from .shapes import overwrite_flat
    zeros = Tensor(np.zeros((n, c, *outs), np.float32))
    # Values: the last write wins. Gradient: every input reads the output gradient at its
    # index (PyTorch gathers), which the scatter-add path provides.
    summed = scatter_add_flat(zeros, positions, x)
    result = overwrite_flat(zeros, positions, x.detach()) + (summed - summed.detach())
    return result.squeeze(0) if unbatched else result


def max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return _max_unpool(input, indices, kernel_size, stride, padding, output_size, 1)


def max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return _max_unpool(input, indices, kernel_size, stride, padding, output_size, 2)


def max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return _max_unpool(input, indices, kernel_size, stride, padding, output_size, 3)


# ---- im2col ------------------------------------------------------------------------------

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """(N, C, H, W) -> (N, C*kh*kw, L) sliding-window columns."""
    x, unbatched = _batched(input, 2)
    kernel, dilation, stride = _tuple(kernel_size, 2), _tuple(dilation, 2), _tuple(stride, 2)
    positions, outs = _window_positions(x.shape, kernel, stride, dilation, [(p, p) for p in _tuple(padding, 2)])
    n, c = x.shape[:2]
    patches = _gather_windows(x, positions, 0.0)                         # (N, C, oh, ow, kh, kw)
    columns = patches.permute(0, 1, 4, 5, 2, 3).reshape(n, c * kernel[0] * kernel[1], outs[0] * outs[1])
    return columns.squeeze(0) if unbatched else columns


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Inverse of unfold: overlapping column entries are summed back into (N, C, H, W)."""
    x, unbatched = (input.unsqueeze(0), True) if len(input.shape) == 2 else (input, False)
    kernel, dilation, stride, padding = (_tuple(v, 2) for v in (kernel_size, dilation, stride, padding))
    out_h, out_w = _tuple(output_size, 2)
    n, ck, locations = x.shape
    c = ck // (kernel[0] * kernel[1])
    positions, outs = _window_positions((n, c, out_h, out_w), kernel, stride, dilation, [(p, p) for p in padding])
    order = positions.transpose(0, 1, 4, 5, 2, 3).reshape(n, ck, locations)
    base = Tensor(np.zeros(n * c * out_h * out_w + 1, np.float32))
    result = scatter_add_flat(base, order, x)[:n * c * out_h * out_w].reshape(n, c, out_h, out_w)
    return result.squeeze(0) if unbatched else result


# ---- padding -----------------------------------------------------------------------------

def pad(input, pad, mode="constant", value=None):
    """Pad trailing dimensions; pairs are (last_left, last_right, previous_left, ...).
    Negative amounts crop."""
    if len(pad) % 2:
        raise ValueError("padding length must be even")
    x = input
    dims = len(pad) // 2
    lefts = [pad[2 * i] for i in range(dims)][::-1]
    rights = [pad[2 * i + 1] for i in range(dims)][::-1]
    first = len(x.shape) - dims
    if mode == "replicate":
        # PyTorch maps each output position to the clamped input index, so a negative side can
        # cut past the other side's edge and still replicate it.
        maps = [np.clip(np.arange(x.shape[first + i] + l + r) - l, 0, x.shape[first + i] - 1)
                for i, (l, r) in enumerate(zip(lefts, rights))]
        grid = np.arange(x.numel()).reshape(x.shape)
        return flat_gather(x, grid[np.ix_(*[np.arange(n) for n in x.shape[:first]], *maps)])
    crop = tuple(slice(None) for _ in range(first)) + tuple(
        slice(max(-l, 0), x.shape[first + i] - max(-r, 0)) for i, (l, r) in enumerate(zip(lefts, rights)))
    x = x[crop]
    lefts, rights = [max(l, 0) for l in lefts], [max(r, 0) for r in rights]
    numpy_mode = {"constant": "constant", "reflect": "reflect", "replicate": "edge", "circular": "wrap"}[mode]
    maps = []
    for i, (l, r) in enumerate(zip(lefts, rights)):
        size = x.shape[first + i]
        if numpy_mode == "constant":
            maps.append(np.concatenate([np.full(l, -1), np.arange(size), np.full(r, -1)]))
        else:
            maps.append(np.pad(np.arange(size), (l, r), mode=numpy_mode))
    grid = np.arange(x.numel()).reshape(x.shape)
    positions = grid[np.ix_(*[np.arange(n) for n in x.shape[:first]], *[np.maximum(m, 0) for m in maps])] \
        if x.numel() else np.zeros(x.shape[:first] + tuple(len(m) for m in maps), np.int64)
    if numpy_mode == "constant":
        invalid = np.zeros(positions.shape, bool)
        for i, m in enumerate(maps):
            shape = [1] * len(positions.shape)
            shape[first + i] = len(m)
            invalid = invalid | (m < 0).reshape(shape)
        positions = np.where(invalid, x.numel(), positions)
        return _gather_windows(x, positions, 0.0 if value is None else float(value))
    return flat_gather(x, positions)


# ---- normalization ----------------------------------------------------------------------

def _stats_shape(x):
    return [1, x.shape[1]] + [1] * (len(x.shape) - 2)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    """Training mode normalizes with batch statistics (biased variance) and updates the
    running ones with the unbiased variance, in place, as PyTorch does."""
    if len(input.shape) < 2 and input.numel() == 0:
        return input.clone()  # PyTorch passes an empty 1-D input through
    dims = tuple(i for i in range(len(input.shape)) if i != 1)
    shape = _stats_shape(input)
    if training:
        mean = input.mean(dims, keepdim=True)
        var = ((input - mean) ** 2).mean(dims, keepdim=True)
        count = input.numel() // input.shape[1]
        if running_mean is not None:
            from .grad_mode import no_grad
            with no_grad():
                unbiased = var.detach().reshape(-1) * (count / max(count - 1, 1))
                running_mean.copy_(running_mean * (1 - momentum) + mean.detach().reshape(-1) * momentum)
                running_var.copy_(running_var * (1 - momentum) + unbiased * momentum)
    else:
        mean, var = running_mean.reshape(shape), running_var.reshape(shape)
    out = (input - mean) / (var + eps).sqrt()
    if weight is not None:
        out = out * weight.reshape(shape)
    return out + bias.reshape(shape) if bias is not None else out


def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True,
                  momentum=0.1, eps=1e-5):
    dims = tuple(range(2, len(input.shape)))
    mean = input.mean(dims, keepdim=True)
    var = ((input - mean) ** 2).mean(dims, keepdim=True)
    out = (input - mean) / (var + eps).sqrt()
    shape = _stats_shape(input)
    if weight is not None:
        out = out * weight.reshape(shape)
    return out + bias.reshape(shape) if bias is not None else out


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    n, c, *rest = input.shape
    grouped = input.reshape(n, num_groups, -1)
    mean = grouped.mean(-1, keepdim=True)
    var = ((grouped - mean) ** 2).mean(-1, keepdim=True)
    out = ((grouped - mean) / (var + eps).sqrt()).reshape(input.shape)
    shape = _stats_shape(input)
    if weight is not None:
        out = out * weight.reshape(shape)
    return out + bias.reshape(shape) if bias is not None else out


def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0):
    """b_c = a_c / (k + alpha/n * sum over a window of n neighbouring channels a^2)^beta."""
    squared = (input * input).unsqueeze(1)
    n, _, c, *rest = squared.shape
    flat = squared.reshape(n, 1, c, -1)
    padded = pad(flat, (0, 0, size // 2, (size - 1) // 2))
    summed = avg_pool2d(padded, (size, 1), stride=1).reshape(n, c, *rest)
    return input / (summed * alpha + k) ** beta


# ---- interpolation ---------------------------------------------------------------------

def _source_coordinates(out, size, scale, align_corners, mode):
    i = np.arange(out, dtype=np.float64)
    if mode in ("nearest", "nearest-exact"):
        ratio = (1.0 / scale) if scale else size / out
        src = np.floor((i + (0.5 if mode == "nearest-exact" else 0.0)) * ratio)
        return np.minimum(src, size - 1).astype(np.int64)
    if align_corners:
        return i * ((size - 1) / (out - 1)) if out > 1 else np.zeros(out)
    ratio = (1.0 / scale) if scale else size / out
    return np.maximum((i + 0.5) * ratio - 0.5, 0.0 if mode != "bicubic" else -np.inf)


def _cubic_weights(t, a=-0.75):
    def near(x):
        return ((a + 2) * x - (a + 3)) * x * x + 1
    def far(x):
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return np.stack([far(t + 1), near(t), near(1 - t), far(2 - t)], -1)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None,
                antialias=False):
    from .functional import gather
    nd = len(input.shape) - 2
    spatial = input.shape[2:]
    scales = _tuple(scale_factor, nd) if scale_factor is not None else (None,) * nd
    outs = _tuple(size, nd) if size is not None else tuple(int(math.floor(s * f)) for s, f in zip(spatial, scales))
    if recompute_scale_factor:
        scales = (None,) * nd
    if mode == "area":
        return _adaptive(input, outs, nd, "avg")[0]
    out = input
    for i, (o, s, f) in enumerate(zip(outs, spatial, scales)):
        axis = 2 + i
        if mode in ("nearest", "nearest-exact"):
            out = gather(out, axis, _source_coordinates(o, s, f, align_corners, mode))
            continue
        src = _source_coordinates(o, s, f, bool(align_corners), mode)
        lower = np.floor(src).astype(np.int64)
        t = (src - lower).astype(np.float32)
        shape = [1] * len(out.shape)
        shape[axis] = o
        if mode == "bicubic":
            taps = [np.clip(lower + offset, 0, s - 1) for offset in (-1, 0, 1, 2)]
            weights = _cubic_weights(t)
            out = sum(gather(out, axis, tap) * Tensor(weights[:, j].reshape(shape)) for j, tap in enumerate(taps))
        else:
            upper = np.minimum(lower + 1, s - 1)
            lower = np.clip(lower, 0, s - 1)
            out = gather(out, axis, lower) * Tensor((1 - t).reshape(shape)) + gather(out, axis, upper) * Tensor(t.reshape(shape))
    return out


def upsample_nearest(input, size=None, scale_factor=None):
    return interpolate(input, size, scale_factor, "nearest")


def upsample_bilinear(input, size=None, scale_factor=None):
    return interpolate(input, size, scale_factor, "bilinear", align_corners=True)


# ---- grid sampling ----------------------------------------------------------------------

def _reflect(c, low2, high2):
    """PyTorch's reflect_coordinates in float64: (value, slope) where slope is +1 or -1."""
    if low2 == high2:
        return np.zeros_like(c), np.zeros_like(c)
    low, span = low2 / 2, (high2 - low2) / 2
    offset = c - low
    distance = np.abs(offset)
    folds = np.floor(distance / span)
    extra = np.fmod(distance, span)
    even = folds % 2 == 0
    return np.where(even, extra + low, span - extra + low), np.where(offset < 0, -1.0, 1.0) * np.where(even, 1.0, -1.0)


def _pad_coordinate(c, size, padding_mode, align):
    """(value, slope) of a float64 source coordinate after padding; clipping has slope 0 at and
    beyond the edges, as in PyTorch."""
    slope = np.ones_like(c)
    if padding_mode == "reflection":
        c, slope = _reflect(c, 0.0, 2.0 * (size - 1)) if align else _reflect(c, -1.0, 2.0 * size - 1)
    if padding_mode in ("border", "reflection"):
        slope = slope * ((c > 0) & (c < size - 1))
        c = np.clip(c, 0, size - 1)
    return c, slope


def _pad_index(i, size, padding_mode, align):
    """The same padding applied to integer tap positions (bicubic)."""
    return _pad_coordinate(i, size, padding_mode, align)[0] if padding_mode != "zeros" else i


def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    """Sample input (N, C, [D,] H, W) at grid (N, [D',] H', W', k) positions in [-1, 1]; grid[..., 0]
    indexes W. Gradients reach input and grid (not for nearest)."""
    align = bool(align_corners)
    spatial = input.shape[2:]
    k = len(spatial)
    n, channels = input.shape[:2]
    out_spatial = grid.shape[1:-1]
    points = int(np.prod(out_spatial))
    # Coordinates are computed in float64 NumPy; the grid gradient enters through
    # delta * slope, where delta = g - g.detach() is zero in value.
    coords, slopes, deltas = [], [], []
    for axis in range(k):  # axis 0 is the first spatial dim; grid component k-1-axis drives it
        size = spatial[axis]
        g = grid[..., k - 1 - axis].reshape(n, 1, points)
        g64 = g._array().astype(np.float64)
        scale = (size - 1) / 2 if align else size / 2
        c = (g64 + 1.0) * scale if align else ((g64 + 1.0) * size - 1.0) / 2
        c, slope = (c, np.ones_like(c)) if mode == "bicubic" else _pad_coordinate(c, size, padding_mode, align)
        coords.append(c)
        slopes.append(slope * scale)
        deltas.append(g - g.detach())

    def offset_from(axis, floor):
        """coordinate - floor as a float32 tensor carrying the grid gradient."""
        return deltas[axis] * Tensor(slopes[axis].astype(np.float32)) + Tensor((coords[axis] - floor).astype(np.float32))
    strides = [int(np.prod(spatial[a + 1:])) for a in range(k)]
    volume = int(np.prod(spatial))
    base = (np.arange(n)[:, None, None] * channels + np.arange(channels)[None, :, None]) * volume

    def taps(indices):
        """input values at integer positions (each (n, 1, points)); outside positions read 0."""
        inside = np.ones(indices[0].shape, bool)
        flat = np.zeros(indices[0].shape, np.int64)
        for i, size, stride in zip(indices, spatial, strides):
            inside &= (i >= 0) & (i < size)
            flat = flat + np.clip(i, 0, size - 1).astype(np.int64) * stride
        picked = flat_gather(input, base + flat)
        return picked * Tensor(inside.astype(np.float32)) if not inside.all() else picked

    if mode == "nearest":
        out = taps([np.rint(c) for c in coords])
    elif mode == "bilinear":
        floors = [np.floor(c) for c in coords]
        fracs = [offset_from(axis, f) for axis, f in enumerate(floors)]
        out = None
        for corner in range(2 ** k):
            bits = [(corner >> (k - 1 - a)) & 1 for a in range(k)]
            weight = None
            for bit, frac in zip(bits, fracs):
                w = frac if bit else 1.0 - frac
                weight = w if weight is None else weight * w
            term = taps([f + bit for f, bit in zip(floors, bits)]) * weight
            out = term if out is None else out + term
    elif mode == "bicubic" and k == 2:
        a = -0.75
        floors = [np.floor(c) for c in coords]
        ts = [offset_from(axis, f) for axis, f in enumerate(floors)]

        def cubic(t):  # weights for taps at -1, 0, 1, 2
            near = lambda x: ((a + 2) * x - (a + 3)) * x * x + 1
            far = lambda x: ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
            return [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
        wy, wx = cubic(ts[0]), cubic(ts[1])
        out = None
        for i in range(4):
            yi = _pad_index(floors[0] - 1 + i, spatial[0], padding_mode, align)
            for j in range(4):
                xj = _pad_index(floors[1] - 1 + j, spatial[1], padding_mode, align)
                term = taps([yi, xj]) * (wy[i] * wx[j])
                out = term if out is None else out + term
    else:
        raise ValueError(f"grid_sample mode {mode!r} is unsupported for {k}-D input")
    return out.reshape(n, channels, *out_spatial)


_GRID_MODES, _GRID_PADDING = ("bilinear", "nearest", "bicubic"), ("zeros", "border", "reflection")


def grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners):
    return grid_sample(input, grid, _GRID_MODES[interpolation_mode], _GRID_PADDING[padding_mode], align_corners)


grid_sampler_3d = grid_sampler_2d


EXPORTS = [name for name in list(globals()) if not name.startswith("_") and callable(globals()[name])
           and name not in ("Tensor", "where", "flat_gather", "scatter_add_flat")]
