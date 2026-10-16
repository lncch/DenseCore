"""torch.fft (NumPy pocketfft), plus stft/istft.

Complex results have no gradients (DenseCore has no complex autograd). The real-output
transforms irfft*/hfft* of real float32 inputs are differentiable: they are linear, and their
adjoint is Re(rfftn(g)) with the one-sided weights (2 for interior frequencies) and the
inverse normalization factor.
"""
import numpy as np

from .tensor import Tensor

_SWAP = {None: "forward", "backward": "forward", "forward": "backward", "ortho": "ortho"}


def _array(x):
    return x._array() if isinstance(x, Tensor) else np.asarray(x)


def _complex_dtype(x):
    return "complex128" if x.dtype in ("float64", "complex128") else "complex64"


def _real_dtype(x):
    return "float64" if x.dtype in ("float64", "complex128") else "float32"


def _complex_out(x, values):
    return Tensor(values, dtype=_complex_dtype(x))


def _dims(x, dim, s, default_all):
    if dim is None:
        if s is not None:
            return list(range(-len(s), 0))
        return list(range(len(x.shape))) if default_all else [-1]
    return [dim] if isinstance(dim, int) else list(dim)


def fft(input, n=None, dim=-1, norm=None):
    return _complex_out(input, np.fft.fft(_array(input), n, dim, norm))


def ifft(input, n=None, dim=-1, norm=None):
    return _complex_out(input, np.fft.ifft(_array(input), n, dim, norm))


def fft2(input, s=None, dim=(-2, -1), norm=None):
    return fftn(input, s, dim, norm)


def ifft2(input, s=None, dim=(-2, -1), norm=None):
    return ifftn(input, s, dim, norm)


def fftn(input, s=None, dim=None, norm=None):
    axes = _dims(input, dim, s, True)
    return _complex_out(input, np.fft.fftn(_array(input), s, axes, norm) if axes else _array(input).astype(complex))


def ifftn(input, s=None, dim=None, norm=None):
    axes = _dims(input, dim, s, True)
    return _complex_out(input, np.fft.ifftn(_array(input), s, axes, norm) if axes else _array(input).astype(complex))


def rfft(input, n=None, dim=-1, norm=None):
    return rfftn(input, None if n is None else [n], [dim], norm)


def rfft2(input, s=None, dim=(-2, -1), norm=None):
    return rfftn(input, s, dim, norm)


def rfftn(input, s=None, dim=None, norm=None):
    if input.dtype.startswith("complex"):
        raise RuntimeError("rfftn expects a real input tensor")
    axes = _dims(input, dim, s, True)
    return _complex_out(input, np.fft.rfftn(_array(input), s, axes, norm))


def _scale(norm, size, inverse):
    """The factor an inverse (or forward) transform applies under a given norm."""
    norm = norm or "backward"
    if norm == "ortho":
        return 1.0 / np.sqrt(size)
    if inverse:
        return 1.0 / size if norm == "backward" else 1.0
    return 1.0 if norm == "backward" else 1.0 / size


def _irfftn(input, s, axes, norm):
    """irfftn whose gradient (for real inputs) is Re(rfftn(g)) * one-sided weights * scale."""
    x = _array(input)
    values = np.fft.irfftn(x, s, axes, norm)
    out_dtype = _real_dtype(input)
    if not (input.dtype == "float32" and input.requires_grad):
        return Tensor(values, dtype=out_dtype)
    from .linalg import _custom
    out_shape = values.shape
    last = axes[-1] % x.ndim
    lengths = [out_shape[a % x.ndim] for a in axes]
    total = int(np.prod(lengths))
    scale = _scale(norm, total, inverse=True)

    def vjp(g):
        spectrum = np.fft.rfftn(g, lengths, axes, norm="backward").real
        n_last = out_shape[last]
        weights = np.full(spectrum.shape[last], 2.0)
        weights[0] = 1.0
        if n_last % 2 == 0:
            weights[-1] = 1.0
        shape = [1] * spectrum.ndim
        shape[last] = -1
        grad = spectrum * weights.reshape(shape) * scale
        # Match the input's extent along each transformed axis (zero-pad or crop).
        result = np.zeros(x.shape)
        region = tuple(slice(0, min(x.shape[i], grad.shape[i])) for i in range(x.ndim))
        result[region] = grad[region]
        return [result]
    return _custom([input], values, vjp)


def irfft(input, n=None, dim=-1, norm=None):
    return _irfftn(input, None if n is None else [n], [dim], norm)


def irfft2(input, s=None, dim=(-2, -1), norm=None):
    return irfftn(input, s, dim, norm)


def irfftn(input, s=None, dim=None, norm=None):
    return _irfftn(input, s, _dims(input, dim, s, True), norm)


def hfft(input, n=None, dim=-1, norm=None):
    return hfftn(input, None if n is None else [n], [dim], norm)


def hfft2(input, s=None, dim=(-2, -1), norm=None):
    return hfftn(input, s, dim, norm)


def hfftn(input, s=None, dim=None, norm=None):
    """PyTorch: hfftn(x) = irfftn(conj(x)) with the normalization swapped."""
    x = input if not input.dtype.startswith("complex") else input.conj()
    return _irfftn(x, s, _dims(input, dim, s, True), _SWAP[norm])


def ihfft(input, n=None, dim=-1, norm=None):
    return ihfftn(input, None if n is None else [n], [dim], norm)


def ihfft2(input, s=None, dim=(-2, -1), norm=None):
    return ihfftn(input, s, dim, norm)


def ihfftn(input, s=None, dim=None, norm=None):
    """PyTorch: ihfftn(x) = conj(rfftn(x)) with the normalization swapped."""
    axes = _dims(input, dim, s, True)
    return _complex_out(input, np.conj(np.fft.rfftn(_array(input), s, axes, _SWAP[norm])))


def fftfreq(n, d=1.0, *, dtype=None, **_):
    return Tensor(np.fft.fftfreq(n, d), dtype=dtype or "float32")


def rfftfreq(n, d=1.0, *, dtype=None, **_):
    return Tensor(np.fft.rfftfreq(n, d), dtype=dtype or "float32")


def fftshift(input, dim=None):
    dims = list(range(len(input.shape))) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    return input.roll([input.shape[d] // 2 for d in dims], dims) if dims else input


def ifftshift(input, dim=None):
    dims = list(range(len(input.shape))) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    return input.roll([-(input.shape[d] // 2) for d in dims], dims) if dims else input


# ---- short-time Fourier transform ---------------------------------------------------

def _window(window, n_fft, win_length):
    win_length = win_length or n_fft
    w = np.ones(win_length) if window is None else _array(window).astype(np.float64)
    left = (n_fft - win_length) // 2
    return np.pad(w, (left, n_fft - win_length - left))


def stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode="reflect",
         normalized=False, onesided=None, return_complex=None, align_to_window=None):
    x = _array(input).astype(np.complex128 if input.dtype.startswith("complex") else np.float64)
    batched = x.ndim == 2
    x = x if batched else x[None]
    hop = hop_length or n_fft // 4
    signal_length = x.shape[-1]
    # Source sample of each padded position (for the gradient of the padding).
    source_index = np.arange(signal_length)
    if center:
        mode = {"reflect": "reflect", "constant": "constant", "replicate": "edge", "circular": "wrap"}[pad_mode]
        x = np.pad(x, ((0, 0), (n_fft // 2, n_fft // 2)), mode=mode)
        source_index = np.pad(source_index, (n_fft // 2, n_fft // 2), mode=mode) if mode != "constant" else \
            np.pad(source_index, (n_fft // 2, n_fft // 2), constant_values=-1)
    w = _window(window, n_fft, win_length)
    frames = 1 + (x.shape[-1] - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    segments = x[:, index] * w                                     # (batch, frames, n_fft)
    one_sided = (onesided if onesided is not None else not np.iscomplexobj(x))
    spectrum = np.fft.rfft(segments, axis=-1) if one_sided else np.fft.fft(segments, axis=-1)
    if normalized:
        spectrum = spectrum / np.sqrt(n_fft)
    out = np.swapaxes(spectrum, -1, -2)                           # (batch, freq, frames)
    out = out if batched else out[0]
    if return_complex is False:
        import warnings
        warnings.warn("stft with return_complex=False is deprecated, as in PyTorch; use view_as_real "
                      "on the complex output instead", UserWarning, stacklevel=2)
        # Deprecated real layout: (..., 2) holding real and imaginary parts. Real in, real out,
        # so it is differentiable: the transform is linear and the adjoint is written below.
        values = np.stack([out.real, out.imag], -1)
        if not (input.dtype == "float32" and input.requires_grad):
            return Tensor(values, dtype=_real_dtype(input))
        from .linalg import _custom
        scale = 1 / np.sqrt(n_fft) if normalized else 1.0

        def vjp(g):
            g = g if batched else g[None]
            spectrum = (g[..., 0] + 1j * g[..., 1]) * scale            # (batch, freq, frames)
            full = np.zeros(spectrum.shape[:-2] + (n_fft, spectrum.shape[-1]), complex)
            full[..., :spectrum.shape[-2], :] = spectrum
            per_frame = (np.fft.ifft(full, axis=-2) * n_fft).real       # sum_k G_k e^{+2 pi i k t / n}
            per_frame = np.swapaxes(per_frame, -1, -2) * w              # (batch, frames, n_fft)
            padded = np.zeros((per_frame.shape[0], len(source_index)))
            for f in range(per_frame.shape[1]):
                padded[:, f * hop:f * hop + n_fft] += per_frame[:, f]
            grad = np.zeros((per_frame.shape[0], signal_length))
            keep = source_index >= 0
            for b in range(grad.shape[0]):
                np.add.at(grad[b], source_index[keep], padded[b, keep])
            return [grad if batched else grad[0]]
        return _custom([input], values, vjp)
    return Tensor(out, dtype="complex128" if input.dtype in ("float64", "complex128") else "complex64")


def istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False,
          onesided=None, length=None, return_complex=False):
    z = _array(input)
    batched = z.ndim == 3
    z = z if batched else z[None]
    hop = hop_length or n_fft // 4
    w = _window(window, n_fft, win_length)
    spectrum = np.swapaxes(z, -1, -2)                              # (batch, frames, freq)
    if normalized:
        spectrum = spectrum * np.sqrt(n_fft)
    one_sided = onesided if onesided is not None else spectrum.shape[-1] != n_fft
    frames = np.fft.irfft(spectrum, n_fft, axis=-1) if one_sided else np.fft.ifft(spectrum, n_fft, axis=-1)
    if not return_complex:
        frames = frames.real
    count = spectrum.shape[-2]
    total = n_fft + hop * (count - 1)
    out = np.zeros((z.shape[0], total), dtype=frames.dtype)
    envelope = np.zeros(total)
    for i in range(count):
        out[:, i * hop:i * hop + n_fft] += frames[:, i] * w
        envelope[i * hop:i * hop + n_fft] += w * w
    start = n_fft // 2 if center else 0
    end = total - n_fft // 2 if center else total
    if length is not None:
        end = start + length
    out, envelope = out[:, start:end], envelope[start:end]
    with np.errstate(all="ignore"):
        out = np.where(envelope > 1e-11, out / np.where(envelope > 1e-11, envelope, 1), 0)
    if length is not None and out.shape[-1] < length:
        out = np.pad(out, ((0, 0), (0, length - out.shape[-1])))
    out = out if batched else out[0]
    real = "float64" if input.dtype == "complex128" else "float32"
    return Tensor(out, dtype=(("complex128" if real == "float64" else "complex64") if return_complex else real))
