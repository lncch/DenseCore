import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)
X = rng.normal(size=(3, 8)).astype(np.float32)
Z = (rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))).astype(np.complex64)


def same(ours, theirs, rtol=1e-5):
    assert ours.dtype == str(theirs.dtype).split('.')[1], (ours.dtype, theirs.dtype)
    assert ours.shape == tuple(theirs.shape), (ours.shape, tuple(theirs.shape))
    np.testing.assert_allclose(ours.numpy(), theirs.resolve_conj().numpy(), rtol=rtol, atol=1e-5)


@pytest.mark.parametrize('norm', [None, 'ortho', 'forward'])
@pytest.mark.parametrize('call', [
    lambda F, x, z, n: F.fft(x, norm=n), lambda F, x, z, n: F.ifft(z, 7, 0, norm=n), lambda F, x, z, n: F.fft2(x, norm=n),
    lambda F, x, z, n: F.fftn(z, s=(4, 6), norm=n), lambda F, x, z, n: F.ifftn(z, norm=n), lambda F, x, z, n: F.rfft(x, 10, norm=n),
    lambda F, x, z, n: F.rfft2(x, norm=n), lambda F, x, z, n: F.irfft(z, norm=n), lambda F, x, z, n: F.irfftn(z, s=(3, 6), norm=n),
    lambda F, x, z, n: F.hfft(z, 9, norm=n), lambda F, x, z, n: F.hfft2(z, norm=n), lambda F, x, z, n: F.ihfft(x, norm=n),
    lambda F, x, z, n: F.ihfftn(x, norm=n), lambda F, x, z, n: F.fftshift(x), lambda F, x, z, n: F.ifftshift(z, dim=1)])
def test_fft_values_match_torch(call, norm):
    same(call(d.fft, d.tensor(X), d.tensor(Z), norm), call(torch.fft, torch.tensor(X), torch.tensor(Z), norm))


@pytest.mark.parametrize('call', [
    lambda F, x: F.irfft(x), lambda F, x: F.irfft(x, 10, norm='ortho'), lambda F, x: F.irfft2(x, s=(3, 5)),
    lambda F, x: F.irfftn(x, norm='forward'), lambda F, x: F.hfft(x, 7), lambda F, x: F.hfftn(x, norm='ortho'),
    lambda F, x: F.fftshift(x, dim=1)])
def test_real_output_transforms_have_torch_gradients(call):
    x, tx = d.tensor(X, requires_grad=True), torch.tensor(X, requires_grad=True)
    out, ref = call(d.fft, x), call(torch.fft, tx)
    same(out, ref.detach())
    w = rng.normal(size=tuple(ref.shape)).astype(np.float32)
    (out * d.tensor(w)).sum().backward()
    (ref * torch.from_numpy(w)).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.filterwarnings("ignore:A window was not provided")  # PyTorch's advice when window=None
@pytest.mark.filterwarnings("ignore:stft with return_complex=False is deprecated")  # both libraries warn
def test_frequencies_stft_istft_and_eig():
    same(d.fft.fftfreq(7, 0.5), torch.fft.fftfreq(7, 0.5))
    same(d.fft.rfftfreq(8), torch.fft.rfftfreq(8))
    signal = rng.normal(size=(2, 64)).astype(np.float32)
    window = np.hanning(16).astype(np.float32)
    for kwargs in ({'n_fft': 16}, {'n_fft': 16, 'hop_length': 4, 'window': window, 'normalized': True},
                   {'n_fft': 16, 'center': False, 'onesided': False}):
        dk = {k: d.tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        tk = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        spec, tspec = d.stft(d.tensor(signal), return_complex=True, **dk), torch.stft(torch.from_numpy(signal), return_complex=True, **tk)
        same(spec, tspec)
        if kwargs.get('center', True):
            same(d.istft(spec, length=64, **dk), torch.istft(tspec, length=64, **tk), rtol=1e-4)
    x, tx = d.tensor(signal, requires_grad=True), torch.tensor(signal, requires_grad=True)
    out, ref = d.stft(x, 16, return_complex=False), torch.stft(tx, 16, return_complex=False)
    same(out, ref.detach())
    w = rng.normal(size=tuple(ref.shape)).astype(np.float32)
    (out * d.tensor(w)).sum().backward()
    (ref * torch.from_numpy(w)).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-4, atol=1e-5)
    a = rng.normal(size=(4, 4)).astype(np.float32)
    values, vectors = d.linalg.eig(d.tensor(a))
    lhs = d.tensor(a).cfloat() @ vectors
    np.testing.assert_allclose(lhs.numpy(), (vectors * values.unsqueeze(0)).numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.sort_complex(d.linalg.eigvals(d.tensor(a)).numpy()),
                               np.sort_complex(torch.linalg.eigvals(torch.from_numpy(a)).numpy()), rtol=1e-5, atol=1e-6)
