import io
import warnings

import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)
Z = (rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))).astype(np.complex64)
R = rng.normal(size=(3, 4)).astype(np.float32)


def same(ours, theirs, rtol=1e-6):
    assert ours.dtype == str(theirs.dtype).split('.')[1], (ours.dtype, theirs.dtype)
    assert ours.shape == tuple(theirs.shape)
    np.testing.assert_allclose(ours.numpy(), theirs.resolve_conj().numpy(), rtol=rtol, atol=1e-6)


def test_creation_inference_views_and_round_trips():
    same(d.tensor(Z), torch.tensor(Z))
    same(d.tensor(Z.astype(np.complex128)), torch.tensor(Z.astype(np.complex128)))
    same(d.tensor([1 + 2j, 3j]), torch.tensor([1 + 2j, 3j]))
    same(d.tensor(Z).T[1:, ::2], torch.tensor(Z).T[1:, ::2])
    same(d.tensor(Z).reshape(4, 3), torch.tensor(Z).reshape(4, 3))
    buffer = io.BytesIO()
    d.save(d.tensor(Z), buffer)
    buffer.seek(0)
    same(d.load(buffer), torch.tensor(Z))
    assert d.tensor(Z).is_complex() and not d.tensor(R).is_complex() and d.cfloat == 'complex64'


def test_casts_parts_and_complex_functions():
    z, tz = d.tensor(Z), torch.tensor(Z)
    for f in [lambda t: t.real, lambda t: t.imag, lambda t: t.conj(), lambda t: t.abs(), lambda t: t.angle(),
              lambda t: t.sgn(), lambda t: t.cdouble(), lambda t: t.exp(), lambda t: t.log(), lambda t: t.sqrt(),
              lambda t: t.sin(), lambda t: t * t.conj(), lambda t: t.sum(), lambda t: t.mean(1), lambda t: t @ t.T]:
        same(f(z), f(tz), rtol=1e-5)
    r, tr = d.tensor(R), torch.tensor(R)
    same(r.cfloat(), tr.cfloat())
    same(d.complex(r, r * 2), torch.complex(tr, tr * 2))
    same(d.polar(r.abs(), r), torch.polar(tr.abs(), tr), rtol=1e-5)
    same(d.view_as_real(z), torch.view_as_real(tz))
    same(d.view_as_complex(d.view_as_real(z)), tz)
    with pytest.raises(RuntimeError):
        r.imag
    with pytest.raises(RuntimeError):
        z.sign()


def test_promotion_with_real_tensors_and_scalars():
    z, tz = d.tensor(Z), torch.tensor(Z)
    r, tr = d.tensor(R), torch.tensor(R)
    for f in [lambda a, b: a + b, lambda a, b: b * a, lambda a, b: a / (b + 3), lambda a, b: a - b.double(),
              lambda a, b: a * 1j, lambda a, b: 2 * a, lambda a, b: b * (1 + 1j), lambda a, b: a == a]:
        same(f(z, r), f(tz, tr), rtol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        np.testing.assert_array_equal(z.float().numpy(), Z.real)
    assert d.arange(5).sum().dtype == 'int64' and d.arange(5).sum().item() == 10
    with pytest.raises(RuntimeError):
        d.arange(5).mean()
    with pytest.raises(RuntimeError):
        z @ d.tensor(R.T)
