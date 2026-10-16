import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)


def well_conditioned(*shape):
    a = rng.normal(size=shape).astype(np.float32)
    square = len(shape) >= 2 and shape[-1] == shape[-2]
    return a + 3 * np.eye(shape[-1], dtype=np.float32) if square else a


def spd(n, batch=()):
    a = rng.normal(size=batch + (n, n)).astype(np.float32)
    return a @ np.swapaxes(a, -1, -2) + n * np.eye(n, dtype=np.float32)


def check(f, *arrays, rtol=1e-4, atol=1e-5):
    """Values against float64 PyTorch; gradients of a fixed random projection."""
    ours = [d.tensor(a, requires_grad=True) for a in arrays]
    theirs = [torch.tensor(a.astype(np.float64), requires_grad=True) for a in arrays]
    o, t = f(d, *ours), f(torch, *theirs)
    o, t = (o, t) if isinstance(t, (tuple, list)) else ((o,), (t,))
    loss, tloss = 0, 0
    for a, b in zip(o, t):
        assert a.shape == tuple(b.shape)
        np.testing.assert_allclose(a.float().numpy(), b.detach().numpy(), rtol=rtol, atol=atol)
        if b.requires_grad and b.dtype == torch.float64:
            w = rng.normal(size=tuple(b.shape))
            loss = loss + (a * d.tensor(w.astype(np.float32))).sum()
            tloss = tloss + (b * torch.from_numpy(w)).sum()
    loss.backward()
    tloss.backward()
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=rtol, atol=atol)


@pytest.mark.parametrize('f,shapes', [
    (lambda m, a: m.linalg.inv(a), [(4, 4)]), (lambda m, a: m.linalg.inv(a), [(2, 3, 3)]),
    (lambda m, a: m.linalg.det(a), [(4, 4)]), (lambda m, a: m.linalg.det(a), [(2, 3, 3)]),
    (lambda m, a: m.linalg.slogdet(a), [(4, 4)]), (lambda m, a: m.logdet(a @ a.mT), [(3, 3)]),
    (lambda m, a, b: m.linalg.solve(a, b), [(4, 4), (4, 2)]), (lambda m, a, b: m.linalg.solve(a, b), [(2, 3, 3), (3,)]),
    (lambda m, a, b: m.linalg.solve(a, b, left=False), [(3, 3), (2, 3)]),
    (lambda m, a, b: m.linalg.solve_triangular(a, b, upper=True), [(4, 4), (4, 3)]),
    (lambda m, a, b: m.linalg.solve_triangular(a, b, upper=False, left=False, unitriangular=True), [(3, 3), (2, 3)]),
    (lambda m, a: m.linalg.qr(a), [(5, 3)]), (lambda m, a: m.linalg.qr(a), [(3, 5)]), (lambda m, a: m.linalg.qr(a), [(4, 4)]),
    (lambda m, a: m.linalg.lu_factor(a)[0], [(4, 4)]), (lambda m, a: m.linalg.lu_factor(a)[0], [(3, 5)]),
    (lambda m, a: m.linalg.lu_factor(a)[0], [(5, 3)]), (lambda m, a: m.linalg.lu(a), [(4, 4)])])
def test_decompositions_and_solves_match_torch(f, shapes):
    check(f, *[well_conditioned(*s) for s in shapes])


def test_cholesky_family():
    a = spd(4)
    check(lambda m, x: m.linalg.cholesky(x @ x.mT + 4 * m.eye(4, dtype=x.dtype)), a)
    check(lambda m, x: m.linalg.cholesky(x @ x.mT + 4 * m.eye(4, dtype=x.dtype), upper=True), a)
    lower = np.linalg.cholesky(spd(3)).astype(np.float32)
    b = rng.normal(size=(3, 2)).astype(np.float32)
    check(lambda m, l, bb: m.cholesky_solve(bb, l), lower, b)
    check(lambda m, l: m.cholesky_inverse(l), lower)


def test_eigh_and_svd_through_sign_invariant_losses():
    a = rng.normal(size=(4, 4)).astype(np.float32)
    # Eigenvalues directly; eigenvectors through Q diag(L) Q^T and Q*Q (sign-invariant).
    check(lambda m, x: m.linalg.eigvalsh(x + x.mT), a)
    check(lambda m, x: (lambda v: (v[1] * v[1], v[1] @ m.diag_embed(v[0]) @ v[1].mT))(m.linalg.eigh(x + x.mT)), a)
    check(lambda m, x: m.linalg.svdvals(x), rng.normal(size=(5, 3)).astype(np.float32))
    check(lambda m, x: (lambda u: (u[0] * u[0], u[2] * u[2], u[0] @ m.diag_embed(u[1]) @ u[2]))(m.linalg.svd(x, full_matrices=False)),
          rng.normal(size=(3, 5)).astype(np.float32))


def test_lu_unpack_solve_and_legacy_names():
    a = well_conditioned(4, 4)
    b = rng.normal(size=(4, 2)).astype(np.float32)
    LU, piv = d.linalg.lu_factor(d.tensor(a))
    tLU, tpiv = torch.linalg.lu_factor(torch.from_numpy(a))
    np.testing.assert_array_equal(piv.numpy(), tpiv.numpy())
    for ours, theirs in zip(d.lu_unpack(LU, piv), torch.lu_unpack(tLU, tpiv)):
        np.testing.assert_allclose(ours.numpy(), theirs.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(d.linalg.lu_solve(LU, piv, d.tensor(b)).numpy(),
                               torch.linalg.lu_solve(tLU, tpiv, torch.from_numpy(b)).numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(d.inverse(d.tensor(a)).numpy(), torch.inverse(torch.from_numpy(a)).numpy(), rtol=1e-4, atol=1e-6)
    u, s, v = d.svd(d.tensor(a))
    np.testing.assert_allclose((u @ d.diag_embed(s) @ v.mT).numpy(), a, rtol=1e-4, atol=1e-5)
    assert d.linalg.inv_ex(d.tensor(a)).info.dtype == 'int32'
