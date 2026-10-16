import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(1)


def check(f, *arrays, rtol=1e-4, atol=1e-5):
    ours = [d.tensor(a, requires_grad=True) for a in arrays]
    theirs = [torch.tensor(a.astype(np.float64), requires_grad=True) for a in arrays]
    o, t = f(d, *ours), f(torch, *theirs)
    o, t = (o, t) if isinstance(t, (tuple, list)) else ((o,), (t,))
    loss, tloss = None, None
    for a, b in zip(o, t):
        assert a.shape == tuple(b.shape), (a.shape, tuple(b.shape))
        np.testing.assert_allclose(a.float().numpy(), b.detach().double().numpy(), rtol=rtol, atol=atol)
        if b.requires_grad and b.dtype == torch.float64 and b.numel():
            w = rng.normal(size=tuple(b.shape))
            term, tterm = (a * d.tensor(w.astype(np.float32))).sum(), (b * torch.from_numpy(w)).sum()
            loss, tloss = (term, tterm) if loss is None else (loss + term, tloss + tterm)
    if loss is not None:
        loss.backward()
        tloss.backward()
        for a, b in zip(ours, theirs):
            if b.grad is not None:
                np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=rtol, atol=atol)


A = rng.normal(size=(4, 4)).astype(np.float32)
TALL = rng.normal(size=(5, 3)).astype(np.float32)
SINGULAR = (rng.normal(size=(4, 2)) @ rng.normal(size=(2, 4))).astype(np.float32)


@pytest.mark.parametrize('f,arrays', [
    (lambda m, a: m.linalg.pinv(a), (TALL,)), (lambda m, a: m.linalg.pinv(a, rtol=1e-5), (SINGULAR,)),  # default rtol depends on the dtype's eps
    (lambda m, a: m.linalg.pinv(a, hermitian=True), (A + A.T,)), (lambda m, a: m.pinverse(a), (TALL.T,)),
    (lambda m, a, b: m.linalg.lstsq(a, b, driver='gelsd'), (TALL, rng.normal(size=(5, 2)).astype(np.float32))),
    (lambda m, a, b: m.linalg.lstsq(a, b).solution, (TALL.T, rng.normal(size=(3, 2)).astype(np.float32))),
    (lambda m, a: m.linalg.matrix_exp(a * 0.5), (A,)), (lambda m, a: m.linalg.matrix_exp(a * 3), (A,)),
    (lambda m, a: m.linalg.matrix_power(a, 3), (A,)), (lambda m, a: m.linalg.matrix_power(a + 3 * m.eye(4, dtype=a.dtype), -2), (A,)),
    (lambda m, a: m.linalg.matrix_power(a, 0), (A,)), (lambda m, a: m.linalg.polar(a), (A + 3 * np.eye(4, dtype=np.float32),)),
    (lambda m, a: m.linalg.matrix_sqrth(a @ a.mT + m.eye(4, dtype=a.dtype)), (A,)),
    (lambda m, a: m.linalg.cond(a + 3 * m.eye(4, dtype=a.dtype)), (A,)),
    (lambda m, a, b, c: m.linalg.multi_dot([a, b, c]), (A, TALL[:4], TALL.T)),
    (lambda m, x, y: m.linalg.vecdot(x, y), (TALL, TALL * 2)), (lambda m, x: m.linalg.vander(x[:, 0], N=4), (TALL,)),
    (lambda m, a: m.linalg.tensorinv(a.reshape(2, 2, 4), ind=2), (A + 3 * np.eye(4, dtype=np.float32),)),
    (lambda m, a, b: m.linalg.tensorsolve(a.reshape(4, 2, 2), b), (A + 3 * np.eye(4, dtype=np.float32), np.ones(4, np.float32))),
    # geqrf is not differentiable in PyTorch, so its outputs enter as leaf inputs.
    (lambda m, h, tau: m.linalg.householder_product(h, tau), [t.numpy() for t in torch.geqrf(torch.from_numpy(TALL))]),
    (lambda m, h, tau, c: m.ormqr(h, tau, c, transpose=True),
     [t.numpy() for t in torch.geqrf(torch.from_numpy(TALL))] + [rng.normal(size=(5, 2)).astype(np.float32)])])
def test_linalg_part2_matches_float64_torch(f, arrays):
    check(f, *arrays)


@pytest.mark.parametrize('ord', ['fro', 'nuc', 2, -2, 1, -1, float('inf'), float('-inf')])
def test_matrix_norms(ord):
    check(lambda m, a: m.linalg.matrix_norm(a, ord), rng.normal(size=(2, 4, 3)).astype(np.float32))


@pytest.mark.parametrize('ord', [2, 1, 0.5, 3.5, -1, float('inf'), float('-inf'), 0])
def test_vector_norms_and_zero_subgradients(ord):
    x = np.array([[0., 1.5, -2., 0.25], [0., 0., 0., 0.]], np.float32)
    check(lambda m, v: m.linalg.vector_norm(v, ord, dim=1), x)
    check(lambda m, v: m.linalg.norm(v, ord, dim=1), x)


def test_cond_fro_values():
    # PyTorch's own cond(A, 'fro') backward raises an in-place error, so values only here.
    a = A + 3 * np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(d.linalg.cond(d.tensor(a), 'fro').item(),
                               torch.linalg.cond(torch.from_numpy(a.astype(np.float64)), 'fro').item(), rtol=1e-5)


def test_matrix_rank_and_legacy_nuclear_norm():
    assert d.linalg.matrix_rank(d.tensor(SINGULAR)).item() == 2 == torch.linalg.matrix_rank(torch.from_numpy(SINGULAR)).item()
    assert d.linalg.matrix_rank(d.tensor(A), atol=10.0).item() == torch.linalg.matrix_rank(torch.from_numpy(A), atol=10.0).item()
    np.testing.assert_allclose(d.norm(d.tensor(A), 'nuc').item(), torch.norm(torch.from_numpy(A), 'nuc').item(), rtol=1e-5)
