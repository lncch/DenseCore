"""softmax, log_softmax (last axis) and cross_entropy match PyTorch bit for bit, values and
gradients, including ignore_index rows."""
import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')


@pytest.mark.mac_exact
def test_softmax_and_log_softmax_bits():
    rng = np.random.default_rng(8)
    bad = []
    for shape in [(3,), (2, 1), (4, 3), (5, 4), (7, 13), (12, 4, 64, 64), (768, 65), (3, 1029)]:
        for name in ("softmax", "log_softmax"):
            x = (rng.normal(size=shape) * 4).astype(np.float32); w = rng.normal(size=shape).astype(np.float32)
            res = []
            for mod in (d, torch):
                F = torch.nn.functional if mod is torch else d.functional
                t = mod.tensor(x, requires_grad=True); y = getattr(F, name)(t, -1); (y * mod.tensor(w)).sum().backward()
                f = (lambda a: a.detach().numpy()) if mod is torch else (lambda a: a.numpy())
                res.append([f(y), f(t.grad)])
            for k, (p, q) in enumerate(zip(*res)):
                if not np.array_equal(p.view(np.uint32), q.view(np.uint32)): bad.append((shape, name, ["y", "dx"][k], int((p.view(np.uint32) != q.view(np.uint32)).sum())))
    assert bad == []


@pytest.mark.mac_exact
def test_cross_entropy_bits():
    rng = np.random.default_rng(10)
    bad = []
    for (n, c) in [(1, 3), (4, 5), (17, 7), (768, 65), (100, 1000), (3000, 10)]:
        for ign in (False, True):
            x = (rng.normal(size=(n, c)) * 3).astype(np.float32); t = rng.integers(0, c, n)
            if ign: t[::3] = -100
            res = []
            for mod in (d, torch):
                F = torch.nn.functional if mod is torch else d.functional
                xt = mod.tensor(x, requires_grad=True)
                loss = F.cross_entropy(xt, mod.tensor(t) if mod is torch else d.tensor(t)); loss.backward()
                f = (lambda a: a.detach().numpy()) if mod is torch else (lambda a: a.numpy())
                res.append([np.asarray(f(loss), np.float32), f(xt.grad)])
            for k, (p, q) in enumerate(zip(*res)):
                if not np.array_equal(p.view(np.uint32), q.view(np.uint32)): bad.append(((n, c), ign, ["loss", "dx"][k], int((p.view(np.uint32) != q.view(np.uint32)).sum())))
    assert bad == []
