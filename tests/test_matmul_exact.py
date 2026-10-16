"""mm, matmul (incl. transposed operands) and linear match PyTorch bit for bit, values and
gradients: DenseCore makes the same Accelerate sgemm_ calls PyTorch makes."""
import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')


def test_matmul_and_linear_bits():
    rng = np.random.default_rng(3)
    bad = []
    def bits(a, b): return bool(np.all(np.asarray(a).view(np.uint32) == np.asarray(b).view(np.uint32)))
    def run(mod, f, arrays, w):
        ts = [mod.tensor(a, requires_grad=True) for a in arrays]
        out = f(mod, *ts); (out * mod.tensor(w)).sum().backward()
        g = (lambda t: t.detach().numpy()) if mod is torch else (lambda t: t.numpy())
        return [g(out)] + [g(t.grad) for t in ts]
    shapes = [(3, 4, 5), (64, 128, 384), (1, 7, 1), (5, 1, 3), (17, 33, 9), (768, 128, 65), (1, 5, 6), (6, 5, 1)]
    for (m, k, n) in shapes:
        a = rng.normal(size=(m, k)).astype(np.float32); b = rng.normal(size=(k, n)).astype(np.float32)
        cases = {
            "mm": ([a, b], lambda M, x, y: x @ y, (m, n)),
            "mm_aT": ([a.T.copy(), b], lambda M, x, y: x.T @ y, (m, n)),
            "mm_bT": ([a, b.T.copy()], lambda M, x, y: x @ y.T, (m, n)),
            "linear": ([a, b.T.copy()], lambda M, x, w: (torch.nn.functional if M is torch else d.functional).linear(x, w), (m, n)),
            "linear_bias": ([a, b.T.copy(), rng.normal(size=n).astype(np.float32)],
                            lambda M, x, w, bb: (torch.nn.functional if M is torch else d.functional).linear(x, w, bb), (m, n)),
            "linear_3d": ([np.stack([a, a * 0.5]), b.T.copy()], lambda M, x, w: (torch.nn.functional if M is torch else d.functional).linear(x, w), (2, m, n)),
        }
        for name, (arrs, f, oshape) in cases.items():
            w = rng.normal(size=oshape).astype(np.float32)
            for i, (x, y) in enumerate(zip(run(d, f, arrs, w), run(torch, f, arrs, w))):
                if not bits(x, y): bad.append(((m, k, n), name, ["out", "g0", "g1", "g2"][i]))
    assert bad == []


@pytest.mark.parametrize('transposed', [False, True])
def test_bmm_bits(transposed):
    # Small products use PyTorch's loop as compiled (fused, or vectorized when both operands
    # are contiguous along k); larger ones use one sgemm_ per batch item.
    rng = np.random.default_rng(1)
    for (bs, m, k, n) in [(10, 5, 10, 5), (2, 3, 4, 5), (48, 64, 32, 64), (3, 1, 7, 1), (5, 4, 16, 6), (3, 3, 17, 7), (12, 64, 32, 64)]:
        a = rng.normal(size=(bs, m, k)).astype(np.float32)
        bt = rng.normal(size=(bs, n, k) if transposed else (bs, k, n)).astype(np.float32)
        w = rng.normal(size=(bs, m, n)).astype(np.float32)
        results = []
        for mod in (d, torch):
            x, y = mod.tensor(a, requires_grad=True), mod.tensor(bt, requires_grad=True)
            yy = (y.permute(0, 2, 1) if mod is d else y.transpose(1, 2)) if transposed else y
            out = mod.bmm(x, yy)
            (out * mod.tensor(w)).sum().backward()
            to_np = (lambda t: t.detach().numpy()) if mod is torch else (lambda t: t.numpy())
            results.append([to_np(out), to_np(x.grad), to_np(y.grad)])
        for got, want in zip(*results):
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (bs, m, k, n)


def test_matmul_routing_bits():
    """`@` routes like PyTorch's _matmul_impl (dot, mv, mm, folded mm/mv, bmm) with the same
    strides, so every case gives PyTorch's bits, gradients included."""
    rng = np.random.default_rng(4); bad = []
    def r(*s): return rng.normal(size=s).astype(np.float32)
    cases = {
        "dot": (r(7), r(7)), "mv": (r(5, 7), r(7)), "vm": (r(7), r(7, 5)), "mm": (r(5, 7), r(7, 3)),
        "3d@2d": (r(4, 5, 7), r(7, 3)), "3d@1d": (r(4, 5, 7), r(7)), "2d@3d": (r(5, 7), r(4, 7, 3)), "1d@3d": (r(7), r(4, 7, 3)),
        "3d@3d": (r(4, 5, 7), r(4, 7, 3)), "bcast": (r(2, 1, 5, 7), r(3, 7, 4)), "4d@4d": (r(2, 3, 16, 8), r(2, 3, 8, 16)),
        "big": (r(12, 64, 128), r(128, 65)), "mv_big": (r(300, 200), r(200)),
    }
    for name, (a, b) in cases.items():
        for kt in (False, True):
            if kt and b.ndim < 2: continue
            bb = np.swapaxes(b, -1, -2).copy() if kt else b
            w = None; res = []
            for mod in (d, torch):
                x, y = mod.tensor(a, requires_grad=True), mod.tensor(bb, requires_grad=True)
                yy = y.transpose(-2, -1) if kt else y
                out = x @ yy
                if w is None: w = r(*out.shape) if out.shape else np.float32(1.3)
                (out * mod.tensor(w)).sum().backward()
                f = (lambda t: t.detach().numpy()) if mod is torch else (lambda t: t.numpy())
                res.append([f(out), f(x.grad), f(y.grad)])
            for i, (p, q) in enumerate(zip(*res)):
                if not np.array_equal(np.asarray(p).view(np.uint32), np.asarray(q).view(np.uint32)):
                    bad.append((name, "kT" if kt else "", ["out", "ga", "gb"][i]))
    assert bad == []
