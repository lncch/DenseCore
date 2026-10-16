"""layer_norm matches PyTorch bit for bit: forward, input, weight and bias gradients, at
1 and 4 threads (Welford moments, two-pass weight/bias reduction, fused scalar steps)."""
import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')


def test_layer_norm_bits():
    rng = np.random.default_rng(6)
    bad = []
    before = (d.get_num_threads(), torch.get_num_threads())
    try:
        for threads in (1, 4):
            torch.set_num_threads(threads); d.set_num_threads(threads)
            for (m, n) in [(3, 1), (2, 3), (5, 4), (7, 13), (6, 64), (768, 128), (9, 65), (4, 1000), (3, 1029), (11, 257)]:
                for affine in ("wb", "w", "none"):
                    x = (rng.normal(size=(m, n)) * 3 + 1).astype(np.float32)
                    wv = rng.normal(size=n).astype(np.float32); bv = rng.normal(size=n).astype(np.float32); gw = rng.normal(size=(m, n)).astype(np.float32)
                    res = []
                    for mod in (d, torch):
                        F = torch.nn.functional if mod is torch else d.functional
                        xt = mod.tensor(x, requires_grad=True)
                        wt = mod.tensor(wv, requires_grad=True) if affine in ("wb", "w") else None
                        bt = mod.tensor(bv, requires_grad=True) if affine == "wb" else None
                        y = F.layer_norm(xt, (n,), wt, bt, 1e-5); (y * mod.tensor(gw)).sum().backward()
                        f = (lambda t: t.detach().numpy()) if mod is torch else (lambda t: t.numpy())
                        res.append([f(y), f(xt.grad)] + ([f(wt.grad)] if wt is not None else []) + ([f(bt.grad)] if bt is not None else []))
                    for k, (p, q) in enumerate(zip(*res)):
                        if not np.array_equal(p.view(np.uint32), q.view(np.uint32)):
                            bad.append((threads, (m, n), affine, ["y", "dx", "dw", "db"][k], int((p.view(np.uint32) != q.view(np.uint32)).sum())))
    finally:
        d.set_num_threads(before[0])
        torch.set_num_threads(before[1])
    assert bad == []
