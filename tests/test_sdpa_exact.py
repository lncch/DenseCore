"""scaled_dot_product_attention (4-D float32, no mask, no dropout) matches PyTorch's CPU flash
attention bit for bit: output and the q/k/v gradients, across both split sizes and >512 keys."""
import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')


@pytest.mark.parametrize("B,H,L,S,E,causal,scale", [
    (2, 2, 7, 7, 5, False, None), (2, 2, 7, 7, 5, True, None), (1, 3, 33, 33, 16, True, None),
    (1, 2, 100, 37, 12, False, None), (1, 1, 200, 200, 8, True, None), (1, 1, 600, 600, 8, False, None),
    (1, 1, 800, 800, 4, True, None), (1, 2, 9, 1030, 6, False, None), (1, 2, 64, 64, 16, True, 0.3),
])
def test_sdpa_bits(B, H, L, S, E, causal, scale):
    rng = np.random.default_rng(L * 31 + S)
    q, g = (rng.normal(size=(B, H, L, E)).astype(np.float32) for _ in range(2))
    k, v = (rng.normal(size=(B, H, S, E)).astype(np.float32) for _ in range(2))
    res = []
    for mod in (d, torch):
        F = torch.nn.functional if mod is torch else d.functional
        ts = [mod.tensor(x, requires_grad=True) for x in (q, k, v)]
        y = F.scaled_dot_product_attention(*ts, is_causal=causal, scale=scale)
        y.backward(mod.tensor(g))
        f = (lambda a: a.detach().numpy()) if mod is torch else (lambda a: a.numpy())
        res.append([f(y)] + [f(t.grad) for t in ts])
    bad = [n for n, p, r in zip(("y", "dq", "dk", "dv"), *res) if not np.array_equal(p.view(np.uint32), r.view(np.uint32))]
    assert bad == []


@pytest.mark.parametrize("B,H,L,S,E", [(2, 2, 7, 7, 5), (1, 2, 200, 37, 12), (1, 1, 9, 1030, 6), (2, 3, 64, 64, 16)])
@pytest.mark.parametrize("kind", ["bool2d", "float2d", "bool4d_b1", "float4d_1h", "row"])
def test_sdpa_masked_bits(B, H, L, S, E, kind):
    rng = np.random.default_rng(L + S)
    q, g = (rng.normal(size=(B, H, L, E)).astype(np.float32) for _ in range(2))
    k, v = (rng.normal(size=(B, H, S, E)).astype(np.float32) for _ in range(2))
    mask = {"bool2d": rng.random((L, S)) > 0.3, "float2d": rng.normal(size=(L, S)).astype(np.float32),
            "bool4d_b1": rng.random((B, 1, L, S)) > 0.5, "float4d_1h": rng.normal(size=(1, H, 1, S)).astype(np.float32),
            "row": np.zeros((L, S), bool)}[kind]
    res = []
    for mod in (d, torch):
        F = torch.nn.functional if mod is torch else d.functional
        ts = [mod.tensor(x, requires_grad=True) for x in (q, k, v)]
        y = F.scaled_dot_product_attention(*ts, attn_mask=mod.tensor(mask))
        y.backward(mod.tensor(g))
        f = (lambda a: a.detach().numpy()) if mod is torch else (lambda a: a.numpy())
        res.append([f(y)] + [f(t.grad) for t in ts])
    bad = [n for n, p, r in zip(("y", "dq", "dk", "dv"), *res) if p.tobytes() != r.tobytes()]
    assert bad == []
