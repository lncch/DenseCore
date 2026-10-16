"""SGD/Adam/AdamW steps and clip_grad_norm_ (L2) match PyTorch's CPU defaults bit for bit, and a
tiny GPT trains to the same bits in both frameworks from the same weights."""
import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')

SHAPES = [(5,), (13,), (64,), (65, 128), (40003,), (1,)]


def _same(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("name,kwargs", [
    ("AdamW", dict(lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)), ("AdamW", {}),
    ("Adam", dict(lr=0.01, weight_decay=0.05)), ("Adam", dict(betas=(0.3, 0.9))),
    ("SGD", dict(lr=0.1)), ("SGD", dict(lr=0.1, momentum=0.9, weight_decay=0.01)),
])
@pytest.mark.parametrize("clip", [None, 1.0])
def test_optimizer_steps_bits(name, kwargs, clip):
    torch.set_num_threads(d.get_num_threads())
    rng = np.random.default_rng(0)
    init = [rng.normal(size=s).astype(np.float32) for s in SHAPES]
    tp = [torch.nn.Parameter(torch.tensor(x)) for x in init]
    dp = [d.tensor(x, requires_grad=True) for x in init]
    to, do = getattr(torch.optim, name)(tp, **kwargs), getattr(d.optim, name)(dp, **kwargs)
    for _ in range(4):
        for t, p, s in zip(tp, dp, SHAPES):
            g = (rng.normal(size=s) * 10).astype(np.float32)
            t.grad, p.grad = torch.tensor(g), d.tensor(g)
        if clip:
            nt, nd = torch.nn.utils.clip_grad_norm_(tp, clip), d.nn.utils.clip_grad_norm_(dp, clip)
            assert _same(nt.numpy(), nd.numpy())
        to.step(); do.step()
        assert all(_same(t.detach().numpy(), p.numpy()) for t, p in zip(tp, dp))


def _tiny_gpt(T, F, vocab=17, n=16, heads=2, block=12):
    """nanoGPT's ops in one block: embeddings, layer norm, attention, GELU MLP, tied head."""
    class GPT(T.nn.Module):
        def __init__(self):
            super().__init__()
            self.wte, self.wpe = T.nn.Embedding(vocab, n), T.nn.Embedding(block, n)
            self.ln1, self.ln2, self.lnf = T.nn.LayerNorm(n), T.nn.LayerNorm(n), T.nn.LayerNorm(n)
            self.attn, self.proj = T.nn.Linear(n, 3 * n, bias=False), T.nn.Linear(n, n, bias=False)
            self.fc, self.out = T.nn.Linear(n, 4 * n), T.nn.Linear(4 * n, n)
            self.head = T.nn.Linear(n, vocab, bias=False)
            self.head.weight = self.wte.weight

        def forward(self, idx, targets):
            B, L = idx.shape
            x = self.wte(idx) + self.wpe(T.arange(0, L, dtype=T.long))
            q, k, v = self.attn(self.ln1(x)).split(n, dim=2)
            q, k, v = (t.view(B, L, heads, n // heads).transpose(1, 2) for t in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, L, n)
            x = x + self.proj(y)
            x = x + self.out(F.gelu(self.fc(self.ln2(x))))
            logits = self.head(self.lnf(x))
            return F.cross_entropy(logits.view(-1, vocab), targets.view(-1), ignore_index=-1)
    return GPT()


def test_tiny_gpt_training_bits():
    torch.set_num_threads(d.get_num_threads())
    torch.manual_seed(0); d.manual_seed(0)
    mt, md = _tiny_gpt(torch, torch.nn.functional), _tiny_gpt(d, d.functional)
    tparams, dparams = dict(mt.named_parameters()), dict(md.named_parameters())
    assert tparams.keys() == dparams.keys()
    # The same seed builds the same initial weights.
    assert all(_same(tparams[k].detach().numpy(), dparams[k].numpy()) for k in tparams)
    ot = torch.optim.AdamW(mt.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1)
    od = d.optim.AdamW(md.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1)
    rng = np.random.default_rng(1)
    for _ in range(5):
        x, y = rng.integers(0, 17, (3, 12)), rng.integers(0, 17, (3, 12))
        lt = mt(torch.tensor(x), torch.tensor(y)); lt.backward()
        ld = md(d.tensor(x), d.tensor(y)); ld.backward()
        assert _same(np.float32(lt.item()), np.float32(ld.item()))
        torch.nn.utils.clip_grad_norm_(mt.parameters(), 1.0); d.nn.utils.clip_grad_norm_(md.parameters(), 1.0)
        ot.step(); od.step(); ot.zero_grad(set_to_none=True); od.zero_grad(set_to_none=True)
        assert all(_same(tparams[k].detach().numpy(), dparams[k].numpy()) for k in tparams)
