"""manual_seed, Generator and the ported samplers give PyTorch's CPU values bit for bit:
rand/randn/randint/randperm, normal_/uniform_/bernoulli_, module init, dropout and
attention with dropout."""
import math

import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _np(t):
    return t.detach().numpy() if hasattr(t, "detach") and not isinstance(t, d.Tensor) else t.numpy()


SIZES = [1, 3, 15, 16, 17, 33, 1000]


@pytest.mark.parametrize("seed", [0, 1337, 2**40 + 7, -5])
def test_sampler_stream_matches_torch(seed):
    torch.manual_seed(seed); d.manual_seed(seed)
    for n in SIZES:
        pairs = [
            (torch.rand(n), d.rand(n)), (torch.randn(n), d.randn(n)),
            (torch.rand(n, dtype=torch.float64), d.rand(n, dtype="float64")),
            (torch.randn(n, dtype=torch.float64), d.randn(n, dtype="float64")),
            (torch.randint(3, 70, (n,)), d.randint(3, 70, (n,))),
            (torch.randint(0, 2**40, (n,)), d.randint(0, 2**40, (n,))),
            (torch.randperm(n), d.randperm(n)),
            (torch.empty(n).normal_(0.5, 0.02), d.zeros(n).normal_(0.5, 0.02)),
            (torch.empty(n).uniform_(-0.3, 0.3), d.zeros(n).uniform_(-0.3, 0.3)),
            (torch.empty(n).bernoulli_(0.8), d.zeros(n).bernoulli_(0.8)),
            (torch.bernoulli(torch.full((n,), 0.3)), d.bernoulli(d.full((n,), 0.3))),
            (torch.normal(torch.zeros(n), torch.ones(n) * 2), d.normal(d.zeros(n), d.ones(n) * 2)),
        ]
        for t, x in pairs:
            assert _same(_np(t), _np(x))
    assert d.initial_seed() == torch.initial_seed()


def test_generator_objects():
    tg, dg = torch.Generator().manual_seed(7), d.Generator().manual_seed(7)
    assert _same(torch.randn(40, generator=tg).numpy(), d.randn(40, generator=dg).numpy())
    assert _same(torch.randperm(30, generator=tg).numpy(), d.randperm(30, generator=dg).numpy())
    assert torch.Generator().initial_seed() == d.Generator().initial_seed()
    state = dg.get_state()
    first = d.rand(5, generator=dg).numpy()
    dg.set_state(state)
    assert _same(first, d.rand(5, generator=dg).numpy())


def test_module_init_matches_torch():
    for seed in (0, 3):
        torch.manual_seed(seed); d.manual_seed(seed)
        pairs = [(torch.nn.Linear(13, 7), d.nn.Linear(13, 7)), (torch.nn.Embedding(5, 20), d.nn.Embedding(5, 20)),
                 (torch.nn.Conv2d(3, 4, 3), d.nn.Conv2d(3, 4, 3)), (torch.nn.Linear(4, 2, bias=False), d.nn.Linear(4, 2, bias=False))]
        for tm, dm in pairs:
            for (n1, tp), (n2, dp) in zip(tm.named_parameters(), dm.named_parameters()):
                assert n1 == n2 and _same(tp.detach().numpy(), dp.numpy())
        for name, kw in [("normal_", dict(std=0.02)), ("uniform_", dict(a=-1, b=2)), ("xavier_uniform_", {}),
                         ("xavier_normal_", {}), ("kaiming_uniform_", dict(a=math.sqrt(5))), ("kaiming_normal_", {})]:
            tw, dw = torch.empty(6, 9), d.zeros(6, 9)
            getattr(torch.nn.init, name)(tw, **kw); getattr(d.nn.init, name)(dw, **kw)
            assert _same(tw.numpy(), dw.numpy()), name


@pytest.mark.parametrize("fn", ["dropout", "dropout1d", "dropout2d", "alpha_dropout"])
def test_dropout_values_and_grads(fn):
    x = np.random.default_rng(0).normal(size=(4, 6, 5, 7)).astype(np.float32)
    if fn == "dropout1d":
        x = x[:, :, 0]
    torch.manual_seed(5); d.manual_seed(5)
    tt, dt = torch.tensor(x, requires_grad=True), d.tensor(x, requires_grad=True)
    yt = getattr(torch.nn.functional, fn)(tt, 0.3, True)
    yd = getattr(d.functional, fn)(dt, 0.3, True)
    g = np.random.default_rng(1).normal(size=x.shape).astype(np.float32)
    yt.backward(torch.tensor(g)); yd.backward(d.tensor(g))
    assert _same(yt.detach().numpy(), yd.numpy()) and _same(tt.grad.numpy(), dt.grad.numpy())


@pytest.mark.mac_exact
@pytest.mark.parametrize("shape", [(2, 4, 64, 32), (1, 2, 7, 5), (2, 3, 9, 4)])
@pytest.mark.parametrize("causal,scale", [(True, None), (False, 0.3), (False, -0.5)])
def test_attention_with_dropout_bits(shape, causal, scale):
    r = np.random.default_rng(0)
    q, k, v, g = (r.normal(size=shape).astype(np.float32) for _ in range(4))
    torch.manual_seed(5); d.manual_seed(5)
    T = [torch.tensor(a, requires_grad=True) for a in (q, k, v)]
    D = [d.tensor(a, requires_grad=True) for a in (q, k, v)]
    yt = torch.nn.functional.scaled_dot_product_attention(*T, dropout_p=0.2, is_causal=causal, scale=scale)
    yd = d.functional.scaled_dot_product_attention(*D, dropout_p=0.2, is_causal=causal, scale=scale)
    yt.backward(torch.tensor(g)); yd.backward(d.tensor(g))
    assert _same(yt.detach().numpy(), yd.numpy())
    assert all(_same(a.grad.numpy(), b.grad.numpy()) for a, b in zip(T, D))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_more_distributions_and_multinomial(dtype):
    rng = np.random.default_rng(0)
    for name, args in [("exponential_", (2.0,)), ("geometric_", (0.3,)), ("cauchy_", (0.5, 2.0)),
                       ("log_normal_", (1.0, 0.5))]:
        torch.manual_seed(4); d.manual_seed(4)
        a = getattr(torch.empty(33, dtype=getattr(torch, dtype)), name)(*args).numpy()
        b = getattr(d.zeros(33, dtype=dtype), name)(*args).numpy()
        assert _same(a, b), name
    for shape in [(5,), (3, 7), (40,)]:
        w = rng.random(shape).astype(dtype)
        for n, replacement in [(1, False), (3, False), (1, True), (9, True)]:
            torch.manual_seed(2); d.manual_seed(2)
            a = torch.multinomial(torch.tensor(w), n, replacement).numpy()
            b = d.multinomial(d.tensor(w), n, replacement).numpy()
            assert _same(a, b)


def test_random_and_randint_forms():
    torch.manual_seed(9); d.manual_seed(9)
    pairs = [(torch.empty((), dtype=torch.int64).random_(), d.empty((), dtype=d.int64).random_()),
             (torch.empty(5).random_(), d.empty(5).random_()),
             (torch.empty(4, dtype=torch.int64).random_(7), d.empty(4, dtype=d.int64).random_(7)),
             (torch.empty(3, dtype=torch.int64).random_(2, 9), d.empty(3, dtype=d.int64).random_(2, 9)),
             (torch.randint(high=5, size=(4,)), d.randint(high=5, size=(4,))),
             (torch.randint(3, (2,)), d.randint(3, (2,)))]
    for t, x in pairs:
        assert _same(t.numpy(), x.numpy())
