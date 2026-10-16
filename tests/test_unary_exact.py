"""Unary math, GELU and atan2 match PyTorch bit for bit (values and gradients).

DenseCore ports SLEEF and PyTorch's own vector math, and reproduces how PyTorch's loops
split work across threads and handle tails, so these hold for any length and thread count.
"""
import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')

RANGES = {"exp": (-20, 20), "exp2": (-20, 20), "expm1": (-5, 5), "log": (0.01, 50), "log2": (0.01, 50),
          "log10": (0.01, 50), "log1p": (-0.9, 50), "sin": (-300, 300), "cos": (-300, 300), "tan": (-3, 3),
          "tanh": (-6, 6), "sigmoid": (-20, 20), "asin": (-0.99, 0.99), "acos": (-0.99, 0.99), "atan": (-20, 20),
          "sinh": (-8, 8), "cosh": (-8, 8), "asinh": (-50, 50), "acosh": (1.01, 50), "atanh": (-0.99, 0.99),
          "sqrt": (0.01, 50), "rsqrt": (0.01, 50), "reciprocal": (0.1, 50), "erf": (-4, 4), "erfc": (-4, 4)}
LENGTHS = (1, 5, 13, 2049, 40001)


@pytest.fixture(params=[1, 4])
def threads(request):
    before = (d.get_num_threads(), torch.get_num_threads())
    d.set_num_threads(request.param)
    torch.set_num_threads(request.param)
    yield request.param
    d.set_num_threads(before[0])
    torch.set_num_threads(before[1])


def same_bits(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return bool(np.all((a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))))


def run(module, f, inputs, weights):
    tensors = [module.tensor(x, requires_grad=True) for x in inputs]
    out = f(module, *tensors)
    (out * module.tensor(weights)).sum().backward()
    np_ = (lambda t: t.detach().numpy()) if module is torch else (lambda t: t.numpy())
    return [np_(out)] + [np_(t.grad) for t in tensors]


@pytest.mark.parametrize('name', RANGES)
def test_unary_values_and_gradients(name, threads):
    rng = np.random.default_rng(len(name) * 7 + threads)
    lo, hi = RANGES[name]
    for n in LENGTHS:
        x = rng.uniform(lo, hi, n).astype(np.float32)
        w = rng.normal(size=n).astype(np.float32)
        f = lambda m, t: getattr(m, name)(t)
        for got, want in zip(run(d, f, [x], w), run(torch, f, [x], w)):
            assert same_bits(got, want), (name, n)


def test_wide_inputs_cover_special_values(threads):
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.integers(0, 2**32, 200000, dtype=np.uint64).astype(np.uint32).view(np.float32),
                        np.array([0., -0., np.inf, -np.inf, np.nan, 88.7, -103.9, -104.1, 100.1], np.float32)])
    for name in ("exp", "log", "sin", "cos", "tan", "tanh", "erf", "erfc", "expm1", "log1p", "atan", "asin"):
        assert same_bits(getattr(d, name)(d.tensor(x)).numpy(), getattr(torch, name)(torch.from_numpy(x)).numpy()), name


def test_gelu_values_and_gradients(threads):
    rng = np.random.default_rng(5)
    for n in (1, 7, 9, 2049, 16385, 40001):
        x = (rng.normal(size=n) * rng.choice([1, 10, 40], size=n)).astype(np.float32)
        w = rng.normal(size=n).astype(np.float32)
        f = lambda m, t: (torch.nn.functional if m is torch else d.functional).gelu(t)
        for got, want in zip(run(d, f, [x], w), run(torch, f, [x], w)):
            assert same_bits(got, want), n


@pytest.mark.parametrize('x_size', ['same', 'one'])
def test_atan2_values_and_gradients(threads, x_size):
    rng = np.random.default_rng(9)
    for n in (1, 13, 2049, 40001):
        y = (rng.normal(size=n) * 10).astype(np.float32)
        x = (rng.normal(size=n if x_size == 'same' else 1) * 10).astype(np.float32)
        y[:min(n, 3)] = np.array([0., -0., np.inf], np.float32)[:min(n, 3)]
        w = rng.normal(size=n).astype(np.float32)
        f = lambda m, a, b: m.atan2(a, b)
        for got, want in zip(run(d, f, [y, x], w), run(torch, f, [y, x], w)):
            assert same_bits(got, want), n
