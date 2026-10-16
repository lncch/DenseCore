import numpy as np
import pytest
import densecore as d

OPS = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b,
       'mul': lambda a, b: a * b, 'div': lambda a, b: a / b}
SPECIAL = np.array([-0., 0., 1., -2.5, np.inf, -np.inf, np.nan, 1e-30, 3e38, -7.], np.float32)


torch = pytest.importorskip('torch')


def _bits_match(got, want):
    """Bit for bit, except that the sign of an invalid-operation NaN is unspecified (approved)."""
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    keep = ~np.isnan(want)
    np.testing.assert_array_equal(got[keep].view(np.uint32), want[keep].view(np.uint32))


def _run(module, name, left, right, weights):
    """Forward and both gradients of OPS[name] under `module` (densecore or torch)."""
    a = module.tensor(left, requires_grad=True)
    b = module.tensor(right, requires_grad=True)
    out = OPS[name](a, b)
    (out * module.tensor(weights)).sum().backward()
    as_np = lambda t: t.detach().numpy() if module is torch else t.numpy()
    return as_np(out), as_np(a.grad), as_np(b.grad)


@pytest.mark.parametrize('name', OPS)
@pytest.mark.parametrize('shape_y', [(4, 6), (1, 4, 6)])
def test_same_shape_and_leading_one_match_torch(name, shape_y):
    # Repointed 2026-10-16 (roadmap: PyTorch is the spec). This used to require the two paths
    # to agree with each other; PyTorch's sum_to turns -0.0 into +0.0 through a size-1 axis.
    rng = np.random.default_rng(0)
    x = np.concatenate([SPECIAL, rng.normal(size=14).astype(np.float32)]).reshape(4, 6)
    y = np.concatenate([SPECIAL[::-1], rng.normal(size=14).astype(np.float32)]).reshape(shape_y)
    weights = np.linspace(-2, 2, 24, dtype=np.float32).reshape(np.broadcast_shapes(x.shape, shape_y))
    with np.errstate(all='ignore'):
        for got, want in zip(_run(d, name, x, y, weights), _run(torch, name, x, y, weights)):
            _bits_match(got, want)


@pytest.mark.parametrize('name', OPS)
@pytest.mark.parametrize('small_shape', [(), (1,), (6,), (4, 6), (1, 1, 6)])
@pytest.mark.parametrize('small_is_left', [False, True])
def test_repeat_broadcast_path_matches_torch(name, small_shape, small_is_left):
    # Repointed 2026-10-16 (roadmap: PyTorch is the spec): the reference used to sum repeats
    # one after another from +0.0; it is now PyTorch's own result (its cascade sum order).
    rng = np.random.default_rng(1)
    big = rng.normal(size=(3, 4, 6)).astype(np.float32)
    big.ravel()[:len(SPECIAL)] = SPECIAL
    small = rng.normal(size=small_shape).astype(np.float32) + np.float32(.5)
    weights = np.linspace(-2, 2, big.size, dtype=np.float32).reshape(big.shape)
    left, right = (small, big) if small_is_left else (big, small)
    for got, want in zip(_run(d, name, left, right, weights), _run(torch, name, left, right, weights)):
        _bits_match(got, want)


def test_same_shape_strided_views_and_empty():
    base = d.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)
    other = d.tensor(np.arange(12, dtype=np.float32).reshape(4, 3) + 1, requires_grad=True)
    out = base.T * other
    np.testing.assert_array_equal(out.numpy(), np.arange(12, dtype=np.float32).reshape(3, 4).T * (np.arange(12).reshape(4, 3) + 1))
    out.sum().backward()
    np.testing.assert_array_equal(base.grad.numpy(), (np.arange(12, dtype=np.float32).reshape(4, 3) + 1).T)
    empty = d.tensor(np.empty((0, 3), np.float32), requires_grad=True)
    (empty + empty).sum().backward()
    assert empty.grad.shape == (0, 3)
