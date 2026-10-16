import numpy as np
import pytest
import densecore as d


def test_masked_fill_values_with_broadcast_and_negative_infinity():
    x = d.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    out = x.masked_fill(np.array([True, False, True]), -np.inf).numpy()
    np.testing.assert_array_equal(out, [[-np.inf, 1, -np.inf], [-np.inf, 4, -np.inf]])
    bool_mask = d.tensor([[False, True, False], [True, False, False]], dtype='bool')
    np.testing.assert_array_equal(x.masked_fill(bool_mask, 9.).numpy(), [[0, 9, 2], [9, 4, 5]])


def test_masked_positions_ignore_nonfinite_inputs_and_get_zero_gradient():
    x = d.tensor([np.inf, 1., np.nan, 2.], requires_grad=True)
    out = x.masked_fill([True, False, True, False], -np.inf)
    np.testing.assert_array_equal(out.numpy(), [-np.inf, 1., -np.inf, 2.])
    (out * d.tensor([3., 4., 5., 6.])).masked_fill([True, False, True, False], 0.).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [0., 4., 0., 6.])


def test_masked_fill_full_jacobian_on_strided_input():
    x = d.tensor(np.random.default_rng(2).normal(size=(3, 4)).astype(np.float32), requires_grad=True)
    mask = np.random.default_rng(3).random((4, 3)) > .5
    assert d.autograd.gradcheck(lambda a: a.T.masked_fill(mask, 1.5), x)


def test_causal_softmax_matches_torch():
    torch = pytest.importorskip('torch')
    scores = np.random.default_rng(4).normal(size=(2, 4, 4)).astype(np.float32)
    causal = np.triu(np.ones((4, 4), bool), 1)
    x = d.tensor(scores, requires_grad=True)
    tx = torch.tensor(scores, requires_grad=True)
    out = x.masked_fill(causal, -np.inf).softmax(-1)
    ref = tx.masked_fill(torch.from_numpy(causal), float('-inf')).softmax(-1)
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-6, atol=1e-7)
    seed = np.random.default_rng(5).normal(size=(2, 4, 4)).astype(np.float32)
    out.backward(d.tensor(seed))
    ref.backward(torch.from_numpy(seed))
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-5, atol=1e-6)


def test_masked_fill_rejections():
    x = d.tensor(np.zeros((2, 3), np.float32))
    with pytest.raises(TypeError, match='boolean'):
        x.masked_fill(np.array([1, 0, 1]), 0.)
    with pytest.raises(TypeError, match='boolean'):
        x.masked_fill(d.tensor([1., 0., 1.]), 0.)
    with pytest.raises(ValueError, match='broadcastable'):
        x.masked_fill(np.array([True, False]), 0.)
    with pytest.raises(TypeError, match='real'):
        x.masked_fill([True, False, True], True)
    with pytest.raises(TypeError, match='float32'):
        d.tensor([True], dtype='bool').masked_fill([True], 0.)
    assert x.masked_fill(np.zeros((2, 3), bool), 1.).shape == (2, 3)
    assert d.tensor(np.empty((0, 3), np.float32)).masked_fill([True, False, True], 1.).shape == (0, 3)
