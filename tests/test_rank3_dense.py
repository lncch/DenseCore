"""Rank>2 Linear, LayerNorm and matmul flatten leading dimensions onto rank-2 kernels."""
import numpy as np
import pytest
import densecore as d

RNG = np.random.default_rng(11)


def _t(*shape, grad=True):
    return d.tensor(RNG.normal(size=shape).astype(np.float32), requires_grad=grad)


def test_linear_rank3_and_rank4_match_rank2_rows():
    d.manual_seed(0)
    layer = d.nn.Linear(4, 3)
    x = _t(2, 5, 4, grad=False)
    out = layer(x)
    assert out.shape == (2, 5, 3)
    np.testing.assert_array_equal(out.numpy().reshape(10, 3), layer(x.reshape(10, 4)).numpy())
    assert layer(_t(2, 3, 2, 4, grad=False)).shape == (2, 3, 2, 3)
    assert d.functional.linear(x, layer.weight, layer.bias).shape == (2, 5, 3)


def test_rank3_full_jacobians():
    w, b, x = _t(3, 4), _t(3), _t(2, 2, 4)
    assert d.autograd.gradcheck(lambda a: d.functional.linear(a, w, b), x)
    assert d.autograd.gradcheck(lambda ww: d.functional.linear(x, ww, b), w)
    nw, nb = _t(4), _t(4)
    assert d.autograd.gradcheck(lambda a: d.functional.layer_norm(a, (4,), nw, nb), x)
    right = _t(4, 3)
    assert d.autograd.gradcheck(lambda a: a @ right, x)
    other = _t(2, 4, 2)
    assert d.autograd.gradcheck(lambda a: a @ other, x)


def test_rank3_ops_match_torch_values_and_gradients():
    torch = pytest.importorskip('torch')
    x_np = RNG.normal(size=(2, 3, 4)).astype(np.float32)
    w_np, b_np = RNG.normal(size=(5, 4)).astype(np.float32), RNG.normal(size=5).astype(np.float32)
    g_np, beta_np = RNG.normal(size=4).astype(np.float32), RNG.normal(size=4).astype(np.float32)
    m_np = RNG.normal(size=(2, 5, 3)).astype(np.float32)
    x, w, b, g, beta, m = (d.tensor(v, requires_grad=True) for v in (x_np, w_np, b_np, g_np, beta_np, m_np))
    tx, tw, tb, tg, tbeta, tm = (torch.tensor(v, requires_grad=True) for v in (x_np, w_np, b_np, g_np, beta_np, m_np))
    out = d.functional.linear(d.functional.layer_norm(x, (4,), g, beta), w, b) @ m
    ref = torch.nn.functional.linear(torch.nn.functional.layer_norm(tx, (4,), tg, tbeta), tw, tb) @ tm
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=2e-5, atol=2e-5)
    out.sum().backward()
    ref.sum().backward()
    for mine, theirs in [(x, tx), (w, tw), (b, tb), (g, tg), (beta, tbeta), (m, tm)]:
        np.testing.assert_allclose(mine.grad.numpy(), theirs.grad.numpy(), rtol=2e-5, atol=2e-5)


def test_layernorm_module_rank3_and_rejections():
    norm = d.nn.LayerNorm(4)
    assert norm(_t(2, 3, 4, grad=False)).shape == (2, 3, 4)
    with pytest.raises(ValueError, match='last dimension'):
        norm(_t(2, 3, 5, grad=False))
    with pytest.raises(ValueError, match='last dimension'):
        norm(_t(5, grad=False))
    with pytest.raises(ValueError):
        _t(2, 3, 4, grad=False) @ _t(5, 2, grad=False)
    with pytest.raises(ValueError):
        _t(2, 3, 4, grad=False) @ _t(3, 4, 2, grad=False)


def test_compiled_model_with_rank3_input_falls_back_to_eager():
    d.manual_seed(1)
    model = d.nn.Sequential(d.nn.Linear(4, 6), d.nn.ReLU(), d.nn.LayerNorm(6), d.nn.Linear(6, 2))
    x3 = _t(2, 3, 4, grad=False)
    eager = model(x3).numpy()
    d.compile(model)
    np.testing.assert_array_equal(model(x3).numpy(), eager)
    x2 = x3.reshape(6, 4)
    np.testing.assert_allclose(model(x2).numpy(), eager.reshape(6, 2), rtol=1e-6, atol=1e-6)
