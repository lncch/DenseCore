import numpy as np
import pytest
import densecore as d

RNG = np.random.default_rng(21)


def _arrays(*shapes):
    return [RNG.normal(size=s).astype(np.float32) for s in shapes]


@pytest.mark.parametrize('shapes', [((2, 4, 3), (2, 5, 3), (2, 5, 2)), ((2, 3, 4, 8), (2, 3, 4, 8), (2, 3, 4, 8))])
@pytest.mark.parametrize('mode', ['none', 'causal', 'bool', 'float'])
def test_attention_matches_torch_values_and_gradients(shapes, mode):
    torch = pytest.importorskip('torch')
    q_np, k_np, v_np = _arrays(*shapes)
    L, S = shapes[0][-2], shapes[1][-2]
    if mode == 'causal' and L != S:
        pytest.skip('causal comparison uses square scores')
    mask, kwargs, tkwargs = None, {}, {}
    if mode == 'causal':
        kwargs = tkwargs = {'is_causal': True}
    elif mode == 'bool':
        mask = RNG.random((L, S)) > .3
        mask[:, 0] = True
        kwargs, tkwargs = {'attn_mask': mask}, {'attn_mask': torch.from_numpy(mask)}
    elif mode == 'float':
        mask = RNG.normal(size=(L, S)).astype(np.float32)
        kwargs, tkwargs = {'attn_mask': d.tensor(mask)}, {'attn_mask': torch.from_numpy(mask)}
    q, k, v = (d.tensor(a, requires_grad=True) for a in (q_np, k_np, v_np))
    tq, tk, tv = (torch.tensor(a, requires_grad=True) for a in (q_np, k_np, v_np))
    out = d.functional.scaled_dot_product_attention(q, k, v, **kwargs)
    ref = torch.nn.functional.scaled_dot_product_attention(tq, tk, tv, **tkwargs)
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-5, atol=1e-6)
    seed = RNG.normal(size=ref.shape).astype(np.float32)
    out.backward(d.tensor(seed))
    ref.backward(torch.from_numpy(seed))
    for mine, theirs in [(q, tq), (k, tk), (v, tv)]:
        np.testing.assert_allclose(mine.grad.numpy(), theirs.grad.numpy(), rtol=1e-4, atol=1e-5)


def test_attention_full_jacobian_with_causal_mask():
    q, k, v = (d.tensor(a, requires_grad=True) for a in _arrays((1, 3, 2), (1, 3, 2), (1, 3, 2)))
    sdpa = d.functional.scaled_dot_product_attention
    assert d.autograd.gradcheck(lambda a: sdpa(a, k, v, is_causal=True), q)
    assert d.autograd.gradcheck(lambda a: sdpa(q, a, v, is_causal=True), k)
    assert d.autograd.gradcheck(lambda a: sdpa(q, k, a, is_causal=True), v)


def test_attention_custom_scale_and_rejections():
    q, k, v = (d.tensor(a) for a in _arrays((1, 2, 4), (1, 3, 4), (1, 3, 5)))
    sdpa = d.functional.scaled_dot_product_attention
    assert sdpa(q, k, v, scale=.1).shape == (1, 2, 5)
    with pytest.raises(ValueError, match='both'):
        sdpa(q, k, v, attn_mask=np.ones((2, 3), bool), is_causal=True)
    with pytest.raises(ValueError, match='shapes'):
        sdpa(q, d.tensor(_arrays((1, 3, 3))[0]), v)
    with pytest.raises(ValueError, match='rank'):
        sdpa(q.reshape(2, 4), k, v)
    with pytest.raises(TypeError):
        sdpa(q, k, v, is_causal=1)
    with pytest.raises(TypeError):
        sdpa(q, k, v, scale='x')
