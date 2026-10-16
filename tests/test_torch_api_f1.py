import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)


def pair(*shape):
    v = rng.normal(size=shape).astype(np.float32)
    return d.tensor(v, requires_grad=True), torch.tensor(v, requires_grad=True)


def check(out, ref, inputs, rtol=1e-5):
    assert out.shape == tuple(ref.shape)
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=rtol, atol=1e-6)
    w = rng.normal(size=tuple(ref.shape)).astype(np.float32)
    (out * d.tensor(w)).sum().backward()
    (ref * torch.from_numpy(w)).sum().backward()
    for a, b in inputs:
        np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('sa,sb', [((4,), (4,)), ((3, 4), (4,)), ((4,), (4, 2)), ((2, 3, 4), (4, 5)),
                                   ((2, 1, 3, 4), (5, 4, 2)), ((4,), (2, 4, 3)), ((2, 3, 4), (4,))])
def test_matmul_follows_torch_rules(sa, sb):
    a, ta = pair(*sa)
    b, tb = pair(*sb)
    check(a @ b, ta @ tb, [(a, ta), (b, tb)])


@pytest.mark.parametrize('f', [lambda x: x.sum(dim=1), lambda x: x.sum(dim=(0, 2), keepdim=True),
                               lambda x: x.mean(dim=[1]), lambda x: x.mean(-1, True),
                               lambda x: x.softmax(dim=0), lambda x: x.log_softmax(1),
                               lambda x: x.expand(5, 2, 3, 4), lambda x: x[:, :1].expand(-1, 5, -1),
                               lambda x: x.repeat(2, 1, 3), lambda x: x.repeat(2, 2, 1, 1),
                               lambda x: x.clone(), lambda x: x.mT, lambda x: x.sum_to_size(3, 1),
                               lambda x: x[:, :1].squeeze((1, 0)), lambda x: x.view_as(x.reshape(6, 4))])
def test_tensor_methods_match_torch(f):
    x, tx = pair(2, 3, 4)
    check(f(x), f(tx), [(x, tx)])


def test_torch_functions_forward_to_methods():
    x, tx = pair(3, 4)
    y, ty = pair(3, 4)
    check(d.add(x, y, alpha=2.5) * d.sub(x, y, alpha=.5), torch.add(tx, ty, alpha=2.5) * torch.sub(tx, ty, alpha=.5),
          [(x, tx), (y, ty)])
    a, b = d.tensor([7., -7., 5.5]), torch.tensor([7., -7., 5.5])
    for mode in ('trunc', 'floor'):
        np.testing.assert_array_equal(d.div(a, 2., rounding_mode=mode).numpy(), torch.div(b, 2., rounding_mode=mode).numpy())
    assert d.sum(d.ones(2, 3), dim=0).shape == (3,)
    with pytest.raises(AttributeError):
        d.numpy
    with pytest.raises(TypeError):
        d.add([1.], 2.)


def test_linear_layer_norm_and_mse_variants():
    x, tx = pair(4)
    w, tw = pair(3, 4)
    check(d.functional.linear(x, w), torch.nn.functional.linear(tx, tw), [(x, tx), (w, tw)])
    y, ty = pair(2, 3, 4)
    g, tg = pair(3, 4)
    check(d.functional.layer_norm(y, (3, 4), g), torch.nn.functional.layer_norm(ty, (3, 4), tg), [(y, ty), (g, tg)])
    v, tv = pair(4)
    check(d.functional.layer_norm(v, 4), torch.nn.functional.layer_norm(tv, (4,)), [(v, tv)])
    p, tp = pair(2, 5)
    q, tq = pair(2, 5)
    for reduction in ('sum', 'none'):
        p.zero_grad(), q.zero_grad()
        tp.grad = tq.grad = None
        check(d.functional.mse_loss(p, q, reduction=reduction),
              torch.nn.functional.mse_loss(tp, tq, reduction=reduction), [(p, tp), (q, tq)])
    assert d.nn.MSELoss(reduction='sum')(p, q).shape == ()


def test_attention_grouped_query_heads():
    q, tq = pair(2, 4, 5, 8)
    k, tk = pair(2, 2, 5, 8)
    v, tv = pair(2, 2, 5, 8)
    check(d.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True),
          torch.nn.functional.scaled_dot_product_attention(tq, tk, tv, is_causal=True, enable_gqa=True),
          [(q, tq), (k, tk), (v, tv)])


def test_casts_new_tensors_and_bfloat16_scalars():
    x = d.tensor(np.array([1.7, -2.2, 300.], np.float32))
    for name in ('byte', 'char', 'short'):
        t = torch.tensor([1.7, -2.2, 300.])
        assert getattr(x, name)().numpy().tolist() == getattr(t, name)().numpy().tolist()
    assert d.tensor(2.5).bfloat16().shape == () and d.tensor(2.5).bfloat16().item() == 2.5
    assert x.new_zeros(2, 3).shape == (2, 3) and x.new_full((2,), 7.).numpy().tolist() == [7., 7.]
    assert x.new_ones((1,), dtype='int64').dtype == 'int64'
    assert x.to(memory_format=torch.preserve_format) is x
    m = d.tensor([1., 2.], requires_grad=True)
    filled = m.masked_fill(np.array([True, False]), d.tensor(5.))
    assert filled.numpy().tolist() == [5., 2.]
