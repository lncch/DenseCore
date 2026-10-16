import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')


def pair(torch_module, **kwargs):
    ours = d.nn.MultiheadAttention(**kwargs)
    theirs = torch_module(**kwargs)
    names = dict(theirs.named_parameters())
    for name, p in ours.named_parameters():
        with d.no_grad():
            p.copy_(d.tensor(names[name].detach().numpy()))
    assert sorted(dict(ours.named_parameters())) == sorted(names)
    return ours, theirs


@pytest.mark.parametrize('kwargs,call', [
    (dict(embed_dim=8, num_heads=2), {}),
    (dict(embed_dim=8, num_heads=4, batch_first=True), {'average_attn_weights': False}),
    (dict(embed_dim=8, num_heads=2, kdim=5, vdim=3, add_bias_kv=True, add_zero_attn=True), {}),
    (dict(embed_dim=8, num_heads=2, bias=False), {'need_weights': False}),
])
def test_multihead_attention_matches_torch(kwargs, call):
    ours, theirs = pair(torch.nn.MultiheadAttention, **kwargs)
    rng = np.random.default_rng(0)
    first = (2, 4) if kwargs.get('batch_first') else (4, 2)
    q = rng.normal(size=first + (8,)).astype(np.float32)
    k = rng.normal(size=(first[0], first[1] + 1) + (kwargs.get('kdim', 8),) if kwargs.get('batch_first') else
                   (first[0] + 1, first[1], kwargs.get('kdim', 8))).astype(np.float32)
    v = rng.normal(size=k.shape[:2] + (kwargs.get('vdim', 8),)).astype(np.float32)
    src = k.shape[1] if kwargs.get('batch_first') else k.shape[0]
    padding = np.zeros((2, src), bool)
    padding[1, -1] = True
    mask = np.triu(np.ones((4, src), bool), 2)
    tq = torch.tensor(q, requires_grad=True)
    out, weights = theirs(tq, torch.tensor(k), torch.tensor(v), key_padding_mask=torch.tensor(padding),
                          attn_mask=torch.tensor(mask), **call)
    dq = d.tensor(q, requires_grad=True)
    got, got_weights = ours(dq, d.tensor(k), d.tensor(v), key_padding_mask=d.tensor(padding),
                            attn_mask=d.tensor(mask), **call)
    np.testing.assert_allclose(got.numpy(), out.detach().numpy(), rtol=1e-5, atol=1e-6)
    if weights is None:
        assert got_weights is None
    else:
        np.testing.assert_allclose(got_weights.numpy(), weights.detach().numpy(), rtol=1e-5, atol=1e-6)
    w = rng.normal(size=out.shape).astype(np.float32)
    (out * torch.tensor(w)).sum().backward()
    (got * d.tensor(w)).sum().backward()
    np.testing.assert_allclose(dq.grad.numpy(), tq.grad.numpy(), rtol=1e-4, atol=1e-5)
    grads = dict(theirs.named_parameters())
    for name, p in ours.named_parameters():
        np.testing.assert_allclose(p.grad.numpy(), grads[name].grad.numpy(), rtol=1e-4, atol=1e-5, err_msg=name)


def test_unbatched_input_and_init_spread():
    d.manual_seed(0)
    m = d.nn.MultiheadAttention(16, 4)
    w = m.in_proj_weight.numpy()
    assert abs(float(w.std()) - np.sqrt(2 / (48 + 16))) < 0.02 and np.all(m.in_proj_bias.numpy() == 0)
    x = d.randn(5, 16)
    out, weights = m(x, x, x)
    assert out.shape == (5, 16) and weights.shape == (5, 5)


def test_nn_init_functions_follow_torch_spreads():
    d.manual_seed(0)
    w = d.zeros(256, 128, requires_grad=True)
    d.nn.init.kaiming_uniform_(w, a=np.sqrt(5))
    bound = d.nn.init.calculate_gain('leaky_relu', np.sqrt(5)) * np.sqrt(3 / 128)
    assert np.abs(w.numpy()).max() <= bound and np.abs(w.numpy()).max() > 0.95 * bound
    d.nn.init.trunc_normal_(w, std=1.0, a=-0.5, b=0.5)
    assert w.numpy().min() >= -0.5 and w.numpy().max() <= 0.5
    d.nn.init.orthogonal_(w)
    np.testing.assert_allclose(w.numpy().T @ w.numpy(), np.eye(128), atol=1e-5)
    d.nn.init.constant_(w, 0.25)
    assert np.all(w.numpy() == 0.25)
    with pytest.raises(ValueError):
        d.nn.init.xavier_uniform_(d.zeros(3, requires_grad=True))


def test_ctc_embedding_bag_and_multilabel_modules_match_torch():
    rng = np.random.default_rng(2)
    lp = torch.randn(9, 3, 6).log_softmax(-1)
    targets, lengths, target_lengths = torch.tensor([[1, 2, 2], [3, 1, 0], [5, 0, 0]]), [9, 7, 4], [3, 2, 1]
    want = torch.nn.CTCLoss()(lp, targets, torch.tensor(lengths), torch.tensor(target_lengths))
    got = d.nn.CTCLoss()(d.tensor(lp.numpy()), d.tensor(targets.numpy()), lengths, target_lengths)
    np.testing.assert_allclose(got.item(), want.item(), rtol=1e-5)
    x, y = rng.normal(size=(4, 5)).astype(np.float32), np.array([[3, 0, -1, 1, 1]] * 4)
    np.testing.assert_allclose(d.nn.MultiLabelMarginLoss()(d.tensor(x), d.tensor(y)).item(),
                               torch.nn.MultiLabelMarginLoss()(torch.tensor(x), torch.tensor(y)).item(), rtol=1e-6)
    bag = d.nn.EmbeddingBag(10, 3, mode='max', padding_idx=2)
    tbag = torch.nn.EmbeddingBag(10, 3, mode='max', padding_idx=2)
    with d.no_grad():
        bag.weight.copy_(d.tensor(tbag.weight.detach().numpy()))
    idx, offsets = np.array([1, 2, 4, 4, 9, 2]), np.array([0, 2, 5])
    np.testing.assert_allclose(bag(d.tensor(idx), d.tensor(offsets)).numpy(),
                               tbag(torch.tensor(idx), torch.tensor(offsets)).detach().numpy(), rtol=1e-6)


def test_requires_grad_toggle_follows_torch():
    x = d.tensor([1., 2.])
    assert x.requires_grad_() is x and x.requires_grad
    (x * x).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [2., 4.])
    y = d.tensor([3.])
    y.requires_grad = True
    assert y.requires_grad
    with pytest.raises(RuntimeError):
        (x * 2).requires_grad_(False)
    with pytest.raises(RuntimeError):
        d.tensor([1, 2]).requires_grad_()
