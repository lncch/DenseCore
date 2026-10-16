import io

import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
rng = np.random.default_rng(0)


def pair(shape, requires_grad=True):
    values = rng.normal(size=shape).astype(np.float32)
    return (d.tensor(values, requires_grad=requires_grad),
            torch.tensor(values, requires_grad=requires_grad))


def backward_both(out, ref):
    weights = rng.normal(size=tuple(ref.shape)).astype(np.float32)
    (out * d.tensor(weights)).sum().backward()
    (ref * torch.from_numpy(weights)).sum().backward()


def test_split_values_and_gradients_match_torch():
    x, tx = pair((5, 6, 4))
    for args in [(2, 1), ([1, 3, 2], 1), (4, -1), (5, 0), (6, 1)]:
        ours, theirs = x.split(*args), tx.split(*args)
        assert len(ours) == len(theirs)
        for a, b in zip(ours, theirs):
            np.testing.assert_array_equal(a.numpy(), b.detach().numpy())
    q, k, v = x.split(2, dim=1)
    tq, tk, tv = tx.split(2, dim=1)
    backward_both(q * k + v, tq * tk + tv)
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        x.split([1, 1], 1)
    with pytest.raises(ValueError):
        x.split(0, 1)


@pytest.mark.parametrize('key', [
    (slice(None), [-1], slice(None)), ([2, 0, 2],), (Ellipsis, [1, 0]), (1, [3, 3, 0]),
    (slice(None), np.array([[0, 1], [5, 2]])), (0, Ellipsis, [2]), (slice(1, None), 2, [0, 3])])
def test_list_indexing_matches_torch_with_gradients(key):
    x, tx = pair((4, 6, 4))
    tkey = tuple(torch.from_numpy(k) if isinstance(k, np.ndarray) else k for k in key)
    out, ref = x[key], tx[tkey]
    assert out.shape == tuple(ref.shape)
    np.testing.assert_array_equal(out.numpy(), ref.detach().numpy())
    backward_both(out, ref)
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-6, atol=1e-6)


def test_tensor_indices_and_storage_dtypes():
    x = d.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(x[:, d.tensor([3, 0], dtype='int64')].numpy(), np.arange(12).reshape(3, 4)[:, [3, 0]])
    assert x[[0]].dtype == 'int64'
    with pytest.raises(IndexError):
        x[:, [4]]
    np.testing.assert_array_equal(x[[0, 2], [1, 3]].numpy(), torch.arange(12).reshape(3, 4)[[0, 2], [1, 3]].numpy())
    with pytest.raises(IndexError):
        x[d.tensor([0.])]


def test_masked_assignment_matches_nanogpt_topk_filter():
    logits, tlogits = pair((3, 10), requires_grad=False)
    v, _ = d.topk(logits, 4)
    tv, _ = torch.topk(tlogits, 4)
    logits[logits < v[:, [-1]]] = -float('inf')
    tlogits[tlogits < tv[:, [-1]]] = -float('inf')
    np.testing.assert_array_equal(logits.numpy(), tlogits.numpy())
    np.testing.assert_allclose(d.functional.softmax(logits, -1).numpy(),
                               torch.softmax(tlogits, -1).numpy(), rtol=1e-6)


def test_masked_assignment_gradients_and_leaf_rule():
    x, tx = pair((4, 5))
    y, ty = x * 2, tx * 2
    mask = y.numpy() > 0
    y[d.tensor(mask, dtype='bool')] = 0.5
    ty[torch.from_numpy(mask)] = 0.5
    np.testing.assert_array_equal(y.numpy(), ty.detach().numpy())
    backward_both(y, ty)
    np.testing.assert_array_equal(x.grad.numpy(), tx.grad.numpy())
    with pytest.raises(RuntimeError, match='leaf'):
        x[mask] = 0.
    ints = d.arange(4)
    ints[ints.numpy() > 1] = 9
    assert ints.numpy().tolist() == [0, 1, 9, 9]
    ints[0] = 5
    ints[1:3] = d.tensor([7, 8], dtype='int64')
    assert ints.numpy().tolist() == [5, 7, 8, 9]
    losses = d.zeros(3)
    losses[1] = 2.5
    assert losses.numpy().tolist() == [0., 2.5, 0.]
    with pytest.raises(IndexError):
        losses[3] = 1.
    with pytest.raises(RuntimeError, match='leaf'):
        x[0] = 1.


def test_copy_into_parameters_and_buffers():
    p = d.nn.Parameter(np.zeros((2, 3), np.float32))
    with pytest.raises(RuntimeError):
        p.copy_(d.ones(2, 3))
    with d.no_grad():
        assert p.copy_(d.arange(3).float()) is p
    np.testing.assert_array_equal(p.numpy(), [[0, 1, 2], [0, 1, 2]])
    assert p.requires_grad and p.is_leaf
    target = d.zeros(3, dtype='int64')
    target.copy_(d.tensor([1.9, -2.7, 3.]))
    np.testing.assert_array_equal(target.numpy(), torch.zeros(3, dtype=torch.int64).copy_(torch.tensor([1.9, -2.7, 3.])).numpy())


@pytest.mark.parametrize('dim,largest', [(-1, True), (0, True), (1, False)])
def test_topk_matches_torch(dim, largest):
    values = rng.permutation(60).astype(np.float32).reshape(3, 4, 5) / 7
    x, tx = d.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)
    out, ref = d.topk(x, 2, dim=dim, largest=largest), torch.topk(tx, 2, dim=dim, largest=largest)
    np.testing.assert_array_equal(out.values.numpy(), ref.values.detach().numpy())
    np.testing.assert_array_equal(out.indices.numpy(), ref.indices.numpy())
    assert out.indices.dtype == 'int64'
    backward_both(out.values, ref.values)
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-6, atol=1e-6)
    v, i = d.topk(d.tensor([1., np.nan, 3., 2.]), 2)
    tv, ti = torch.topk(torch.tensor([1., np.nan, 3., 2.]), 2)
    np.testing.assert_array_equal(i.numpy(), ti.numpy())
    assert d.topk(d.ones(3), 0).values.shape == (0,)
    with pytest.raises(ValueError):
        d.topk(d.ones(3), 4)


def test_multinomial_and_randint_are_seeded_and_well_formed():
    d.manual_seed(1)
    a = d.multinomial(d.tensor([[0., 0., 1.], [1., 1., 2.]]), 1)
    b = d.randint(5, 9, (4, 2))
    d.manual_seed(1)
    assert d.multinomial(d.tensor([[0., 0., 1.], [1., 1., 2.]]), 1).numpy().tolist() == a.numpy().tolist()
    np.testing.assert_array_equal(d.randint(5, 9, (4, 2)).numpy(), b.numpy())
    assert a.dtype == b.dtype == 'int64' and a.shape == (2, 1) and a.numpy()[0, 0] == 2
    assert b.shape == (4, 2) and b.numpy().min() >= 5 and b.numpy().max() < 9
    assert d.randint(3, (5,)).numpy().max() < 3
    counts = np.bincount(d.multinomial(d.tensor([1., 3.]), 4000, replacement=True).numpy(), minlength=2)
    assert abs(counts[1] / 4000 - .75) < .03
    no_repeat = d.multinomial(d.tensor([1., 1., 1., 1.]), 4).numpy()
    assert sorted(no_repeat.tolist()) == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="element < 0"):  # PyTorch's error type and message
        d.multinomial(d.tensor([0., -1.]), 1)


@pytest.mark.parametrize('max_norm,norm_type', [(0.5, 2.0), (100., 2.0), (0.3, float('inf')), (1.0, 1.0)])
def test_clip_grad_norm_matches_torch(max_norm, norm_type):
    shapes = [(3, 4), (4,), (2, 2)]
    ours = [d.nn.Parameter(rng.normal(size=s).astype(np.float32)) for s in shapes]
    theirs = [torch.nn.Parameter(torch.from_numpy(p.numpy().copy())) for p in ours]
    for p, tp in zip(ours, theirs):
        (p * p * 1.7).sum().backward()
        (tp * tp * 1.7).sum().backward()
    total = d.nn.utils.clip_grad_norm_(ours, max_norm, norm_type)
    ttotal = torch.nn.utils.clip_grad_norm_(theirs, max_norm, norm_type)
    np.testing.assert_allclose(total.item(), ttotal.item(), rtol=1e-6)
    for p, tp in zip(ours, theirs):
        np.testing.assert_allclose(p.grad.numpy(), tp.grad.numpy(), rtol=1e-6, atol=1e-7)


def test_grad_assignment_rules():
    p = d.nn.Parameter(np.ones(2, np.float32))
    p.grad = d.tensor([3., 4.])
    np.testing.assert_array_equal(p.grad.numpy(), [3., 4.])
    p.grad = None
    assert p.grad is None
    with pytest.raises(RuntimeError, match='shape'):
        p.grad = d.ones(3)
    with pytest.raises(RuntimeError, match='leaf'):
        (p * 2).grad = d.ones(2)
    with pytest.raises(TypeError):
        p.grad = d.ones(2, dtype='int64')


def _adamw_pair(groups):
    shapes = [(4, 3), (3,), (2, 5)]
    ours = [d.nn.Parameter(rng.normal(size=s).astype(np.float32)) for s in shapes]
    theirs = [torch.nn.Parameter(torch.from_numpy(p.numpy().copy())) for p in ours]
    if groups:
        spec = lambda ps: [{'params': [ps[0], ps[2]], 'weight_decay': 0.1}, {'params': [ps[1]], 'weight_decay': 0.0}]
        return ours, theirs, d.optim.AdamW(spec(ours), lr=3e-2, betas=(.9, .95)), torch.optim.AdamW(spec(theirs), lr=3e-2, betas=(.9, .95))
    return ours, theirs, d.optim.AdamW(ours, lr=3e-2), torch.optim.AdamW(theirs, lr=3e-2)


def _step(params, optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    sum(((p * p).sum() * (i + 1) for i, p in enumerate(params)), d.tensor(0.) if isinstance(params[0], d.Tensor) else 0).backward()
    optimizer.step()


@pytest.mark.parametrize('groups', [False, True])
def test_adamw_with_groups_and_lr_schedule_matches_torch(groups):
    ours, theirs, opt, topt = _adamw_pair(groups)
    for lr in (3e-2, 2e-2, 1e-2, 5e-3, 1e-3):
        _step(ours, opt, lr)
        _step(theirs, topt, lr)
    for p, tp in zip(ours, theirs):
        np.testing.assert_allclose(p.numpy(), tp.detach().numpy(), rtol=1e-6, atol=1e-7)
    assert len(opt.param_groups) == (2 if groups else 1)
    assert opt.param_groups[-1]['lr'] == 1e-3


def test_grouped_adamw_state_round_trips_through_save_load():
    ours, _, opt, _ = _adamw_pair(True)
    _step(ours, opt, 1e-2)
    buffer = io.BytesIO()
    d.save({'optimizer': opt.state_dict(), 'params': ours}, buffer)
    _step(ours, opt, 1e-2)
    expected = [p.numpy() for p in ours]
    buffer.seek(0)
    saved = d.load(buffer)
    params = [d.nn.Parameter(p.numpy()) for p in saved['params']]
    spec = [{'params': [params[0], params[2]], 'weight_decay': 0.1}, {'params': [params[1]], 'weight_decay': 0.0}]
    fresh = d.optim.AdamW(spec, lr=1e-2, betas=(.9, .95))
    fresh.load_state_dict(saved['optimizer'])
    _step(params, fresh, 1e-2)
    for p, e in zip(params, expected):
        np.testing.assert_array_equal(p.numpy(), e)
    with pytest.raises(ValueError):
        fresh.load_state_dict(d.optim.AdamW(params).state_dict())
    with pytest.raises(ValueError, match='more than one group'):
        d.optim.AdamW([{'params': [params[0]]}, {'params': [params[0]]}])
    model = d.nn.Linear(2, 2)
    grouped = d.optim.AdamW([{'params': [model.weight]}, {'params': [model.bias], 'weight_decay': 0.}])
    with pytest.raises(ValueError, match='single-group'):
        d.checkpoints.save_checkpoint(io.BytesIO(), model, grouped)


def test_cross_entropy_ignore_index_matches_torch():
    x, tx = pair((6, 5))
    labels = np.array([1, -1, 4, -1, 0, 2], np.int64)
    loss = d.functional.cross_entropy(x, d.from_numpy(labels), ignore_index=-1)
    tloss = torch.nn.functional.cross_entropy(tx, torch.from_numpy(labels), ignore_index=-1)
    np.testing.assert_allclose(loss.item(), tloss.item(), rtol=1e-6)
    loss.backward()
    tloss.backward()
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(d.functional.cross_entropy(x, labels.clip(0)).item(),
                               torch.nn.functional.cross_entropy(tx, torch.from_numpy(labels.clip(0))).item(), rtol=1e-6)
    with pytest.raises(ValueError):
        d.functional.cross_entropy(x, labels)  # -1 is not ignored by default (-100)


def test_attention_dropout_p_is_seeded_and_scaled():
    q, tq = pair((2, 3, 4, 8), requires_grad=False)
    F = d.functional
    plain = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    np.testing.assert_allclose(plain.numpy(), torch.nn.functional.scaled_dot_product_attention(
        tq, tq, tq, is_causal=True).numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(F.scaled_dot_product_attention(q, q, q, None, 0.0, True).numpy(), plain.numpy())
    d.manual_seed(3)
    dropped = F.scaled_dot_product_attention(q, q, q, dropout_p=0.5, is_causal=True).numpy()
    d.manual_seed(3)
    np.testing.assert_array_equal(F.scaled_dot_product_attention(q, q, q, dropout_p=0.5, is_causal=True).numpy(), dropped)
    assert not np.array_equal(dropped, plain.numpy())
    assert not F.scaled_dot_product_attention(q, q, q, dropout_p=1.0).numpy().any()


@pytest.mark.parametrize('key', [(None, Ellipsis), (slice(None), None), (Ellipsis, None), (None, 1, None),
                                 (None, slice(None), [2, 0]), (0, None, slice(1, None))])
def test_none_indexing_matches_torch_with_gradients(key):
    x, tx = pair((3, 4))
    out, ref = x[key], tx[key]
    assert out.shape == tuple(ref.shape)
    np.testing.assert_array_equal(out.numpy(), ref.detach().numpy())
    backward_both(out, ref)
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=1e-6, atol=1e-6)
    ids = d.tensor([5, 1, 2], dtype='int64')[None, ...]
    assert ids.shape == (1, 3) and ids.dtype == 'int64'


def test_tolist_and_cpu_only_cuda_seeding():
    assert d.tensor([[1., 2.]]).tolist() == torch.tensor([[1., 2.]]).tolist()
    assert d.arange(3).tolist() == [0, 1, 2] and d.tensor(2.5).tolist() == 2.5
    assert d.tensor([1.5], dtype='bfloat16').tolist() == [1.5]
    assert d.cuda.manual_seed(1) is None and d.cuda.manual_seed_all(1) is None
