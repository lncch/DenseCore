import io
import pickle

import numpy as np
import pytest
import densecore as d


def test_creation_functions_match_torch():
    torch = pytest.importorskip('torch')
    np.testing.assert_array_equal(d.ones(2, 3).numpy(), torch.ones(2, 3).numpy())
    np.testing.assert_array_equal(d.zeros((4,)).numpy(), torch.zeros(4).numpy())
    assert d.ones(2, dtype=d.bool).dtype == 'bool'
    assert d.zeros(3, requires_grad=True).requires_grad
    for args in [(5,), (2, 7), (-3, 9, 4), (0, 0)]:
        theirs = torch.arange(*args)
        assert d.arange(*args).dtype == 'int64' == str(theirs.dtype).split('.')[1]
        np.testing.assert_array_equal(d.arange(*args).numpy(), theirs.numpy())
    np.testing.assert_array_equal(d.arange(4, dtype=d.long).numpy(), np.arange(4))
    assert d.arange(4, dtype=d.float32).dtype == 'float32'
    with pytest.raises(ValueError):
        d.arange(0, 1, .5, dtype=d.long)
    for args in [(5.,), (2, 7.), (0, 1, .1), (1., -2., -.3), (-1e3, 1e3, .77)]:
        start, end, step = (0, args[0], 1) if len(args) == 1 else (args + (1,))[:3]
        count = max(0, int(np.ceil((end - start) / step)))
        # Exact: the correctly rounded value of start + i*step.
        ref = (start + np.arange(count, dtype=np.float64) * step).astype(np.float32)
        np.testing.assert_array_equal(d.arange(*args).numpy(), ref)
        # PyTorch's float32 kernel rounds partway (seen up to 5 ulp off); approved bound: 8 ulp.
        theirs = torch.arange(*args, dtype=torch.float32).numpy()
        assert theirs.shape == ref.shape
        np.testing.assert_array_max_ulp(ref, theirs, maxulp=8)
    with pytest.raises(ValueError):
        d.arange(0, 3, 0)
    with pytest.raises(RuntimeError):
        torch.arange(3, 1)
    with pytest.raises(ValueError):
        d.arange(3, 1)


def test_tril_matches_torch_and_zeroes_infinities_exactly():
    torch = pytest.importorskip('torch')
    values = np.random.default_rng(0).normal(size=(2, 3, 4)).astype(np.float32)
    values[0, 0, 3] = np.inf
    for diagonal in (-1, 0, 2):
        np.testing.assert_array_equal(d.tril(d.tensor(values), diagonal).numpy(),
                                      torch.tril(torch.from_numpy(values), diagonal).numpy())
    mask = np.ones((3, 3), bool)
    np.testing.assert_array_equal(d.tril(d.tensor(mask, dtype='bool')).numpy(), np.tril(mask))
    x = d.tensor(values, requires_grad=True)
    d.tril(x).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.broadcast_to(np.tril(np.ones((3, 4))), (2, 3, 4)))
    with pytest.raises(ValueError):
        d.tril(d.ones(3))


def test_from_numpy_copies_and_keeps_every_supported_dtype():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = d.from_numpy(array)
    array[0, 0] = 99
    assert t.numpy()[0, 0] == 0 and t.shape == (2, 3)
    assert d.from_numpy(np.array([True, False])).dtype == 'bool'
    for dtype in (np.float64, np.float16, np.int8, np.int16, np.int32, np.int64, np.uint8):
        source = (np.arange(-3, 3) * 7).astype(dtype)
        converted = d.from_numpy(source)
        assert converted.dtype == source.dtype.name
        np.testing.assert_array_equal(converted.numpy(), source)
    with pytest.raises(TypeError, match='uint16'):
        d.from_numpy(np.zeros(3, np.uint16))
    with pytest.raises(TypeError):
        d.from_numpy([1., 2.])


def test_tensor_shape_methods_match_torch():
    torch = pytest.importorskip('torch')
    values = np.random.default_rng(1).normal(size=(2, 4, 6)).astype(np.float32)
    x, tx = d.tensor(values), torch.from_numpy(values)
    assert x.size() == tuple(tx.size()) and x.size(-1) == tx.size(-1) and x.size(1) == 4
    assert x.dim() == tx.dim() == x.ndim
    np.testing.assert_array_equal(x.view(2, 4, 2, 3).transpose(1, 2).contiguous().view(2, -1).numpy(),
                                  tx.view(2, 4, 2, 3).transpose(1, 2).contiguous().view(2, -1).numpy())
    np.testing.assert_array_equal(x[0].t().numpy(), tx[0].t().numpy())
    assert x[0, 0].t() is not None and d.tensor(3.).t().shape == ()
    with pytest.raises(ValueError):
        x.t()
    with pytest.raises(IndexError):
        x.size(3)
    assert d.tensor([2.5]).sum().item() == 2.5 and isinstance(d.tensor(1.).item(), float)
    assert d.tensor(True, dtype='bool').item() is True
    with pytest.raises(RuntimeError):
        x.item()


def test_to_accepts_cpu_and_converts_dtypes():
    x = d.tensor([1., 2.], requires_grad=True)
    assert x.to('cpu') is x and x.to('cpu', non_blocking=True) is x
    assert x.to(dtype=d.float32) is x and x.to(x) is x
    with pytest.raises(ValueError):
        x.to('cuda')
    assert x.to(d.float16).dtype == 'float16' and x.to('cpu', d.long).dtype == 'int64'
    with pytest.raises(ValueError):
        x.to('complex32')
    with pytest.raises(TypeError):
        x.to(layout=None)


def test_functional_softmax_and_nn_functional_alias():
    torch = pytest.importorskip('torch')
    from densecore.nn import functional as F
    import densecore.nn.functional as F2
    assert F is d.functional and F2 is d.functional
    values = np.random.default_rng(2).normal(size=(3, 5)).astype(np.float32)
    for dim in (0, -1):
        np.testing.assert_allclose(F.softmax(d.tensor(values), dim=dim).numpy(),
                                   torch.softmax(torch.from_numpy(values), dim=dim).numpy(), rtol=1e-6)
        np.testing.assert_allclose(F.log_softmax(d.tensor(values), dim).numpy(),
                                   torch.log_softmax(torch.from_numpy(values), dim).numpy(), rtol=1e-6, atol=1e-6)
    with pytest.raises(TypeError):
        F.softmax(d.tensor(values), -1, dtype='float16')


def test_dtype_names_and_builtins_stay_intact():
    assert d.float32 == d.float == 'float32' and d.bool == 'bool' and d.long == 'int64'
    assert d.cat([d.ones(1), d.ones(1)], 0).shape == (2,)  # module code still sees builtin bool
    with pytest.raises(AttributeError):
        d.not_a_dtype


def test_module_list_dict_apply_and_buffers():
    nn = d.nn
    blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
    blocks.append(nn.GELU())
    assert len(blocks) == 4 and isinstance(blocks[-1], nn.GELU) and len(blocks[1:3]) == 2
    assert [type(m) for m in blocks][:1] == [nn.Linear]
    root = nn.ModuleDict(dict(wte=nn.Embedding(5, 2), h=blocks, ln=nn.LayerNorm(2)))
    assert root.wte is root['wte'] and 'h' in root and list(root) == ['wte', 'h', 'ln']
    names = [n for n, _ in root.named_parameters()]
    assert names[0] == 'wte.weight' and 'h.0.weight' in names and 'ln.bias' in names
    with pytest.raises(TypeError):
        nn.ModuleList([1])
    with pytest.raises(TypeError):
        root['x'] = 3
    visited = []
    root.apply(lambda m: visited.append(type(m).__name__))
    assert visited[-1] == 'ModuleDict' and visited.index('Linear') < visited.index('ModuleList')
    attn = nn.Linear(2, 2)
    attn.register_buffer('mask', d.tril(d.ones(3, 3)))
    assert attn.mask.shape == (3, 3) and 'mask' not in dict(attn.named_parameters())
    with pytest.raises(ValueError):
        attn.register_buffer('weight', d.ones(1))


def test_init_functions_replace_values_in_place_with_seeded_rng():
    nn = d.nn
    d.manual_seed(5)
    layer = nn.Linear(300, 200)
    optimizer = d.optim.SGD(layer.parameters(), lr=.1)
    weight = layer.weight
    assert nn.init.normal_(layer.weight, mean=0.0, std=0.02) is weight
    values = weight.numpy()
    assert abs(values.mean()) < 1e-3 and abs(values.std() - 0.02) < 1e-3
    nn.init.zeros_(layer.bias)
    assert not layer.bias.numpy().any() and layer.weight.requires_grad
    d.manual_seed(5)
    again = nn.Linear(300, 200)
    np.testing.assert_array_equal(nn.init.normal_(again.weight, 0.0, 0.02).numpy(), values)
    layer(d.ones(1, 300)).sum().backward()
    optimizer.step()
    assert not np.array_equal(layer.weight.numpy(), values)
    with pytest.raises(TypeError):
        nn.init.zeros_(d.tensor([True], dtype='bool'))


def test_param_group_lr_matches_torch_schedule():
    torch = pytest.importorskip('torch')
    start = np.random.default_rng(3).normal(size=(4, 3)).astype(np.float32)
    p, tp = d.nn.Parameter(start), torch.nn.Parameter(torch.from_numpy(start.copy()))
    opt, topt = d.optim.Adam([p], lr=1e-2), torch.optim.Adam([tp], lr=1e-2)
    for lr in (1e-2, 5e-3, 1e-3):
        for group in opt.param_groups:
            group['lr'] = lr
        for group in topt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        topt.zero_grad(set_to_none=True)
        (p * p).sum().backward()
        (tp * tp).sum().backward()
        opt.step()
        topt.step()
    np.testing.assert_allclose(p.numpy(), tp.detach().numpy(), rtol=1e-6, atol=1e-7)
    assert opt.param_groups[0]['lr'] == 1e-3 and opt.state_dict()['hyperparameters']['lr'] == 1e-3
    assert opt.param_groups[0]['params'][0] is p
    opt.param_groups[0]['betas'] = topt.param_groups[0]['betas'] = (.8, .9)
    (p * p).sum().backward()
    (tp * tp).sum().backward()
    opt.step()
    topt.step()
    np.testing.assert_allclose(p.numpy(), tp.detach().numpy(), rtol=1e-6, atol=1e-7)
    assert opt.param_groups[0]['betas'] == (.8, .9)
    with pytest.raises(NotImplementedError):
        opt.param_groups[0]['params'] = []
    with pytest.raises(ValueError):
        opt.param_groups[0]['lr'] = -1.
    assert opt.param_groups[0]['lr'] == 1e-3
    with pytest.raises(NotImplementedError):
        opt.zero_grad(set_to_none=False)


def test_cuda_backends_amp_stubs_behave_like_cpu_only_torch():
    assert d.cuda.is_available() is False and d.cuda.is_bf16_supported() is False
    d.backends.cuda.matmul.allow_tf32 = True
    d.backends.cudnn.allow_tf32 = True
    with d.amp.autocast(device_type='cpu', enabled=False):
        pass
    with pytest.raises(NotImplementedError):
        d.autocast(device_type='cpu', dtype=d.bfloat16)
    with pytest.warns(UserWarning, match='Disabling'):
        scaler = d.cuda.amp.GradScaler(enabled=True)
    p = d.nn.Parameter(np.ones(2, np.float32))
    opt = d.optim.SGD([p], lr=.5)
    loss = (p * p).sum()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    np.testing.assert_array_equal(p.numpy(), [0., 0.])
    assert not d.amp.GradScaler(enabled=False).is_enabled()


def test_save_load_round_trips_checkpoint_dicts_bit_exactly(tmp_path):
    d.manual_seed(0)
    model = d.nn.Sequential(d.nn.Linear(3, 4), d.nn.GELU(), d.nn.Linear(4, 2))
    opt = d.optim.Adam(model.parameters(), lr=1e-3)
    model(d.ones(2, 3)).sum().backward()
    opt.step()
    special = d.tensor([np.nan, -0., np.inf, 1e-45], requires_grad=True)
    checkpoint = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'iter_num': 7,
                  'best_val_loss': d.tensor(1.5), 'config': {'betas': (.9, .95)}, 'special': special,
                  'mask': d.tensor([True, False], dtype='bool')}
    path = tmp_path / 'ckpt.pt'
    d.save(checkpoint, path)
    loaded = d.load(path, map_location='cpu')
    restored = d.nn.Sequential(d.nn.Linear(3, 4), d.nn.GELU(), d.nn.Linear(4, 2))
    restored.load_state_dict(loaded['model'])
    for a, b in zip(model.parameters(), restored.parameters()):
        np.testing.assert_array_equal(a.numpy().view(np.uint32), b.numpy().view(np.uint32))
    fresh = d.optim.Adam(restored.parameters(), lr=1e-3)
    fresh.load_state_dict(loaded['optimizer'])
    assert loaded['iter_num'] == 7 and loaded['best_val_loss'].item() == 1.5
    assert loaded['config'] == {'betas': (.9, .95)} and loaded['mask'].dtype == 'bool'
    np.testing.assert_array_equal(loaded['special'].numpy().view(np.uint32), special.numpy().view(np.uint32))
    assert loaded['special'].requires_grad
    buffer = io.BytesIO()
    d.save([d.ones(2)], buffer)
    buffer.seek(0)
    assert d.load(buffer)[0].shape == (2,)


def test_load_refuses_arbitrary_pickled_classes(tmp_path):
    path = tmp_path / 'evil.pt'
    path.write_bytes(pickle.dumps({'x': io.BytesIO()}))
    with pytest.raises(pickle.UnpicklingError, match='refusing'):
        d.load(path)
    with pytest.raises(ValueError):
        d.load(path, map_location='cuda')
