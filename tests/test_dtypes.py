import io

import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
ALL = ['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16', 'int32', 'int64', 'uint8', 'bool']
FLOATS = ['float32', 'float64', 'float16', 'bfloat16']


def tdt(name):
    return getattr(torch, name)


def bits(t):
    """Bit pattern of a densecore or torch tensor, comparable across libraries."""
    if isinstance(t, d.Tensor):
        if t.dtype == 'bfloat16':
            return np.frombuffer(t._native.to_bytes(), '<u2')
        return t.numpy().ravel().view(np.uint8)
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().ravel().view(np.uint16)
    return t.contiguous().numpy().ravel().view(np.uint8)


def values_for(src, dst):
    """Values whose conversion is defined in both libraries (no NaN/inf/out-of-range to int)."""
    ints = [0, 1, 2, 7, 100, 127]
    if src in ('int8', 'int16', 'int32', 'int64') and dst not in ('uint8',):
        ints += [-1, -128]
    if src in FLOATS and dst in FLOATS + ['bool']:
        return [0., -0., 1., -2.5, 1e-3, 3.140625, 65504., 1e-8, np.inf, -np.inf, np.nan, 1 / 3]
    if src in FLOATS:
        return [0., 1.75, -0.0, 2.5, 99.9] + ([-3.7, -1.5] if dst != 'uint8' else [])
    return [v for v in ints if src != 'uint8' or v >= 0] if src != 'bool' else [True, False]


@pytest.mark.parametrize('src', ALL)
@pytest.mark.parametrize('dst', ALL)
def test_every_conversion_matches_torch_bit_for_bit(src, dst):
    values = values_for(src, dst)
    if src == 'float16' or (src == 'bfloat16' and dst != 'float32'):
        values = [v for v in values if not isinstance(v, float) or abs(v) < 6e4]
    ours = d.tensor(np.array(values, np.float64), dtype=src).to(dst)
    theirs = torch.tensor(values, dtype=torch.float64).to(tdt(src)).to(tdt(dst))
    assert ours.dtype == dst and ours.shape == tuple(theirs.shape)
    np.testing.assert_array_equal(bits(ours), bits(theirs))


@pytest.mark.parametrize('source', ['float32', 'float64'])
def test_bfloat16_rounding_matches_torch_including_ties(source):
    rng = np.random.default_rng(0)
    base = rng.normal(size=4000) * np.exp(rng.uniform(-80, 80, 4000))
    halfway = (np.arange(1, 400, dtype=np.uint32) << 16 | 0x8000).view(np.float32).astype(np.float64)
    edges = [3.3895313892515355e38, 3.4e38, 1e39, -1e39, 1e-40, -1e-45, 5e-324, np.nan, -np.nan, np.inf, -0.]
    values = np.concatenate([base, halfway, -halfway, edges, halfway + 2 ** -60])
    if source == 'float32':
        with np.errstate(over='ignore'):
            values = values.astype(np.float32)
    ours = d.tensor(values, dtype='bfloat16')
    theirs = torch.from_numpy(np.asarray(values)).to(torch.bfloat16)
    np.testing.assert_array_equal(bits(ours), bits(theirs))
    np.testing.assert_array_equal(ours.float().numpy(), theirs.float().numpy())


@pytest.mark.parametrize('dtype', [t for t in ALL if t not in ('float32', 'bool')])
def test_storage_dtypes_support_views_like_torch(dtype):
    source = (np.arange(24) % 11 - 3).reshape(2, 3, 4)
    if dtype == 'uint8':
        source = np.abs(source)
    ours = d.tensor(source.astype(np.float64), dtype=dtype)
    theirs = torch.tensor(source, dtype=tdt(dtype))
    for f in [lambda x: x.permute(2, 0, 1), lambda x: x[1, ::2, 1:], lambda x: x.transpose(0, 2).contiguous().view(-1),
              lambda x: x[:, 1], lambda x: x.reshape(6, 4)[::3].t()]:
        out, ref = f(ours), f(theirs)
        assert out.shape == tuple(ref.shape) and out.dtype == dtype
        np.testing.assert_array_equal(bits(out), bits(ref.contiguous()))
    view = ours.permute(2, 0, 1)
    assert view.shares_storage(ours) and not view.is_contiguous()
    assert ours.size() == (2, 3, 4) and ours.dim() == 3 and ours.numel() == 24


def test_numpy_item_truth_and_bfloat16_numpy_refusal():
    assert d.tensor([7], dtype='int64').item() == 7 and isinstance(d.tensor([7], dtype='int8').item(), int)
    assert d.tensor([2.5], dtype='float16').item() == 2.5 and d.tensor([2.5], dtype='bfloat16').item() == 2.5
    assert bool(d.tensor([3], dtype='int32')) and not bool(d.tensor([0], dtype='uint8'))
    assert d.tensor([1, 2], dtype='int64').numpy().dtype == np.int64
    with pytest.raises(TypeError, match='bfloat16'):
        d.tensor([1.], dtype='bfloat16').numpy()
    with pytest.raises(TypeError):
        torch.tensor([1.], dtype=torch.bfloat16).numpy()
    with pytest.raises(OverflowError):
        d.tensor([300], dtype='int8')


@pytest.mark.parametrize('dtype', ['float64', 'int64', 'float16', 'uint8'])
def test_storage_dtypes_reject_gradients_and_math_explicitly(dtype):
    with pytest.raises(ValueError, match='float32'):
        d.tensor([1.], dtype=dtype, requires_grad=True)
    x = d.tensor([1, 2], dtype=dtype)
    with pytest.raises((TypeError, RuntimeError)):
        x.backward()
    # sum works for every dtype now, with PyTorch's result dtype (integers -> int64).
    ours, theirs = x.sum(), torch.tensor([1, 2]).to(tdt(dtype)).sum()
    assert ours.dtype == str(theirs.dtype).split('.')[1] and ours.item() == theirs.item()


OPERATIONS = [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / b,
              lambda a, b: a // b, lambda a, b: a % b, lambda a, b: a < b, lambda a, b: a == b]


@pytest.mark.parametrize('left', ['int64', 'int32', 'int8', 'uint8', 'float64', 'float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize('right', ['int64', 'int16', 'float64', 'bfloat16', 'float32', 3, 2.5])
def test_storage_arithmetic_matches_torch_values_and_promotion(left, right):
    if left == right == 'float32' or (left == 'float32' and right in (3, 2.5)):
        pytest.skip('float32-only arithmetic is the native path, tested elsewhere')
    a_values = np.array([7, 3, 9, 4], np.float64)
    b_values = np.array([2, 5, 3, 1], np.float64)
    a, ta = d.tensor(a_values, dtype=left), torch.tensor(a_values).to(tdt(left))
    b, tb = ((d.tensor(b_values, dtype=right), torch.tensor(b_values).to(tdt(right)))
             if isinstance(right, str) else (right, right))
    for i, op in enumerate(OPERATIONS):
        if i in (4, 5) and 'float32' in (left, right):
            continue  # floor division and remainder are not wired for float32 yet
        ours, theirs = op(a, b), op(ta, tb)
        assert ours.dtype == str(theirs.dtype).split('.')[1], (i, ours.dtype, theirs.dtype)
        np.testing.assert_array_equal(bits(ours), bits(theirs))
    reflected, treflected = 10 - a, 10 - ta
    assert reflected.dtype == str(treflected.dtype).split('.')[1]
    np.testing.assert_array_equal(bits(reflected), bits(treflected))
    np.testing.assert_array_equal(bits(-a), bits(-ta))


def test_float32_mixed_with_int_keeps_gradients():
    x = d.tensor([1., 2.], requires_grad=True)
    (x * d.tensor([3, 4], dtype='int64')).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [3., 4.])
    with pytest.raises(RuntimeError, match='float64'):
        x * d.tensor([1., 2.], dtype='float64')
    with d.no_grad():
        assert (x * d.tensor([1., 2.], dtype='float64')).dtype == 'float64'


def test_cat_stack_and_save_load_keep_dtypes(tmp_path):
    a, b = d.arange(3), d.arange(3, 6)
    assert d.cat([a, b]).dtype == 'int64'
    np.testing.assert_array_equal(d.cat([a, b]).numpy(), np.arange(6))
    np.testing.assert_array_equal(d.stack([a, b], 1).numpy(), np.stack([np.arange(3), np.arange(3, 6)], 1))
    with pytest.raises(TypeError, match='dtype'):
        d.cat([a, b.float()])
    half = d.cat([d.tensor([1.5], dtype='bfloat16'), d.tensor([2.], dtype='bfloat16')])
    assert half.dtype == 'bfloat16' and half.float().numpy().tolist() == [1.5, 2.]
    saved = {name: d.tensor(np.array([1, 2, 3], np.float64), dtype=name) for name in ALL}
    saved['nan16'] = d.tensor([np.nan, -0.], dtype='bfloat16')
    buffer = io.BytesIO()
    d.save(saved, buffer)
    buffer.seek(0)
    loaded = d.load(buffer)
    for name, value in saved.items():
        assert loaded[name].dtype == value.dtype
        np.testing.assert_array_equal(bits(loaded[name]), bits(value))


def test_int64_tensors_drive_embedding_and_cross_entropy_like_torch():
    d.manual_seed(0)
    emb = d.nn.Embedding(10, 4)
    temb = torch.nn.Embedding(10, 4)
    with torch.no_grad():
        temb.weight.copy_(torch.from_numpy(emb.weight.numpy()))
    idx = np.array([[1, 9, 9], [0, 3, 1]], np.int64)
    out, ref = emb(d.from_numpy(idx)), temb(torch.from_numpy(idx))
    np.testing.assert_array_equal(out.numpy(), ref.detach().numpy())
    targets = d.from_numpy(np.array([3, 0, 9, 9, 1, 2], np.int64))
    logits = out.reshape(-1, 4) @ d.tensor(np.eye(4, 10, dtype=np.float32))
    tlogits = ref.reshape(-1, 4) @ torch.eye(4, 10)
    loss = d.functional.cross_entropy(logits, targets)
    tloss = torch.nn.functional.cross_entropy(tlogits, torch.from_numpy(targets.numpy()))
    loss.backward()
    tloss.backward()
    np.testing.assert_allclose(loss.item(), tloss.item(), rtol=1e-6)
    np.testing.assert_allclose(emb.weight.grad.numpy(), temb.weight.grad.numpy(), rtol=1e-5, atol=1e-7)
    for bad in [d.tensor([1.]), d.tensor([1.], dtype='float64'), d.tensor([True], dtype='bool')]:
        with pytest.raises(TypeError, match='integer'):
            emb(bad)


def test_float32_byte_transfer_keeps_exact_bits_and_views():
    raw = np.array([0x7FC00001, 0xFFC12345, 0x80000000, 0x00000001, 0x7F7FFFFF, 0xFF800000], np.uint32)
    values = raw.view(np.float32)
    t = d.tensor(values, requires_grad=True)
    np.testing.assert_array_equal(t.numpy().view(np.uint32), raw)
    grid = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    view = d.tensor(grid).permute(2, 0, 1)[1:, :, ::2]
    np.testing.assert_array_equal(view.numpy(), grid.transpose(2, 0, 1)[1:, :, ::2])
    assert d.tensor(np.asfortranarray(grid[0])).numpy().tolist() == grid[0].tolist()
    with pytest.raises(ValueError):
        d._native._Tensor.from_bytes(b'abc', [1])


@pytest.mark.parametrize('data', [[1, 2], [1., 2], [True, False], [[1], [2]], 3, 2.5, True, [], [True, 2],
                                  np.array([1, 2], np.int32), np.array([1.5]), np.array([1.5], np.float16),
                                  np.array([True]), np.int64(4), np.float64(0.5)])
def test_tensor_infers_dtype_like_torch(data):
    ours, theirs = d.tensor(data), torch.tensor(data)
    assert ours.dtype == str(theirs.dtype).split('.')[1]
    np.testing.assert_array_equal(ours.float().numpy(), theirs.float().numpy())


def test_tensor_constructor_and_legacy_class():
    assert d.Tensor([1, 2]).dtype == 'float32'  # like the legacy torch.Tensor constructor
    assert d.tensor([1, 2], dtype='float32').dtype == 'float32'
    assert d.tensor(d.tensor([1, 2])).dtype == 'int64'
    with pytest.raises(ValueError):
        d.tensor([1, 2], requires_grad=True)  # PyTorch: only floating point tensors can require grad
