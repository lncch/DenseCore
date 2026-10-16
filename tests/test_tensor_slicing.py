import numpy as np
import pytest
import densecore as d


@pytest.mark.parametrize("key", [slice(None), (slice(1, 4, 2), slice(1, 5, 2)),
    (1, slice(None)), (Ellipsis, 2), (slice(-4, -1), -1),
    (slice(3, 1), slice(None)), (), (Ellipsis,), (slice(None), slice(5, 5))])
def test_basic_slices_match_numpy_and_share_storage(key):
    a = np.arange(20, dtype=np.float32).reshape(4, 5)
    x = d.tensor(a)
    y = x[key]
    np.testing.assert_array_equal(y.numpy(), a[key])
    assert x.shares_storage(y)
    if y.numel():
        assert y.storage_offset() == (a[key].ctypes.data - a.ctypes.data) // 4
    del x
    np.testing.assert_array_equal(y.numpy(), a[key])


def test_chained_strided_slice_backward_and_retained_graph():
    a = np.arange(20, dtype=np.float32).reshape(4, 5)
    x = d.tensor(a, requires_grad=True)
    y = x.T[1:5:2, 1:4:2]
    assert y.stride() == (2, 10)
    (y * y).sum().backward(retain_graph=True)
    want = np.zeros_like(a)
    want.T[1:5:2, 1:4:2] = 2 * a.T[1:5:2, 1:4:2]
    np.testing.assert_array_equal(x.grad.numpy(), want)
    y.sum().backward()
    want.T[1:5:2, 1:4:2] += 1
    np.testing.assert_array_equal(x.grad.numpy(), want)


def test_scalar_and_empty_slice_backward():
    x = d.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    (x[1, 0] + x[:0].sum()).backward()
    np.testing.assert_array_equal(x.grad.numpy(), [[0., 0.], [1., 0.]])
    s = d.tensor(2., requires_grad=True)
    s[()].backward()
    assert s.grad.numpy().item() == 1
    assert d.tensor(np.empty((0, 3), dtype=np.float32))[:, 1:].shape == (0, 2)


@pytest.mark.parametrize("key", [4, -5, (0, 0, 0)])
def test_out_of_range_indices(key):
    with pytest.raises(IndexError):
        d.tensor(np.zeros((4, 5), dtype=np.float32))[key]


@pytest.mark.parametrize("key", [object(), [1.5, 2.5], "a", 1.5, (Ellipsis, Ellipsis)])
def test_unsupported_index_forms(key):
    with pytest.raises((TypeError, IndexError)):
        d.tensor(np.zeros((4, 5), dtype=np.float32))[key]


def test_invalid_slice_steps_and_bounds():
    x = d.tensor([1., 2., 3.])
    with pytest.raises(ValueError): x[::0]
    with pytest.raises(ValueError): x[::-1]
    with pytest.raises(TypeError): x[1.5:]
    with pytest.raises(TypeError): x[::True]


def test_slice_saved_version_and_numpy_copy():
    p = d.nn.Parameter([1., 2., 3., 4.])
    y = (p[::2] * p[::2]).sum()
    y.backward(retain_graph=True)
    d.optim.SGD([p], lr=.1).step()
    with pytest.raises(RuntimeError, match="modified"): y.backward()
    view = p[1:]
    a = view.numpy()
    a[:] = 99
    assert view.numpy()[0] == p.numpy()[1]
