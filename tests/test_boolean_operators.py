import numpy as np
import pytest
import densecore as d


COMPARISONS = [("eq", np.equal), ("ne", np.not_equal), ("lt", np.less),
               ("le", np.less_equal), ("gt", np.greater), ("ge", np.greater_equal)]


@pytest.mark.parametrize("name,reference", COMPARISONS)
def test_comparison_broadcast_strided_and_nondifferentiable(name, reference):
    av = np.array([[0., np.nan], [-0., np.inf], [2., -3.]], dtype="f")
    bv = np.array([[0.], [1.]], dtype="f")
    a = d.tensor(av, requires_grad=True)
    b = d.tensor(bv, requires_grad=True)
    for out in [getattr(a.T, name)(b), getattr(d, name)(a.T, b)]:
        assert out.dtype == "bool" and not out.requires_grad and out.grad is None
        np.testing.assert_array_equal(out.numpy(), reference(av.T, bv))
    assert a.grad is None and b.grad is None
    # Producing a mask does not consume the differentiable input graph.
    a.sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), np.ones_like(av))


@pytest.mark.parametrize("name,reference", COMPARISONS)
def test_comparison_scalars_empty_and_errors(name, reference):
    x = d.tensor([0., 1., 2.])
    np.testing.assert_array_equal(getattr(x, name)(1).numpy(), reference(x.numpy(), np.float32(1)))
    empty = d.tensor(np.empty((0, 3), dtype="f"))
    assert getattr(d, name)(empty, d.tensor([1., 2., 3.])).shape == (0, 3)
    with pytest.raises(TypeError):
        getattr(x, name)(d.tensor([True], dtype="bool"))
    with pytest.raises(ValueError):
        getattr(x, name)(d.tensor([1., 2.]))
    with pytest.raises(TypeError):
        getattr(x, name)(np.ones(3))


@pytest.mark.parametrize("name,reference", [("logical_and", np.logical_and),
                                            ("logical_or", np.logical_or),
                                            ("logical_xor", np.logical_xor)])
def test_boolean_binary_broadcast_strided_and_errors(name, reference):
    av = np.array([[True, False], [False, True], [True, True]])
    bv = np.array([[False], [True]])
    a, b = d.tensor(av, dtype="bool"), d.tensor(bv, dtype="bool")
    for out in [getattr(a.T, name)(b), getattr(d, name)(a.T, b)]:
        assert out.dtype == "bool" and not out.requires_grad
        np.testing.assert_array_equal(out.numpy(), reference(av.T, bv))
    np.testing.assert_array_equal(getattr(a, name)(True).numpy(), reference(av, True))
    assert getattr(d, name)(d.tensor(np.empty((0, 2)), dtype="bool"), b.T).shape == (0, 2)
    for invalid in [1, np.ones(2)]:
        with pytest.raises(TypeError):
            getattr(a, name)(invalid)
    # PyTorch's logical_* methods accept any dtype (nonzero counts as True).
    floats = np.array([0., 2.5], np.float32)
    np.testing.assert_array_equal(getattr(d.tensor(floats), name)(b).numpy(), reference(floats != 0, bv))
    np.testing.assert_array_equal(getattr(a.T, name)(d.tensor(np.array([[0.], [3.]], np.float32))).numpy(),
                                  reference(av.T, np.array([[0.], [3.]]) != 0))


def test_broadcast_empty_huge_dimensions_skips_unreachable_products():
    from densecore._native import _Tensor, _BoolTensor
    size = np.iinfo(np.uintp).max
    shape = [size, 2, 0]
    x = d.Tensor._wrap(_Tensor([], shape))
    b = d.Tensor._wrap(_BoolTensor([], shape))
    assert (x + 1).shape == tuple(shape)
    assert x.eq(0).shape == tuple(shape)
    assert b.logical_and(True).shape == tuple(shape)


def test_comparison_torch_parity():
    torch = pytest.importorskip("torch")
    av = np.array([[0., np.nan, np.inf], [-0., -np.inf, 2.]], dtype="f")
    bv = np.array([[0.], [2.]], dtype="f")
    for name, _ in COMPARISONS:
        actual = getattr(d, name)(d.tensor(av), d.tensor(bv))
        expected = getattr(torch, name)(torch.tensor(av), torch.tensor(bv))
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
