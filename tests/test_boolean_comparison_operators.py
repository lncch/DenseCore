import operator
import numpy as np
import pytest
import densecore as d


COMPS = [("eq", operator.eq, np.equal), ("ne", operator.ne, np.not_equal),
         ("lt", operator.lt, np.less), ("le", operator.le, np.less_equal),
         ("gt", operator.gt, np.greater), ("ge", operator.ge, np.greater_equal)]


@pytest.mark.parametrize("name,op,reference", COMPS)
def test_bool_comparisons_broadcast_views_and_reflected_scalars(name, op, reference):
    av = np.array([[True, False], [False, True], [True, True]])
    bv = np.array([[False], [True]])
    a, b = d.tensor(av, dtype="bool"), d.tensor(bv, dtype="bool")
    for out in [op(a.T, b), getattr(a.T, name)(b), getattr(d, name)(a.T, b)]:
        assert out.dtype == "bool" and not out.requires_grad
        np.testing.assert_array_equal(out.numpy(), reference(av.T, bv))
    for scalar in [True, np.bool_(False)]:
        np.testing.assert_array_equal(op(scalar, a).numpy(), reference(scalar, av))
        np.testing.assert_array_equal(op(a, scalar).numpy(), reference(av, scalar))
    for invalid in [1, d.tensor([1.])]:
        with pytest.raises(TypeError):
            op(a, invalid)


@pytest.mark.parametrize("op,reference", [(operator.and_, np.logical_and),
                                         (operator.or_, np.logical_or),
                                         (operator.xor, np.logical_xor)])
def test_bool_bitwise_operators_and_reflection(op, reference):
    av = np.array([[True, False], [False, True]])
    a = d.tensor(av, dtype="bool").T
    b = d.tensor([[False], [True]], dtype="bool")
    np.testing.assert_array_equal(op(a, b).numpy(), reference(av.T, b.numpy()))
    for scalar in [True, np.bool_(False)]:
        np.testing.assert_array_equal(op(scalar, a).numpy(), reference(scalar, av.T))
        np.testing.assert_array_equal(op(a, scalar).numpy(), reference(av.T, scalar))
    for invalid in [1, 1., d.tensor([1.]), np.array([True, False])]:
        with pytest.raises(TypeError):
            op(a, invalid)


def test_bool_empty_and_mask_chain_keep_gradients_disabled():
    mask = d.tensor(np.empty((0, 3)), dtype="bool")
    assert (mask == True).shape == (0, 3)
    assert (mask & False).shape == (0, 3)
    x = d.tensor([-1., 0., 2.], requires_grad=True)
    out = (x > 0) & (x < 3)
    np.testing.assert_array_equal(out.numpy(), [False, False, True])
    assert not out.requires_grad and out.grad is None and x.grad is None
    with pytest.raises(RuntimeError):
        out.backward()
    x.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [1., 1., 1.])


def test_bool_numpy_bridge_rejects_kwargs_and_mixed_types():
    a = d.tensor([True, False], dtype="bool")
    with pytest.raises(TypeError):
        np.bitwise_and(a, True, out=np.empty(2, dtype=bool))
    with pytest.raises(TypeError):
        np.bitwise_and.reduce(a)
    with pytest.raises(TypeError):
        np.array(1) & a


def test_numpy_boolean_scalar_preserves_float_scalar_comparison_behavior():
    x = d.tensor([0., 1.])
    np.testing.assert_array_equal((x == np.bool_(True)).numpy(), [False, True])
    np.testing.assert_array_equal((np.bool_(True) == x).numpy(), [False, True])


def test_bool_comparison_empty_huge_layout_and_shape_error():
    from densecore._native import _BoolTensor
    shape = [np.iinfo(np.uintp).max, 2, 0]
    x = d.Tensor._wrap(_BoolTensor([], shape))
    assert (x == True).shape == tuple(shape)
    assert (False | x).shape == tuple(shape)
    with pytest.raises(ValueError):
        d.tensor([[True, False]], dtype="bool") == d.tensor([True, False, True], dtype="bool")


def test_boolean_comparison_and_bitwise_torch_parity():
    torch = pytest.importorskip("torch")
    av, bv = [[True, False], [False, True]], [[False], [True]]
    a, b = d.tensor(av, dtype="bool"), d.tensor(bv, dtype="bool")
    ta, tb = torch.tensor(av), torch.tensor(bv)
    for name, op, _ in COMPS:
        np.testing.assert_array_equal(op(a.T, b).numpy(), getattr(torch, name)(ta.T, tb).numpy())
    for op in [operator.and_, operator.or_, operator.xor]:
        np.testing.assert_array_equal(op(a.T, b).numpy(), op(ta.T, tb).numpy())
