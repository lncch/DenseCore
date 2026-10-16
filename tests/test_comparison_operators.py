import operator
import numpy as np
import pytest
import densecore as d


@pytest.mark.parametrize("op,reference", [(operator.eq, np.equal), (operator.ne, np.not_equal),
    (operator.lt, np.less), (operator.le, np.less_equal), (operator.gt, np.greater),
    (operator.ge, np.greater_equal)])
def test_comparison_operators_and_reflected_scalars(op, reference):
    values = np.array([[0., np.nan], [-0., 2.]], dtype="f")
    x = d.tensor(values, requires_grad=True)
    for actual, expected in [(op(x.T, 1), reference(values.T, 1)),
                             (op(1, x.T), reference(1, values.T))]:
        assert isinstance(actual, d.Tensor) and actual.dtype == "bool"
        assert not actual.requires_grad
        np.testing.assert_array_equal(actual.numpy(), expected)


def test_identity_hash_and_shared_parameter_registration():
    p = d.nn.Parameter([[1., 2.], [3., 4.]])
    q = d.nn.Parameter(p)
    assert hash(p) == object.__hash__(p)
    keys = {p: "p", q: "q"}
    assert len(keys) == 2 and keys[p] == "p" and keys[q] == "q"
    module = d.nn.Module()
    module.a, module.b, module.c = p, p, q
    params = list(module.parameters())
    assert len(params) == 2 and params[0] is p and params[1] is q
    mask = p == q
    assert isinstance(mask, d.Tensor)
    with pytest.raises(RuntimeError, match="ambiguous"):
        bool(mask)
    with pytest.raises(RuntimeError, match="ambiguous"):
        q in [p]


def test_unsupported_equality_uses_python_fallback():
    x = d.tensor([1., 2.])
    assert (x == None) is False
    assert (x != None) is True
    assert (x == object()) is False
    with pytest.raises(TypeError):
        x < object()


def test_parameter_identity_survives_compiled_training_with_operators():
    model = d.compile(d.nn.Linear(2, 1))
    params = list(model.parameters())
    keys = {p: index for index, p in enumerate(params)}
    assert (params[0] == params[0]).dtype == "bool"
    opt = d.optim.SGD(params, lr=.01)
    x, y = d.tensor([[1., 2.]]), d.tensor([[.5]])
    d.compiled_train_step(model, x, y, opt)
    for index, p in enumerate(model.parameters()):
        assert p is params[index] and keys[p] == index


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("op,reference", [(operator.eq, np.equal), (operator.ne, np.not_equal),
    (operator.lt, np.less), (operator.le, np.less_equal), (operator.gt, np.greater),
    (operator.ge, np.greater_equal)])
def test_reflected_numpy_real_scalars_use_supported_predicates(scalar_type, op, reference):
    x = d.tensor([0., 2.])
    actual = op(scalar_type(1), x)
    assert isinstance(actual, d.Tensor)
    np.testing.assert_array_equal(actual.numpy(), reference(np.float32(1), x.numpy()))
    with pytest.raises(TypeError):
        op(np.array([1., 1.]), x)


def test_distinct_scalar_tensor_list_membership_uses_values():
    left, right = d.tensor(1.), d.tensor(1.)
    assert left is not right
    assert left in [right]


def test_numeric_zero_dimensional_comparison_bridge_rejects_unsupported_ufunc_forms():
    x = d.tensor([0., 2.])
    np.testing.assert_array_equal((np.array(1.) < x).numpy(), [False, True])
    with pytest.raises(TypeError):
        np.less(np.array(1.), x, out=np.empty(2, dtype=bool))
    with pytest.raises(TypeError):
        np.less.reduce(x)
    with pytest.raises(TypeError):
        np.equal(np.array("1"), x)
