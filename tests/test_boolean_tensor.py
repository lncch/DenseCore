import numpy as np
import pytest
import densecore as d


def test_boolean_construction_is_real_dtype_and_copies():
    source = np.array([[True, False], [False, True]])
    x = d.tensor(source, dtype="bool")
    source[:] = False
    assert x.dtype == "bool" and x.device == "cpu"
    assert x.numpy().dtype == np.bool_
    np.testing.assert_array_equal(x.numpy(), [[True, False], [False, True]])
    exported = x.numpy()
    exported[:] = False
    assert x.numpy()[0, 0]
    assert not x.requires_grad and x.is_leaf and x.grad is None
    with pytest.raises((TypeError, ValueError), match="bool|gradient"):
        d.tensor([True], dtype="bool", requires_grad=True)
    with pytest.raises(RuntimeError):
        x.backward()


def test_boolean_views_share_typed_storage():
    x = d.tensor([[True, False, False], [False, True, True]], dtype="bool")
    view = x.T[1:, ::1]
    assert x.shares_storage(view)
    assert view.dtype == "bool"
    assert view.stride() == (1, 3)
    np.testing.assert_array_equal(view.numpy(), [[False, True], [False, True]])
    copy = view.reshape(4)
    assert not x.shares_storage(copy)
    np.testing.assert_array_equal(copy.numpy(), [False, True, False, True])
    assert x.shares_storage(x.reshape(6))
    assert x.shares_storage(x.detach())
    assert not x.shares_storage(d.tensor([[1., 0., 0.], [0., 1., 1.]]))
    assert not d.tensor([1.]).shares_storage(d.tensor([True], dtype="bool"))


@pytest.mark.parametrize("shape", [(), (0, 3), (2, 0), (2, 3)])
def test_logical_not_is_native_boolean_operation(shape):
    values = np.ones(shape, dtype=np.bool_)
    x = d.tensor(values, dtype="bool")
    for out in [x.logical_not(), ~x, d.logical_not(x)]:
        assert out.dtype == "bool" and not out.requires_grad
        np.testing.assert_array_equal(out.numpy(), np.logical_not(values))
    # PyTorch's logical_not accepts any dtype: nonzero -> False.
    assert d.tensor([1., 0.]).logical_not().numpy().tolist() == [False, True]


@pytest.mark.parametrize("op", [lambda x: x + x, lambda x: x * 2, lambda x: -x,
                                 lambda x: x.exp(), lambda x: x.relu(), lambda x: x.sum(),
                                 lambda x: x.mean(), lambda x: x.softmax(), lambda x: x @ x,
                                 lambda x: d.functional.linear(x, d.tensor([[1., 2.]])),
                                 lambda x: d.functional.layer_norm(x, 2),
                                 lambda x: d.functional.cross_entropy(x, [0])])
def test_boolean_unsupported_math_rejected_explicitly(op):
    with pytest.raises(TypeError):
        op(d.tensor([[True, False]], dtype="bool"))


def test_boolean_neural_network_and_joining_boundaries():
    x = d.tensor([[True, False]], dtype="bool")
    for op in [lambda: d.nn.Parameter(x), lambda: d.cat([x]), lambda: d.stack([x]),
               lambda: d.compile(d.nn.ReLU())(x)]:
        with pytest.raises(TypeError):
            op()


def test_tensor_truth_requires_single_element():
    for dtype in ["bool", "float32"]:
        assert bool(d.tensor([1], dtype=dtype))
        assert not bool(d.tensor(0, dtype=dtype))
        for value in [[], [0, 1]]:
            with pytest.raises(RuntimeError, match="ambiguous"):
                bool(d.tensor(value, dtype=dtype))


def test_boolean_numeric_truth_conversion_and_invalid_sources():
    values = np.array([0., -0., .5, -2., np.nan, np.inf])
    np.testing.assert_array_equal(d.tensor(values, dtype="bool").numpy(),
                                  [False, False, True, True, True, True])
    for value in [["no"], np.array([object()], dtype=object)]:
        with pytest.raises(TypeError):
            d.tensor(value, dtype="bool")
    x = d.tensor([[True, False, True], [False, True, False]], dtype="bool").T[::2]
    np.testing.assert_array_equal((~x).numpy(), np.logical_not(x.numpy()))
    assert not x.shares_storage(~x)
    with pytest.raises(RuntimeError):
        x.zero_grad()


def test_boolean_state_loading_is_rejected_without_mutation():
    model = d.nn.Linear(2, 1)
    before = {name: value.numpy().copy() for name, value in model.state_dict().items()}
    state = model.state_dict()
    state["weight"] = d.tensor([[True, False]], dtype="bool")
    with pytest.raises(TypeError, match="float32"):
        model.load_state_dict(state)
    for name, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), before[name])


def test_boolean_numeric_and_view_torch_parity():
    torch = pytest.importorskip("torch")
    values = np.array([[0., .5, np.nan], [-2., np.inf, -0.]], dtype=np.float32)
    actual = d.tensor(values, dtype="bool").T[::2]
    expected = torch.tensor(values, dtype=torch.bool).T[::2]
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    np.testing.assert_array_equal((~actual).numpy(), (~expected).numpy())


def test_scalar_truth_does_not_require_numpy_supported_rank():
    from densecore._native import _BoolTensor, _Tensor
    for native in [_BoolTensor([True], [1] * 100), _Tensor([1.], [1] * 100)]:
        assert bool(d.Tensor._wrap(native))


def test_boolean_seed_rejection_preserves_float_backward_graph():
    x = d.tensor([2.], requires_grad=True)
    loss = (x * x).sum()
    with pytest.raises(TypeError):
        loss.backward(d.tensor(True, dtype="bool"))
    loss.backward()
    np.testing.assert_array_equal(x.grad.numpy(), [4.])


def test_gradcheck_rejects_boolean_constants_instead_of_casting():
    x = d.tensor([.3], requires_grad=True)
    mask = d.tensor([True], dtype="bool")
    with pytest.raises(TypeError, match="float32"):
        d.autograd.gradcheck(lambda value, flag: value * value, (x, mask))


def test_eager_linear_rejects_boolean_receiver():
    with pytest.raises(TypeError):
        d.nn.Linear(2, 1)(d.tensor([[True, False]], dtype="bool"))


@pytest.mark.parametrize("kind,field", [("sgd", "momentum_buffer"),
                                        ("adam", "first_moment"),
                                        ("adam", "second_moment")])
def test_boolean_optimizer_state_rejected_atomically(kind, field):
    p = d.nn.Parameter([1., -2.])
    opt = d.optim.SGD([p], lr=.02, momentum=.8) if kind == "sgd" else d.optim.Adam([p], lr=.02)
    (p * p).sum().backward()
    opt.step()
    before = opt.state_dict()
    bad = opt.state_dict()
    bad["hyperparameters"]["lr"] = .123
    bad["state"][0][field] = d.tensor([True, False], dtype="bool")
    with pytest.raises((TypeError, ValueError), match="float32"):
        opt.load_state_dict(bad)
    after = opt.state_dict()
    assert after["hyperparameters"] == before["hyperparameters"]
    for name, value in before["state"][0].items():
        got = after["state"][0][name]
        if isinstance(value, d.Tensor):
            np.testing.assert_array_equal(got.numpy(), value.numpy())
        else:
            assert got == value


@pytest.mark.parametrize("op", [lambda a, b: a + b, lambda a, b: a @ b,
                                 lambda a, b: d.functional.mse_loss(a, b),
                                 lambda a, b: d.functional.linear(a, b)])
def test_boolean_right_operand_cannot_enter_float_kernels(op):
    with pytest.raises(TypeError):
        op(d.tensor([[1., 2.], [3., 4.]]),
           d.tensor([[True, False], [False, True]], dtype="bool"))
