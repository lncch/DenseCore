"""Cross-operation coverage for view scattering and backward lifecycle semantics."""
import numpy as np
import pytest
import densecore as d


UNARY = ["neg", "exp", "log", "sqrt", "sin", "cos", "sigmoid", "tanh", "relu"]


def expected_derivative(name, values):
    if name == "neg":
        return -np.ones_like(values)
    if name == "exp":
        return np.exp(values)
    if name == "log":
        return 1 / values
    if name == "sqrt":
        return .5 / np.sqrt(values)
    if name == "sin":
        return np.cos(values)
    if name == "cos":
        return -np.sin(values)
    if name == "sigmoid":
        sigmoid = 1 / (1 + np.exp(-values))
        return sigmoid * (1 - sigmoid)
    if name == "tanh":
        return 1 - np.tanh(values) ** 2
    if name == "relu":
        return (values > 0).astype(np.float32)
    raise AssertionError(name)


@pytest.mark.parametrize("name", UNARY)
@pytest.mark.parametrize("empty", [False, True])
def test_unary_shared_strided_retained_and_disconnected_lifecycle(name, empty):
    values = np.empty((0, 3), np.float32) if empty else (
        np.arange(6, dtype=np.float32).reshape(2, 3) / 5 + .3
    )
    x = d.tensor(values, requires_grad=True, dtype="float32")
    disconnected = d.tensor(9., requires_grad=True)
    view = x.T[::2]
    shared = -view if name == "neg" else getattr(view, name)()
    output = shared + shared * 2.
    seed = d.tensor(np.ones(output.shape, np.float32), dtype="float32")
    expected = np.zeros_like(values)
    expected[:, ::2] = 3 * expected_derivative(name, values[:, ::2])

    output.backward(seed, retain_graph=True)
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=3e-6, atol=1e-7)
    first = x.grad
    assert disconnected.grad is None
    with pytest.raises(RuntimeError, match="shape"):
        output.backward(d.tensor(1.))
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=3e-6, atol=1e-7)

    output.backward(seed)
    np.testing.assert_allclose(x.grad.numpy(), 2 * expected, rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(first.numpy(), expected, rtol=3e-6, atol=1e-7)
    assert disconnected.grad is None
    with pytest.raises(RuntimeError):
        output.backward(seed)
    np.testing.assert_allclose(x.grad.numpy(), 2 * expected, rtol=3e-6, atol=1e-7)


def assert_retained_lifecycle(output, leaves):
    """Check publication, retry and consumption against independent expectations."""
    seed = d.tensor(np.ones(output.shape, np.float32), dtype="float32")
    output.backward(seed, retain_graph=True)
    snapshots = []
    for leaf, expected in leaves:
        if expected is None:
            assert leaf.grad is None
            snapshots.append(None)
        else:
            np.testing.assert_allclose(leaf.grad.numpy(), expected, rtol=3e-6, atol=1e-7)
            snapshots.append(leaf.grad)
    # An extra leading dimension always violates the exact seed-shape contract.
    with pytest.raises(RuntimeError, match="shape"):
        output.backward(d.tensor(np.ones((1,) + output.shape, np.float32), dtype="float32"))
    for leaf, expected in leaves:
        if expected is not None:
            np.testing.assert_allclose(leaf.grad.numpy(), expected, rtol=3e-6, atol=1e-7)
    output.backward(seed)
    for (leaf, expected), snapshot in zip(leaves, snapshots):
        if expected is None:
            assert leaf.grad is None
        else:
            np.testing.assert_allclose(leaf.grad.numpy(), 2 * expected, rtol=3e-6, atol=1e-7)
            np.testing.assert_allclose(snapshot.numpy(), expected, rtol=3e-6, atol=1e-7)
    with pytest.raises(RuntimeError):
        output.backward(seed)
    for leaf, expected in leaves:
        if expected is not None:
            np.testing.assert_allclose(leaf.grad.numpy(), 2 * expected, rtol=3e-6, atol=1e-7)


BINARY = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y,
          "mul": lambda x, y: x * y, "div": lambda x, y: x / y}


@pytest.mark.parametrize("name", BINARY)
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("frozen", [None, "left", "right"])
def test_binary_broadcast_strided_frozen_lifecycle(name, empty, frozen):
    values = np.empty((0, 3), np.float32) if empty else (
        np.arange(6, dtype=np.float32).reshape(2, 3) / 5 + .3
    )
    bias = np.array([[1.5], [2.5]], np.float32)
    x = d.tensor(values, requires_grad=frozen != "left", dtype="float32")
    y = d.tensor(bias, requires_grad=frozen != "right", dtype="float32")
    view = values.T[::2]
    if name in ("add", "sub"):
        dx = np.ones_like(view)
        dy = np.ones_like(view) * (1 if name == "add" else -1)
    elif name == "mul":
        dx = np.broadcast_to(bias, view.shape)
        dy = view
    else:
        dx = np.broadcast_to(1 / bias, view.shape)
        dy = -view / bias ** 2
    expected_x = np.zeros_like(values)
    expected_x[:, ::2] = 3 * dx.T
    expected_y = 3 * dy.sum(axis=1, keepdims=True)
    shared = BINARY[name](x.T[::2], y)
    output = shared + shared * 2
    assert_retained_lifecycle(output, [
        (x, None if frozen == "left" else expected_x),
        (y, None if frozen == "right" else expected_y),
    ])


@pytest.mark.parametrize("name", BINARY)
def test_binary_identical_native_parent_lifecycle(name):
    values = np.array([[.5, 1.5], [2., 3.]], np.float32)
    x = d.tensor(values, requires_grad=True, dtype="float32")
    expected = {"add": np.full_like(values, 2), "sub": np.zeros_like(values),
                "mul": 2 * values, "div": np.zeros_like(values)}[name]
    assert_retained_lifecycle(BINARY[name](x, x), [(x, expected)])


@pytest.mark.parametrize("batched,empty_axis", [
    (batched, axis) for batched in [False, True]
    for axis in [None, "rows", "inner", "columns", "batch"]
    if batched or axis != "batch"
])
@pytest.mark.parametrize("frozen", [None, "left", "right"])
def test_matrix_strided_empty_frozen_lifecycle(batched, empty_axis, frozen):
    batch, m, k, n = 2, 3, 4, 2
    if empty_axis == "rows":
        m = 0
    elif empty_axis == "inner":
        k = 0
    elif empty_axis == "columns":
        n = 0
    elif empty_axis == "batch":
        batch = 0
    shape_a = (batch, k, m) if batched else (k, m)
    shape_b = (batch, n, k) if batched else (n, k)
    a = np.arange(np.prod(shape_a), dtype=np.float32).reshape(shape_a) / 8 + .3
    b = np.arange(np.prod(shape_b), dtype=np.float32).reshape(shape_b) / 9 + .4
    x = d.tensor(a, requires_grad=frozen != "left", dtype="float32")
    y = d.tensor(b, requires_grad=frozen != "right", dtype="float32")
    av, bv = np.swapaxes(a, -1, -2), np.swapaxes(b, -1, -2)
    g = np.ones(np.matmul(av, bv).shape, np.float32)
    dx = np.swapaxes(np.matmul(g, b), -1, -2)
    dy = np.swapaxes(np.matmul(a, g), -1, -2)
    xv = x.permute(0, 2, 1) if batched else x.T
    yv = y.permute(0, 2, 1) if batched else y.T
    output = d.bmm(xv, yv) if batched else xv @ yv
    assert_retained_lifecycle(output, [
        (x, None if frozen == "left" else dx),
        (y, None if frozen == "right" else dy),
    ])


@pytest.mark.parametrize("batched", [False, True])
def test_matrix_identical_native_parent_lifecycle(batched):
    values = np.array([[.5, 1.5], [2., 3.]], np.float32)
    if batched:
        values = values[None]
    x = d.tensor(values, requires_grad=True, dtype="float32")
    g = np.ones_like(values)
    expected = np.matmul(g, np.swapaxes(values, -1, -2)) + np.matmul(
        np.swapaxes(values, -1, -2), g
    )
    assert_retained_lifecycle(d.bmm(x, x) if batched else x @ x, [(x, expected)])


@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_axis_reduction_strided_shared_lifecycle(mean, empty):
    shape = (0, 3, 4) if empty else (2, 3, 4)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 5
    x = d.tensor(values, requires_grad=True, dtype="float32")
    view = x.permute(2, 0, 1)[::2]
    output = view.mean(axis=(0, 2)) if mean else view.sum(axis=(0, 2))
    output = output + output
    expected = np.zeros_like(values)
    expected[:, :, ::2] = 2 / 6 if mean else 2
    assert_retained_lifecycle(output, [(x, expected)])


@pytest.mark.parametrize("family", ["binary", "matmul", "bmm", "reduction"])
def test_version_failure_does_not_publish_unrelated_leaf(family):
    p = d.nn.Parameter([[1., 2.], [3., 4.]])
    unrelated = d.tensor(3., requires_grad=True)
    if family == "binary":
        old = p / p.T
    elif family == "matmul":
        old = p @ p
    elif family == "bmm":
        old = d.bmm(p.reshape(1, 2, 2), p.reshape(1, 2, 2))
    else:
        old = p.T.mean(axis=0)
    output = old.sum() + unrelated * 2
    p.sum().backward()
    before = p.grad.numpy().copy()
    d.optim.SGD([p], lr=.1).step()
    with pytest.raises(RuntimeError, match="modified"):
        output.backward()
    np.testing.assert_array_equal(p.grad.numpy(), before)
    assert unrelated.grad is None


@pytest.mark.parametrize("mean", [False, True])
def test_empty_reduced_axis_nonempty_output_lifecycle(mean):
    x = d.tensor(np.empty((2, 0, 3), np.float32), requires_grad=True, dtype="float32")
    scale = d.tensor(.5, requires_grad=True)
    view = x.permute(2, 1, 0) * scale
    output = view.mean(axis=1) if mean else view.sum(axis=1)
    assert output.shape == (3, 2)
    if mean:
        assert np.isnan(output.numpy()).all()
    else:
        np.testing.assert_array_equal(output.numpy(), np.zeros((3, 2), np.float32))
    assert_retained_lifecycle(output, [
        (x, np.empty((2, 0, 3), np.float32)),
        (scale, np.array(0., np.float32)),
    ])


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mean", [False, True])
def test_empty_whole_reduction_lifecycle(compiled, mean):
    x = d.tensor(np.empty((2, 0, 3), np.float32), requires_grad=True, dtype="float32")
    if compiled:
        from densecore._native import _GraphExecutor
        executor = _GraphExecutor(1, [("mean" if mean else "sum", 0, None, None)])
        output = d.Tensor._wrap(executor.execute([x._native]))
    else:
        output = x.mean() if mean else x.sum()
    assert output.shape == ()
    if mean:
        assert np.isnan(output.numpy()).item()
    else:
        assert output.numpy().item() == 0.
    assert_retained_lifecycle(output, [(x, np.empty(x.shape, np.float32))])


@pytest.mark.mac_exact
@pytest.mark.parametrize("kind", ["linear", "layer_norm"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("frozen", [None, "input", "weight"])
def test_dense_module_analytic_strided_lifecycle(kind, compiled, empty, frozen):
    rows = 0 if empty else 2
    width = 2 if kind == "linear" else 3
    values = np.arange(width * rows, dtype=np.float32).reshape(width, rows) / 7 + .2
    logical = values.T
    x = d.tensor(values, requires_grad=frozen != "input", dtype="float32")
    if kind == "linear":
        weight = np.array([[.2, .7], [-.3, .4], [.6, -.1]], np.float32)
        bias = np.array([.1, .2, -.2], np.float32)
        module = d.nn.Linear(2, 3)
        expected_value = logical @ weight.T + bias
        dx = np.ones((rows, 3), np.float32) @ weight
        dw = np.ones((3, rows), np.float32) @ logical
    else:
        weight = np.array([.7, 1.2, 1.9], np.float32)
        bias = np.array([-.1, .3, .2], np.float32)
        module = d.nn.LayerNorm(3, eps=.1)
        # Repointed 2026-10-16 (roadmap: PyTorch's bits are the spec; owner approved such
        # repoints): the reference used to be a float64 formula; it is now PyTorch float32.
        torch = pytest.importorskip("torch")
        tx = torch.tensor(logical, requires_grad=True)
        tw, tb = torch.tensor(weight, requires_grad=True), torch.tensor(bias)
        ty = torch.nn.functional.layer_norm(tx, (3,), tw, tb, eps=.1)
        ty.sum().backward()
        expected_value = ty.detach().numpy()
        dx = tx.grad.numpy() if rows else np.zeros((rows, 3), np.float32)
        dw = tw.grad.numpy()
    module.weight = d.nn.Parameter(weight, requires_grad=frozen != "weight")
    module.bias = d.nn.Parameter(bias)
    if compiled:
        d.compile(module)
    shared = module(x.T)
    if compiled:
        assert module._compiled_runner.executor is not None
    np.testing.assert_allclose(shared.numpy(), expected_value, rtol=3e-6, atol=1e-7)
    assert_retained_lifecycle(shared + shared, [
        (x, None if frozen == "input" else (2 * dx.T).astype(np.float32)),
        (module.weight, None if frozen == "weight" else (2 * dw).astype(np.float32)),
        (module.bias, np.full(3, 2 * rows, np.float32)),
    ])


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("frozen", [None, "prediction", "target"])
def test_mse_both_operands_strided_lifecycle(compiled, frozen):
    values = np.arange(12, dtype=np.float32).reshape(3, 4) / 8 + .2
    targets = values / 2 - .1
    x = d.tensor(values, requires_grad=frozen != "prediction", dtype="float32")
    y = d.tensor(targets, requires_grad=frozen != "target", dtype="float32")
    a, b = x.T[1::2], y.T[1::2]
    if compiled:
        from densecore._native import _GraphExecutor
        executor = _GraphExecutor(2, [("mse", 0, 1, None)])
        output = d.Tensor._wrap(executor.execute([a._native, b._native]))
    else:
        output = d.functional.mse_loss(a, b)
    expected = np.zeros_like(values)
    expected[:, 1::2] = 2 * (values[:, 1::2] - targets[:, 1::2]) / 6
    assert_retained_lifecycle(output, [
        (x, None if frozen == "prediction" else expected),
        (y, None if frozen == "target" else -expected),
    ])


@pytest.mark.parametrize("compiled", [False, True])
def test_cross_entropy_strided_lifecycle(compiled):
    values = np.arange(12, dtype=np.float32).reshape(4, 3) / 7 - .3
    x = d.tensor(values, requires_grad=True, dtype="float32")
    logits = x.T[:, ::2]
    labels = [0, 1, 0]
    if compiled:
        from densecore._native import _GraphExecutor
        executor = _GraphExecutor(2, [("cross_entropy", 0, 1, None)])
        output = d.Tensor._wrap(executor.execute([logits._native, d.tensor(labels, dtype="float32")._native]))
    else:
        output = d.functional.cross_entropy(logits, labels)
    control = values.T[:, ::2]
    exponential = np.exp(control - control.max(axis=1, keepdims=True))
    expected_logit = exponential / exponential.sum(axis=1, keepdims=True)
    expected_logit[np.arange(3), labels] -= 1
    expected = np.zeros_like(values)
    expected[::2, :] = expected_logit.T / 3
    assert_retained_lifecycle(output, [(x, expected)])


@pytest.mark.parametrize("kind", ["linear", "layer_norm"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
def test_public_model_late_input_hook_failure_preserves_gradients(kind, compiled, loss_name):
    d.manual_seed(71)
    module = d.nn.Linear(3, 3) if kind == "linear" else d.nn.LayerNorm(3, eps=.1)
    if compiled:
        d.compile(module)
    x = d.tensor([[.2, .7, -.3], [.4, -.1, .9]], requires_grad=True)
    prediction = module(x)
    if compiled:
        assert module._compiled_runner.executor is not None
    loss = (d.functional.mse_loss(prediction, d.tensor(np.zeros((2, 3), np.float32), dtype="float32"))
            if loss_name == "mse" else d.functional.cross_entropy(prediction, [0, 2]))
    loss.backward(retain_graph=True)
    leaves = [x, *module.parameters()]
    first = [leaf.grad for leaf in leaves]
    values = [leaf.numpy() for leaf in leaves]
    calls = []

    def fail(gradient):
        calls.append(gradient.shape)
        raise ValueError("late model hook failure")

    handle = x.register_hook(fail)
    with pytest.raises(ValueError, match="late model hook failure"):
        loss.backward()
    assert calls == [x.shape]
    for leaf, gradient, value in zip(leaves, first, values):
        np.testing.assert_array_equal(leaf.grad.numpy(), gradient.numpy())
        np.testing.assert_array_equal(leaf.numpy(), value)
    handle.remove()
    loss.backward()
    for leaf, gradient in zip(leaves, first):
        np.testing.assert_allclose(leaf.grad.numpy(), 2 * gradient.numpy(), rtol=3e-6, atol=1e-7)


def assert_state_equal(actual, expected):
    if isinstance(expected, d.Tensor):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
@pytest.mark.parametrize("parameter_index", [0, 2])
def test_fused_dense_norm_hook_rejection_preserves_populated_adam_state(loss_name, parameter_index):
    d.manual_seed(72)
    module = d.compile(d.nn.Sequential(d.nn.Linear(3, 3), d.nn.LayerNorm(3, eps=.1)))
    optimizer = d.optim.Adam(module.parameters(), lr=.002)
    x = d.tensor([[.2, .7, -.3], [.4, -.1, .9]])
    target = d.tensor(np.zeros((2, 3), np.float32), dtype="float32") if loss_name == "mse" else [0, 2]
    d.compiled_train_step(module, x, target, optimizer, loss=loss_name)
    prediction = module(x)
    loss = (d.functional.mse_loss(prediction, target) if loss_name == "mse"
            else d.functional.cross_entropy(prediction, target))
    loss.backward()
    parameters = list(module.parameters())
    gradients = [p.grad for p in parameters]
    state = optimizer.state_dict()
    model_state = module.state_dict()
    calls = []
    handle = parameters[parameter_index].register_hook(lambda g: calls.append(g))
    with pytest.raises(ValueError, match="hook"):
        d.compiled_train_step(module, x, target, optimizer, loss=loss_name)
    assert calls == []
    assert_state_equal(optimizer.state_dict(), state)
    assert_state_equal(module.state_dict(), model_state)
    for p, gradient in zip(parameters, gradients):
        np.testing.assert_array_equal(p.grad.numpy(), gradient.numpy())
    handle.remove()
    d.compiled_train_step(module, x, target, optimizer, loss=loss_name)
