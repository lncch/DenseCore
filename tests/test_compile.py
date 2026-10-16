import numpy as np
import pytest
import densecore as dc


def models():
    dc.manual_seed(41)
    eager = dc.nn.Sequential(dc.nn.Linear(2, 3), dc.nn.ReLU(), dc.nn.Linear(3, 1))
    compiled = dc.nn.Sequential(dc.nn.Linear(2, 3), dc.nn.ReLU(), dc.nn.Linear(3, 1))
    compiled.load_state_dict(eager.state_dict())
    assert dc.compile(compiled) is compiled
    return eager, compiled


def test_compile_preserves_forward_backward_and_upstream_eager_graph():
    eager, compiled = models()
    for model in (eager, compiled):
        x = dc.tensor([[1., 2.], [-1., 3.]], requires_grad=True)
        output = model(x * 2.)
        output.sum().backward()
        model.input_grad = x.grad.numpy()
    np.testing.assert_allclose(compiled.input_grad, eager.input_grad, rtol=1e-5, atol=1e-6)
    for a, b in zip(eager.parameters(), compiled.parameters()):
        np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=1e-5, atol=1e-6)
    assert set(eager.state_dict()) == set(compiled.state_dict())


def test_live_forwards_and_retained_backward_keep_distinct_activations():
    eager, compiled = models()
    for model in (eager, compiled):
        first = model(dc.tensor([[1., 2.]])).sum()
        second = model(dc.tensor([[-3., 1.]])).sum()
        second.backward()
        first.backward(retain_graph=True)
        first.backward()
    for a, b in zip(eager.parameters(), compiled.parameters()):
        np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=1e-5, atol=1e-6)


def test_no_grad_output_ownership_shape_specialization_and_updated_weights():
    eager, compiled = models()
    with dc.no_grad():
        first = compiled(dc.tensor([[1., 2.]]))
        original = first.numpy()
        for batch in range(1, 7):
            x = dc.tensor(np.ones((batch, 2), dtype=np.float32))
            np.testing.assert_allclose(compiled(x).numpy(), eager(x).numpy(), rtol=1e-5, atol=1e-6)
        state = eager.state_dict()
        state['0.weight'] = dc.tensor(np.ones((3, 2), dtype=np.float32))
        eager.load_state_dict(state)
        compiled.load_state_dict(state)
        x = dc.tensor([[2., 3.]])
        np.testing.assert_allclose(compiled(x).numpy(), eager(x).numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(first.numpy(), original)
        assert not first.requires_grad


def test_specialization_eviction_keeps_live_autograd_graph_valid():
    eager, compiled = models()
    for model in (eager, compiled):
        outputs = [model(dc.tensor(np.full((batch, 2), .3, dtype=np.float32))).sum() for batch in range(1, 7)]
        for output in reversed(outputs):
            output.backward()
    for a, b in zip(eager.parameters(), compiled.parameters()):
        np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=1e-5, atol=1e-6)


def test_compiled_model_rejects_stale_parameter_saved_in_forward():
    _, model = models()
    loss = model(dc.tensor([[1., 2.]])).sum()
    model.load_state_dict(model.state_dict())
    with pytest.raises(RuntimeError, match='modified'):
        loss.backward()
    assert all(p.grad is None for p in model.parameters())


def test_custom_modules_and_forward_replacement_use_eager_behavior():
    class Custom(dc.nn.Module):
        def forward(self, x):
            return x * 3.
    custom = dc.compile(Custom())
    np.testing.assert_array_equal(custom(dc.tensor([2.])).numpy(), [6.])
    _, model = models()
    model.forward = lambda x: x * 4.
    np.testing.assert_array_equal(model(dc.tensor([2.])).numpy(), [8.])
    del model.forward
    model[0].forward = lambda x: dc.tensor(np.ones((x.shape[0], 3), dtype=np.float32))
    x = dc.tensor([[2., 3.]])
    np.testing.assert_array_equal(model(x).numpy(), model.forward(x).numpy())


def test_shared_modules_parameters_and_structural_replacement():
    layer = dc.nn.Linear(2, 2, bias=False)
    model = dc.nn.Sequential(layer, dc.nn.ReLU(), layer)
    x = dc.tensor([[1., 2.]], requires_grad=True)
    model(x).sum().backward()
    expected = layer.weight.grad.numpy()
    model.zero_grad()
    dc.compile(model)
    model(x).sum().backward()
    np.testing.assert_allclose(layer.weight.grad.numpy(), expected, rtol=1e-5, atol=1e-6)
    model.__setattr__('2', dc.nn.Linear(2, 1, bias=False))
    with dc.no_grad():
        assert model(x).shape == (1, 1)
        np.testing.assert_array_equal(model(x).numpy(), model.forward(x).numpy())


def test_compiled_training_updates_match_eager_and_invalid_shapes_error():
    eager, compiled = models()
    x = dc.tensor([[1., 2.], [-1., 3.]])
    target = dc.tensor([[.5], [1.]])
    for model in (eager, compiled):
        optimizer = dc.optim.Adam(model.parameters(), lr=.01)
        for _ in range(5):
            optimizer.zero_grad()
            dc.functional.mse_loss(model(x), target).backward()
            optimizer.step()
    for a, b in zip(eager.parameters(), compiled.parameters()):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        compiled(dc.tensor([[1., 2., 3.]]))


def test_compiled_argument_parameter_alias_accumulates_both_slots():
    model = dc.nn.Linear(2, 2, bias=False)
    model(model.weight).sum().backward()
    expected = model.weight.grad.numpy()
    model.zero_grad()
    assert model.compile() is None
    alias = dc.Tensor._wrap(model.weight._native)
    model(alias).sum().backward()
    np.testing.assert_allclose(model.weight.grad.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_compiled_frozen_result_preserves_version_provenance():
    model = dc.nn.Linear(2, 1, bias=False)
    model.weight = dc.nn.Parameter([[1., 2.]], requires_grad=False)
    dc.compile(model)
    frozen = model(dc.tensor([[2., 3.]]))
    factor = dc.tensor([[1.]], requires_grad=True)
    loss = (frozen * factor).sum()
    model.load_state_dict(model.state_dict())
    with pytest.raises(RuntimeError, match='modified'):
        loss.backward()
    assert factor.grad is None


def test_class_forward_patch_and_empty_sequential_fall_back(monkeypatch):
    model = dc.compile(dc.nn.Sequential(dc.nn.ReLU()))
    monkeypatch.setattr(dc.nn.ReLU, 'forward', lambda self, x: x * 5.)
    np.testing.assert_array_equal(model(dc.tensor([2.])).numpy(), [10.])
    empty = dc.compile(dc.nn.Sequential())
    x = dc.tensor([2.], requires_grad=True)
    assert empty(x) is x


def test_tensor_subclass_and_tensor_method_patch_preserve_eager_dispatch(monkeypatch):
    class CustomTensor(dc.Tensor):
        def relu(self):
            return dc.tensor([99.])
    model = dc.compile(dc.nn.ReLU())
    np.testing.assert_array_equal(model(CustomTensor([-1.])).numpy(), [99.])
    monkeypatch.setattr(dc.Tensor, 'relu', lambda self: dc.tensor([77.]))
    np.testing.assert_array_equal(model(dc.tensor([-1.])).numpy(), [77.])


def test_tensor_wrap_patch_preserves_each_eager_operation(monkeypatch):
    model = dc.compile(dc.nn.Sequential(dc.nn.ReLU(), dc.nn.ReLU()))
    original = dc.Tensor._wrap
    monkeypatch.setattr(dc.Tensor, '_wrap', staticmethod(lambda native, **kwargs: dc.tensor(original(native, **kwargs).numpy() + 1.)))
    np.testing.assert_array_equal(model(dc.tensor([2.])).numpy(), [4.])


def test_no_grad_compilation_accepts_immutable_stale_provenance_snapshot():
    source = dc.nn.Linear(1, 1, bias=False)
    source.weight = dc.nn.Parameter([[2.]], requires_grad=False)
    old = source(dc.tensor([[3.]]))
    source.load_state_dict(source.state_dict())
    model = dc.compile(dc.nn.ReLU())
    with dc.no_grad():
        result = model(old)
        np.testing.assert_array_equal(result.numpy(), [[6.]])
        assert not result.requires_grad


def test_compiled_frozen_stale_input_matches_eager_reads_but_rejects_backward():
    source = dc.nn.Linear(1, 1, bias=False)
    source.weight = dc.nn.Parameter([[2.]], requires_grad=False)
    old = source(dc.tensor([[3.]]))
    source.load_state_dict(source.state_dict())
    model = dc.compile(dc.nn.ReLU())
    result = model(old)
    np.testing.assert_array_equal(result.numpy(), [[6.]])
    assert not result.requires_grad
    factor = dc.tensor([[1.]], requires_grad=True)
    with pytest.raises(RuntimeError, match='modified'):
        (result * factor).sum().backward()
    assert factor.grad is None


def test_native_binary_graph_shared_argument_and_empty_broadcast_gradients():
    from densecore._native import _GraphExecutor
    square = _GraphExecutor(1, [('mul', 0, 0, None)])
    x = dc.tensor([1., 2.], requires_grad=True)
    output = dc.Tensor._wrap(square.execute([x._native]))
    output.backward(dc.tensor([2., 3.]))
    np.testing.assert_array_equal(x.grad.numpy(), [4., 12.])
    row_add = _GraphExecutor(2, [('add', 0, 1, None)])
    empty = dc.tensor(np.zeros((0, 3), dtype=np.float32), requires_grad=True)
    row = dc.tensor([1., 2., 3.], requires_grad=True)
    dc.Tensor._wrap(row_add.execute([empty._native, row._native])).sum().backward()
    np.testing.assert_array_equal(row.grad.numpy(), [0., 0., 0.])
    assert empty.grad.shape == (0, 3)


def test_compiled_fast_path_signature_mutations():
    """Cached compilation must detect supported model mutations."""
    import densecore as dc
    from densecore.compilation import _signature_matches

    model = dc.compile(dc.nn.Sequential(
        dc.nn.Linear(4, 8),
        dc.nn.ReLU(),
        dc.nn.Linear(8, 2),
    ))

    x = dc.tensor([[1.0, 2.0, 3.0, 4.0]])
    model(x)

    runner = model._compiled_runner

    assert runner.executor is not None
    assert _signature_matches(model, runner.signature)

    model.eval()
    assert not _signature_matches(model, runner.signature)

    model.train()
    assert _signature_matches(model, runner.signature)

    original_child = model._modules["1"]
    model._modules["1"] = dc.nn.ReLU()
    assert not _signature_matches(model, runner.signature)

    model._modules["1"] = original_child
    assert _signature_matches(model, runner.signature)

    linear = model._modules["0"]
    original_weight = linear.weight

    linear.weight = dc.nn.Parameter(original_weight.numpy().copy())
    assert not _signature_matches(model, runner.signature)

    linear.weight = original_weight
    assert _signature_matches(model, runner.signature)


def test_iterative_compilation_guard_detects_mutations(monkeypatch):
    import densecore as dc
    from densecore.compilation import _signature_matches

    model = dc.compile(dc.nn.Sequential(
        dc.nn.Linear(4, 4),
        dc.nn.LayerNorm(4),
        dc.nn.ReLU(),
    ))

    x = dc.tensor([[1., 2., 3., 4.]])
    model(x)

    runner = model._compiled_runner
    signature = runner.signature

    assert _signature_matches(model, signature)

    # Reordering children must invalidate the cached graph.
    children = model._modules
    original_items = list(children.items())

    children.clear()
    children.update(reversed(original_items))

    assert not _signature_matches(model, signature)

    children.clear()
    children.update(original_items)
    assert _signature_matches(model, signature)

    # Replacing a parameter with the same shape must invalidate.
    linear = model[0]
    original_weight = linear.weight

    linear.weight = dc.nn.Parameter(original_weight.numpy().copy())
    assert not _signature_matches(model, signature)

    linear.weight = original_weight
    assert _signature_matches(model, signature)

    # LayerNorm configuration changes must invalidate.
    norm = model[1]
    original_eps = norm.eps

    norm.eps = 1e-3
    assert not _signature_matches(model, signature)

    norm.eps = original_eps
    assert _signature_matches(model, signature)

    # Monkey-patching forward must disable the cached path.
    monkeypatch.setattr(
        dc.nn.ReLU,
        "forward",
        lambda self, x: x * 2,
    )

    assert not _signature_matches(model, signature)
