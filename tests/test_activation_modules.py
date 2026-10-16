import numpy as np
import pytest
import densecore as d


@pytest.mark.parametrize('name', ['Sigmoid', 'Tanh'])
def test_module_functional_state_and_modes(name):
    module = getattr(d.nn, name)()
    function = getattr(d.functional, name.lower())
    x = d.tensor([-.2, .3], requires_grad=True)
    np.testing.assert_array_equal(module(x).numpy(), function(x).numpy())
    assert list(module.parameters()) == [] and module.state_dict() == {}
    module.load_state_dict({})
    module.eval()
    assert module.training is False
    module.train()
    assert module.training is True
    assert d.autograd.gradcheck(module, x)
    with pytest.raises(TypeError): function([1., 2.])
    with pytest.raises(TypeError): module(d.tensor(True, dtype='bool'))


@pytest.mark.parametrize('name', ['Sigmoid', 'Tanh'])
def test_sequential_compiler_fallback_preserves_training(name):
    model = d.nn.Sequential(d.nn.Linear(2, 2), getattr(d.nn, name)())
    x = d.tensor([[.2, -.4]], requires_grad=True)
    expected = model(x).numpy()
    compiled = d.compile(model)
    actual = compiled(x)
    assert compiled._compiled_runner.executor is None
    np.testing.assert_array_equal(actual.numpy(), expected)
    actual.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in model.parameters())
    before = [p.numpy().copy() for p in model.parameters()]
    d.optim.SGD(model.parameters(), lr=.1).step()
    assert any(not np.array_equal(old, p.numpy()) for old, p in zip(before, model.parameters()))
