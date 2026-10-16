import math

import numpy as np
import pytest
import densecore as d

MODES = ['none', 'tanh']


def _reference(x, approximate):
    x = np.asarray(x, dtype=np.float64)
    if approximate == 'none':
        erfc = np.vectorize(math.erfc)
        return 0.5 * x * erfc(-x / math.sqrt(2.0))
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    # 0.5 * (1 + tanh(u)) == 1 / (1 + exp(-2u)); this form avoids tail cancellation.
    with np.errstate(over='ignore'):
        return x / (1.0 + np.exp(-2.0 * inner))


@pytest.mark.mac_exact
@pytest.mark.parametrize('approximate', MODES)
def test_gelu_matches_float64_reference_including_negative_tail(approximate):
    values = np.concatenate([np.linspace(-12, 12, 481), [-30., -9., -6., -5., -3., 1e-30, -1e-30]])
    x = d.tensor(values.astype(np.float32), dtype="float32")
    out = x.gelu(approximate).numpy()
    if approximate == 'none':
        # Repointed 2026-10-16 (roadmap: PyTorch's bits are the spec; approved by the owner).
        # PyTorch's float32 erf rounds the far negative tail to -0.0, where the old test
        # wanted the float64 value (e.g. -1.3e-30 at x = -12).
        torch = pytest.importorskip('torch')
        ref = torch.nn.functional.gelu(torch.from_numpy(values.astype(np.float32))).numpy()
        np.testing.assert_array_equal(out.view(np.uint32), ref.view(np.uint32))
        return
    ref = _reference(values.astype(np.float32), approximate)
    np.testing.assert_allclose(out, ref, rtol=2e-6, atol=1e-30)


@pytest.mark.parametrize('approximate', MODES)
def test_gelu_strided_full_jacobian(approximate):
    x = d.tensor([[-2.5, -.4, .3], [.9, 1.7, -1.1]], requires_grad=True)
    assert d.autograd.gradcheck(lambda a: a.T[::2].gelu(approximate), x)


@pytest.mark.mac_exact
@pytest.mark.parametrize('approximate', MODES)
def test_gelu_values_and_gradients_match_torch_with_special_values(approximate):
    torch = pytest.importorskip('torch')
    values = np.array([-np.inf, -100., -20., -5., -1., -0., 0., .5, 1., 5., 20., 100., np.inf, np.nan], np.float32)
    x = d.tensor(values, requires_grad=True, dtype="float32")
    tx = torch.tensor(values, requires_grad=True)
    out = x.gelu(approximate)
    ref = torch.nn.functional.gelu(tx, approximate=approximate)
    out.sum().backward()
    ref.sum().backward()
    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=2e-6, atol=1e-7, equal_nan=True)
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=2e-6, atol=1e-6, equal_nan=True)


def test_gelu_signed_zero_and_nonfinite_without_optional_torch():
    x = d.tensor([-np.inf, -0., 0., np.inf, np.nan], requires_grad=True)
    out = x.gelu()
    values = out.numpy()
    assert np.isnan(values[0]) and np.isnan(values[4])
    np.testing.assert_array_equal(values[1:4], [-0., 0., np.inf])
    assert np.signbit(values[1]) and not np.signbit(values[2])
    out.sum().backward()
    gradient = x.grad.numpy()
    np.testing.assert_array_equal(gradient[1:3], [.5, .5])
    assert np.isnan(gradient[0]) and np.isnan(gradient[3]) and np.isnan(gradient[4])


@pytest.mark.parametrize('approximate', MODES)
def test_gelu_module_functional_empty_nograd_and_bool_rejection(approximate):
    module = d.nn.GELU(approximate)
    x = d.tensor(np.empty((0, 3)), requires_grad=True, dtype="float32")
    module(x).sum().backward()
    assert x.grad.shape == x.shape
    with d.no_grad():
        out = d.functional.gelu(d.tensor(0., requires_grad=True), approximate)
    assert not out.requires_grad and out.numpy().item() == 0.
    with pytest.raises(TypeError, match='float32'):
        d.tensor(True, dtype='bool').gelu(approximate)


def test_gelu_rejects_unknown_approximation_and_non_tensor_input():
    with pytest.raises(ValueError, match='approximate'):
        d.tensor([1.]).gelu('erf')
    with pytest.raises(ValueError, match='approximate'):
        d.nn.GELU('fast')
    with pytest.raises(TypeError, match='Tensor'):
        d.functional.gelu([1.])


def test_gelu_model_trains_and_compile_falls_back_to_eager_results():
    d.manual_seed(3)
    model = d.nn.Sequential(d.nn.Linear(4, 8), d.nn.GELU(), d.nn.Linear(8, 2))
    x = d.tensor(np.random.default_rng(1).normal(size=(5, 4)).astype(np.float32), dtype="float32")
    eager = model(x).numpy()
    compiled = d.compile(model)
    np.testing.assert_array_equal(compiled(x).numpy(), eager)
    target = d.tensor(np.zeros((5, 2), np.float32), dtype="float32")
    optimizer = d.optim.SGD(model.parameters(), lr=.1)
    loss_fn = d.nn.MSELoss()
    first = float(loss_fn(model(x), target).numpy())
    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()
    assert float(loss.numpy()) < first


def test_fused_training_rejects_gelu_model_before_mutation():
    d.manual_seed(4)
    model = d.compile(d.nn.Sequential(d.nn.Linear(3, 4), d.nn.GELU(), d.nn.Linear(4, 1)))
    optimizer = d.optim.SGD(model.parameters(), lr=.1)
    before = [p.numpy().copy() for p in model.parameters()]
    x = d.tensor(np.ones((2, 3), np.float32), dtype="float32")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        d.compiled_train_step(model, x, d.tensor(np.zeros((2, 1), np.float32), dtype="float32"), optimizer)
    for old, p in zip(before, model.parameters()):
        np.testing.assert_array_equal(p.numpy(), old)
