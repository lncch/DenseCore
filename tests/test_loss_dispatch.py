import numpy as np
import pytest
import densecore as d
from densecore._native import _GraphExecutor


def _compiled(op, x, target):
    executor = _GraphExecutor(2, [(op, 0, 1, None)])
    return d.Tensor._wrap(executor.execute([x._native, target._native]))


@pytest.mark.parametrize("magnitude", [1e30, -1e30])
def test_compiled_cross_entropy_keeps_small_loss_with_large_equal_logits(magnitude):
    x = d.tensor(np.full((2, 3), magnitude, dtype=np.float32), requires_grad=True)
    eager = d.functional.cross_entropy(x, [0, 2])
    got = _compiled("cross_entropy", x, d.tensor([0., 2.]))
    np.testing.assert_allclose(got.numpy(), eager.numpy(), rtol=1e-6)
    np.testing.assert_allclose(got.numpy(), np.log(3), rtol=1e-6)
    got.backward()
    expected = np.full((2, 3), 1/6, dtype=np.float32)
    expected[0, 0] -= .5
    expected[1, 2] -= .5
    np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-7)


@pytest.mark.parametrize("compiled", [False, True])
def test_mse_strided_operands_and_both_gradients_match_torch(compiled):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    arrays = [rng.normal(size=(3, 4)).astype(np.float32) for _ in range(2)]
    x, target = [d.tensor(a, requires_grad=True) for a in arrays]
    rx, rt = [torch.tensor(a, requires_grad=True) for a in arrays]
    a, b = x.T[1:, ::2], target.T[1:, ::2]
    got = _compiled("mse", a, b) if compiled else d.functional.mse_loss(a, b)
    want = torch.nn.functional.mse_loss(rx.T[1:, ::2], rt.T[1:, ::2])
    got.backward(d.tensor(1.3))
    want.backward(torch.tensor(1.3))
    np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=2e-6)
    for actual, reference in [(x, rx), (target, rt)]:
        np.testing.assert_allclose(actual.grad.numpy(), reference.grad.numpy(), rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("compiled", [False, True])
def test_cross_entropy_strided_logits_and_label_values_match_torch(compiled):
    torch = pytest.importorskip("torch")
    a = np.random.default_rng(13).normal(size=(6, 6)).astype(np.float32)
    x = d.tensor(a, requires_grad=True)
    rx = torch.tensor(a, requires_grad=True)
    labels = [0, 2, 1]
    view = x[1::2, ::2].T
    # Unselected invalid entries must not be read as logical labels.
    target = d.tensor([0., 99., 2., 99., 1., 99.])[::2]
    got = _compiled("cross_entropy", view, target) if compiled else d.functional.cross_entropy(view, labels)
    want = torch.nn.functional.cross_entropy(rx[1::2, ::2].T, torch.tensor(labels))
    got.backward(d.tensor(.7))
    want.backward(torch.tensor(.7))
    np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=2e-6)
    np.testing.assert_allclose(x.grad.numpy(), rx.grad.numpy(), rtol=2e-6, atol=1e-7)
    assert target.grad is None


@pytest.mark.parametrize("compiled", [False, True])
def test_loss_full_jacobians_and_shared_storage(compiled):
    x = d.tensor([[.2, .4, .9], [.7, -.3, .1], [.4, .8, -.2]], requires_grad=True)
    def mse(a):
        return _compiled("mse", a, a.T) if compiled else d.functional.mse_loss(a, a.T)
    def ce(a):
        return _compiled("cross_entropy", a.T, d.tensor([0., 2., 1.])) if compiled else d.functional.cross_entropy(a.T, [0, 2, 1])
    assert d.autograd.gradcheck(mse, x)
    assert d.autograd.gradcheck(ce, x)


@pytest.mark.parametrize("labels", [[-.1, 0.], [0., 3.], [0., .5], [0., float("nan")], [0., float("inf")]])
def test_compiled_invalid_labels_and_workspace_recovery(labels):
    executor = _GraphExecutor(2, [("cross_entropy", 0, 1, None)])
    x = d.tensor([[.2, .4, .9], [.7, -.3, .1]], requires_grad=True)
    with pytest.raises(ValueError):
        executor.execute([x._native, d.tensor(labels)._native])
    got = d.Tensor._wrap(executor.execute([x._native, d.tensor([0., 2.])._native]))
    expected = d.functional.cross_entropy(x, [0, 2])
    np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=1e-6)
    got.backward()
    assert x.grad is not None


def test_empty_batch_contracts_and_mse_shape_validation():
    x = d.tensor(np.empty((0, 3), dtype=np.float32), requires_grad=True)
    eager = d.functional.cross_entropy(x, [])
    assert np.isnan(eager.numpy())
    eager.backward()
    assert x.grad.shape == (0, 3)
    with pytest.raises(ValueError): _compiled("cross_entropy", x, d.tensor([]))
    for compiled in [False, True]:
        with pytest.raises(ValueError):
            _compiled("mse", x, x) if compiled else d.functional.mse_loss(x, x)
        a, b = d.tensor([[1., 2.]]), d.tensor([1., 2.])
        with pytest.raises(ValueError):
            _compiled("mse", a, b) if compiled else d.functional.mse_loss(a, b)


@pytest.mark.parametrize("row", [[float("nan"), 0., 1.], [float("inf"), 0., 1.],
                                 [float("-inf"), 0., 1.], [float("-inf")] * 3])
def test_cross_entropy_special_values_match_between_execution_paths(row):
    roots = [d.tensor([row], requires_grad=True) for _ in range(2)]
    eager = d.functional.cross_entropy(roots[0], [0])
    compiled = _compiled("cross_entropy", roots[1], d.tensor([0.]))
    np.testing.assert_allclose(compiled.numpy(), eager.numpy(), equal_nan=True)
    eager.backward()
    compiled.backward()
    np.testing.assert_allclose(roots[1].grad.numpy(), roots[0].grad.numpy(), equal_nan=True)
