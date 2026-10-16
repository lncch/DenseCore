import numpy as np
import pytest
import densecore as d
from densecore._native import _GraphExecutor


@pytest.mark.parametrize("axis", [None, 0, -1, (0, 2), ()])
@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("keepdims", [False, True])
def test_registered_axis_reductions_on_offset_strided_views(axis, mean, keepdims):
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
    x = d.tensor(a, requires_grad=True)
    view = x.permute(2, 0, 1)[:, 1:, :]
    op = "mean" if mean else "sum"
    got = getattr(view, op)(axis=axis, keepdims=keepdims)
    expected = getattr(np, op)(a.transpose(2, 0, 1)[:, 1:, :], axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(got.numpy(), expected, rtol=2e-6, atol=1e-6)
    assert got.dtype == "float32" and got.device == "cpu"
    assert d.autograd.gradcheck(
        lambda z: getattr(z.permute(2, 0, 1)[:, 1:, :], op)(axis=axis, keepdims=keepdims), x)


@pytest.mark.parametrize("mean", [False, True])
def test_compiled_and_eager_whole_reductions_share_values_and_gradients(mean):
    torch = pytest.importorskip("torch")
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
    results = []
    grads = []
    op = "mean" if mean else "sum"
    for compiled in [False, True]:
        x = d.tensor(a, requires_grad=True)
        view = x.permute(2, 0, 1)[:, 1:, :]
        if compiled:
            executor = _GraphExecutor(1, [(op, 0, None, None)])
            result = d.Tensor._wrap(executor.execute([view._native]))
        else:
            result = getattr(view, op)()
        result.backward(d.tensor(.7))
        results.append(result.numpy())
        grads.append(x.grad.numpy())
    ref = torch.tensor(a, requires_grad=True)
    want = getattr(ref.permute(2, 0, 1)[:, 1:, :], op)()
    want.backward(torch.tensor(.7))
    for result, gradient in zip(results, grads):
        np.testing.assert_allclose(result, want.detach().numpy(), rtol=2e-6)
        np.testing.assert_allclose(gradient, ref.grad.numpy(), rtol=2e-6)


@pytest.mark.parametrize("mean", [False, True])
def test_empty_and_scalar_compiled_reduction_contract(mean):
    op = "mean" if mean else "sum"
    executor = _GraphExecutor(1, [(op, 0, None, None)])
    for a in [np.empty((2, 0, 3), dtype=np.float32), np.array(3., dtype=np.float32)]:
        x = d.tensor(a, requires_grad=True)
        y = d.Tensor._wrap(executor.execute([x._native]))
        eager = getattr(d.tensor(a), op)()
        np.testing.assert_allclose(y.numpy(), eager.numpy(), equal_nan=True)
        y.backward()
        assert x.grad.shape == x.shape


def test_empty_mean_skips_unrepresentable_unused_divisor_in_backward():
    from densecore._native import _Tensor
    maximum = int(np.iinfo(np.uintp).max)
    x = d.Tensor._wrap(_Tensor([], [0, maximum, 2], True))
    y = x.mean(axis=(1, 2))
    assert y.shape == (0,)
    y.backward(d.tensor(np.empty((0,), dtype=np.float32)))
    assert x.grad.shape == x.shape
    assert x.grad.numel() == 0
