import numpy as np
import pytest
import densecore as dc
from densecore._native import _GraphExecutor, _Tensor


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_empty_layout_skips_unreachable_products_in_compiled_workspaces(grad_enabled):
    shape = [np.iinfo(np.uintp).max, 2, 0]
    x = _Tensor([], shape, True)
    executor = _GraphExecutor(1, [("relu", 0, None, None)])
    with dc.enable_grad() if grad_enabled else dc.no_grad():
        out = executor.execute([x])
    assert out.shape == shape
    if grad_enabled:
        out.backward(_Tensor([], shape), retain_graph=False)
        assert x.grad.shape == shape
    else:
        assert not out.requires_grad


def test_specialization_accepts_layout_changes_and_recovers_after_shape_error():
    executor = _GraphExecutor(2, [("matmul", 0, 1, None)])
    weight = dc.tensor([[1., 2.], [3., 4.]])
    base = dc.tensor([[.2, .3], [.4, .5]], requires_grad=True)
    for view in [base, base.T, base]:
        out = dc.Tensor._wrap(executor.execute([view._native, weight._native]))
        np.testing.assert_allclose(out.numpy(), view.numpy() @ weight.numpy(), rtol=1e-6)
        out.sum().backward()
    with pytest.raises(ValueError):
        executor.execute([dc.tensor([[1., 2., 3.]])._native, weight._native])
    with dc.no_grad():
        out = dc.Tensor._wrap(executor.execute([base.T._native, weight._native]))
    np.testing.assert_allclose(out.numpy(), base.numpy().T @ weight.numpy(), rtol=1e-6)
    assert not out.requires_grad
