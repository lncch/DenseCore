import numpy as np
import pytest
import densecore as d
from densecore._native import _GraphExecutor, _Tensor


@pytest.mark.parametrize("kind", ["add", "mul"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("zero_axis", [0, 1, 2])
def test_huge_empty_compiled_broadcast_returns_empty_and_zero_scalar_vjp(kind, enabled, zero_axis):
    shape = [np.iinfo(np.uintp).max, 2]
    shape.insert(zero_axis, 0)
    x = _Tensor([], shape, True)
    scalar = _Tensor([3.], [], True)
    executor = _GraphExecutor(2, [(kind, 0, 1, None)])
    with d.enable_grad() if enabled else d.no_grad():
        actual = executor.execute([x, scalar])
    assert actual.shape == shape and actual.values == []
    assert actual.requires_grad is enabled
    if enabled:
        actual.backward(_Tensor([], shape))
        assert x.grad.shape == shape and x.grad.values == []
        assert scalar.grad.shape == [] and scalar.grad.values == [0.]
