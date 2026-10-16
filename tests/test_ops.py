import numpy as np
import pytest
from densecore import tensor

def test_dense_forward():
    x = tensor([[1., -2.], [3., 4.]])
    b = tensor([10., 20.])
    np.testing.assert_array_equal((x + b).numpy(), [[11, 18], [13, 24]])
    np.testing.assert_array_equal((b * x).numpy(), [[10, -40], [30, 80]])
    np.testing.assert_array_equal((2 * x + 1).numpy(), [[3, -3], [7, 9]])
    np.testing.assert_array_equal((x @ x.T).numpy(), [[5, -5], [-5, 25]])
    np.testing.assert_array_equal(x.relu().numpy(), [[1, 0], [3, 4]])
    assert x.sum().numpy() == 6
    assert x.mean().numpy() == 1.5

def test_rejects_extra_broadcasting():
    np.testing.assert_array_equal(
        (tensor(np.ones((2, 3)), dtype="float32") + tensor(np.array([[1], [2]]), dtype="float32")).numpy(),
        np.array([[2, 2, 2], [3, 3, 3]]),
    )
    with pytest.raises(ValueError):
        tensor(np.ones((2, 3)), dtype="float32") + tensor(np.ones((4, 1)), dtype="float32")
    with pytest.raises(ValueError):
        tensor([1., 2.]).transpose()
    with pytest.raises(ValueError):
        tensor([1., 2.]) @ tensor([1., 2., 3.])
    with pytest.raises(ValueError):
        tensor(np.ones((2, 3)), dtype="float32") @ tensor(np.ones((2, 3)), dtype="float32")
    with pytest.raises(TypeError):
        tensor([1.]) + [1]

def test_empty_and_scalar_forward():
    x = tensor(np.empty((0, 3), dtype=np.float32), dtype="float32")
    assert (x + tensor([1., 2., 3.])).shape == (0, 3)
    assert (x * 2).shape == (0, 3)
    assert x.sum().numpy() == 0
    assert np.isnan(x.mean().numpy())
    assert (tensor(np.empty((2, 0)), dtype="float32") @ tensor(np.empty((0, 3)), dtype="float32")).shape == (2, 3)
    np.testing.assert_array_equal((tensor(np.empty((2, 0)), dtype="float32") @ tensor(np.empty((0, 3)), dtype="float32")).numpy(), np.zeros((2, 3)))
    assert (tensor(3.) * tensor(2.)).numpy() == 6

def test_nonsquare_transpose_and_equal_shapes():
    x = tensor([[1., 2., 3.], [4., 5., 6.]])
    np.testing.assert_array_equal(x.T.numpy(), [[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal((x + x).numpy(), [[2, 4, 6], [8, 10, 12]])

@pytest.mark.parametrize("op", ["add", "mul"])
@pytest.mark.parametrize("reverse", [False, True])
def test_numpy_operand_rejected_without_object_fallback(op, reverse):
    import operator
    x = tensor([1.0, 2.0], requires_grad=True)
    a = np.array([3.0, 4.0], dtype=np.float32)
    left, right = (a, x) if reverse else (x, a)
    with pytest.raises(TypeError):
        getattr(operator, op)(left, right)

@pytest.mark.parametrize("value", [np.float32(2), np.float64(2), np.int64(2)])
def test_numpy_real_scalar_preserves_gradient(value):
    x = tensor([1., 2.], requires_grad=True)
    y = value * x + value
    np.testing.assert_array_equal(y.numpy(), [4, 6])
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [2, 2])

def test_large_empty_transpose_finishes_without_dimension_loop():
    import subprocess,sys
    code="import numpy as np; from densecore import tensor; x=tensor(np.empty((10**12,0),dtype=np.float32)); assert x.T.shape==(0,10**12)"
    subprocess.run([sys.executable,"-c",code],check=True,timeout=10)
