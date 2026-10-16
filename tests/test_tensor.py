import numpy as np
import pytest
from densecore import tensor

def test_copy_ownership():
    source = np.array([[1, 2], [3, 4]], dtype=np.float32)
    t = tensor(source)
    source[0, 0] = 99
    output = t.numpy()
    assert output[0, 0] == 1
    output[0, 0] = 88
    assert t.numpy()[0, 0] == 1
    assert t.shape == (2, 2)
    assert t.dtype == "float32" and t.device == "cpu"

def test_scalar_empty_and_noncontiguous():
    assert tensor(2.0).shape == ()
    assert tensor(np.empty((0, 3), dtype=np.float32)).shape == (0, 3)
    source = np.arange(12, dtype=np.float32).reshape(3, 4).T
    np.testing.assert_array_equal(tensor(source).numpy(), source)

def test_reshape_and_rejections():
    t = tensor([[1, 2], [3, 4]])
    np.testing.assert_array_equal(t.reshape(4).numpy(), [1, 2, 3, 4])
    with pytest.raises(ValueError):
        t.reshape(3)
    np.testing.assert_array_equal(
        t.reshape(-1).numpy(),
        [1, 2, 3, 4],
    )
    with pytest.raises(ValueError):
        t.reshape(-2)
    with pytest.raises(TypeError):
        tensor([1], dtype="complex32")
    with pytest.raises(ValueError):
        tensor([1], device="cuda")

@pytest.mark.parametrize("shape", [(True,), (1.5,), ("2",)])
def test_invalid_dimensions(shape):
    with pytest.raises(TypeError):
        tensor([1, 2]).reshape(*shape)

def test_numpy_integer_shape_and_scalar_reshape():
    assert tensor([1, 2]).reshape(np.int64(2)).shape == (2,)
    assert tensor([2]).reshape().numpy().shape == ()

def test_native_shape_validation():
    from densecore._native import _Tensor
    with pytest.raises(ValueError, match="requires 2 elements, got 1"):
        _Tensor([1.0], [2])
    assert tensor([1]).numpy()[0] == 1

def test_oversized_dimension_has_shape_error():
    with pytest.raises(ValueError, match="shape dimension exceeds"):
        tensor([1]).reshape(1 << 100)
