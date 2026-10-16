import numpy as np
import pytest
from densecore import tensor

def test_shared_edges_and_bias_gradient():
    x = tensor([[1., 2.], [3., 4.]], requires_grad=True)
    b = tensor([5., 6.], requires_grad=True)
    y = ((x * x) + x + b).sum()
    y.backward()
    np.testing.assert_array_equal(x.grad.numpy(), [[3, 5], [7, 9]])
    np.testing.assert_array_equal(b.grad.numpy(), [2, 2])
    assert y.grad is None
    x.zero_grad()
    assert x.grad is None

def test_unequal_shared_paths():
    x = tensor([2.0], requires_grad=True)
    y = x * x
    (y + y * y).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [36])

def test_matmul_transpose_reshape_seed():
    a = tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    b = tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
    y = (a @ b).T.reshape(4)
    y.backward(tensor([1., 2., 3., 4.]))
    np.testing.assert_array_equal(a.grad.numpy(), [[7, 15, 23], [10, 22, 34]])
    np.testing.assert_array_equal(b.grad.numpy(), [[9, 19], [12, 26], [15, 33]])

def test_scalar_and_mean_gradient():
    x = tensor([1., 2., 3.], requires_grad=True)
    s = tensor(2.0, requires_grad=True)
    (x * s + s).mean().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2/3]*3)
    np.testing.assert_allclose(s.grad.numpy(), 3)

def test_relu_gradient_and_leaf_seed():
    x = tensor([-1., 0., 2.], requires_grad=True)
    x.relu().backward(tensor([3., 4., 5.]))
    np.testing.assert_array_equal(x.grad.numpy(), [0, 0, 5])
    x.backward(tensor([1., 1., 1.]))
    np.testing.assert_array_equal(x.grad.numpy(), [1, 1, 6])

def test_finite_difference():
    values = np.array([0.7, -1.2, 2.3], dtype=np.float32)
    x = tensor(values, requires_grad=True, dtype="float32")
    (x * x * x + x * 2).mean().backward()
    numerical = []
    for i in range(3):
        lo, hi = values.copy(), values.copy()
        lo[i] -= 1e-3
        hi[i] += 1e-3
        numerical.append((np.mean(hi**3 + 2*hi)-np.mean(lo**3 + 2*lo))/2e-3)
    np.testing.assert_allclose(x.grad.numpy(), numerical, rtol=2e-2, atol=2e-3)

def test_retained_then_consumed_backward():
    x = tensor([2.0], requires_grad=True)
    y = (x * x).sum()
    y.backward(retain_graph=True)
    y.backward()
    np.testing.assert_array_equal(x.grad.numpy(), [8])
    assert y.numpy() == 4
    with pytest.raises(RuntimeError):
        y.backward()
    np.testing.assert_array_equal(x.grad.numpy(), [8])

def test_consumed_shared_path_is_atomic():
    x = tensor([2.0], requires_grad=True)
    shared = x * x
    left, right = shared.sum(), (shared * 3).sum()
    left.backward()
    with pytest.raises(RuntimeError):
        right.backward()
    np.testing.assert_array_equal(x.grad.numpy(), [4])

def test_invalid_seeds_and_flags():
    x = tensor([2.0, 3.0], requires_grad=True)
    x.backward(tensor([1., 1.]))
    y = x * x
    for seed in [None, tensor(1.), tensor([1., 1.], requires_grad=True),
                 tensor([1., 1.], requires_grad=True) * 2]:
        with pytest.raises(RuntimeError):
            y.backward(seed)
        np.testing.assert_array_equal(x.grad.numpy(), [1, 1])
    with pytest.raises(RuntimeError):
        y.zero_grad()
    with pytest.raises(RuntimeError):
        tensor(1.).backward()
    with pytest.raises(TypeError):
        y.backward([1, 1])
    with pytest.raises(TypeError):
        y.backward(tensor([1., 1.]), retain_graph=1)
    with pytest.raises(TypeError):
        tensor(1., requires_grad=1)
    y.backward(tensor([1., 1.]))
    np.testing.assert_array_equal(x.grad.numpy(), [5, 7])

def test_grad_copy_is_detached_and_flags():
    x = tensor([2.], requires_grad=True)
    y = x * x
    assert x.is_leaf and not y.is_leaf and y.requires_grad
    assert (tensor([1.]) + 2).is_leaf
    y.sum().backward()
    g = x.grad
    assert g.is_leaf and not g.requires_grad
    g.numpy()[0] = 999
    np.testing.assert_array_equal(x.grad.numpy(), [4])

def test_empty_gradients():
    x = tensor(np.empty((0, 3)), requires_grad=True, dtype="float32")
    b = tensor([1., 2., 3.], requires_grad=True)
    (x + b).mean().backward()
    assert x.grad.shape == (0, 3)
    np.testing.assert_array_equal(b.grad.numpy(), [0, 0, 0])
