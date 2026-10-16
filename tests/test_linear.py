import numpy as np
import pytest
from densecore import tensor, functional as F

def test_fused_linear_forward_and_gradients():
    x=tensor([[1.,2.],[3.,4.]],requires_grad=True)
    w=tensor([[2.,3.],[4.,5.]],requires_grad=True)
    b=tensor([1.,2.],requires_grad=True)
    y=F.linear(x,w,b)
    np.testing.assert_array_equal(y.numpy(),[[9,16],[19,34]])
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),[[6,8],[6,8]])
    np.testing.assert_array_equal(w.grad.numpy(),[[4,6],[4,6]])
    np.testing.assert_array_equal(b.grad.numpy(),[2,2])

def test_fused_linear_shared_input_and_weight():
    x=tensor([[1.,2.],[3.,4.]],requires_grad=True)
    F.linear(x,x).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),[[8,12],[8,12]])

def test_linear_empty_and_zero_feature_dimensions():
    x=tensor(np.empty((0,2)),requires_grad=True, dtype="float32")
    w=tensor(np.ones((3,2)),requires_grad=True, dtype="float32")
    F.linear(x,w).sum().backward()
    assert x.grad.shape==(0,2)
    np.testing.assert_array_equal(w.grad.numpy(),np.zeros((3,2)))
    y=F.linear(tensor(np.empty((2,0)), dtype="float32"),tensor(np.empty((3,0)), dtype="float32"),tensor([1.,2.,3.]))
    np.testing.assert_array_equal(y.numpy(),[[1,2,3],[1,2,3]])

def test_linear_invalid_inputs():
    with pytest.raises(ValueError):F.linear(tensor([1.,2.,3.]),tensor([[1.,2.]]))
    with pytest.raises(ValueError):F.linear(tensor([[1.,2.]]),tensor([[1.,2.]]),tensor(1.))
    with pytest.raises(TypeError):F.linear([[1,2]],tensor([[1.,2.]]))
