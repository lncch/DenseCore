import numpy as np
import pytest
from densecore import tensor, nn, manual_seed

def test_registration_shared_parameter_and_plain_results():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter([[1, 2], [3, 4]])
            self.alias = self.weight
            self.bias = nn.Parameter([5, 6])
        def forward(self, x):
            return x @ self.weight.T + self.bias
    m=Model()
    assert list(m.parameters()) == [m.weight, m.bias]
    assert list(m.named_parameters()) == [("weight",m.weight),("bias",m.bias)]
    y=m(tensor([[2., 3.]]))
    np.testing.assert_array_equal(y.numpy(), [[13,24]])
    assert type(y) is type(tensor(1.))
    y.sum().backward()
    np.testing.assert_array_equal(m.weight.grad.numpy(), [[2,3],[2,3]])
    m.zero_grad()
    assert m.weight.grad is None

def test_linear_seed_sequential_and_modes():
    manual_seed(42)
    a=nn.Linear(3,2)
    manual_seed(42)
    b=nn.Linear(3,2)
    np.testing.assert_array_equal(a.weight.numpy(),b.weight.numpy())
    assert a.weight.shape==(2,3) and a.bias.shape==(2,)
    m=nn.Sequential(a,nn.ReLU(),nn.Linear(2,1,bias=False))
    assert list(dict(m.named_parameters())) == ["0.weight","0.bias","2.weight"]
    assert m(tensor([[1.,2.,3.]])).shape==(1,1)
    m.eval()
    assert not m.training and not a.training
    m.train()
    assert m.training and a.training

@pytest.mark.parametrize("args",[(0,2),(-1,2),(True,2),(1.5,2)])
def test_linear_invalid_sizes(args):
    with pytest.raises((ValueError,TypeError)):
        nn.Linear(*args)

def test_module_cycles_and_replacement():
    m=nn.Module()
    with pytest.raises(ValueError):
        m.child=m
    child=nn.Module()
    m.child=child
    with pytest.raises(ValueError):
        child.parent=m
    m.weight=nn.Parameter([1])
    m.weight=None
    assert list(m.parameters())==[]

def test_parameter_replacement_preserves_registration_order():
    m=nn.Linear(2,1)
    m.weight=nn.Parameter([[3,4]])
    assert list(dict(m.named_parameters()))==["weight","bias"]
    with pytest.raises(TypeError):
        m.weight=tensor([[1.,2.]])
    np.testing.assert_array_equal(m.weight.numpy(),[[3,4]])

def test_reserved_registration_name_rejected():
    m=nn.Module()
    with pytest.raises(ValueError):
        m._parameters=nn.Parameter([1])
    assert list(m.parameters())==[]
