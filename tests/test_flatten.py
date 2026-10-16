import numpy as np
import pytest
import densecore as d


@pytest.mark.parametrize('shape,start,end,want', [((2,3,4),0,-1,(24,)), ((2,3,4),1,-1,(2,12)), ((2,3,4),0,1,(6,4)), ((2,3,4),-2,-1,(2,12)), ((),0,-1,(1,)), ((2,0,3),1,-1,(2,0))])
def test_flatten_shape_values_and_gradients(shape,start,end,want):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    x = d.tensor(values, requires_grad=True, dtype="float32")
    out = x.flatten(start,end)
    assert out.shape == want
    np.testing.assert_array_equal(out.numpy(), values.reshape(want))
    out.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones(shape))


def test_flatten_layout_bool_and_custom_owners():
    x = d.tensor([[1.,2.],[3.,4.]], requires_grad=True)
    assert x.flatten().shares_storage(x)
    view = x.T
    out = view.flatten()
    assert not out.shares_storage(view)
    np.testing.assert_array_equal(out.numpy(), [1.,3.,2.,4.])
    assert x.flatten(1,1) is x
    assert d.autograd.gradcheck(lambda a: a.T.flatten(), x)
    mask = d.tensor([[True,False]], dtype='bool')
    assert mask.flatten().dtype == 'bool' and mask.flatten().shares_storage(mask)
    class Identity(d.autograd.Function):
        @staticmethod
        def forward(ctx, x): return x
        @staticmethod
        def backward(ctx, grad): return grad
    Identity.apply(x).flatten().sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize('start,end,error', [(True,-1,TypeError), (0,False,TypeError), (.5,-1,TypeError), (3,-1,IndexError), (-4,-1,IndexError), (2,1,ValueError)])
def test_flatten_invalid_axes(start,end,error):
    with pytest.raises(error):
        d.tensor(np.zeros((2,3,4)), dtype="float32").flatten(start,end)


def test_flatten_module_sequential_and_compiler_fallback():
    module = d.nn.Flatten()
    assert list(module.parameters()) == [] and module.state_dict() == {}
    model = d.compile(d.nn.Sequential(module, d.nn.Linear(6,2)))
    x = d.tensor(np.ones((2,2,3)), requires_grad=True, dtype="float32")
    model(x).sum().backward()
    assert model._compiled_runner.executor is None
    assert x.grad.shape == (2,2,3)
    with pytest.raises(TypeError):
        d.nn.Flatten(start_dim=True)


def test_flatten_torch_parity():
    torch = pytest.importorskip('torch')
    x = d.tensor(np.arange(24, dtype=np.float32).reshape(2,3,4), requires_grad=True, dtype="float32")
    tx = torch.tensor(x.numpy(), requires_grad=True)
    actual = x.permute(2,0,1).flatten(1,2)
    expected = tx.permute(2,0,1).flatten(1,2)
    actual.sum().backward()
    expected.sum().backward()
    np.testing.assert_array_equal(actual.numpy(), expected.detach().numpy())
    np.testing.assert_array_equal(x.grad.numpy(), tx.grad.numpy())
