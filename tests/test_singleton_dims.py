import numpy as np
import pytest
import densecore as d


@pytest.mark.parametrize('shape', [(), (1,), (2,1,3), (0,1,3)])
def test_singleton_dimension_values_and_gradients(shape):
    x = d.tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape), requires_grad=True)
    for dim in range(-len(shape)-1,len(shape)+1):
        y = x.unsqueeze(dim)
        assert y.shape == np.expand_dims(x.numpy(), dim).shape
        np.testing.assert_array_equal(y.numpy(), np.expand_dims(x.numpy(),dim))
        assert y.shares_storage(x)
    out = x.unsqueeze(0).squeeze()
    np.testing.assert_array_equal(out.numpy(), np.squeeze(x.numpy()))
    out.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones(shape))


def test_squeeze_selected_axis_noop_and_strided_gradcheck():
    x = d.tensor(np.arange(6, dtype=np.float32).reshape(2,1,3), requires_grad=True)
    assert x.squeeze(0) is x
    assert x.squeeze(1).shape == (2,3)
    assert x.squeeze(-2).shape == (2,3)
    assert d.autograd.gradcheck(lambda a: a.permute(2,1,0).squeeze(1).unsqueeze(-1), x)
    scalar = d.tensor(2.)
    assert scalar.squeeze(0) is scalar and scalar.squeeze(-1) is scalar


@pytest.mark.parametrize('name,dim,error', [('unsqueeze',True,TypeError), ('squeeze',False,TypeError), ('unsqueeze',.5,TypeError), ('squeeze',(0,1.5),TypeError), ('unsqueeze',3,IndexError), ('squeeze',2,IndexError)])
def test_singleton_dimension_invalid_axes(name,dim,error):
    with pytest.raises(error): getattr(d.tensor([[1.]]),name)(dim)


def test_bool_and_custom_owner_propagation():
    mask = d.tensor([[True,False]], dtype='bool')
    assert mask.unsqueeze(0).squeeze().dtype == 'bool'
    class Identity(d.autograd.Function):
        @staticmethod
        def forward(ctx,x): return x
        @staticmethod
        def backward(ctx,g): return g
    x = d.tensor([2.], requires_grad=True)
    Identity.apply(x).unsqueeze(0).squeeze().backward()
    assert x.grad.numpy().item() == 1.


def test_singleton_dimension_torch_parity():
    torch = pytest.importorskip('torch')
    x = d.tensor(np.arange(6, dtype=np.float32).reshape(2,1,3), requires_grad=True)
    tx = torch.tensor(x.numpy(), requires_grad=True)
    result = x.permute(2,1,0).squeeze(1).unsqueeze(-1)
    expected = tx.permute(2,1,0).squeeze(1).unsqueeze(-1)
    result.sum().backward()
    expected.sum().backward()
    np.testing.assert_array_equal(result.numpy(), expected.detach().numpy())
    np.testing.assert_array_equal(x.grad.numpy(), tx.grad.numpy())
