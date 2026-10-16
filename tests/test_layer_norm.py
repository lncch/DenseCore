import numpy as np
import pytest
import densecore as dc


def test_layer_norm_singleton_affine_gradients_and_registration():
    model=dc.nn.LayerNorm(1)
    x=dc.tensor([[4.],[-2.]],requires_grad=True)
    output=model(x);output.backward(dc.tensor([[2.],[3.]]))
    np.testing.assert_array_equal(output.numpy(),[[0.],[0.]])
    np.testing.assert_array_equal(x.grad.numpy(),[[0.],[0.]])
    np.testing.assert_array_equal(model.weight.grad.numpy(),[0.])
    np.testing.assert_array_equal(model.bias.grad.numpy(),[5.])
    assert set(model.state_dict())=={'weight','bias'}


def test_layer_norm_defaults_no_affine_empty_batch_and_constant_row():
    module=dc.nn.LayerNorm((3,),elementwise_affine=False)
    assert list(module.parameters())==[]
    x=dc.tensor([[5.,5.,5.]],requires_grad=True)
    output=module(x);output.sum().backward()
    np.testing.assert_array_equal(output.numpy(),[[0.,0.,0.]])
    np.testing.assert_array_equal(x.grad.numpy(),[[0.,0.,0.]])
    assert module(dc.tensor(np.zeros((0,3),dtype=np.float32))).shape==(0,3)
    with pytest.raises(ValueError):module(dc.tensor([[1.,2.]]))


@pytest.mark.parametrize('shape',[0,-1,True,(),(2,3)])
def test_layer_norm_rejects_unsupported_shape(shape):
    with pytest.raises((ValueError,TypeError)):dc.nn.LayerNorm(shape)


@pytest.mark.parametrize('eps',[0.,-1.,float('nan'),float('inf'),1e-60,True])
def test_layer_norm_rejects_bad_epsilon(eps):
    with pytest.raises((ValueError,TypeError)):dc.nn.LayerNorm(2,eps=eps)


def test_compiled_layer_norm_keeps_graphs_state_and_changed_epsilon():
    dc.manual_seed(8)
    model=dc.nn.Sequential(dc.nn.Linear(2,3),dc.nn.LayerNorm(3),dc.nn.ReLU())
    x=dc.tensor([[1.,2.]],requires_grad=True)
    baseline=model(x).sum();baseline.backward()
    expected=[p.grad.numpy() for p in model.parameters()]
    model.zero_grad();x.zero_grad();dc.compile(model)
    first=model(x).sum();second=model(dc.tensor([[3.,-1.]])).sum()
    first.backward()
    for p,want in zip(model.parameters(),expected):np.testing.assert_allclose(p.grad.numpy(),want,rtol=1e-4,atol=1e-5)
    model.zero_grad();second.backward()
    with dc.no_grad():
        model[1].eps=.2
        np.testing.assert_allclose(model(x).numpy(),model.forward(x).numpy(),rtol=1e-5,atol=1e-6)
    frozen=model(x).sum();model.load_state_dict(model.state_dict())
    with pytest.raises(RuntimeError,match='modified'):frozen.backward()


def test_compiled_non_affine_shape_contract_and_function_override(monkeypatch):
    model=dc.compile(dc.nn.LayerNorm(3,elementwise_affine=False))
    with pytest.raises(ValueError):model(dc.tensor([[1.,2.,3.,4.]]))
    model.normalized_shape=(4,)
    with pytest.raises(ValueError):model(dc.tensor([[1.,2.,3.]]))
    monkeypatch.setattr(dc.functional,'layer_norm',lambda x,*args,**kwargs:x*0.)
    np.testing.assert_array_equal(model(dc.tensor([[1.,2.,3.]] )).numpy(),[[0.,0.,0.]])
