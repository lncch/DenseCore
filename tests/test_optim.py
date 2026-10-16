import copy
import numpy as np
import pytest
from densecore import tensor, nn, optim

def test_sgd_momentum_decay_and_preserved_grad():
    p=nn.Parameter([1.0,2.0])
    opt=optim.SGD([p],lr=.1,momentum=.9,weight_decay=.1)
    p.backward(tensor([2.,3.]))
    opt.step()
    np.testing.assert_allclose(p.numpy(),[.79,1.68],rtol=1e-6)
    np.testing.assert_array_equal(p.grad.numpy(),[2,3])
    assert opt.state_dict()["state"][0]["step"]==1
    opt.zero_grad()
    assert p.grad is None
    before=p.numpy()
    opt.step()
    np.testing.assert_array_equal(p.numpy(),before)
    assert opt.state_dict()["state"][0]["step"]==1

def test_adam_fixture_and_state_copy():
    p=nn.Parameter([1.0,-2.0])
    opt=optim.Adam([p],lr=.1)
    p.backward(tensor([2.,-3.]))
    opt.step()
    np.testing.assert_allclose(p.numpy(),[.9,-1.9],rtol=1e-5,atol=1e-6)
    state=opt.state_dict()
    np.testing.assert_allclose(state["state"][0]["first_moment"].numpy(),[.2,-.3],rtol=1e-6)
    state["state"][0]["first_moment"]._native=tensor([9.,9.])._native
    np.testing.assert_allclose(opt.state_dict()["state"][0]["first_moment"].numpy(),[.2,-.3],rtol=1e-6)

def test_versioned_update_invalidates_retained_graph():
    p=nn.Parameter([2.0])
    y=(p*p).sum()
    y.backward(retain_graph=True)
    optim.SGD([p],lr=.1).step()
    np.testing.assert_allclose(p.numpy(),[1.6])
    with pytest.raises(RuntimeError,match="modified"):
        y.backward()
    np.testing.assert_array_equal(p.grad.numpy(),[4])

def test_module_strict_load_and_shared_alias():
    m=nn.Module()
    m.a=nn.Parameter([1,2])
    m.b=m.a
    identity=id(m.a)
    state=m.state_dict()
    m.load_state_dict({"a":tensor([3.,4.]),"b":tensor([3.,4.])})
    assert id(m.a)==identity and m.a is m.b
    np.testing.assert_array_equal(state["a"].numpy(),[1,2])
    with pytest.raises(ValueError):
        m.load_state_dict({"a":tensor([5.,6.]),"b":tensor([7.,8.])})
    with pytest.raises(ValueError):
        m.load_state_dict({"a":tensor([1.,2.])})
    np.testing.assert_array_equal(m.a.numpy(),[3,4])

def test_frozen_transpose_parent_load_invalidates_graph():
    m=nn.Module()
    m.weight=nn.Parameter([[2.0]],requires_grad=False)
    x=tensor([[3.0]],requires_grad=True)
    y=(x@m.weight.T).sum()
    m.load_state_dict({"weight":tensor([[4.0]])})
    with pytest.raises(RuntimeError,match="modified"):
        y.backward()
    assert x.grad is None

@pytest.mark.parametrize("kind",["sgd","adam"])
def test_resume_and_bad_state_atomicity(kind):
    make=lambda p: optim.SGD([p],lr=.02,momentum=.8) if kind=="sgd" else optim.Adam([p],lr=.02)
    p=nn.Parameter([1.,-2.])
    opt=make(p)
    for _ in range(3):
        opt.zero_grad(); (p*p).sum().backward(); opt.step()
    saved=opt.state_dict()
    q=nn.Parameter(p)
    resumed=make(q)
    resumed.load_state_dict(saved)
    for _ in range(3):
        for value, engine in [(p,opt),(q,resumed)]:
            engine.zero_grad(); (value*value).sum().backward(); engine.step()
    np.testing.assert_array_equal(q.numpy(),p.numpy())
    bad=copy.copy(saved); bad["state"]=[]
    with pytest.raises(ValueError):
        resumed.load_state_dict(bad)
    assert resumed.state_dict()["state"][0]["step"]==6

@pytest.mark.parametrize("kwargs",[{"lr":-1},{"lr":float("nan")},{"lr":float("inf")},{"lr":True},{"momentum":-1},{"weight_decay":-1}])
def test_sgd_bad_hyperparameters(kwargs):
    config={"lr":.01}; config.update(kwargs)
    with pytest.raises((ValueError,TypeError)):
        optim.SGD([nn.Parameter([1])],**config)

def test_optimizer_rejects_duplicates_and_nonleaves():
    p=nn.Parameter([1.])
    with pytest.raises(ValueError):
        optim.SGD([p,p],lr=.1)
    with pytest.raises(ValueError):
        optim.Adam([p*2])
    with pytest.raises(ValueError):
        optim.Adam([])

def test_native_stale_handle_and_duplicate_alias_rejected():
    from densecore._native import _Optimizer, _replace_leaves
    p=nn.Parameter([1.])
    old=p._native
    engine=optim.SGD([p],lr=.1)
    p.backward(tensor([1.])); engine.step()
    with pytest.raises(RuntimeError,match="modified"):
        _replace_leaves([old],[tensor([99.])._native])
    with pytest.raises(ValueError):
        _Optimizer("sgd",[p._native,p._native],.1)
    np.testing.assert_allclose(p.numpy(),[.9])

def test_model_failed_second_shape_load_is_atomic():
    m=nn.Module(); m.a=nn.Parameter([1]); m.b=nn.Parameter([2,3])
    with pytest.raises(ValueError):
        m.load_state_dict({"a":tensor([5.]),"b":tensor([6.])})
    np.testing.assert_array_equal(m.a.numpy(),[1])
    np.testing.assert_array_equal(m.b.numpy(),[2,3])

@pytest.mark.parametrize("config",[{"betas":(1,.9)},{"betas":(-.1,.9)},
    {"betas":(.9,1)},{"eps":0},{"eps":-1},{"eps":1e-100}])
def test_adam_invalid_hyperparameters(config):
    with pytest.raises(ValueError):
        optim.Adam([nn.Parameter([1])],**config)

def test_zero_gradient_decay_and_frozen_parameter():
    p=nn.Parameter([2]); frozen=nn.Parameter([7],requires_grad=False)
    opt=optim.SGD([p,frozen],lr=.1,weight_decay=.5)
    p.backward(tensor([0.])); opt.step()
    np.testing.assert_allclose(p.numpy(),[1.9])
    np.testing.assert_array_equal(frozen.numpy(),[7])
    assert opt.state_dict()["state"][1]["step"]==0

def test_mixed_versions_in_frozen_expression_rejected():
    m=nn.Module(); m.weight=nn.Parameter([[2.]],requires_grad=False)
    old=m.weight.T
    m.load_state_dict({"weight":tensor([[3.]])})
    mixed=m.weight.T+old
    x=tensor([[1.]],requires_grad=True)
    y=(x@mixed).sum()
    with pytest.raises(RuntimeError,match="modified"):
        y.backward()
    assert x.grad is None

def test_nonzero_hyperparameter_underflow_rejected():
    with pytest.raises(ValueError):
        optim.SGD([nn.Parameter([1])],lr=1e-100)

def test_native_dependent_leaf_batch_prepared_before_commit():
    from densecore._native import _replace_leaves
    p=tensor([1.])
    dependent=p*2
    updated=_replace_leaves([p._native,dependent._native],[tensor([3.])._native,tensor([4.])._native])
    assert updated[0].values==[3.] and updated[1].values==[4.]
    assert updated[0].grad is None and updated[1].grad is None

@pytest.mark.parametrize("kind",["sgd","adam"])
def test_disabled_decay_does_not_multiply_infinite_parameter(kind):
    p=nn.Parameter([float("inf")])
    engine=optim.SGD([p],lr=.1) if kind=="sgd" else optim.Adam([p],lr=.1)
    p.backward(tensor([1.]));engine.step()
    assert np.isposinf(p.numpy()[0])
