import gc
import weakref
import numpy as np
import pytest
import densecore as d


def test_late_leaf_and_nonleaf_hooks_accumulate_once():
    x = d.tensor(2., requires_grad=True)
    y = x * x
    out = y + y
    seen = []
    y.register_hook(lambda g: seen.append(g.numpy().item()) or g * 3)
    x.register_hook(lambda g: g + 1)
    out.backward()
    assert seen == [2.] and x.grad.numpy().item() == 25.


def test_hook_order_removal_and_snapshot():
    x = d.tensor(2., requires_grad=True)
    events = []
    second = None
    def first(g):
        events.append('first')
        second.remove()
        return g * 2
    x.register_hook(first)
    second = x.register_hook(lambda g: events.append('second') or g + 3)
    x.backward()
    assert x.grad.numpy().item() == 5. and events == ['first','second']
    x.zero_grad()
    x.backward()
    assert x.grad.numpy().item() == 2.
    second.remove()


@pytest.mark.parametrize('bad', [lambda g: 2., lambda g: d.tensor(True,dtype='bool'), lambda g: d.tensor([1.]), lambda g: d.tensor(1.,requires_grad=True)])
def test_invalid_hook_result_preserves_graph_and_retry(bad):
    x = d.tensor(2., requires_grad=True)
    out = x * x
    handle = out.register_hook(bad)
    with pytest.raises((RuntimeError,TypeError,ValueError)): out.backward()
    assert x.grad is None
    handle.remove()
    out.backward()
    assert x.grad.numpy().item() == 4.


def test_hook_exception_preserves_existing_gradients():
    x = d.tensor(2., requires_grad=True)
    x.backward()
    y = x*x
    def fail(g): raise ValueError('hook sentinel')
    handle = x.register_hook(fail)
    with pytest.raises(ValueError,match='hook sentinel'): y.backward()
    assert x.grad.numpy().item() == 1.
    handle.remove()
    y.backward()
    assert x.grad.numpy().item() == 5.


def test_parameter_hooks_survive_optimizer_and_state_replacement():
    model = d.nn.Linear(1,1,bias=False)
    p = model.weight
    calls = []
    p.register_hook(lambda g: calls.append(g.numpy().copy()))
    for _ in range(2):
        model(d.tensor([[2.]])).sum().backward()
        d.optim.SGD(model.parameters(),lr=.1).step()
        model.load_state_dict(model.state_dict())
        p.zero_grad()
    assert len(calls) == 2
    model = d.compile(model)
    before = p.numpy().copy()
    with pytest.raises(ValueError,match='hook'):
        d.compiled_train_step(model,d.tensor([[2.]]),d.tensor([[0.]]),d.optim.SGD(model.parameters(),lr=.1))
    np.testing.assert_array_equal(p.numpy(),before)


def test_custom_hook_order_and_gc_cycle():
    events = []
    class F(d.autograd.Function):
        @staticmethod
        def forward(ctx,x): return x*x
        @staticmethod
        def backward(ctx,g):
            events.append('custom')
            return g*4
    x = d.tensor(2.,requires_grad=True)
    y = F.apply(x)
    class Hook:
        def __init__(self, value): self.value = value
        def __call__(self,g): events.append('hook'); return g*2
    callback = Hook(y)
    reference = weakref.ref(callback)
    y.register_hook(callback)
    y.backward()
    assert events == ['hook','custom'] and x.grad.numpy().item() == 8.
    del callback,y
    gc.collect()
    assert reference() is None


def test_hook_registration_validation():
    with pytest.raises(RuntimeError): d.tensor(2.).register_hook(lambda g:g)
    with pytest.raises(RuntimeError): d.tensor(True,dtype='bool').register_hook(lambda g:g)
    with pytest.raises(TypeError): d.tensor(2.,requires_grad=True).register_hook(2)


def test_each_hook_revalidates_before_next_callback():
    model = d.nn.Linear(1,1,bias=False)
    p = model.weight
    out = model(d.tensor([[2.]]))
    calls = []
    def mutate(g): model.load_state_dict(model.state_dict()); return g
    out.register_hook(mutate)
    out.register_hook(lambda g: calls.append('incorrect'))
    with pytest.raises(RuntimeError,match='modified'): out.backward()
    assert calls == [] and p.grad is None


def test_compiled_outputs_late_hooks_and_fused_identity_refresh():
    model = d.compile(d.nn.Linear(1,1,bias=False))
    x = d.tensor([[2.]],requires_grad=True)
    out = model(x)
    model.weight.register_hook(lambda g:g*3)
    out.sum().backward()
    assert model.weight.grad.numpy().item() == 6.
    other = d.compile(d.nn.Linear(1,1,bias=False))
    d.compiled_train_step(other,d.tensor([[1.]]),d.tensor([[0.]]),d.optim.SGD(other.parameters(),lr=.1))
    other.weight.register_hook(lambda g:g*2)
    other(d.tensor([[3.]])).sum().backward()
    assert other.weight.grad.numpy().item() == 6.


def test_hook_reentrant_backward_rejects_and_disconnected_hook_not_called():
    x = d.tensor(2.,requires_grad=True)
    other = d.tensor(3.,requires_grad=True)
    other.register_hook(lambda g: (_ for _ in ()).throw(AssertionError('disconnected')))
    handle = x.register_hook(lambda g: other.backward())
    with pytest.raises(RuntimeError,match='reentrant'): x.backward()
    assert x.grad is None and other.grad is None
    handle.remove()
    x.backward()
    assert x.grad.numpy().item() == 1.


def test_twenty_thousand_node_hook_graph_and_drop():
    x = d.tensor(1.,requires_grad=True)
    y = x
    for _ in range(20000): y = y + 1.
    x.register_hook(lambda g:g*2)
    y.backward()
    assert x.grad.numpy().item() == 2.
    del y
    gc.collect()


def test_direct_native_backward_refuses_active_hook_graph():
    x = d.tensor(2.,requires_grad=True)
    y = x*x
    handle = x.register_hook(lambda g:g*2)
    with pytest.raises(RuntimeError,match='Python driver'): y._native.backward()
    assert x.grad is None
    handle.remove()
    y.backward()
    assert x.grad.numpy().item() == 4.


def test_native_fused_boundary_rejects_hook_before_update():
    from densecore.tensor import Tensor
    model = d.compile(d.nn.Linear(1,1,bias=False))
    optimizer = d.optim.Adam(model.parameters())
    x,target = d.tensor([[2.]]),d.tensor([[0.]])
    d.compiled_train_step(model,x,target,optimizer)
    runner = model._compiled_training_runner
    p = model.weight
    p.register_hook(lambda g:g*2)
    before = p.numpy().copy()
    state = optimizer.state_dict()
    with pytest.raises(ValueError,match='hook'):
        runner.executor.train_step([x._native,p._native,target._native],[1],optimizer._native,((p,),Tensor.__dict__['_native']))
    np.testing.assert_array_equal(p.numpy(),before)
    assert optimizer.state_dict()['state'][0]['step'] == state['state'][0]['step']


def test_owner_slots_do_not_keep_consumed_native_ancestors_alive():
    x = d.tensor(2.,requires_grad=True)
    y = x*x
    x_slot,y_slot = x._hook_slot,y._hook_slot
    z = y*y
    del x,y
    assert x_slot.identity.alive and y_slot.identity.alive
    z.backward()
    assert not x_slot.identity.alive and not y_slot.identity.alive


def test_hook_survives_deleted_wrapper_and_retained_graph():
    x = d.tensor(2.,requires_grad=True)
    y = x*x
    calls = []
    y.register_hook(lambda g: calls.append(g.numpy().item()) or g*2)
    z = y*y
    del y
    gc.collect()
    z.backward(retain_graph=True)
    z.backward()
    assert calls == [8.,8.] and x.grad.numpy().item() == 128.


def test_same_hook_context_concurrent_backward_is_bounded():
    import os,subprocess,sys
    script = '''
import threading
import densecore as d
entered,release=threading.Event(),threading.Event()
x=d.tensor(2.,requires_grad=True)
y=x*x
errors=[]
def hook(g):
    entered.set()
    assert release.wait(5)
    return g*2
y.register_hook(hook)
def run():
    try:y.backward()
    except BaseException as e:errors.append(e)
thread=threading.Thread(target=run)
thread.start()
try:
    assert entered.wait(5)
    try:y.backward()
    except RuntimeError as e:assert 'already running' in str(e)
    else:raise AssertionError('concurrent hook accepted')
finally:
    release.set()
    thread.join(5)
assert not thread.is_alive() and not errors,errors
assert x.grad.numpy().item()==8.
'''
    subprocess.run([sys.executable,'-c',script],env=os.environ.copy(),check=True,timeout=15)


def test_checkpoint_native_replacement_preserves_hook(tmp_path):
    model = d.nn.Linear(1,1,bias=False)
    optimizer = d.optim.Adam(model.parameters())
    path = tmp_path/'state.safetensors'
    d.checkpoints.save_checkpoint(path,model,optimizer)
    model.weight.register_hook(lambda g:g*3)
    d.checkpoints.load_checkpoint(path,model,optimizer)
    model(d.tensor([[2.]])).sum().backward()
    assert model.weight.grad.numpy().item() == 6.


def test_torch_hook_parity_on_strided_shared_retained_graph():
    torch = pytest.importorskip('torch')
    values = np.array([[1.,2.],[3.,4.]],np.float32)
    x,tx = d.tensor(values,requires_grad=True, dtype="float32"),torch.tensor(values,requires_grad=True)
    y,ty = x.T*x,tx.T*tx
    y.register_hook(lambda g:g.T)
    ty.register_hook(lambda g:g.T)
    x.register_hook(lambda g:g*2)
    tx.register_hook(lambda g:g*2)
    seed = np.array([[.1,.2],[.3,.4]],np.float32)
    for retain in [True,False]:
        y.backward(d.tensor(seed, dtype="float32"),retain_graph=retain)
        ty.backward(torch.tensor(seed),retain_graph=retain)
    np.testing.assert_allclose(x.grad.numpy(),tx.grad.numpy(),rtol=1e-6)


def test_native_fused_guard_survives_parameter_replacement():
    from densecore.tensor import Tensor
    model = d.compile(d.nn.Linear(1,1,bias=False))
    optimizer = d.optim.SGD(model.parameters(),lr=.1)
    x,target = d.tensor([[2.]]),d.tensor([[0.]])
    d.compiled_train_step(model,x,target,optimizer)
    runner = model._compiled_training_runner
    p = model.weight
    p.register_hook(lambda g:g*2)
    p.backward()
    optimizer.step()
    before = p.numpy().copy()
    with pytest.raises(ValueError,match='hook'):
        runner.executor.train_step([x._native,p._native,target._native],[1],optimizer._native,((p,),Tensor.__dict__['_native']))
    np.testing.assert_array_equal(p.numpy(),before)


@pytest.mark.parametrize('replacement',['optimizer','state','checkpoint'])
def test_private_native_guard_after_every_parameter_replacement(replacement,tmp_path):
    model = d.nn.Linear(1,1,bias=False)
    p = model.weight
    handle = p.register_hook(lambda g:g*2)
    if replacement == 'optimizer':
        p.backward()
        d.optim.SGD(model.parameters(),lr=.1).step()
    elif replacement == 'state':
        model.load_state_dict(model.state_dict())
    else:
        path = tmp_path/'model.safetensors'
        d.checkpoints.save_checkpoint(path,model)
        d.checkpoints.load_checkpoint(path,model)
    old_grad = None if p.grad is None else p.grad.numpy().copy()
    with pytest.raises(RuntimeError,match='Python driver'): p._native.backward()
    if old_grad is None: assert p.grad is None
    else: np.testing.assert_array_equal(p.grad.numpy(),old_grad)
    handle.remove()
    p.zero_grad()
    p.backward()
    assert p.grad.numpy().item() == 1.


def test_empty_strided_and_detached_hook_boundaries():
    x = d.tensor(np.empty((0,2)),requires_grad=True, dtype="float32")
    seen = []
    x.register_hook(lambda g: seen.append(g.shape))
    x.sum().backward()
    assert seen == [(0,2)] and x.grad.shape == (0,2)
    assert not x.detach()._contexts and x.grad._hook_slot is None
    with d.no_grad(): out = x + 1.
    assert not out._contexts and out._hook_slot is None


def test_compiled_extraction_retry_after_parent_hook_failure():
    model=d.compile(d.nn.Linear(2,1,bias=False))
    x=d.tensor([[2.,3.]])
    output=model(x).sum()
    def fail(g):raise ValueError('after compiled extraction')
    handle=model.weight.register_hook(fail)
    with pytest.raises(ValueError,match='after compiled extraction'):output.backward()
    assert model.weight.grad is None
    handle.remove()
    output.backward(retain_graph=True)
    first=model.weight.grad.numpy().copy()
    output.backward()
    np.testing.assert_array_equal(model.weight.grad.numpy(),first*2)
