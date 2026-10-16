import concurrent.futures
import numpy as np
import pytest
import densecore as dc


def models(shared=False, norm=False):
    dc.manual_seed(19)
    layers=[dc.nn.Linear(3,3),dc.nn.ReLU()]
    if norm: layers.append(dc.nn.LayerNorm(3))
    layers.append(layers[0] if shared else dc.nn.Linear(3,3))
    a=dc.nn.Sequential(*layers)
    dc.manual_seed(19)
    layers=[dc.nn.Linear(3,3),dc.nn.ReLU()]
    if norm: layers.append(dc.nn.LayerNorm(3))
    layers.append(layers[0] if shared else dc.nn.Linear(3,3))
    b=dc.compile(dc.nn.Sequential(*layers))
    return a,b


def compare(a,b):
    if isinstance(a,dc.Tensor): np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=1e-4,atol=1e-6)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: compare(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare(x,y)
    else: assert a==b


@pytest.mark.parametrize('kind',['sgd','momentum','adam'])
@pytest.mark.parametrize('shared',[False,True])
def test_consecutive_state_parity(kind,shared):
    a,b=models(shared)
    make=lambda m: dc.optim.Adam(m.parameters(),lr=.002,weight_decay=.01) if kind=='adam' else dc.optim.SGD(m.parameters(),lr=.01,momentum=.9 if kind=='momentum' else 0,weight_decay=.01)
    oa,ob=make(a),make(b)
    ids=[id(p) for p in b.parameters()]
    for step in range(12):
        x=dc.tensor(np.random.default_rng(step).normal(size=(4,3)), dtype="float32")
        y=dc.tensor(np.random.default_rng(step+100).normal(size=(4,3)), dtype="float32")
        oa.zero_grad();loss=dc.functional.mse_loss(a(x),y);loss.backward();oa.step()
        fused=dc.compiled_train_step(b,x,y,ob)
        compare(loss,fused);compare(a.state_dict(),b.state_dict());compare(oa.state_dict(),ob.state_dict())
        assert not fused.requires_grad and [id(p) for p in b.parameters()]==ids
        with pytest.raises((ValueError,RuntimeError)): fused.backward()
    # Continue through ordinary training with the same optimizer state.
    for m,o in [(a,oa),(b,ob)]:
        o.zero_grad();dc.functional.mse_loss(m(x),y).backward();o.step()
    compare(a.state_dict(),b.state_dict());compare(oa.state_dict(),ob.state_dict())


def test_preserve_gradients_and_invalidate_old_graph():
    _,m=models();o=dc.optim.Adam(m.parameters());x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    old=dc.functional.mse_loss(m(x),y);old.backward(retain_graph=True)
    gradients=[p.grad for p in m.parameters()]
    dc.compiled_train_step(m,x,y,o)
    for p,g in zip(m.parameters(),gradients):compare(p.grad,g)
    with pytest.raises((ValueError,RuntimeError)):old.backward()


@pytest.mark.parametrize('case',['order','different','shape','frozen','grad','input_grad','target_grad','custom','replacement'])
def test_rejection_is_atomic(case):
    _,m=models(norm=case=='norm');params=list(m.parameters());o=dc.optim.Adam(params)
    x=dc.tensor([[1.,2.,3.]],requires_grad=case=='input_grad');y=dc.tensor([[0.,1.,0.]],requires_grad=case=='target_grad')
    if case=='order':o=dc.optim.Adam(reversed(params))
    if case=='different':o=dc.optim.Adam([dc.nn.Parameter(p.numpy()) for p in params])
    if case=='shape':y=dc.tensor([[1.,2.]])
    if case=='frozen':m._modules['0'].weight=dc.nn.Parameter(params[0],requires_grad=False)
    if case=='custom':m.forward=lambda x:x
    if case=='replacement':m._modules['0'].weight=dc.nn.Parameter(params[0])
    before=m.state_dict();state=o.state_dict()
    with dc.no_grad() if case=='grad' else dc.enable_grad():
        with pytest.raises((ValueError,TypeError,RuntimeError)):dc.compiled_train_step(m,x,y,o)
    compare(before,m.state_dict());compare(state,o.state_dict())


def test_failure_overflow_preserves_every_parameter():
    _,m=models();o=dc.optim.Adam(m.parameters());x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    dc.compiled_train_step(m,x,y,o)
    state=o.state_dict();state['state'][-1]['step']=2**64-1;o.load_state_dict(state)
    before=m.state_dict();state=o.state_dict()
    with pytest.raises(RuntimeError,match='overflow'):dc.compiled_train_step(m,x,y,o)
    compare(before,m.state_dict());compare(state,o.state_dict())


def test_fused_threads_serialize():
    a,b=models();oa=dc.optim.Adam(a.parameters());ob=dc.optim.Adam(b.parameters());x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    for _ in range(20):
        oa.zero_grad();dc.functional.mse_loss(a(x),y).backward();oa.step()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _:dc.compiled_train_step(b,x,y,ob),range(20)))
    compare(a.state_dict(),b.state_dict());compare(oa.state_dict(),ob.state_dict())


def test_guard_mutation_and_cache_reuse():
    _,m=models();o=dc.optim.SGD(m.parameters(),lr=.01);x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    dc.compiled_train_step(m,x,y,o);executor=m._compiled_training_runner.executor
    dc.compiled_train_step(m,x,y,o);assert m._compiled_training_runner.executor is executor
    m.eval();dc.compiled_train_step(m,x,y,o);assert m._compiled_training_runner.executor is not executor
    m._modules['1']=dc.nn.LayerNorm(3)
    before=m.state_dict();state=o.state_dict()
    with pytest.raises(ValueError):dc.compiled_train_step(m,x,y,o)
    compare(before,m.state_dict());compare(state,o.state_dict())


def test_dispatch_subclasses_and_alias_rejected(monkeypatch):
    _,m=models();o=dc.optim.SGD(m.parameters(),lr=.01);x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    with monkeypatch.context() as patch:
        patch.setattr(dc.nn.Linear,'forward',lambda self,x:x)
        with pytest.raises(ValueError):dc.compiled_train_step(m,x,y,o)
    class Custom(dc.nn.Linear):pass
    custom=dc.compile(Custom(3,3));custom_o=dc.optim.SGD(custom.parameters(),lr=.01)
    with pytest.raises(ValueError):dc.compiled_train_step(custom,x,y,custom_o)
    # Distinct Parameter objects must not claim the same native logical input.
    params=list(m.parameters());params[2]._native=params[0]._native
    with pytest.raises(RuntimeError,match='alias'):dc.compiled_train_step(m,x,y,o)


def test_native_publication_preflight():
    _,m=models();o=dc.optim.Adam(m.parameters());x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    dc.compiled_train_step(m,x,y,o);params=list(m.parameters());runner=m._compiled_training_runner
    inputs=[x._native]+[p._native for p in params]+[y._native]
    before=m.state_dict();state=o.state_dict()
    with pytest.raises(ValueError,match='descriptor'):
        runner.executor.train_step(inputs,list(range(1,len(params)+1)),o._native,(params,object()))
    compare(before,m.state_dict());compare(state,o.state_dict())


def test_native_wrong_owner_descriptor_is_atomic():
    _,m=models();o=dc.optim.Adam(m.parameters());x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    dc.compiled_train_step(m,x,y,o);params=list(m.parameters());runner=m._compiled_training_runner
    class Other:
        __slots__=('_native',)
    before=m.state_dict();state=o.state_dict()
    with pytest.raises(TypeError):
        runner.executor.train_step([x._native]+[p._native for p in params]+[y._native],list(range(1,len(params)+1)),o._native,(params,Other.__dict__['_native']))
    compare(before,m.state_dict());compare(state,o.state_dict())


@pytest.mark.parametrize('mutation',['extra_parameter','extra_child','registration_order'])
def test_hot_cache_registry_guards(mutation):
    _,m=models();o=dc.optim.Adam(m.parameters());x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    dc.compiled_train_step(m,x,y,o)
    linear=m._modules['0']
    if mutation=='extra_parameter':linear.extra=dc.nn.Parameter([1])
    if mutation=='extra_child':linear.extra=dc.nn.ReLU()
    if mutation=='registration_order':linear._parameters={'bias':linear.bias,'weight':linear.weight}
    before=m.state_dict();state=o.state_dict()
    with pytest.raises(ValueError):dc.compiled_train_step(m,x,y,o)
    compare(before,m.state_dict());compare(state,o.state_dict())


def test_python_trace_sees_complete_publication():
    import sys
    _,m=models();o=dc.optim.Adam(m.parameters());x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    dc.compiled_train_step(m,x,y,o)
    params=list(m.parameters());old=tuple(id(p._native) for p in params);seen=[]
    def trace(frame,event,arg):
        if event=='line' and frame.f_code.co_filename.endswith('native_training.py'):
            seen.append(tuple(id(p._native) for p in params))
        return trace
    previous=sys.gettrace();sys.settrace(trace)
    try:dc.compiled_train_step(m,x,y,o)
    finally:sys.settrace(previous)
    new=tuple(id(p._native) for p in params)
    assert old!=new and old in seen and new in seen
    assert all(snapshot in (old,new) for snapshot in seen)


@pytest.mark.parametrize('enabled',[False,True])
def test_gc_mode_restored_on_success_and_failure(enabled):
    import gc
    _,m=models();o=dc.optim.Adam(m.parameters());x=dc.tensor([[1.,2.,3.]]);y=dc.tensor([[0.,1.,0.]])
    previous=gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        dc.compiled_train_step(m,x,y,o)
        assert gc.isenabled()==enabled
        state=o.state_dict();state['state'][-1]['step']=2**64-1;o.load_state_dict(state)
        with pytest.raises(RuntimeError):dc.compiled_train_step(m,x,y,o)
        assert gc.isenabled()==enabled
    finally:(gc.enable if previous else gc.disable)()


def test_shapes_eviction_failure_recovery_and_retained_losses():
    a,b=models();oa=dc.optim.Adam(a.parameters());ob=dc.optim.Adam(b.parameters());snapshots=[]
    for step,batch in enumerate([1,32,2,3,4,5,1,32]):
        x=dc.tensor(np.random.default_rng(step).normal(size=(batch,3)), dtype="float32");y=dc.tensor(np.random.default_rng(step+100).normal(size=(batch,3)), dtype="float32")
        oa.zero_grad();loss=dc.functional.mse_loss(a(x),y);loss.backward();oa.step()
        fused=dc.compiled_train_step(b,x,y,ob);snapshots.append((fused,fused.numpy().copy()))
        compare(loss,fused);compare(a.state_dict(),b.state_dict());compare(oa.state_dict(),ob.state_dict())
    saved=ob.state_dict();bad=ob.state_dict();bad['state'][-1]['step']=2**64-1;ob.load_state_dict(bad)
    with pytest.raises(RuntimeError,match='overflow'):dc.compiled_train_step(b,x,y,ob)
    ob.load_state_dict(saved)
    oa.zero_grad();loss=dc.functional.mse_loss(a(x),y);loss.backward();oa.step()
    compare(loss,dc.compiled_train_step(b,x,y,ob));compare(a.state_dict(),b.state_dict());compare(oa.state_dict(),ob.state_dict())
    for loss,snapshot in snapshots:np.testing.assert_array_equal(loss.numpy(),snapshot)


def test_gc_callback_can_inspect_optimizer_without_deadlock():
    import os,subprocess,sys
    code="""
import gc
import densecore as dc
m=dc.compile(dc.nn.Linear(2,1));o=dc.optim.Adam(m.parameters())
x=dc.tensor([[1.,2.]]);y=dc.tensor([[1.]])
calls=[]
def callback(phase,info):
    if phase=='start':
        o._native.snapshot()
        calls.append(1)
gc.callbacks.append(callback)
gc.set_threshold(1,1,1)
for _ in range(5):dc.compiled_train_step(m,x,y,o)
gc.callbacks.remove(callback)
assert calls and gc.isenabled()
print('gc callback completed')
"""
    result=subprocess.run([sys.executable,'-c',code],env=os.environ,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert 'gc callback completed' in result.stdout


@pytest.mark.parametrize("optimizer_type", [dc.optim.SGD, dc.optim.Adam])
def test_frozen_parameter_rejected_with_matching_optimizer(optimizer_type):
    model = dc.compile(dc.nn.Linear(2, 1))
    model.weight = dc.nn.Parameter(model.weight, requires_grad=False)
    optimizer = optimizer_type(model.parameters(), lr=0.01)
    before_model = model.state_dict()
    before_optimizer = optimizer.state_dict()
    with pytest.raises(RuntimeError, match="training requires unique trainable input parameter slots"):
        dc.compiled_train_step(model, dc.tensor([[1.0, 2.0]]), dc.tensor([[1.0]]), optimizer)
    compare(before_model, model.state_dict())
    compare(before_optimizer, optimizer.state_dict())


@pytest.mark.parametrize('activation', [dc.nn.Sigmoid, dc.nn.Tanh, dc.nn.Flatten])
@pytest.mark.parametrize('kind', ['sgd', 'adam'])
def test_new_eager_modules_reject_fused_training_before_mutation(activation, kind):
    model = dc.compile(dc.nn.Sequential(dc.nn.Linear(3, 3), activation()))
    optimizer = dc.optim.Adam(model.parameters()) if kind == 'adam' else dc.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    x, target = dc.tensor([[1.,2.,3.]]), dc.tensor([[0.,1.,0.]])
    dc.functional.mse_loss(model(x), target).backward(retain_graph=True)
    parameters = list(model.parameters())
    handles = [p._native for p in parameters]
    gradients = [p.grad for p in parameters]
    before, state = model.state_dict(), optimizer.state_dict()
    with pytest.raises(ValueError, match='supported built-in modules'):
        dc.compiled_train_step(model, x, target, optimizer)
    compare(before, model.state_dict())
    compare(state, optimizer.state_dict())
    for p, handle, gradient in zip(parameters, handles, gradients):
        assert p._native is handle
        compare(p.grad, gradient)
