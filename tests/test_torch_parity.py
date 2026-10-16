import numpy as np
import pytest
from densecore import tensor

torch = pytest.importorskip("torch")

def compare(actual, reference):
    np.testing.assert_allclose(actual.numpy(), reference.detach().numpy(), rtol=1e-4, atol=1e-6, equal_nan=True)

@pytest.mark.parametrize("seed", [1, 42, 81])
def test_dense_graph_matches_torch(seed):
    rng = np.random.default_rng(seed)
    arrays = [rng.normal(size=s).astype(np.float32) for s in [(3, 4), (4, 2), (2,)]]
    ours = [tensor(a, requires_grad=True, dtype="float32") for a in arrays]
    ref = [torch.tensor(a, requires_grad=True) for a in arrays]
    y = ((ours[0] @ ours[1]) + ours[2]).relu().mean()
    z = ((ref[0] @ ref[1]) + ref[2]).relu().mean()
    compare(y, z)
    y.backward()
    z.backward()
    for a, b in zip(ours, ref):
        compare(a.grad, b.grad)

@pytest.mark.parametrize("shapes", [((2, 3), (2, 3)), ((2, 3), (3,)), ((3,), (2, 3)),
                                    ((), (2, 3)), ((2, 3), ()), ((), ()),
                                    ((0, 3), (3,)), ((2, 0), (0,)),
                                    ((2, 1, 3), (1, 4, 1)), ((3, 1), (1, 4)),
                                    ((1, 0, 3), (1, 1, 3)), ((1,), (0,))])
@pytest.mark.parametrize("op", ["add", "mul"])
def test_binary_broadcast_parity(shapes, op):
    rng = np.random.default_rng(83)
    arrays = [rng.normal(size=s).astype(np.float32) for s in shapes]
    a, b = [tensor(x, requires_grad=True, dtype="float32") for x in arrays]
    ta, tb = [torch.tensor(x, requires_grad=True) for x in arrays]
    y, z = (a+b, ta+tb) if op == "add" else (a*b, ta*tb)
    compare(y,z)
    seed = rng.normal(size=y.shape).astype(np.float32)
    y.backward(tensor(seed, dtype="float32"))
    z.backward(torch.tensor(seed))
    compare(a.grad,ta.grad)
    compare(b.grad,tb.grad)

@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (0, 3)])
@pytest.mark.parametrize("op", ["sum", "mean", "relu", "reshape"])
def test_unary_parity(shape, op):
    rng=np.random.default_rng(14)
    values=rng.normal(size=shape).astype(np.float32)
    a=tensor(values, requires_grad=True, dtype="float32")
    b=torch.tensor(values, requires_grad=True)
    if op=="reshape":
        y,z=a.reshape(values.size),b.reshape(values.size)
    else:
        y,z=getattr(a,op)(),getattr(b,op)()
    compare(y,z)
    seed=rng.normal(size=y.shape).astype(np.float32)
    y.backward(tensor(seed, dtype="float32"))
    z.backward(torch.tensor(seed))
    compare(a.grad,b.grad)

@pytest.mark.parametrize("dims", [(2, 3, 4), (0, 3, 4), (2, 0, 3), (2, 3, 0)])
def test_matmul_transpose_parity(dims):
    m,k,n=dims
    rng=np.random.default_rng(3)
    arrays=[rng.normal(size=(m,k)).astype(np.float32), rng.normal(size=(k,n)).astype(np.float32)]
    a,b=[tensor(v,requires_grad=True, dtype="float32") for v in arrays]
    ta,tb=[torch.tensor(v,requires_grad=True) for v in arrays]
    y,z=(a@b).T,(ta@tb).T
    compare(y,z)
    seed=rng.normal(size=y.shape).astype(np.float32)
    y.backward(tensor(seed, dtype="float32")); z.backward(torch.tensor(seed))
    compare(a.grad,ta.grad); compare(b.grad,tb.grad)

def test_shared_retained_graph_parity():
    a=tensor([2.0,-1.0],requires_grad=True)
    b=torch.tensor([2.0,-1.0],requires_grad=True)
    ay,by=a*a,b*b
    y,z=(ay+ay*ay).sum(),(by+by*by).sum()
    y.backward(retain_graph=True); z.backward(retain_graph=True)
    y.backward(); z.backward()
    compare(a.grad,b.grad)

def test_relu_nan_seed_signed_zero_parity():
    values=np.array([-1.,-0.,0.,1.,np.nan],dtype=np.float32)
    seed=np.array([np.nan,np.inf,np.nan,np.inf,2.],dtype=np.float32)
    a=tensor(values,requires_grad=True, dtype="float32")
    b=torch.tensor(values,requires_grad=True)
    y,z=a.relu(),b.relu()
    compare(y,z)
    np.testing.assert_array_equal(np.signbit(y.numpy()[:4]),np.signbit(z.detach().numpy()[:4]))
    y.backward(tensor(seed, dtype="float32")); z.backward(torch.tensor(seed))
    compare(a.grad,b.grad)

def test_registered_dense_module_parity():
    from densecore import nn
    rng=np.random.default_rng(33)
    arrays=[rng.normal(size=s).astype(np.float32) for s in [(3,4),(3,), (2,3),(2,)]]
    model=nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,2))
    ref=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.ReLU(),torch.nn.Linear(3,2))
    for layer,index in [(0,0),(2,2)]:
        model[layer].weight=nn.Parameter(arrays[index])
        model[layer].bias=nn.Parameter(arrays[index+1])
        with torch.no_grad():
            ref[layer].weight.copy_(torch.tensor(arrays[index]))
            ref[layer].bias.copy_(torch.tensor(arrays[index+1]))
    data=rng.normal(size=(5,4)).astype(np.float32)
    y,z=model(tensor(data, dtype="float32")),ref(torch.tensor(data))
    compare(y,z)
    y.sum().backward(); z.sum().backward()
    for a,b in zip(model.parameters(),ref.parameters()):
        compare(a.grad,b.grad)

@pytest.mark.parametrize("values,labels", [([[0,0],[0,0]],[0,1]),
    ([[10000,9999,-10000],[-10000,10000,9998]],[1,2]),
    ([[float("nan"),1],[float("inf"),2]],[0,1]),
    (np.empty((0,3),dtype=np.float32),[])])
def test_cross_entropy_loss_parity(values,labels):
    from densecore import functional as F
    a=tensor(values,requires_grad=True, dtype="float32")
    b=torch.tensor(np.asarray(values,dtype=np.float32),requires_grad=True)
    y=F.cross_entropy(a,labels)
    z=torch.nn.functional.cross_entropy(b,torch.tensor(labels,dtype=torch.int64))
    compare(y,z)
    y.backward(); z.backward()
    compare(a.grad,b.grad)

def test_mse_target_gradient_parity():
    from densecore import functional as F
    rng=np.random.default_rng(72)
    values=[rng.normal(size=(3,4)).astype(np.float32) for _ in range(2)]
    a,b=[tensor(x,requires_grad=True, dtype="float32") for x in values]
    ta,tb=[torch.tensor(x,requires_grad=True) for x in values]
    y,z=F.mse_loss(a,b),torch.nn.functional.mse_loss(ta,tb)
    compare(y,z)
    y.backward(); z.backward()
    compare(a.grad,ta.grad); compare(b.grad,tb.grad)

@pytest.mark.parametrize("kind,config",[("sgd",{"lr":.03}),
    ("sgd",{"lr":.03,"momentum":.9,"weight_decay":.02}),
    ("adam",{"lr":.01}),
    ("adam",{"lr":.01,"betas":(.8,.95),"eps":1e-6,"weight_decay":.03})])
def test_native_optimizer_multistep_parity(kind,config):
    from densecore import nn, optim
    rng=np.random.default_rng(532)
    initial=rng.normal(size=(3,4)).astype(np.float32)
    p=nn.Parameter(initial)
    q=torch.tensor(initial,requires_grad=True)
    constructor=optim.SGD if kind=="sgd" else optim.Adam
    oracle=torch.optim.SGD if kind=="sgd" else torch.optim.Adam
    opt=constructor([p],**config)
    ref=oracle([q],foreach=False,**config)
    for step in range(12):
        opt.zero_grad(); ref.zero_grad()
        if step!=4:
            grad=rng.normal(size=initial.shape).astype(np.float32)
            p.backward(tensor(grad, dtype="float32")); q.backward(torch.tensor(grad))
        opt.step(); ref.step()
        np.testing.assert_allclose(p.numpy(),q.detach().numpy(),rtol=1e-5,atol=1e-6)
        record=opt.state_dict()["state"][0]
        if kind=="sgd" and config.get("momentum",0):
            np.testing.assert_allclose(record["momentum_buffer"].numpy(),ref.state[q]["momentum_buffer"].numpy(),rtol=1e-5,atol=1e-6)
        if kind=="adam":
            assert record["step"]==int(ref.state[q]["step"])
            np.testing.assert_allclose(record["first_moment"].numpy(),ref.state[q]["exp_avg"].numpy(),rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(record["second_moment"].numpy(),ref.state[q]["exp_avg_sq"].numpy(),rtol=1e-5,atol=1e-6)

@pytest.mark.parametrize("kind",["sgd","adam"])
@pytest.mark.parametrize("loss_kind",["mse","ce"])
def test_complete_training_loop_parity(kind,loss_kind):
    from densecore import nn, optim, manual_seed
    manual_seed(19)
    width=1 if loss_kind=="mse" else 2
    model=nn.Sequential(nn.Linear(2,4),nn.ReLU(),nn.Linear(4,width))
    oracle=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.ReLU(),torch.nn.Linear(4,width))
    oracle.load_state_dict({k:torch.tensor(v.numpy()) for k,v in model.state_dict().items()})
    rng=np.random.default_rng(55)
    data=rng.normal(size=(16,2)).astype(np.float32)
    targets=(data[:,0]+data[:,1]).reshape(-1,1).astype(np.float32) if loss_kind=="mse" else (data[:,0]>0).astype(np.int64)
    a,b=tensor(data, dtype="float32"),torch.tensor(data)
    opt=(optim.SGD(model.parameters(),lr=.02,momentum=.8) if kind=="sgd" else optim.Adam(model.parameters(),lr=.02))
    ref=(torch.optim.SGD(oracle.parameters(),lr=.02,momentum=.8,foreach=False) if kind=="sgd" else torch.optim.Adam(oracle.parameters(),lr=.02,foreach=False))
    loss=nn.MSELoss() if loss_kind=="mse" else nn.CrossEntropyLoss()
    tloss=torch.nn.MSELoss() if loss_kind=="mse" else torch.nn.CrossEntropyLoss()
    for _ in range(15):
        opt.zero_grad(); ref.zero_grad()
        y=loss(model(a),tensor(targets, dtype="float32") if loss_kind=="mse" else targets)
        z=tloss(oracle(b),torch.tensor(targets))
        compare(y,z)
        y.backward(); z.backward()
        opt.step(); ref.step()
        for p,q in zip(model.parameters(),oracle.parameters()):
            np.testing.assert_allclose(p.numpy(),q.detach().numpy(),rtol=1e-5,atol=1e-6)

@pytest.mark.parametrize("flags",[(True,True,True),(True,False,False),(False,True,True)])
def test_functional_linear_gradient_flags_parity(flags):
    from densecore import functional as F
    rng=np.random.default_rng(72)
    arrays=[rng.normal(size=s).astype(np.float32) for s in [(4,3),(2,3),(2,)]]
    ours=[tensor(a,requires_grad=f, dtype="float32") for a,f in zip(arrays,flags)]
    ref=[torch.tensor(a,requires_grad=f) for a,f in zip(arrays,flags)]
    y,z=F.linear(*ours),torch.nn.functional.linear(*ref)
    compare(y,z);y.sum().backward();z.sum().backward()
    for a,b,flag in zip(ours,ref,flags):
        if flag:compare(a.grad,b.grad)
        else:assert a.grad is None
