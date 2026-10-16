import numpy as np
import pytest
import densecore as dc

torch = pytest.importorskip('torch')

@pytest.mark.parametrize('shape',[(1,1),(3,7),(8,32),(2,129),(2,1025),(0,5)])
@pytest.mark.parametrize('affine,bias',[(True,True),(True,False),(False,False)])
@pytest.mark.parametrize('compiled',[False,True])
def test_layer_norm_torch_forward_and_vjp(shape,affine,bias,compiled):
    rng=np.random.default_rng(32)
    values=rng.normal(size=shape).astype(np.float32)
    seeds=rng.normal(size=shape).astype(np.float32)
    ours=dc.nn.LayerNorm(shape[1],elementwise_affine=affine,bias=bias)
    ref=torch.nn.LayerNorm(shape[1],elementwise_affine=affine,bias=bias)
    if affine:
        weight=rng.normal(size=shape[1]).astype(np.float32)
        ours.weight=dc.nn.Parameter(weight)
        ref.weight.data.copy_(torch.tensor(weight))
        if bias:
            b=rng.normal(size=shape[1]).astype(np.float32)
            ours.bias=dc.nn.Parameter(b);ref.bias.data.copy_(torch.tensor(b))
    if compiled:dc.compile(ours)
    x=dc.tensor(values,requires_grad=True);t=torch.tensor(values,requires_grad=True)
    got=ours(x);want=ref(t)
    if compiled:assert ours._compiled_runner.executor is not None, 'normalization fell back to eager'
    np.testing.assert_allclose(got.numpy(),want.detach().numpy(),rtol=1e-4,atol=1e-5)
    got.backward(dc.tensor(seeds));want.backward(torch.tensor(seeds))
    np.testing.assert_allclose(x.grad.numpy(),t.grad.numpy(),rtol=1e-4,atol=1e-5)
    if affine:
        np.testing.assert_allclose(ours.weight.grad.numpy(),ref.weight.grad.numpy(),rtol=1e-4,atol=1e-5)
        if bias:np.testing.assert_allclose(ours.bias.grad.numpy(),ref.bias.grad.numpy(),rtol=1e-4,atol=1e-5)

@pytest.mark.mac_exact
@pytest.mark.parametrize('offset',[0.,100.,10000.,1e6])
def test_layer_norm_near_constant_rows_and_offsets(offset):
    # Repointed 2026-10-16 (roadmap: PyTorch's bits are the spec; owner approved such repoints):
    # the reference used to be float64 PyTorch; float32 PyTorch loses digits on these rows.
    values=np.array([[offset+.1,offset+.2,offset+.3,offset+.4]],dtype=np.float32)
    x=dc.tensor(values,requires_grad=True);t=torch.tensor(values,requires_grad=True)
    g=dc.functional.layer_norm(x,(4,));w=torch.nn.functional.layer_norm(t,(4,),eps=float(np.float32(1e-5)))
    np.testing.assert_array_equal(g.numpy().view(np.uint32),w.detach().numpy().view(np.uint32))
    seed=np.array([[.2,-.3,.4,.7]],dtype=np.float32)
    g.backward(dc.tensor(seed));w.backward(torch.tensor(seed))
    np.testing.assert_array_equal(x.grad.numpy().view(np.uint32),t.grad.numpy().view(np.uint32))
