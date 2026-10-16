import numpy as np
import pytest
import densecore as dc
from densecore._native import _GraphExecutor

@pytest.mark.parametrize('mean',[False,True])
@pytest.mark.parametrize('shape',[(2,3),(0,3),(2,0),()])
def test_compiled_whole_reductions_match_eager(shape,mean):
    values=np.arange(np.prod(shape,dtype=int),dtype=np.float32).reshape(shape)
    eager=dc.tensor(values,requires_grad=True)
    compiled=dc.tensor(values,requires_grad=True)
    expected=eager.mean() if mean else eager.sum()
    executor=_GraphExecutor(1,[('mean' if mean else 'sum',0,None,None)])
    got=dc.Tensor._wrap(executor.execute([compiled._native]))
    np.testing.assert_allclose(got.numpy(),expected.numpy(),equal_nan=True)
    seed=dc.tensor(2.)
    expected.backward(seed);got.backward(seed)
    np.testing.assert_allclose(compiled.grad.numpy(),eager.grad.numpy(),equal_nan=True)

@pytest.mark.parametrize('m,k,n',[(2,3,4),(1,1,1),(0,3,2),(2,0,3),(2,3,0)])
def test_compiled_matmul_and_branch_reduction_match_eager(m,k,n):
    a=np.arange(m*k,dtype=np.float32).reshape(m,k)*.1
    b=np.arange(k*n,dtype=np.float32).reshape(k,n)*.2
    pairs=[(dc.tensor(a,requires_grad=True),dc.tensor(b,requires_grad=True)) for _ in range(2)]
    x,w=pairs[0]; expected=(x@w).sum(); expected.backward(dc.tensor(.7))
    x,w=pairs[1]; executor=_GraphExecutor(2,[('matmul',0,1,None),('sum',2,None,None)])
    got=dc.Tensor._wrap(executor.execute([x._native,w._native]));got.backward(dc.tensor(.7))
    np.testing.assert_allclose(got.numpy(),expected.numpy(),rtol=1e-5,atol=1e-6)
    for a,b in zip(pairs[0],pairs[1]):np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy(),rtol=1e-5,atol=1e-6)

def test_compiled_shared_matmul_operand_and_invalid_shapes():
    x=dc.tensor([[1.,2.],[3.,4.]],requires_grad=True)
    executor=_GraphExecutor(1,[('matmul',0,0,None)])
    got=dc.Tensor._wrap(executor.execute([x._native]));got.sum().backward()
    expected=dc.tensor([[1.,2.],[3.,4.]],requires_grad=True);(expected@expected).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),expected.grad.numpy())
    with pytest.raises(ValueError):executor.execute([dc.tensor([[1.,2.,3.]])._native])
