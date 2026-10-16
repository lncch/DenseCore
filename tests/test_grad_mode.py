import threading
import numpy as np
import pytest
from densecore import tensor, nn, optim, no_grad, enable_grad, is_grad_enabled

def test_modes_and_factory_exception():
    x=tensor([2.],requires_grad=True)
    assert is_grad_enabled()
    with no_grad():
        assert not is_grad_enabled()
        assert not (x*x).requires_grad
        assert tensor([1.],requires_grad=True).requires_grad
        with enable_grad():
            assert is_grad_enabled() and (x*x).requires_grad
        assert not is_grad_enabled()
    assert is_grad_enabled()

def test_reused_nested_context_and_exception_restore():
    guard=no_grad()
    with guard:
        with guard:
            assert not is_grad_enabled()
        assert not is_grad_enabled()
    assert is_grad_enabled()
    with pytest.raises(ValueError):
        with no_grad():
            raise ValueError("example")
    assert is_grad_enabled()

def test_decorator_and_thread_isolation():
    @no_grad()
    def inference(x):return x*x
    x=tensor([2.],requires_grad=True)
    assert not inference(x).requires_grad
    seen=[]
    with no_grad():
        thread=threading.Thread(target=lambda:seen.append(is_grad_enabled()))
        thread.start();thread.join()
    assert seen==[True]
    assert is_grad_enabled()

def test_detach_and_inference_snapshot_survive_updates():
    p=nn.Parameter([2.])
    with no_grad():snapshot=p*p
    detached=p.detach()
    assert not detached.requires_grad and detached.is_leaf
    p.backward(tensor([1.]));optim.SGD([p],lr=.1).step()
    np.testing.assert_array_equal(snapshot.numpy(),[4])
    np.testing.assert_array_equal(detached.numpy(),[2])
    x=tensor([3.],requires_grad=True)
    (x*snapshot).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),[4])

def test_recorded_backward_inside_no_grad():
    x=tensor([2.],requires_grad=True)
    y=(x*x).sum()
    with no_grad():y.backward()
    np.testing.assert_array_equal(x.grad.numpy(),[4])

def test_lazy_async_and_generator_decorators_rejected():
    async def coroutine():return 1
    async def async_generator():yield 1
    def generator():yield 1
    for fn in [coroutine,async_generator,generator]:
        with pytest.raises(TypeError):no_grad()(fn)
