import numpy as np
import pytest
from densecore import tensor, nn, functional as F

def test_mse_both_operand_gradients():
    x=tensor([1.,3.],requires_grad=True)
    target=tensor([2.,1.],requires_grad=True)
    loss=nn.MSELoss()(x,target)
    np.testing.assert_allclose(loss.numpy(),2.5)
    loss.backward()
    np.testing.assert_array_equal(x.grad.numpy(),[-1,2])
    np.testing.assert_array_equal(target.grad.numpy(),[1,-2])
    with pytest.raises(ValueError):
        F.mse_loss(x,tensor(1.))
    with pytest.raises(ValueError):
        nn.MSELoss(reduction="bogus")

def test_cross_entropy_fixture():
    x=tensor([[0.,0.],[0.,0.]],requires_grad=True)
    loss=nn.CrossEntropyLoss()(x,[0,1])
    np.testing.assert_allclose(loss.numpy(),np.log(2),rtol=1e-6)
    loss.backward()
    np.testing.assert_array_equal(x.grad.numpy(),[[-.25,.25],[.25,-.25]])

@pytest.mark.parametrize("labels",[[2,0],[-1,0],[0.0,1.0],[True,False],[[0],[1]],[0]])
def test_invalid_cross_entropy_targets(labels):
    with pytest.raises((ValueError,TypeError)):
        F.cross_entropy(tensor([[1.,2.],[3.,4.]]),labels)

def test_cross_entropy_empty_batch_and_invalid_logits():
    x=tensor(np.empty((0,3)),requires_grad=True, dtype="float32")
    loss=F.cross_entropy(x,[])
    assert np.isnan(loss.numpy())
    loss.backward()
    assert x.grad.shape==(0,3)
    with pytest.raises(ValueError):
        F.cross_entropy(tensor(np.empty((2,0)), dtype="float32"),[0,0])
    with pytest.raises(ValueError):
        F.cross_entropy(tensor([1.,2.]),[0])
    with pytest.raises(TypeError):
        F.cross_entropy(tensor([[1.,2.]]),tensor([0.]))
    with pytest.raises(ValueError):
        nn.CrossEntropyLoss(reduction="bogus")

def test_mixed_boolean_class_labels_rejected_before_coercion():
    with pytest.raises(TypeError):
        F.cross_entropy(tensor([[0.,1.],[1.,0.]]),[True,0])
