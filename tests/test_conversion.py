import numpy as np
import pytest
from densecore import tensor, nn, checkpoints

torch=pytest.importorskip("torch")

def test_torch_dense_weights_conversion(tmp_path):
    source=torch.nn.Linear(2,1)
    path=tmp_path/"torch.pt";output=tmp_path/"weights.safetensors"
    torch.save(source.state_dict(),path)
    target=nn.Linear(2,1)
    checkpoints.convert_torch_weights(path,output,model=target)
    checkpoints.load_weights(output,target)
    x=np.array([[1,2],[3,4]],dtype=np.float32)
    np.testing.assert_allclose(target(tensor(x, dtype="float32")).numpy(),source(torch.tensor(x)).detach().numpy(),rtol=1e-5,atol=1e-6)

def test_wrapped_checkpoint_name_map(tmp_path):
    path=tmp_path/"source.pt";output=tmp_path/"weights.safetensors"
    torch.save({"state_dict":{"dense.weight":torch.tensor([[3.,4.]]),"dense.bias":torch.tensor([5.])},"epoch":3},path)
    m=nn.Linear(2,1)
    checkpoints.convert_torch_weights(path,output,model=m,key_map={"dense.weight":"weight","dense.bias":"bias"})
    checkpoints.load_weights(output,m)
    np.testing.assert_array_equal(m(tensor([[1.,2.]])).numpy(),[[16]])

def test_invalid_conversion_preserves_destination(tmp_path):
    path=tmp_path/"source.pt";output=tmp_path/"weights.safetensors"
    output.write_bytes(b"original")
    torch.save({"weight":torch.ones((1,2),dtype=torch.float64)},path)
    with pytest.raises(ValueError):
        checkpoints.convert_torch_weights(path,output)
    assert output.read_bytes()==b"original"
    torch.save({"weight":torch.ones((1,2)),"bias":torch.ones(1)},path)
    for mapping in [{"weight":"x"},{"weight":"x","bias":"x"}]:
        with pytest.raises(ValueError):
            checkpoints.convert_torch_weights(path,output,key_map=mapping)
        assert output.read_bytes()==b"original"
    with pytest.raises(ValueError):
        checkpoints.convert_torch_weights(path,output,model=nn.Linear(3,1))
    assert output.read_bytes()==b"original"
