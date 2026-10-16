import json
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file
from densecore import tensor, nn, optim, checkpoints

def arrays(model):
    return {k:v.numpy() for k,v in model.state_dict().items()}

def assert_model_equal(a,b):
    for k,v in arrays(a).items():
        np.testing.assert_array_equal(v,arrays(b)[k])

def test_weights_roundtrip_and_external_import(tmp_path):
    m=nn.Linear(2,1)
    path=tmp_path/"weights.safetensors"
    checkpoints.save_weights(path,m)
    n=nn.Linear(2,1)
    checkpoints.load_weights(path,n)
    assert_model_equal(m,n)
    save_file({"weight":np.array([[3,4]],dtype=np.float32),"bias":np.array([5],dtype=np.float32)},str(path),metadata={"format":"pt"})
    checkpoints.load_weights(path,n)
    np.testing.assert_array_equal(n(tensor([[1.,2.]])).numpy(),[[16]])

def test_scalar_empty_shared_alias_weights(tmp_path):
    m=nn.Module();m.scalar=nn.Parameter(2.);m.empty=nn.Parameter(np.empty((0,3)));m.alias=m.scalar
    path=tmp_path/"weights.safetensors"
    checkpoints.save_weights(path,m)
    m.load_state_dict({"scalar":tensor(3.),"alias":tensor(3.),"empty":tensor(np.empty((0,3)), dtype="float32")})
    checkpoints.load_weights(path,m)
    assert m.scalar.shape==() and m.empty.shape==(0,3) and m.alias is m.scalar
    assert m.scalar.numpy()==2.

def test_weight_bad_dtype_names_and_shape_are_atomic(tmp_path):
    m=nn.Linear(2,1)
    before=arrays(m)
    path=tmp_path/"bad.safetensors"
    for payload in [{"weight":np.ones((1,2),dtype=np.float64),"bias":np.ones(1,dtype=np.float32)},
                    {"weight":np.ones((1,3),dtype=np.float32),"bias":np.ones(1,dtype=np.float32)},
                    {"weight":np.ones((1,2),dtype=np.float32)}]:
        save_file(payload,str(path))
        with pytest.raises(ValueError):
            checkpoints.load_weights(path,m)
        for name,value in before.items():
            np.testing.assert_array_equal(arrays(m)[name],value)

@pytest.mark.parametrize("kind",["sgd","adam"])
def test_file_checkpoint_resume(tmp_path,kind):
    m=nn.Linear(2,1)
    make=lambda model: optim.SGD(model.parameters(),lr=.02,momentum=.8) if kind=="sgd" else optim.Adam(model.parameters(),lr=.02)
    engine=make(m)
    x=tensor([[1.,2.],[2.,1.]]);target=tensor([[3.],[4.]])
    def step(model,opt):
        opt.zero_grad();nn.MSELoss()(model(x),target).backward();opt.step()
    for _ in range(4):step(m,engine)
    path=tmp_path/"checkpoint.safetensors"
    checkpoints.save_checkpoint(path,m,engine)
    n=nn.Linear(2,1);other=make(n)
    checkpoints.load_checkpoint(path,n,other)
    for _ in range(5):step(m,engine);step(n,other)
    assert_model_equal(m,n)
    assert engine.state_dict()["state"][0]["step"]==other.state_dict()["state"][0]["step"]

def test_bad_optimizer_manifest_preserves_model_and_state(tmp_path):
    m=nn.Linear(2,1);engine=optim.Adam(m.parameters())
    path=tmp_path/"checkpoint.safetensors"
    checkpoints.save_checkpoint(path,m,engine)
    with safe_open(str(path),framework="numpy") as f:
        payload={k:f.get_tensor(k).copy() for k in f.keys()}
        metadata=f.metadata()
    manifest=json.loads(metadata["manifest"])
    manifest["optimizer"]["hyperparameters"]["eps"]=-1
    metadata["manifest"]=json.dumps(manifest)
    save_file(payload,str(path),metadata=metadata)
    target=nn.Linear(2,1);other=optim.Adam(target.parameters())
    before=arrays(target)
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(path,target,other)
    for k,v in before.items():np.testing.assert_array_equal(arrays(target)[k],v)
    assert all(s["step"]==0 for s in other.state_dict()["state"])

def test_unexpected_payload_and_missing_optimizer_rejected(tmp_path):
    m=nn.Linear(2,1);engine=optim.Adam(m.parameters())
    path=tmp_path/"checkpoint.safetensors"
    checkpoints.save_checkpoint(path,m,engine)
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(path,m)
    with safe_open(str(path),framework="numpy") as f:
        payload={k:f.get_tensor(k).copy() for k in f.keys()};metadata=f.metadata()
    payload["unclaimed"]=np.array([1],dtype=np.float32)
    save_file(payload,str(path),metadata=metadata)
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(path,m,engine)

def test_failed_write_keeps_destination_and_cleans_temp(tmp_path,monkeypatch):
    path=tmp_path/"weights.safetensors";path.write_bytes(b"original")
    def interrupted(arrays,filename,metadata=None):
        from pathlib import Path
        Path(filename).write_bytes(b"partial")
        raise OSError("simulated interrupted serialization")
    monkeypatch.setattr(checkpoints,"save_file",interrupted)
    with pytest.raises(OSError):
        checkpoints.save_weights(path,nn.Linear(2,1))
    assert path.read_bytes()==b"original"
    assert list(tmp_path.iterdir())==[path]

def test_checkpoint_without_optimizer_and_empty_model(tmp_path):
    path=tmp_path/"checkpoint.safetensors"
    m=nn.Linear(2,1);n=nn.Linear(2,1)
    checkpoints.save_checkpoint(path,m)
    checkpoints.load_checkpoint(path,n)
    assert_model_equal(m,n)
    empty=nn.Module()
    checkpoints.save_checkpoint(path,empty)
    checkpoints.load_checkpoint(path,empty)
    assert empty.state_dict()=={}

def test_scalar_and_empty_optimizer_buffers_resume(tmp_path):
    m=nn.Module();m.scalar=nn.Parameter(2.);m.empty=nn.Parameter(np.empty((0,3)))
    engine=optim.Adam(m.parameters(),lr=.01)
    m.scalar.backward();m.empty.backward(tensor(np.empty((0,3)), dtype="float32"));engine.step()
    path=tmp_path/"checkpoint.safetensors"
    checkpoints.save_checkpoint(path,m,engine)
    n=nn.Module();n.scalar=nn.Parameter(0.);n.empty=nn.Parameter(np.empty((0,3)))
    other=optim.Adam(n.parameters())
    checkpoints.load_checkpoint(path,n,other)
    assert_model_equal(m,n)
    records=other.state_dict()["state"]
    assert records[0]["first_moment"].shape==()
    assert records[1]["first_moment"].shape==(0,3)

def test_bad_parameter_order_and_version_preserve_target(tmp_path):
    m=nn.Linear(2,1);engine=optim.Adam(m.parameters())
    path=tmp_path/"checkpoint.safetensors"
    checkpoints.save_checkpoint(path,m,engine)
    with safe_open(str(path),framework="numpy") as f:
        payload={k:f.get_tensor(k).copy() for k in f.keys()};metadata=f.metadata()
    manifest=json.loads(metadata["manifest"])
    manifest["parameter_names"].reverse()
    metadata["manifest"]=json.dumps(manifest)
    save_file(payload,str(path),metadata=metadata)
    target=nn.Linear(2,1);other=optim.Adam(target.parameters());before=arrays(target)
    with pytest.raises(ValueError):checkpoints.load_checkpoint(path,target,other)
    for k,v in before.items():np.testing.assert_array_equal(arrays(target)[k],v)
    metadata["version"]="99";save_file(payload,str(path),metadata=metadata)
    with pytest.raises(ValueError):checkpoints.load_checkpoint(path,target,other)

def test_model_only_checkpoint_validates_parameter_metadata(tmp_path):
    m=nn.Linear(2,1);path=tmp_path/"checkpoint.safetensors"
    checkpoints.save_checkpoint(path,m)
    with safe_open(str(path),framework="numpy") as f:
        payload={k:f.get_tensor(k).copy() for k in f.keys()};metadata=f.metadata()
    manifest=json.loads(metadata["manifest"]);manifest["parameter_names"]=123
    metadata["manifest"]=json.dumps(manifest);save_file(payload,str(path),metadata=metadata)
    with pytest.raises(ValueError):checkpoints.load_checkpoint(path,m)

def test_unknown_internal_weight_format_rejected(tmp_path):
    m=nn.Linear(2,1);path=tmp_path/"weights.safetensors"
    save_file({k:v.numpy() for k,v in m.state_dict().items()},str(path),
              metadata={"format":"densecore.unknown","version":"1"})
    with pytest.raises(ValueError):checkpoints.load_weights(path,m)
