"""Float32 safetensors weights and versioned, non-executable checkpoints."""
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from .tensor import Tensor
from .nn import Module
from ._native import _replace_leaves

def _module(model):
    if not isinstance(model, Module):
        raise TypeError("model must be a Module")

def _write(path, arrays, metadata):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix="."+path.name+".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        save_file(arrays, name, metadata=metadata)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

def _read(path):
    with safe_open(str(path), framework="numpy") as handle:
        metadata = handle.metadata() or {}
        arrays = {}
        for key in handle.keys():
            value = handle.get_tensor(key)
            if value.dtype != np.dtype(np.float32):
                raise ValueError("only float32 checkpoint tensors are supported")
            arrays[key] = value.copy(order="C")
    return arrays, metadata

def save_weights(path, model):
    _module(model)
    arrays = {name: value.numpy() for name,value in model.state_dict().items()}
    _write(path, arrays, {"format":"densecore.weights", "version":"1"})

def load_weights(path, model):
    _module(model)
    arrays, metadata = _read(path)
    if metadata.get("format") == "densecore.checkpoint":
        raise ValueError("use load_checkpoint for a training checkpoint")
    if metadata.get("format", "").startswith("densecore.") and metadata.get("format") != "densecore.weights":
        raise ValueError("unsupported internal weight format")
    if metadata.get("format") == "densecore.weights" and metadata.get("version") != "1":
        raise ValueError("unsupported weight format version")
    model.load_state_dict({k:Tensor(v) for k,v in arrays.items()})

def _optimizer_for_model(model, optimizer):
    from .optim import _Base
    if not isinstance(optimizer, _Base):
        raise TypeError("optimizer must be a supported optimizer")
    if optimizer._grouped():
        raise ValueError("checkpoints support single-group optimizers; use densecore.save for groups")
    registered = list(model.parameters())
    if len(registered) != len(optimizer._params) or any(a is not b for a,b in zip(registered,optimizer._params)):
        raise ValueError("checkpoint optimizer must cover model parameters in registration order")

def save_checkpoint(path, model, optimizer=None):
    _module(model)
    model_state=model.state_dict()
    arrays = {"model/"+k:v.numpy() for k,v in model_state.items()}
    manifest = {"model":list(model_state),
                "parameter_names":[name for name,_ in model.named_parameters()],
                "optimizer":None}
    if optimizer is not None:
        _optimizer_for_model(model, optimizer)
        state = optimizer.state_dict()
        for i,record in enumerate(state["state"]):
            for name,value in list(record.items()):
                if isinstance(value, Tensor):
                    key=f"optimizer/{i}/{name}"
                    arrays[key]=value.numpy()
                    record[name]=key
        manifest["optimizer"]=state
    _write(path,arrays,{"format":"densecore.checkpoint", "version":"1",
                       "manifest":json.dumps(manifest,allow_nan=False,separators=(",",":"))})

def _unique_pairs(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint manifest field")
        result[key]=value
    return result

def _invalid_constant(value):
    raise ValueError("nonfinite JSON manifest number")

def load_checkpoint(path, model, optimizer=None):
    _module(model)
    arrays, metadata = _read(path)
    if metadata.get("format") != "densecore.checkpoint" or metadata.get("version") != "1":
        raise ValueError("unsupported checkpoint format/version")
    if set(metadata) != {"format","version","manifest"}:
        raise ValueError("unexpected checkpoint metadata")
    manifest = json.loads(metadata["manifest"],object_pairs_hook=_unique_pairs,parse_constant=_invalid_constant)
    if not isinstance(manifest,dict) or set(manifest)!={"model","parameter_names","optimizer"}:
        raise ValueError("invalid checkpoint manifest")
    names=manifest["model"]
    if not isinstance(names,list) or any(not isinstance(n,str) for n in names) or len(set(names))!=len(names):
        raise ValueError("invalid checkpoint model names")
    parameter_names=manifest["parameter_names"]
    if (not isinstance(parameter_names,list) or any(not isinstance(n,str) for n in parameter_names)
            or len(set(parameter_names))!=len(parameter_names)
            or not set(parameter_names).issubset(names)):
        raise ValueError("invalid checkpoint parameter metadata")
    used=set()
    values={}
    for name in names:
        key="model/"+name
        if key not in arrays:
            raise ValueError("missing model tensor")
        used.add(key)
        values[name]=Tensor(arrays[key])
    params,weights=model._prepare_state_dict(values)
    saved=manifest["optimizer"]
    prepared=None
    if (saved is None)!=(optimizer is None):
        raise ValueError("checkpoint and requested optimizer presence must match")
    if optimizer is not None:
        _optimizer_for_model(model, optimizer)
        if manifest["parameter_names"]!=[name for name,_ in model.named_parameters()]:
            raise ValueError("optimizer parameter order mismatch")
        if not isinstance(saved,dict) or not isinstance(saved.get("state"),list):
            raise ValueError("invalid optimizer manifest")
        for i,record in enumerate(saved["state"]):
            if not isinstance(record,dict):
                raise ValueError("invalid optimizer state record")
            for name in ("momentum_buffer","first_moment","second_moment"):
                if name not in record or record[name] is None:
                    continue
                key=record[name]
                if not isinstance(key,str) or key!=f"optimizer/{i}/{name}" or key not in arrays:
                    raise ValueError("invalid optimizer tensor reference")
                record[name]=Tensor(arrays[key])
                used.add(key)
        prepared=optimizer._prepare_state_dict(saved)
    if set(arrays)!=used:
        raise ValueError("unexpected checkpoint payload tensors")
    updated=_replace_leaves([p._native for p in params],[v._native for v in weights])
    for p,native in zip(params,updated):
        p._native=native
        if p._hook_slot is not None and p._hook_slot.callbacks:
            p._sync_hook_slot()
    if optimizer is not None:
        optimizer._native,optimizer._config=prepared

def convert_torch_weights(source, destination, *, model=None, key_map=None):
    """Convert a trusted CPU float32 tensor state_dict without importing model code."""
    from collections.abc import Mapping
    import torch
    state=torch.load(source,map_location="cpu",weights_only=True)
    if isinstance(state,Mapping) and isinstance(state.get("state_dict"),Mapping):
        state=state["state_dict"]
    if not isinstance(state,Mapping) or any(not isinstance(k,str) for k in state):
        raise ValueError("source must contain a named tensor state_dict")
    if key_map is None:
        key_map={key:key for key in state}
    if not isinstance(key_map,Mapping) or set(key_map)!=set(state):
        raise ValueError("name map must cover every source tensor exactly")
    if any(not isinstance(k,str) or not k for k in key_map.values()) or len(set(key_map.values()))!=len(key_map):
        raise ValueError("mapped names must be unique nonempty strings")
    arrays={}
    for key,value in state.items():
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.layout!=torch.strided:
            raise ValueError("only dense float32 Torch tensors are supported")
        arrays[key_map[key]]=value.detach().cpu().numpy().copy(order="C")
    if model is not None:
        _module(model)
        model._prepare_state_dict({k:Tensor(v) for k,v in arrays.items()})
    _write(destination,arrays,{"format":"densecore.weights","version":"1"})
