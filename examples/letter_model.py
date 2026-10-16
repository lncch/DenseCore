"""Shared architecture, initialization and schedule for real-task comparisons."""
import hashlib
import numpy as np


def initial_weights(width):
    rng=np.random.default_rng(73)
    weights={}
    for prefix,inputs,outputs in [('0',16,width),('3',width,26)]:
        bound=1/np.sqrt(inputs)
        weights[prefix+'.weight']=rng.uniform(-bound,bound,(outputs,inputs)).astype(np.float32)
        weights[prefix+'.bias']=rng.uniform(-bound,bound,outputs).astype(np.float32)
    weights['1.weight']=np.ones(width,dtype=np.float32)
    weights['1.bias']=np.zeros(width,dtype=np.float32)
    return weights


def weight_hash(state):
    h=hashlib.sha256()
    for key in sorted(state):h.update(key.encode());h.update(np.ascontiguousarray(state[key]).tobytes())
    return h.hexdigest()


def build(engine,width,weights=None):
    weights=initial_weights(width) if weights is None else weights
    if engine.startswith('densecore'):
        import densecore as d
        model=d.nn.Sequential(d.nn.Linear(16,width),d.nn.LayerNorm(width),d.nn.ReLU(),d.nn.Linear(width,26))
        model.load_state_dict({k:d.tensor(v) for k,v in weights.items()})
        if engine=='densecore_compiled':d.compile(model)
        return model
    import torch
    model=torch.nn.Sequential(torch.nn.Linear(16,width),torch.nn.LayerNorm(width),torch.nn.ReLU(),torch.nn.Linear(width,26))
    model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()})
    return model


def epoch_schedule(indices,epochs=100,batch=128):
    rng=np.random.default_rng(991)
    for _ in range(epochs):
        order=rng.permutation(indices)
        yield [order[start:start+batch] for start in range(0,len(order),batch)]


def metrics(logits,labels):
    predicted=logits.argmax(axis=1)
    confusion=np.zeros((26,26),dtype=np.int64)
    np.add.at(confusion,(labels,predicted),1)
    tp=np.diag(confusion);den=confusion.sum(0)+confusion.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(26,dtype=np.float64),where=den!=0)
    shifted=logits.astype(np.float64)-logits.max(axis=1,keepdims=True)
    loss=np.mean(np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)),labels])
    return dict(accuracy=float(np.mean(predicted==labels)),macro_f1=float(f1.mean()),loss=float(loss))
