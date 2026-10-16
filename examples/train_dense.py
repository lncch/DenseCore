"""Deterministic correctness demos; these are not performance benchmarks."""
import json
import numpy as np
from densecore import tensor, nn, optim, manual_seed

def regression():
    rng=np.random.default_rng(37)
    x=rng.normal(size=(64,2)).astype(np.float32)
    heldout=rng.normal(size=(32,2)).astype(np.float32)
    def targets(v):
        return (3*v[:,0]-2*v[:,1]+.5).reshape(-1,1).astype(np.float32)
    manual_seed(7)
    model=nn.Linear(2,1)
    optimizer=optim.SGD(model.parameters(),lr=.08)
    loss=nn.MSELoss()
    initial=float(loss(model(tensor(x)),tensor(targets(x))).numpy())
    for _ in range(100):
        optimizer.zero_grad()
        objective=loss(model(tensor(x)),tensor(targets(x)))
        objective.backward()
        optimizer.step()
    final=float(loss(model(tensor(heldout)),tensor(targets(heldout))).numpy())
    return {"initial_train_mse":initial,"heldout_mse":final,"steps":100,
            "passed":final<1e-3}

def classification():
    rng=np.random.default_rng(111)
    def data(n):
        x=np.concatenate([rng.normal(-2,.6,size=(n,2)),rng.normal(2,.6,size=(n,2))]).astype(np.float32)
        return x,np.concatenate([np.zeros(n,dtype=np.int64),np.ones(n,dtype=np.int64)])
    x,labels=data(64)
    test,truth=data(32)
    manual_seed(11)
    model=nn.Sequential(nn.Linear(2,8),nn.ReLU(),nn.Linear(8,2))
    optimizer=optim.Adam(model.parameters(),lr=.03)
    loss=nn.CrossEntropyLoss()
    initial=float(loss(model(tensor(x)),labels).numpy())
    for _ in range(100):
        optimizer.zero_grad()
        objective=loss(model(tensor(x)),labels)
        objective.backward()
        optimizer.step()
    final=float(loss(model(tensor(x)),labels).numpy())
    accuracy=float(np.mean(model(tensor(test)).numpy().argmax(axis=1)==truth))
    return {"initial_train_cross_entropy":initial,"final_train_cross_entropy":final,
            "heldout_accuracy":accuracy,"steps":100,
            "passed":accuracy>=.95 and final<initial*.5}

if __name__=="__main__":
    results={"regression":regression(),"classification":classification()}
    print(json.dumps(results,indent=2))
    if not all(r["passed"] for r in results.values()):
        raise SystemExit("predeclared toy-task quality gate failed")
