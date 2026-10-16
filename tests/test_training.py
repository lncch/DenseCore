import importlib.util
from pathlib import Path
import numpy as np
from densecore import tensor, nn, optim, manual_seed

def demos():
    path=Path(__file__).resolve().parents[1]/"examples"/"train_dense.py"
    spec=importlib.util.spec_from_file_location("training_demo",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_regression_quality_gate():
    assert demos().regression()["passed"]

def test_classification_quality_gate():
    assert demos().classification()["passed"]

def test_model_and_adam_resume():
    manual_seed(7)
    model=nn.Linear(2,1)
    engine=optim.Adam(model.parameters(),lr=.02)
    x=tensor([[1.,2.],[2.,1.]])
    target=tensor([[3.],[4.]])
    def step(m,o):
        o.zero_grad(); nn.MSELoss()(m(x),target).backward(); o.step()
    for _ in range(4):
        step(model,engine)
    resumed=nn.Linear(2,1)
    resumed.load_state_dict(model.state_dict())
    second=optim.Adam(resumed.parameters())
    second.load_state_dict(engine.state_dict())
    for _ in range(5):
        step(model,engine); step(resumed,second)
    for p,q in zip(model.parameters(),resumed.parameters()):
        np.testing.assert_array_equal(p.numpy(),q.numpy())
