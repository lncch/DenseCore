"""File resume example using a temporary directory."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from densecore import tensor, nn, optim, checkpoints, manual_seed

manual_seed(7)
model=nn.Linear(2,1)
engine=optim.Adam(model.parameters(),lr=.01)
x=tensor([[1,2],[2,1]])
target=tensor([[3],[4]])

def step(model,optimizer):
    optimizer.zero_grad()
    nn.MSELoss()(model(x),target).backward()
    optimizer.step()

for _ in range(5):step(model,engine)
with TemporaryDirectory() as directory:
    path=Path(directory)/"training.safetensors"
    checkpoints.save_checkpoint(path,model,engine)
    restored=nn.Linear(2,1)
    resumed=optim.Adam(restored.parameters())
    checkpoints.load_checkpoint(path,restored,resumed)
    step(model,engine);step(restored,resumed)
    for p,q in zip(model.parameters(),restored.parameters()):
        np.testing.assert_array_equal(p.numpy(),q.numpy())
print("Model + Adam file resume matches uninterrupted update.")
