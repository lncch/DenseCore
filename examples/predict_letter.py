"""Run the packaged letter-feature classifier without PyTorch or training data."""
import argparse
import json
from pathlib import Path
import numpy as np
import densecore as dc


def predict(features,model_directory):
    model_directory=Path(model_directory)
    metadata=json.loads((model_directory/'letter128.json').read_text())
    features=np.asarray(features,dtype=np.float32)
    if features.shape!=(16,) or not np.all(np.isfinite(features)):raise ValueError('provide exactly16finite numeric features')
    inputs=(features-np.asarray(metadata['mean'],dtype=np.float32))/np.asarray(metadata['scale'],dtype=np.float32)
    model=dc.nn.Sequential(dc.nn.Linear(16,128),dc.nn.LayerNorm(128),dc.nn.ReLU(),dc.nn.Linear(128,26))
    dc.checkpoints.load_weights(model_directory/'letter128.safetensors',model)
    model.eval();dc.compile(model)
    with dc.no_grad():logits=model(dc.tensor(inputs[None,:])).numpy()[0]
    shifted=logits.astype(np.float64)-logits.max();probabilities=np.exp(shifted);probabilities/=probabilities.sum()
    index=int(logits.argmax())
    return dict(letter=chr(ord('A')+index),probability=float(probabilities[index]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('features',nargs=16,type=float);p.add_argument('--model-directory',type=Path,default=Path(__file__).parents[1]/'models');a=p.parse_args();print(json.dumps(predict(a.features,a.model_directory),indent=2))
