"""Validation-selected letter classifier. Test evaluation is a separate step."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from letter_data import prepare
from letter_model import build,epoch_schedule,metrics,weight_hash,initial_weights


def train(engine,width,cache,out,epochs=100):
    manifest=prepare(cache)
    with np.load(Path(cache)/'prepared.npz',allow_pickle=False) as dataset:
        x=dataset['x'];y=dataset['y'];train_ids=dataset['train'];valid=dataset['validation']
    start=time.perf_counter()
    if engine.startswith('densecore'):
        import densecore as d
        model=build(engine,width);optimizer=d.optim.Adam(model.parameters(),lr=.001)
        xv=d.tensor(x[valid])
        def update(ids):
            optimizer.zero_grad();loss=d.functional.cross_entropy(model(d.tensor(x[ids])),y[ids]);loss.backward();optimizer.step();return float(loss.numpy())
        def evaluate():
            with d.no_grad():return model(xv).numpy()
        def state():return {k:v.numpy() for k,v in model.state_dict().items()}
        version=__import__('importlib.metadata',fromlist=['version']).version('densecore')
    else:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        model=build(engine,width);optimizer=torch.optim.Adam(model.parameters(),lr=.001,foreach=False)
        xv=torch.from_numpy(x[valid]);yt=torch.from_numpy(y)
        def update(ids):
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(x[ids])),yt[ids]);loss.backward();optimizer.step();return float(loss.detach())
        def evaluate():
            with torch.no_grad():return model(xv).numpy()
        def state():return {k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
        version=torch.__version__
    history=[];best=None;best_state=None
    for epoch,batches in enumerate(epoch_schedule(train_ids,epochs),1):
        losses=[update(ids) for ids in batches]
        if not np.all(np.isfinite(losses)):raise ValueError('nonfinite training loss')
        validation=metrics(evaluate(),y[valid])
        row=dict(epoch=epoch,training_loss=float(np.mean(losses)),validation=validation);history.append(row)
        if best is None or validation['accuracy']>best['validation']['accuracy']:
            best=row;best_state=state()
        if epoch%20==0:print(f'{engine} width{width} epoch{epoch} val{validation["accuracy"]:.4f}',flush=True)
    passed=best['validation']['accuracy']>=(.8 if width==32 else .9) and history[-1]['training_loss']<=history[0]['training_loss']*.5
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez(out/f'{engine}-{width}-weights.npz',**best_state)
    result=dict(engine=engine,width=width,version=version,architecture=f'16-{width}-LayerNorm-ReLU-26',optimizer=dict(kind='Adam',lr=.001,betas=[.9,.999],eps=1e-8),
                epochs=epochs,batch=128,initial_weight_hash=weight_hash(initial_weights(width)),best_weight_hash=weight_hash(best_state),dataset=manifest,
                validation_gate=passed,best_epoch=best['epoch'],best_validation=best['validation'],training_seconds=time.perf_counter()-start,history=history)
    (out/f'{engine}-{width}-quality.json').write_text(json.dumps(result,indent=2)+'\n')
    if not passed:raise RuntimeError('validation quality gate failed; test remains unopened')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine',choices=['densecore','densecore_compiled','torch_eager'],required=True);p.add_argument('--width',choices=[32,128],type=int,required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--cache',type=Path,default=Path(__file__).parents[1]/'data/letter');p.add_argument('--out',type=Path,default=Path(__file__).parents[1]/'benchmarks/letter-quality');args=p.parse_args()
    result=train(args.engine,args.width,args.cache,args.out,args.epochs);print(json.dumps({k:v for k,v in result.items() if k not in ['history','dataset']},indent=2))
