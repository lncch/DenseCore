"""Fresh-process public API comparisons on real letter-recognition minibatches."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
from letter_model import build,initial_weights,weight_hash

ENGINES=['densecore','densecore_compiled','torch_eager','torch_optimized_eager','torch_compile']
THREADS=dict(VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')


def child(engine,width,batch,cache,validate_only=False):
    import numpy as np
    start=time.perf_counter()
    ours=engine.startswith('densecore')
    if ours:import densecore as d
    else:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import_seconds=time.perf_counter()-start
    with np.load(Path(cache)/'prepared.npz',allow_pickle=False) as data:
        ids=data['train'][:batch];inputs=data['x'][ids];targets=data['y'][ids]
    model=build(engine,width)
    if ours:
        x=d.tensor(inputs);y=targets.copy();optimizer=d.optim.Adam(model.parameters(),lr=.001)
        context=d.no_grad;infer=lambda:model(x)
        def train():
            optimizer.zero_grad();loss=d.functional.cross_entropy(model(x),y);loss.backward();optimizer.step();return loss
        def values(x):return x.numpy()
        def state():return {k:v.numpy() for k,v in model.state_dict().items()}
        def gradients():return {k:p.grad.numpy() for k,p in model.named_parameters()}
        def moments():
            s=optimizer.state_dict()
            return s
        version=__import__('importlib.metadata',fromlist=['version']).version('densecore')
    else:
        x=torch.from_numpy(inputs.copy());y=torch.from_numpy(targets.copy())
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,foreach=engine=='torch_optimized_eager')
        context=torch.no_grad
        forward=torch.compile(model) if engine=='torch_compile' else model
        if engine=='torch_optimized_eager':
            hidden=torch.empty((batch,width));output=torch.empty((batch,26))
            w1,b1=model[0].weight.detach(),model[0].bias.detach()
            w2,b2=model[3].weight.detach(),model[3].bias.detach()
            gamma,beta=model[1].weight.detach(),model[1].bias.detach()
            def infer():
                torch.addmm(b1,x,w1.t(),out=hidden)
                normalized=torch.nn.functional.layer_norm(hidden,[width],gamma,beta).relu_()
                torch.addmm(b2,normalized,w2.t(),out=output)
                return output.clone()
        else:infer=lambda:forward(x)
        def train():
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(forward(x),y);loss.backward();optimizer.step();return loss
        def values(x):return x.detach().numpy()
        def state():return {k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
        def gradients():return {k:p.grad.numpy() for k,p in model.named_parameters()}
        version=torch.__version__
    with context():
        start=time.perf_counter();initial=values(infer()).tolist();first_inference=time.perf_counter()-start
    start=time.perf_counter();first_loss=float(values(train()));first_training=time.perf_counter()-start
    first_gradients={k:v.ravel().tolist() for k,v in gradients().items()}
    for _ in range(19):train()
    checked={k:v.ravel().tolist() for k,v in state().items()}
    if engine=='densecore_compiled' and model._compiled_runner.executor is None:raise RuntimeError('compiler fell back to eager')
    row=dict(engine=engine,width=width,batch=batch,version=version,weight_hash=weight_hash(initial_weights(width)),data_hash=hashlib.sha256(inputs.tobytes()+targets.tobytes()).hexdigest(),
             initial_output=initial,first_loss=first_loss,first_gradients=first_gradients,weights_after_20=checked,first_inference_seconds=first_inference,first_training_seconds=first_training,import_seconds=import_seconds)
    if validate_only:return row
    with context():
        for _ in range(100):infer()
    inference=[];training=[]
    for _ in range(5):
        with context():
            enabled=gc.isenabled();gc.disable()
            try:
                start=time.perf_counter()
                for _ in range(300):infer()
                inference.append((time.perf_counter()-start)/300)
            finally:
                if enabled:gc.enable()
    for _ in range(20):train()
    for _ in range(5):
        enabled=gc.isenabled();gc.disable()
        try:
            start=time.perf_counter()
            for _ in range(100):train()
            training.append((time.perf_counter()-start)/100)
        finally:
            if enabled:gc.enable()
    with context():final=values(infer())
    z=final.astype(np.float64)-final.max(axis=1,keepdims=True)
    row.update(inference_seconds=inference,training_seconds=training,final_loss=float(np.mean(np.log(np.exp(z).sum(1))-z[np.arange(batch),targets])),
               final_weights={k:v.ravel().tolist() for k,v in state().items()},peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024))
    return row


def close(a,b,rtol=1e-4,atol=1e-5):
    import numpy as np
    return np.asarray(a).shape==np.asarray(b).shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and bool(np.allclose(a,b,rtol=rtol,atol=atol))


def validate(row,reference,final=False):
    checks=dict(hashes=row['weight_hash']==reference['weight_hash'] and row['data_hash']==reference['data_hash'],
                initial_output=close(row['initial_output'],reference['initial_output']),initial_loss=close([row['first_loss']],[reference['first_loss']]),
                first_gradients=all(close(row['first_gradients'][k],v) for k,v in reference['first_gradients'].items()),
                weights_after_20=all(close(row['weights_after_20'][k],v) for k,v in reference['weights_after_20'].items()))
    if final:checks['final_loss']=close([row['final_loss']],[reference['final_loss']],rtol=1e-3,atol=1e-5)
    return dict(passed=all(checks.values()),checks=checks)


def main(args):
    if args.child:
        print(json.dumps(child(args.child,args.width,args.batch,args.cache,args.validate_only),allow_nan=False));return
    runs=[];pre=[]
    def run(engine,width,batch,validation=False):
        cmd=[sys.executable,str(Path(__file__).resolve()),'--child',engine,'--width',str(width),'--batch',str(batch),'--cache',str(args.cache)]
        if validation:cmd.append('--validate-only')
        with tempfile.TemporaryDirectory(prefix='letter-compile-') as directory:
            env=dict(os.environ,**THREADS,TORCHINDUCTOR_CACHE_DIR=directory,TRITON_CACHE_DIR=directory+'/triton')
            p=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=240)
        if p.returncode:raise RuntimeError(engine+' failed: '+p.stderr[-1500:])
        return json.loads(p.stdout)
    for width in [32,128]:
        for batch in [32,128]:
            reference=run('torch_eager',width,batch,True)
            for engine in ENGINES:
                row=reference if engine=='torch_eager' else run(engine,width,batch,True)
                check=validate(row,reference);pre.append(dict(engine=engine,width=width,batch=batch,**check))
                if not check['passed']:raise RuntimeError(f'pre-timing gate failed {engine}/{width}/{batch}: {check}')
            print(f'validated width{width} batch{batch}',flush=True)
            for repetition in range(args.repetitions):
                block=[]
                for engine in (ENGINES if repetition%2==0 else list(reversed(ENGINES))):
                    row=run(engine,width,batch);row['repetition']=repetition;block.append(row)
                ref=next(r for r in block if r['engine']=='torch_eager')
                for row in block:row['validation']=validate(row,ref,True)
                runs.extend(block);print(f'timed width{width} batch{batch} repetition{repetition+1}',flush=True)
    summary=[]
    for width in [32,128]:
        for batch in [32,128]:
            for engine in ENGINES:
                rows=[r for r in runs if (r['width'],r['batch'],r['engine'])==(width,batch,engine)]
                summary.append(dict(engine=engine,width=width,batch=batch,inference_us=statistics.median(statistics.median(r['inference_seconds']) for r in rows)*1e6,
                                    training_us=statistics.median(statistics.median(r['training_seconds']) for r in rows)*1e6,peak_rss_mib=statistics.median(r['peak_rss_mib'] for r in rows)))
    result=dict(platform=platform.platform(),threads=THREADS,accepted=all(r['validation']['passed'] for r in runs),pre_timing=pre,summary=summary,runs=runs)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(accepted=result['accepted'],summary=summary),indent=2))
    if not result['accepted']:raise RuntimeError('post-timing correctness gate failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',choices=ENGINES);p.add_argument('--width',type=int,default=128);p.add_argument('--batch',type=int,default=128);p.add_argument('--cache',type=Path,default=Path(__file__).parents[1]/'data/letter');p.add_argument('--validate-only',action='store_true');p.add_argument('--repetitions',type=int,default=5);p.add_argument('--output',type=Path,default=Path(__file__).with_name('letter-results.json'));main(p.parse_args())
