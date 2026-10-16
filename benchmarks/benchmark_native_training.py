"""Fresh-process, matched MSE/optimizer benchmark for opt-in fused training."""
import argparse,gc,hashlib,json,os,platform,resource,statistics,subprocess,sys,time
from pathlib import Path
THREAD_ENV={"VECLIB_MAXIMUM_THREADS":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
ENGINES=["densecore","densecore_compiled","densecore_fused","densecore_fused_uncached","torch_optimized_eager"]


def child(engine,batch,kind,validate=False):
    import numpy as np
    start=time.perf_counter()
    dense=engine.startswith('densecore')
    if dense:
        import densecore as dc
        if engine=='densecore_fused_uncached':
            import functools
            import densecore.native_training as training
            training._GraphExecutor=functools.partial(training._GraphExecutor,cache_training_workspace=False)
    else:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
    imported=time.perf_counter()-start
    rng=np.random.default_rng(20261014)
    arrays=[(rng.normal(size=s)*.05).astype(np.float32) for s in [(128,64),(128,),(32,128),(32,)]]
    xx=rng.normal(size=(batch,64)).astype(np.float32);yy=rng.normal(size=(batch,32)).astype(np.float32)
    names=['0.weight','0.bias','2.weight','2.bias'];setup=time.perf_counter()
    momentum=.9 if kind=='momentum' else 0
    if dense:
        m=dc.nn.Sequential(dc.nn.Linear(64,128),dc.nn.ReLU(),dc.nn.Linear(128,32))
        m.load_state_dict({k:dc.tensor(v) for k,v in zip(names,arrays)})
        if engine!='densecore':dc.compile(m)
        x,y=dc.tensor(xx),dc.tensor(yy)
        o=dc.optim.Adam(m.parameters(),lr=.001) if kind=='adam' else dc.optim.SGD(m.parameters(),lr=.001,momentum=momentum)
        def step():
            if engine in ('densecore_fused','densecore_fused_uncached'):return dc.compiled_train_step(m,x,y,o)
            o.zero_grad();loss=dc.functional.mse_loss(m(x),y);loss.backward();o.step();return loss
        def values(t):return t.numpy()
        def weights():return [p.numpy().ravel().tolist() for p in m.parameters()]
        def states():
            result=[]
            for r in o.state_dict()['state']:
                result.append({k:v.numpy().ravel().tolist() if isinstance(v,dc.Tensor) else v for k,v in r.items()})
            return result
    else:
        m=torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.ReLU(),torch.nn.Linear(128,32))
        m.load_state_dict({k:torch.from_numpy(v) for k,v in zip(names,arrays)})
        x,y=torch.from_numpy(xx),torch.from_numpy(yy)
        o=torch.optim.Adam(m.parameters(),lr=.001,foreach=True) if kind=='adam' else torch.optim.SGD(m.parameters(),lr=.001,momentum=momentum,foreach=True)
        def step():
            o.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(m(x),y);loss.backward();o.step();return loss
        def values(t):return t.detach().numpy()
        def weights():return [values(p).ravel().tolist() for p in m.parameters()]
        def states():
            result=[]
            for p in m.parameters():
                r=o.state[p]
                if kind=='adam':result.append({'step':int(r['step'].item()),'first_moment':values(r['exp_avg']).ravel().tolist(),'second_moment':values(r['exp_avg_sq']).ravel().tolist()})
                else:result.append({'step':12,'momentum_buffer':values(r['momentum_buffer']).ravel().tolist() if momentum else None})
            return result
    setup_seconds=time.perf_counter()-setup
    first=time.perf_counter();loss=step();first_seconds=time.perf_counter()-first
    losses=[float(values(loss))]
    for _ in range(11):losses.append(float(values(step())))
    record=dict(engine=engine,batch=batch,optimizer=kind,initial_weight_sha256=hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest(),data_sha256=hashlib.sha256(xx.tobytes()+yy.tobytes()).hexdigest(),losses=losses,weights=weights(),state=states(),import_seconds=imported,setup_seconds=setup_seconds,first_step_seconds=first_seconds)
    if validate:return record
    for _ in range(50):step()
    samples=[]
    enabled=gc.isenabled();gc.disable()
    try:
        for _ in range(5):
            before=time.perf_counter()
            for _ in range(300):loss=step()
            samples.append((time.perf_counter()-before)/300)
    finally:
        if enabled:gc.enable()
    record['training_seconds']=samples
    record['final_loss']=float(values(loss))
    record['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return record


def compare(a,b):
    import numpy as np
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list) and a and isinstance(a[0],(dict,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif a is None:assert b is None
    else:np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-6)


def main(args):
    if args.child:print(json.dumps(child(args.child,args.batch,args.optimizer,args.validate_only),allow_nan=False));return
    if args.output.exists():raise SystemExit('refusing to overwrite benchmark artifact')
    def run(engine,batch,kind,validate=False):
        cmd=[sys.executable,str(Path(__file__).resolve()),'--child',engine,'--batch',str(batch),'--optimizer',kind]
        if validate:cmd+=['--validate-only']
        result=subprocess.run(cmd,env=dict(os.environ,**THREAD_ENV),capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError(result.stderr[-2000:])
        return json.loads(result.stdout)
    gates=[];references={}
    for kind in ['sgd','momentum','adam']:
        for batch in [1,32]:
            ref=run('torch_optimized_eager',batch,kind,True);references[kind,batch]=ref
            for engine in ENGINES:
                row=ref if engine=='torch_optimized_eager' else run(engine,batch,kind,True)
                for k in ['initial_weight_sha256','data_sha256']:assert row[k]==ref[k]
                for k in ['losses','weights','state']:compare(row[k],ref[k])
                gates.append(dict(engine=engine,batch=batch,optimizer=kind,passed=True))
            print(f'validated {kind} batch{batch}',flush=True)
    rows=[]
    for kind in ['sgd','momentum','adam']:
        for batch in [1,32]:
            for repeat in range(args.repeats):
                for engine in ENGINES if repeat%2==0 else list(reversed(ENGINES)):
                    row=run(engine,batch,kind);ref=references[kind,batch]
                    for k in ['losses','weights','state']:compare(row[k],ref[k])
                    row.pop('weights');row.pop('state');row['repeat']=repeat;row['parity_passed']=True;rows.append(row)
                print(f'timed {kind} batch{batch} repeat{repeat+1}',flush=True)
    final_checks=[]
    for kind in ['sgd','momentum','adam']:
        for batch in [1,32]:
            ref=next(r for r in rows if r['optimizer']==kind and r['batch']==batch and r['engine']=='torch_optimized_eager')
            for row in [r for r in rows if r['optimizer']==kind and r['batch']==batch]:
                assert row['initial_weight_sha256']==ref['initial_weight_sha256'] and row['data_sha256']==ref['data_sha256']
                assert abs(row['final_loss']-ref['final_loss']) <= 1e-5+1e-3*abs(ref['final_loss'])
                final_checks.append(dict(engine=row['engine'],optimizer=kind,batch=batch,repeat=row['repeat'],passed=True))
    summary=[]
    for kind in ['sgd','momentum','adam']:
        for batch in [1,32]:
            for engine in ENGINES:
                selected=[r for r in rows if (r['optimizer'],r['batch'],r['engine'])==(kind,batch,engine)]
                med=[statistics.median(r['training_seconds'])*1e6 for r in selected]
                summary.append(dict(engine=engine,optimizer=kind,batch=batch,training_us=statistics.median(med),min_process_us=min(med),max_process_us=max(med),peak_rss_mib=statistics.median(r['peak_rss_bytes'] for r in selected)/2**20))
    result=dict(accepted=True,platform=platform.platform(),python=sys.version,threads=THREAD_ENV,repeats=args.repeats,pre_timing_validation=gates,final_checks=final_checks,runs=rows,summary=summary)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',choices=ENGINES);p.add_argument('--batch',type=int,default=1);p.add_argument('--optimizer',choices=['sgd','momentum','adam'],default='sgd');p.add_argument('--validate-only',action='store_true');p.add_argument('--repeats',type=int,default=5);p.add_argument('--output',type=Path,default=Path(__file__).with_name('native-training-baseline.json'));main(p.parse_args())
