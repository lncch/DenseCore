"""Fresh-process benchmark of the public densecore API and declared controls."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import tempfile

THREAD_ENV={"VECLIB_MAXIMUM_THREADS":"1","OMP_NUM_THREADS":"1",
            "OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}

def thread_environment(threads):
    if threads <= 0:
        raise ValueError("threads must be positive")
    return {name: str(threads) for name in THREAD_ENV}


def child(engine,batch,validation_only=False,features=(64,128,32),validation_output=None,threads=1):
    controls = thread_environment(threads)
    os.environ.update(controls)
    input_features,hidden_features,output_features=features
    started=time.perf_counter()
    import numpy as np
    if engine in {"densecore","densecore_compiled"}:
        import densecore as d
    else:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    import_seconds=time.perf_counter()-started
    thread_metadata = dict(requested_threads=threads, thread_environment=controls,
                           torch_intraop_threads=None if engine.startswith("densecore") else torch.get_num_threads(),
                           torch_interop_threads=None if engine.startswith("densecore") else torch.get_num_interop_threads())
    setup=time.perf_counter()
    rng=np.random.default_rng(20261014)
    names=["0.weight","0.bias","2.weight","2.bias"]
    arrays=[(rng.normal(size=s)*.05).astype(np.float32) for s in [(hidden_features,input_features),(hidden_features,),(output_features,hidden_features),(output_features,)]]
    inputs=rng.normal(size=(batch,input_features)).astype(np.float32)
    targets=rng.normal(size=(batch,output_features)).astype(np.float32)
    weight_hash=hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
    data_hash=hashlib.sha256(inputs.tobytes()+targets.tobytes()).hexdigest()
    compile_setup=0.0
    if engine in {"densecore","densecore_compiled"}:
        model=d.nn.Sequential(d.nn.Linear(input_features,hidden_features),d.nn.ReLU(),d.nn.Linear(hidden_features,output_features))
        model.load_state_dict({k:d.tensor(v) for k,v in zip(names,arrays)})
        if engine=="densecore_compiled":
            before=time.perf_counter();model=d.compile(model);compile_setup=time.perf_counter()-before
        x,y=d.tensor(inputs),d.tensor(targets)
        optimizer=d.optim.SGD(model.parameters(),lr=.001)
        loss_fn=d.nn.MSELoss()
        mode=d.no_grad
        infer=lambda:model(x)
        def train():
            optimizer.zero_grad()
            loss=loss_fn(model(x),y)
            loss.backward()
            optimizer.step()
            return loss
        def state():return {k:v.numpy().copy() for k,v in model.state_dict().items()}
        def values(v):return v.numpy()
        backend=d.cpu_backend()
        version=__import__("importlib.metadata",fromlist=["version"]).version("densecore")
    else:
        model=torch.nn.Sequential(torch.nn.Linear(input_features,hidden_features),torch.nn.ReLU(),torch.nn.Linear(hidden_features,output_features))
        model.load_state_dict({k:torch.from_numpy(v) for k,v in zip(names,arrays)})
        x,y=torch.from_numpy(inputs),torch.from_numpy(targets)
        loss_fn=torch.nn.MSELoss()
        mode=torch.no_grad
        if engine=="torch_optimized_eager":
            def update(p):
                with torch.no_grad():p.add_(p.grad,alpha=-.001)
                p.grad=None
            hooks=[p.register_post_accumulate_grad_hook(update) for p in model.parameters()]
            hidden=torch.empty((batch,hidden_features));output=torch.empty((batch,output_features))
            w1,b1,w2,b2=list(model.parameters())
            wt1,wt2=w1.detach().t(),w2.detach().t()
            def infer():
                torch.addmm(b1.detach(),x,wt1,out=hidden)
                hidden.relu_()
                torch.addmm(b2.detach(),hidden,wt2,out=output)
                return output.clone()
            def train():
                loss=loss_fn(model(x),y)
                loss.backward()
                return loss
        else:
            optimizer=torch.optim.SGD(model.parameters(),lr=.001,foreach=False)
            if engine=="torch_compile_forward":
                before=time.perf_counter();forward=torch.compile(model);compile_setup=time.perf_counter()-before
            else:forward=model
            infer=lambda:forward(x)
            def train():
                optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(forward(x),y)
                loss.backward()
                optimizer.step()
                return loss
        def state():return {k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
        def values(v):return v.detach().numpy()
        backend="torch CPU"
        version=torch.__version__
    setup_seconds=time.perf_counter()-setup
    with mode():
        before=time.perf_counter();initial=values(infer()).copy();first_infer=time.perf_counter()-before
    before=time.perf_counter();first_loss=values(train());first_train=time.perf_counter()-before
    for _ in range(19):train()
    checked_state=state()
    validation=validation_record(initial,checked_state,validation_output)
    if validation_only:
        return dict(engine=engine,batch=batch,initial_weight_hash=weight_hash,data_hash=data_hash,
                    features=list(features),first_loss=float(first_loss),**thread_metadata,**validation)
    with mode():
        for _ in range(100):infer()
    inference=[]
    for _ in range(5):
        with mode():
            was_enabled=gc.isenabled();gc.disable()
            try:
                before=time.perf_counter()
                for _ in range(500):infer()
                inference.append((time.perf_counter()-before)/500)
            finally:
                if was_enabled:gc.enable()
    for _ in range(20):train()
    training=[]
    for _ in range(3):
        was_enabled=gc.isenabled();gc.disable()
        try:
            before=time.perf_counter()
            for _ in range(200):train()
            training.append((time.perf_counter()-before)/200)
        finally:
            if was_enabled:gc.enable()
    with mode():final_loss=float(values(loss_fn(infer(),y)))
    usage=resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes=usage.ru_maxrss if sys.platform=="darwin" else usage.ru_maxrss*1024
    return dict(engine=engine,batch=batch,version=version,backend=backend,
                import_seconds=import_seconds,setup_seconds=setup_seconds,compile_setup_seconds=compile_setup,
                first_inference_seconds=first_infer,first_training_seconds=first_train,
                inference_seconds=inference,training_seconds=training,
                peak_rss_bytes=rss_bytes,cpu_seconds=usage.ru_utime+usage.ru_stime,
                initial_weight_hash=weight_hash,data_hash=data_hash,
                features=list(features),
                first_loss=float(first_loss),final_loss=final_loss,**thread_metadata,**validation)

def validation_record(initial,weights,output=None):
    if output is None:
        return dict(initial_output=initial.ravel().tolist(),weights_after_20_steps={k:v.ravel().tolist() for k,v in weights.items()})
    import numpy as np
    output=Path(output)
    with output.open("xb") as stream:
        np.savez(stream,initial_output=initial,**weights)
    return dict(validation_payload=dict(path=str(output.resolve()),sha256=hashlib.sha256(output.read_bytes()).hexdigest()))


def validation_equal(a,b):
    if ("validation_payload" in a) != ("validation_payload" in b):
        return False
    if "validation_payload" not in a:
        return (close(a["initial_output"],b["initial_output"]) and a["weights_after_20_steps"].keys()==b["weights_after_20_steps"].keys() and all(close(a["weights_after_20_steps"][name],value) for name,value in b["weights_after_20_steps"].items()))
    import numpy as np
    for row in (a,b):
        payload=row["validation_payload"]
        if hashlib.sha256(Path(payload["path"]).read_bytes()).hexdigest()!=payload["sha256"]:
            raise ValueError("validation payload checksum mismatch")
    with np.load(a["validation_payload"]["path"],allow_pickle=False) as left, np.load(b["validation_payload"]["path"],allow_pickle=False) as right:
        return (set(left.files)==set(right.files) and all(left[k].shape==right[k].shape and np.isfinite(left[k]).all() and np.isfinite(right[k]).all() and np.allclose(left[k],right[k],rtol=1e-4,atol=1e-6,equal_nan=False) for k in left.files))


def close(a,b,rtol=1e-4,atol=1e-6):
    if len(a)!=len(b):return False
    return all(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=atol+rtol*abs(y) for x,y in zip(a,b))

def main(args):
    controls = thread_environment(args.threads)
    if args.child:
        print(json.dumps(child(args.child,args.batch,args.validate_only,args.features,args.validation_output,args.threads),allow_nan=False))
        return
    if not args.batches or any(batch <= 0 for batch in args.batches):
        raise ValueError("benchmark batches must be positive")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite benchmark artifact")
    engines=["densecore","torch_eager","torch_optimized_eager"]
    if args.include_densecore_compile:engines.insert(1,"densecore_compiled")
    if args.include_compile:engines.append("torch_compile_forward")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.cache_root is not None:args.cache_root.mkdir(parents=True,exist_ok=True)
    payload_root=Path(tempfile.mkdtemp(prefix=args.output.stem+"-arrays-",dir=args.output.parent))
    def run_once(engine,batch,validate=False):
        command=[sys.executable,str(Path(__file__).resolve()),"--child",engine,"--batch",str(batch),"--threads",str(args.threads)]
        payload=payload_root/f"{engine}-{batch}-{time.time_ns()}.npz"
        command += ["--features",*[str(v) for v in args.features],"--validation-output",str(payload)]
        if validate:command.append("--validate-only")
        with tempfile.TemporaryDirectory(prefix="densecore-inductor-",dir=args.cache_root) as cache:
            env=dict(os.environ,**controls,TORCHINDUCTOR_CACHE_DIR=cache,
                     TRITON_CACHE_DIR=str(Path(cache)/"triton"),TORCH_EXTENSIONS_DIR=str(Path(cache)/"extensions"))
            run=subprocess.run(command,capture_output=True,text=True,env=env,timeout=180)
        if run.returncode:raise RuntimeError(engine+" failed: "+run.stderr[-2000:])
        row = json.loads(run.stdout)
        if row["requested_threads"] != args.threads or row["thread_environment"] != controls:
            raise RuntimeError("child thread controls do not match request")
        if engine.startswith("torch") and (row["torch_intraop_threads"] != args.threads or row["torch_interop_threads"] != 1):
            raise RuntimeError("Torch reported unexpected thread settings")
        return row
    validation=[]
    for batch in args.batches:
        reference=run_once("torch_eager",batch,True)
        for engine in engines:
            row=reference if engine=="torch_eager" else run_once(engine,batch,True)
            passed=(row["initial_weight_hash"]==reference["initial_weight_hash"] and row["data_hash"]==reference["data_hash"]
                    and validation_equal(row,reference)
                    and close([row["first_loss"]],[reference["first_loss"]]))
            validation.append(dict(engine=engine,batch=batch,passed=passed))
            if not passed:raise SystemExit(f"pre-timing numerical gate failed: {engine} batch{batch}")
            print(f"validated {engine} batch{batch}",flush=True)
    rows=[]
    for batch in args.batches:
        for repeat in range(3):
            order=engines if repeat%2==0 else list(reversed(engines))
            for engine in order:
                row=run_once(engine,batch);row["repeat"]=repeat;rows.append(row)
            print(f"timed batch{batch} repetition{repeat+1}",flush=True)
    checks=[]
    for batch in args.batches:
        reference=next(r for r in rows if r["batch"]==batch and r["engine"]=="torch_eager")
        for row in [r for r in rows if r["batch"]==batch]:
            passed=(row["initial_weight_hash"]==reference["initial_weight_hash"] and row["data_hash"]==reference["data_hash"]
                    and validation_equal(row,reference)
                    and close([row["final_loss"]],[reference["final_loss"]],rtol=1e-3,atol=1e-5))
            checks.append(dict(engine=row["engine"],batch=batch,repeat=row["repeat"],passed=passed))
    summary=[]
    for batch in args.batches:
        for engine in engines:
            selected=[r for r in rows if r["batch"]==batch and r["engine"]==engine]
            summary.append(dict(engine=engine,batch=batch,
                inference_us=statistics.median(statistics.median(r["inference_seconds"]) for r in selected)*1e6,
                training_us=statistics.median(statistics.median(r["training_seconds"]) for r in selected)*1e6,
                import_seconds=statistics.median(r["import_seconds"] for r in selected),
                peak_rss_mib=statistics.median(r["peak_rss_bytes"] for r in selected)/(1024**2)))
    result=dict(platform=platform.platform(),python=sys.version,requested_threads=args.threads,thread_environment=controls,
                features=list(args.features),validation_payload_directory=str(payload_root),
                accepted=all(c["passed"] for c in checks),pre_timing_validation=validation,checks=checks,summary=summary,runs=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(accepted=result["accepted"],summary=summary),indent=2))
    if not result["accepted"]:raise SystemExit("benchmark correctness gate failed")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--child",choices=["densecore","densecore_compiled","torch_eager","torch_optimized_eager","torch_compile_forward"])
    parser.add_argument("--features",type=int,nargs=3,default=[64,128,32],metavar=("INPUT","HIDDEN","OUTPUT"))
    parser.add_argument("--validation-output",type=Path,help="child binary validation snapshot")
    parser.add_argument("--threads",type=int,default=1,help="requested BLAS/OpenMP and Torch intra-op threads; Torch inter-op stays one")
    parser.add_argument("--batch",type=int,default=1)
    parser.add_argument("--batches",type=int,nargs="+",default=[1,32],help="parent-run batch sizes; child uses --batch")
    parser.add_argument("--include-compile",action="store_true")
    parser.add_argument("--include-densecore-compile",action="store_true")
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--cache-root",type=Path)
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("results.json"))
    args=parser.parse_args()
    if args.threads <= 0:parser.error("threads must be positive")
    if any(v<=0 for v in args.features):parser.error("features must be positive")
    main(args)
