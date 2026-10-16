"""Paired fresh-process compiled-training comparison using matched CPU gates."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


def main(args):
    if args.repeats <= 0 or any(v <= 0 for v in [*args.features,*args.batches]):
        raise ValueError('repeats, features and batches must be positive')
    if len(set(args.batches)) != len(args.batches):
        raise ValueError("comparison batches must be unique")
    if args.output.exists():
        raise FileExistsError('refusing to overwrite comparison artifact')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    payload_root=Path(tempfile.mkdtemp(prefix=args.output.stem+'-arrays-',dir=args.output.parent))
    script=Path(__file__).with_name('benchmark_cpu.py').resolve()
    spec=importlib.util.spec_from_file_location('densecore_cpu_comparison',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    interpreters={'baseline':args.baseline_python,'candidate':args.candidate_python}
    environments={}
    identities={}
    identity_probe='import densecore,densecore._native as n,hashlib,json;from pathlib import Path;print(json.dumps(dict(package=densecore.__file__,extension_sha256=hashlib.sha256(Path(n.__file__).read_bytes()).hexdigest())))'
    for label,python in interpreters.items():
        env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(module.THREAD_ENV)
        if label=='candidate' and args.candidate_pythonpath:
            env['PYTHONPATH']=str(Path(args.candidate_pythonpath).resolve())
        environments[label]=env
        identities[label]=json.loads(subprocess.check_output([python,'-c',identity_probe],env=env,text=True,timeout=30))
    def run(label,batch,suffix,validate=False):
        payload=payload_root/f'{label}-{batch}-{suffix}.npz'
        command=[interpreters[label],str(script),'--child','densecore_compiled','--batch',str(batch),'--features',*[str(v) for v in args.features],'--validation-output',str(payload)]
        if validate:command.append('--validate-only')
        return json.loads(subprocess.check_output(command,env=environments[label],text=True,timeout=180))
    rows=[]
    pre_timing_gates=[]
    references={}
    for batch in args.batches:
        reference=run('baseline',batch,'pre',True)
        candidate=run('candidate',batch,'pre',True)
        if (candidate['initial_weight_hash']!=reference['initial_weight_hash'] or candidate['data_hash']!=reference['data_hash'] or not module.validation_equal(candidate,reference) or not module.close([candidate['first_loss']],[reference['first_loss']])):
            raise ValueError('paired pre-timing numerical gate failed')
        references[batch]=reference
        pre_timing_gates.append(dict(batch=batch,passed=True))
    for batch in args.batches:
        reference=references[batch]
        final_reference=None
        for repeat in range(args.repeats):
            for label in (['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']):
                row=run(label,batch,str(repeat))
                if final_reference is None:final_reference=row
                if (row['initial_weight_hash']!=reference['initial_weight_hash'] or row['data_hash']!=reference['data_hash'] or not module.validation_equal(row,reference) or not module.close([row['final_loss']],[final_reference['final_loss']],rtol=1e-3,atol=1e-5)):
                    raise ValueError('paired numerical gate failed')
                row.update(label=label,repeat=repeat,parity_passed=True);rows.append(row)
            print(f'paired batch{batch} repetition{repeat+1}',flush=True)
    summary=[]
    for batch in args.batches:
        for label in interpreters:
            values=[statistics.median(r['training_seconds'])*1e6 for r in rows if r['batch']==batch and r['label']==label]
            summary.append(dict(batch=batch,label=label,training_us=statistics.median(values),min_us=min(values),max_us=max(values)))
    result=dict(pre_timing_gates=pre_timing_gates,features=args.features,repeats=args.repeats,threads=module.THREAD_ENV,identities=identities,script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,summary=summary)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-python',required=True)
    parser.add_argument('--candidate-python',required=True)
    parser.add_argument('--candidate-pythonpath')
    parser.add_argument('--features',nargs=3,type=int,default=[256,1024,256])
    parser.add_argument('--batches',nargs='+',type=int,default=[1,32])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
