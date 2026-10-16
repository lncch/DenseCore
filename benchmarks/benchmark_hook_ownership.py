"""Fresh-process no-hook ownership regression probe, not a PyTorch comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def child(nodes, memory):
    import gc
    import resource
    import time
    import tracemalloc
    import densecore as d
    if memory:
        tracemalloc.start()
    start = time.perf_counter()
    x = d.tensor(1., requires_grad=True)
    y = x
    for _ in range(nodes):
        y = y + 1.
    built = time.perf_counter()
    python_peak = tracemalloc.get_traced_memory()[1] if memory else None
    y.backward()
    finished = time.perf_counter()
    assert x.grad.numpy().item() == 1.
    gc.collect()
    # macOS reports bytes; Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(build_seconds=built-start, backward_seconds=finished-built,
                python_peak_bytes=python_peak,
                rss_peak_bytes=rss if sys.platform == 'darwin' else rss*1024,
                python=sys.version, package_path=d.__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--nodes', type=int, default=3000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline-python')
    parser.add_argument('--candidate-python')
    parser.add_argument('--candidate-pythonpath')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.nodes <= 0 or args.repeats <= 0:
        parser.error('nodes and repeats must be positive')
    if args.child:
        print(json.dumps(child(args.nodes, args.memory)))
        return
    if not all((args.baseline_python, args.candidate_python, args.output)):
        parser.error('baseline-python, candidate-python and output are required')
    script = Path(__file__).resolve()
    result = dict(nodes=args.nodes, repeats=args.repeats,
                  script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                  note='Timing runs disable tracemalloc; allocation runs are separate. Peak RSS includes imports and is not incremental tensor memory.', results={})
    for name, python in [('baseline',args.baseline_python), ('candidate',args.candidate_python)]:
        env = os.environ.copy()
        env.pop('PYTHONPATH',None)
        if name == 'candidate' and args.candidate_pythonpath:
            env['PYTHONPATH'] = str(Path(args.candidate_pythonpath).resolve())
        groups = {}
        for memory in (False,True):
            command = [python,str(script),'--child','--nodes',str(args.nodes)]
            if memory:
                command.append('--memory')
            groups['allocation' if memory else 'timing'] = [json.loads(subprocess.check_output(command,env=env,text=True,timeout=60)) for _ in range(args.repeats)]
        result['results'][name] = groups
        result['results'][name]['medians'] = {key:statistics.median(row[key] for row in groups[group]) for key,group in [('build_seconds','timing'),('backward_seconds','timing'),('python_peak_bytes','allocation'),('rss_peak_bytes','allocation')]}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:data['medians'] for name,data in result['results'].items()},indent=2))


if __name__ == '__main__':
    main()
