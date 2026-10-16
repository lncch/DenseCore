"""Fresh-process startup and declared-dependency installed payload measurements.

Use installed wheels. Requires packaging in the selected interpreters for inventory.
No cache eviction: fresh processes still benefit from operating-system file caches.
"""
import argparse
import csv
import io
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

THREAD_ENV = {name: "1" for name in (
    "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"
)}

# This code runs under -I, so neither repository Python sources nor PYTHONPATH
# can substitute for the installed package being measured.
CHILD = r'''
import json, sys, time
engine, mode = sys.argv[1:]
started = time.perf_counter()
import numpy as np
if mode == "numpy":
    print(json.dumps({"numpy": np.__version__}))
    raise SystemExit
if engine == "densecore":
    import densecore as framework
else:
    import torch as framework
    framework.set_num_threads(1)
    framework.set_num_interop_threads(1)
import_seconds = time.perf_counter() - started
result = {"import_seconds": import_seconds, "package_path": framework.__file__}
if mode == "forward":
    weight = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    bias = np.array([1., 2.], dtype=np.float32)
    values = np.array([[2., 3.]], dtype=np.float32)
    model = framework.nn.Linear(2, 2)
    if engine == "densecore":
        model.load_state_dict({"weight": framework.tensor(weight), "bias": framework.tensor(bias)})
        x = framework.tensor(values)
    else:
        model.load_state_dict({"weight": framework.from_numpy(weight), "bias": framework.from_numpy(bias)})
        x = framework.from_numpy(values)
    with framework.no_grad():
        output = model(x)
    result["output"] = output.numpy().tolist()
print(json.dumps(result))
'''


def validate_forward(payload):
    expected = [[9., 20.]]
    actual = payload.get("output")
    if not isinstance(actual, list) or len(actual) != 1 or not isinstance(actual[0], list) or len(actual[0]) != 2:
        raise ValueError("first-forward shape gate failed")
    for got, want in zip(actual[0], expected[0]):
        if not isinstance(got, (int, float)) or not math.isfinite(got) or not math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("first-forward numerical gate failed")


def installed_payload(roots):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pending = [(name, set()) for name in roots]
    processed = {}
    distributions = {}
    paths = set()
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        if name in processed and extras <= processed[name]:
            continue
        extras |= processed.get(name, set())
        processed[name] = extras
        dist = metadata.distribution(name)
        direct = dist.read_text("direct_url.json")
        if direct and json.loads(direct).get("dir_info", {}).get("editable"):
            raise ValueError(f"editable distribution cannot provide installed payload: {name}")
        # Distribution.files may silently filter missing files. Read the wheel
        # RECORD directly so an incomplete installation cannot look smaller.
        record = dist.read_text("RECORD")
        if record is None:
            raise ValueError(f"missing installed file inventory: {name}")
        rows = list(csv.reader(io.StringIO(record)))
        if not rows or any(len(row) != 3 or not row[0] for row in rows):
            raise ValueError(f"invalid installed file inventory: {name}")
        dist_paths = {Path(dist.locate_file(row[0])).resolve() for row in rows}
        missing = [str(path) for path in dist_paths if not path.is_file()]
        if missing:
            raise ValueError(f"missing recorded files for {name}: {missing[:3]}")
        paths.update(dist_paths)
        distributions[name] = {"version": dist.version, "files": len(dist_paths),
                               "payload_bytes": sum(path.stat().st_size for path in dist_paths)}
        for declaration in dist.requires or []:
            req = Requirement(declaration)
            if req.marker and not any(req.marker.evaluate({"extra": extra}) for extra in {"", *extras}):
                continue
            dependency = metadata.distribution(req.name)
            if req.specifier and not req.specifier.contains(dependency.version, prereleases=True):
                raise ValueError(f"unsatisfied installed requirement: {declaration}")
            pending.append((req.name, set(req.extras)))
    return {"roots": roots, "distributions": distributions, "unique_files": len(paths),
            "payload_bytes": sum(path.stat().st_size for path in paths)}


def inventory(engine):
    import importlib
    framework = importlib.import_module(engine)
    native = importlib.import_module("densecore._native" if engine == "densecore" else "torch._C")
    result = installed_payload([engine, "numpy"])
    result.update(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  package_path=framework.__file__, native_entry_path=native.__file__,
                  native_entry_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
    return result


def measure(python, engine, mode, env):
    code = "pass" if mode == "empty" else CHILD
    command = [python, "-I", "-c", code]
    if mode != "empty":
        command += [engine, mode]
    start = time.perf_counter()
    output = subprocess.check_output(command, env=env, text=True, timeout=120)
    elapsed = time.perf_counter() - start
    payload = {} if mode == "empty" else json.loads(output)
    if mode == "forward":
        validate_forward(payload)
    return {"engine": engine, "mode": mode, "wall_seconds": elapsed, "child": payload}


def main(args):
    if args.repeats < 2:
        raise ValueError("at least two repetitions required")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite startup results")
    interpreters = {"densecore": os.path.abspath(args.densecore_python),
                    "torch": os.path.abspath(args.torch_python)}
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(THREAD_ENV)
    inventories = {}
    for engine, python in interpreters.items():
        inventories[engine] = json.loads(subprocess.check_output(
            [python, "-I", str(Path(__file__).resolve()), "--inventory", engine],
            env=env, text=True, timeout=120,
        ))
    # Both numerical gates complete before any recorded timing samples.
    for engine, python in interpreters.items():
        measure(python, engine, "forward", env)
    cases = [(engine, mode) for engine in interpreters for mode in ("empty", "numpy", "import", "forward")]
    rows = []
    for repeat in range(args.repeats):
        for engine, mode in cases if repeat % 2 == 0 else reversed(cases):
            row = measure(interpreters[engine], engine, mode, env)
            row["repeat"] = repeat
            rows.append(row)
    summary = []
    for engine, mode in cases:
        values = [r["wall_seconds"] for r in rows if r["engine"] == engine and r["mode"] == mode]
        summary.append({"engine": engine, "mode": mode, "median_seconds": statistics.median(values),
                        "min_seconds": min(values), "max_seconds": max(values)})
    result = {"inventories": inventories, "pre_timing_forward_gates": 2,
              "repeats": args.repeats, "threads": THREAD_ENV, "rows": rows, "summary": summary,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "boundaries": "wall time includes process launch, mode work, JSON output and process exit; no OS cache eviction; import mode includes NumPy and Torch thread setup"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"summary": summary, "payload_bytes": {k: v["payload_bytes"] for k, v in inventories.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", choices=["densecore", "torch"])
    parser.add_argument("--densecore-python")
    parser.add_argument("--torch-python")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(inventory(args.inventory)))
    elif not (args.densecore_python and args.torch_python and args.output):
        parser.error("provide both interpreters and --output")
    else:
        main(args)
