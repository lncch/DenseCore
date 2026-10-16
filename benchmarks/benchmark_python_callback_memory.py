"""Measure Python-managed memory during plain, hook and custom-VJP backward.

Tensor graphs are built before tracing starts. `tracemalloc` reports Python-managed
bytes only; native TensorData backing Vecs are measured separately by Rust tests.
Each recorded sample runs in a fresh interpreter.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tracemalloc


CASES = ("plain", "hook", "custom_vjp")
THREAD_ENV = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def child(case):
    import densecore as dc

    x = dc.tensor([[1.0] * 128 for _ in range(32)], requires_grad=True)
    if case == "plain":
        loss = (x * x).sum()
        expected = 2.0
    elif case == "hook":
        product = x * x
        product.register_hook(lambda gradient: gradient * 2.0)
        loss = product.sum()
        expected = 4.0
    elif case == "custom_vjp":
        class SquareWithScaledVjp(dc.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value * value

            @staticmethod
            def backward(ctx, gradient):
                return gradient * 2.0

        loss = SquareWithScaledVjp.apply(x).sum()
        expected = 2.0
    else:
        raise ValueError(f"unknown callback memory case: {case}")

    gc.collect()
    tracemalloc.start(10)
    current_before, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    loss.backward()
    current_after, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gradient_sample = float(x.grad.numpy()[0, 0])
    if gradient_sample != expected:
        raise ValueError(f"{case} gradient gate failed: {gradient_sample} != {expected}")
    return {
        "case": case,
        "shape": [32, 128],
        "current_before_bytes": current_before,
        "current_after_bytes": current_after,
        "peak_bytes": peak_bytes,
        "peak_delta_bytes": max(0, peak_bytes - current_before),
        "current_delta_bytes": current_after - current_before,
        "gradient_sample": gradient_sample,
        "native_tensor_storage_included": False,
    }


def main(args):
    if args.repeats < 3:
        raise ValueError("at least three fresh-process repetitions required")
    if args.output is None:
        raise ValueError("parent mode requires --output")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite callback memory results: {args.output}")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(THREAD_ENV)
    rows = []
    for case in CASES:
        for repeat in range(args.repeats):
            output = subprocess.check_output(
                [sys.executable, str(Path(__file__).resolve()), "--child", case],
                env=env,
                text=True,
                timeout=120,
            )
            row = json.loads(output)
            if row["case"] != case or row["native_tensor_storage_included"]:
                raise ValueError(f"invalid child measurement: {row}")
            row["repeat"] = repeat
            rows.append(row)

    summary = []
    for case in CASES:
        samples = [row for row in rows if row["case"] == case]
        values = [row["peak_delta_bytes"] for row in samples]
        summary.append({
            "case": case,
            "median_peak_delta_bytes": statistics.median(values),
            "min_peak_delta_bytes": min(values),
            "max_peak_delta_bytes": max(values),
        })
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "shape": [32, 128],
        "repeats": args.repeats,
        "threads": THREAD_ENV,
        "rows": rows,
        "summary": summary,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "tracemalloc Python-managed high-water during backward only; graphs are built before tracing; native TensorData Vecs, allocator and RSS are excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=CASES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.child:
        print(json.dumps(child(options.child)))
    else:
        main(options)
