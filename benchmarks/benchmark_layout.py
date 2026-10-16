"""Local layout timings and isolated retained-view peak-RSS measurements.

Run with the installed candidate or PYTHONPATH=python. Output is JSON on stdout;
redirect it outside the repository. Timings include Python API overhead. RSS is
process peak RSS, including imports and construction, not live allocation size.
"""
import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--retained", type=int, default=32)
    parser.add_argument("--memory", choices=["views", "materialized"])
    args = parser.parse_args()
    if min(args.size, args.iterations, args.repeats, args.retained) <= 0:
        parser.error("sizes and repetition counts must be positive")
    import numpy as np
    import densecore as d
    array = np.arange(args.size ** 2, dtype=np.float32).reshape(args.size, args.size)
    x = d.tensor(array)
    if args.memory:
        before = peak_bytes()
        held = [x.T if args.memory == "views" else x.T.contiguous() for _ in range(args.retained)]
        after = peak_bytes()
        assert len(held) == args.retained
        print(json.dumps({"peak_before_bytes": before, "peak_after_bytes": after,
                          "peak_increase_bytes": after - before}))
        return
    operations = {
        "creation": lambda: d.tensor(array),
        "reshape": lambda: x.reshape(args.size // 1, args.size),
        "transpose": lambda: x.T,
        "permute": lambda: x.permute(1, 0),
        "slice": lambda: x[::2, 1::2],
        "cold_transpose_contiguous": lambda: x.T.contiguous(),
    }
    timings = {}
    for name, operation in operations.items():
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                result = operation()
            samples.append((time.perf_counter_ns() - start) / args.iterations)
            del result
        timings[name] = {"median_ns": statistics.median(samples), "samples_ns": samples}
    memory = {}
    for mode in ("views", "materialized"):
        child = subprocess.run([sys.executable, __file__, "--size", str(args.size),
                                "--retained", str(args.retained), "--memory", mode],
                               check=True, text=True, capture_output=True)
        memory[mode] = json.loads(child.stdout)
    print(json.dumps({"platform": platform.platform(), "python": sys.version,
                      "size": args.size, "iterations": args.iterations,
                      "repeats": args.repeats, "retained": args.retained,
                      "timings": timings, "memory": memory}, indent=2))


if __name__ == "__main__":
    main()
