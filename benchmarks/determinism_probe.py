"""Check whether eager training gives bit-identical weights across fresh processes and thread settings.

Measurement only. Each child trains one seeded MLP with Adam and prints a SHA-256 of every
final parameter's float32 bytes. The parent compares hashes across repeats and thread counts.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

THREAD_VARS = ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
WORKLOADS = {"small": ((64, 128, 32), 32), "large": ((256, 1024, 256), 32), "large-b256": ((256, 1024, 256), 256)}


def child(workload, steps):
    from densecore import manual_seed, nn, optim, tensor

    features, batch = WORKLOADS[workload]
    rng = np.random.default_rng(1234)
    x = rng.normal(size=(batch, features[0])).astype(np.float32)
    y = rng.normal(size=(batch, features[-1])).astype(np.float32)
    manual_seed(5)
    model = nn.Sequential(nn.Linear(features[0], features[1]), nn.ReLU(), nn.Linear(features[1], features[2]))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(tensor(x)), tensor(y))
        loss.backward()
        optimizer.step()
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(parameter.numpy(), dtype=np.float32).tobytes())
    return {"workload": workload, "final_loss": float(loss.numpy()), "weights_sha256": digest.hexdigest()}


def run(workload, threads, steps):
    env = dict(os.environ, **{name: str(threads) for name in THREAD_VARS})
    out = subprocess.run([sys.executable, __file__, "--child", workload, "--steps", str(steps)],
                         env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child(args.child, args.steps)))
        return
    report = {}
    for workload in WORKLOADS:
        rows = [dict(run(workload, t, args.steps), threads=t, repeat=r)
                for t in args.threads for r in range(args.repeats)]
        hashes = {row["weights_sha256"] for row in rows}
        by_thread = {t: {row["weights_sha256"] for row in rows if row["threads"] == t} for t in args.threads}
        report[workload] = {
            "runs": len(rows),
            "distinct_hashes": len(hashes),
            "same_thread_setting_repeatable": all(len(h) == 1 for h in by_thread.values()),
            "identical_across_thread_settings": len(hashes) == 1,
            "rows": rows,
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
