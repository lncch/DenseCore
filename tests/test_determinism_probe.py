"""Validate the determinism probe child: fresh processes must report equal weight hashes."""
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/determinism_probe.py"


def _child(threads):
    import os
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    output = subprocess.check_output(
        [sys.executable, str(SCRIPT), "--child", "small", "--steps", "5"], text=True, timeout=60, env=env
    )
    return json.loads(output)


def test_small_workload_weights_repeat_bitwise_across_processes_and_threads():
    first, second, threaded = _child(1), _child(1), _child(4)
    assert len(first["weights_sha256"]) == 64
    assert first["weights_sha256"] == second["weights_sha256"]
    assert first["weights_sha256"] == threaded["weights_sha256"]
