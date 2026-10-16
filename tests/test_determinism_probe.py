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


def test_small_workload_weights_repeat_bitwise_across_processes():
    # Same settings, fresh processes: identical weights. Across BLAS thread counts the bits may
    # change, as they do in PyTorch: matmuls go through the same Accelerate calls PyTorch makes,
    # and Accelerate's rounding depends on its thread count on some Macs.
    first, second = _child(1), _child(1)
    assert len(first["weights_sha256"]) == 64
    assert first["weights_sha256"] == second["weights_sha256"]
    threaded, threaded_again = _child(4), _child(4)
    assert threaded["weights_sha256"] == threaded_again["weights_sha256"]
