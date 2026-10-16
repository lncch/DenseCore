"""Validate Python-managed callback memory benchmark child behavior."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_python_callback_memory.py"


@pytest.mark.parametrize(
    ("case", "expected_gradient"),
    [("plain", 2.0), ("hook", 4.0), ("custom_vjp", 2.0)],
)
def test_child_measures_python_memory_and_checks_gradient(case, expected_gradient):
    output = subprocess.check_output(
        [sys.executable, str(SCRIPT), "--child", case], text=True, timeout=60
    )
    result = json.loads(output)
    assert result["case"] == case
    assert result["shape"] == [32, 128]
    assert result["gradient_sample"] == expected_gradient
    assert result["peak_delta_bytes"] >= 0
    assert result["peak_bytes"] >= result["current_after_bytes"]
    assert result["native_tensor_storage_included"] is False
