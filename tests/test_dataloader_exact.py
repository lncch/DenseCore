"""torch.utils.data on DenseCore gives PyTorch's batches, including the random draws made by
the loader, the samplers and the (emulated) workers, compared against real PyTorch workers."""
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip('torch')

HERE = os.path.dirname(__file__)
CASES = ["plain", "shuffle", "shuffle_gen", "drop_last_none", "workers2", "workers3_init", "persistent",
         "prefetch5", "iter0", "iter_workers", "iter_workers_none", "split", "subset_sampler", "weighted"]


def _run(lib):
    import densecore
    # The subprocess imports the same DenseCore as this process (source tree or installed wheel).
    package_root = os.path.dirname(os.path.dirname(densecore.__file__))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([package_root, HERE]))
    out = subprocess.run([sys.executable, os.path.join(HERE, "_dataloader_cases.py"), lib, *CASES],
                         capture_output=True, text=True, cwd=HERE, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


def test_dataloader_matches_recorded_torch():
    """Against torch 2.14.1's output, recorded with `python _dataloader_cases.py torch`."""
    with open(os.path.join(HERE, "dataloader_torch_2.14.1.json")) as f:
        recorded = json.load(f)
    dense_out = _run("densecore")
    assert [c for c in CASES if recorded[c] != dense_out[c]] == []


@pytest.mark.skipif(not os.environ.get("DENSECORE_LIVE_TORCH"), reason="spawns real torch workers (~100 s)")
def test_dataloader_matches_live_torch():
    torch_out, dense_out = _run("torch"), _run("densecore")
    assert [c for c in CASES if torch_out[c] != dense_out[c]] == []


def test_exports_match_torch():
    import torch
    import densecore as d
    expected = {n for n in torch.utils.data.__all__ if "DataPipe" not in n and n not in {
        "DataChunk", "argument_validation", "functional_datapipe", "guaranteed_datapipes_determinism",
        "non_deterministic", "runtime_validation", "runtime_validation_disabled"}}
    assert expected == set(d.utils.data.__all__)
