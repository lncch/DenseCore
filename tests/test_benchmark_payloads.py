import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'benchmarks'/'benchmark_cpu.py'


def test_child_configurable_features_and_binary_validation(tmp_path):
    payload = tmp_path/'validation.npz'
    command = [sys.executable,str(SCRIPT),'--child','densecore','--batch','2','--validate-only','--features','2','3','1','--validation-output',str(payload)]
    row = json.loads(subprocess.check_output(command,env=os.environ.copy(),text=True,timeout=30))
    assert row['features'] == [2,3,1] and row['validation_payload']['sha256']
    assert 'weights_after_20_steps' not in row and 'initial_output' not in row
    with np.load(payload,allow_pickle=False) as arrays:
        assert arrays['initial_output'].shape == (2,1)
        assert arrays['0.weight'].shape == (3,2)
        assert arrays['2.weight'].shape == (1,3)


def test_payload_equality_checks_shape_and_contents(tmp_path):
    spec = importlib.util.spec_from_file_location('densecore_benchmark_cpu',SCRIPT)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    one,two = tmp_path/'one.npz',tmp_path/'two.npz'
    a = module.validation_record(np.ones((2,1)),{'w':np.ones((2,3))},one)
    b = module.validation_record(np.ones((2,1)),{'w':np.ones((2,3))},two)
    assert module.validation_equal(a,b)
    changed = module.validation_record(np.ones((1,2)),{'w':np.ones((2,3))},tmp_path/'three.npz')
    assert not module.validation_equal(a,changed)


def test_payload_tampering_is_rejected(tmp_path):
    import pytest
    spec = importlib.util.spec_from_file_location('densecore_benchmark_cpu',SCRIPT)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'payload.npz'
    row=module.validation_record(np.ones((1,1)),{'w':np.ones((1,1))},path)
    with path.open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(ValueError,match='checksum'):module.validation_equal(row,row)


def test_payload_path_exact_and_existing_file_preserved(tmp_path):
    import pytest
    spec=importlib.util.spec_from_file_location('densecore_benchmark_cpu',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'no-extension'
    row=module.validation_record(np.ones((1,1)),{},path)
    assert Path(row['validation_payload']['path']) == path.resolve()
    assert path.exists()
    before=path.read_bytes()
    with pytest.raises(FileExistsError): module.validation_record(np.zeros((1,1)),{},path)
    assert path.read_bytes() == before


def test_binary_gate_rejects_matching_infinities(tmp_path):
    spec=importlib.util.spec_from_file_location('densecore_benchmark_cpu',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    left=module.validation_record(np.array([[np.inf]]),{'w':np.ones((1,1))},tmp_path/'left.npz')
    right=module.validation_record(np.array([[np.inf]]),{'w':np.ones((1,1))},tmp_path/'right.npz')
    assert not module.validation_equal(left,right)


def test_legacy_direct_child_json_is_preserved():
    command=[sys.executable,str(SCRIPT),'--child','densecore','--batch','1','--validate-only','--features','2','3','1']
    row=json.loads(subprocess.check_output(command,env=os.environ.copy(),text=True,timeout=30))
    assert len(row['initial_output']) == 1
    assert len(row['weights_after_20_steps']['0.weight']) == 6
    assert len(row['weights_after_20_steps']['2.weight']) == 3
    assert 'validation_payload' not in row


def test_thread_controls_reach_child_and_preserve_numerical_payload(tmp_path):
    spec = importlib.util.spec_from_file_location('densecore_thread_benchmark', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for threads in [1, 2]:
        command = [sys.executable, str(SCRIPT), '--child', 'densecore', '--batch', '2',
                   '--validate-only', '--features', '2', '3', '1', '--threads', str(threads),
                   '--validation-output', str(tmp_path / f'threads-{threads}.npz')]
        row = json.loads(subprocess.check_output(command, text=True, timeout=30))
        assert row['requested_threads'] == threads
        assert set(row['thread_environment'].values()) == {str(threads)}
        assert row['torch_intraop_threads'] is None
        rows.append(row)
    assert module.validation_equal(*rows)
    assert rows[0]['initial_weight_hash'] == rows[1]['initial_weight_hash']
    assert rows[0]['data_hash'] == rows[1]['data_hash']


def test_torch_reports_requested_intraop_and_fixed_interop_threads(tmp_path):
    import pytest
    pytest.importorskip('torch')
    command = [sys.executable, str(SCRIPT), '--child', 'torch_eager', '--batch', '2',
               '--validate-only', '--features', '2', '3', '1', '--threads', '2',
               '--validation-output', str(tmp_path / 'torch.npz')]
    row = json.loads(subprocess.check_output(command, text=True, timeout=30))
    assert row['requested_threads'] == row['torch_intraop_threads'] == 2
    assert row['torch_interop_threads'] == 1


def test_invalid_thread_count_creates_no_artifact(tmp_path):
    output = tmp_path / 'invalid.json'
    result = subprocess.run([sys.executable, str(SCRIPT), '--threads', '0', '--output', str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'threads must be positive' in result.stderr
    assert not output.exists()
