"""Guard numerical gating and dependency inventory boundaries in startup reports."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("densecore_startup_benchmark", ROOT / "benchmarks/benchmark_startup.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


@pytest.mark.parametrize("output", [[[float("nan"), 20.]], [[9., float("inf")]], [[9., 21.]], [9., 20.], [[9.]]])
def test_forward_gate_rejects_bad_values_and_shapes(output):
    with pytest.raises(ValueError, match="gate failed"):
        BENCH.validate_forward({"output": output})


def test_forward_gate_accepts_exact_public_linear_result():
    BENCH.validate_forward({"output": [[9., 20.]]})


def distributions(tmp_path, monkeypatch, *, editable=False, missing=False, bad_version=False):
    from importlib.metadata import Distribution
    definitions = {
        "root": ["dep[feature]>=2" if bad_version else "dep[feature]>=1", 'absent; python_version < "1"'],
        "dep": ['leaf; extra == "feature"', "root>=1"],
        "leaf": [],
    }
    found = {}
    (tmp_path / "shared.bin").write_bytes(b"abc")
    for name, requires in definitions.items():
        folder = tmp_path / (name + "-1.0.dist-info")
        folder.mkdir()
        (folder / "METADATA").write_text("Metadata-Version: 2.1\nName: " + name + "\nVersion: 1.0\n" + "".join("Requires-Dist: " + req + "\n" for req in requires))
        (tmp_path / (name + ".bin")).write_bytes(name.encode())
        (folder / "RECORD").write_text(name + ".bin,,\nshared.bin,,\n" + ("missing.bin,,\n" if missing and name == "root" else ""))
        if editable and name == "root":
            (folder / "direct_url.json").write_text('{"dir_info": {"editable": true}}')
        found[name] = Distribution.at(folder)
    monkeypatch.setattr(BENCH.metadata, "distribution", lambda name: found[name])


def test_inventory_follows_extras_markers_cycles_and_deduplicates_files(tmp_path, monkeypatch):
    distributions(tmp_path, monkeypatch)
    result = BENCH.installed_payload(["root"])
    assert set(result["distributions"]) == {"root", "dep", "leaf"}
    assert result["unique_files"] == 4
    assert result["payload_bytes"] == len(b"rootdepleafabc")


@pytest.mark.parametrize("option,message", [("editable", "editable distribution"), ("missing", "missing recorded files"), ("bad_version", "unsatisfied installed requirement")])
def test_inventory_rejects_incomplete_or_incompatible_installations(tmp_path, monkeypatch, option, message):
    distributions(tmp_path, monkeypatch, **{option: True})
    with pytest.raises(ValueError, match=message):
        BENCH.installed_payload(["root"])
