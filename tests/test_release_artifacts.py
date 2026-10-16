"""Source archive contracts for reproducible benchmark source and excluded results."""
import importlib.util
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "densecore_release_artifacts", ROOT / "scripts/check_release_artifacts.py"
)
ARTIFACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARTIFACTS)
SOURCE_FILES = [
    "Cargo.toml", "Cargo.lock", "pyproject.toml", "README.md", "LICENSE",
    "THIRD_PARTY_NOTICES.md", "python/densecore/__init__.py", "src/graph_api.rs",
    "tests/test_native_training.py", "examples/predict_letter.py",
    "models/letter128.safetensors", "docs/DATASET_ATTRIBUTION.md",
    "benchmarks/benchmark_native_training.py", "scripts/check_release_artifacts.py",
    "benchmarks/benchmark_gemm_initialization.py", "benchmarks/gemm_initialization_probe.rs",
]


def archive(tmp_path, *, omit=None, extra=None):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as output:
        for name in SOURCE_FILES:
            if name != omit:
                output.add(ROOT / name, arcname="densecore-0.10.0/" + name)
        if extra:
            import io
            member = tarfile.TarInfo("densecore-0.10.0/" + extra)
            member.size = 1
            output.addfile(member, io.BytesIO(b"x"))
    return path


def test_source_archive_preserves_standalone_benchmark_pair(tmp_path):
    result = ARTIFACTS.check_sdist(archive(tmp_path), ROOT)
    assert result["scratch_excluded"]


@pytest.mark.parametrize("omitted", [
    "benchmarks/benchmark_gemm_initialization.py", "benchmarks/gemm_initialization_probe.rs",
])
def test_source_archive_requires_benchmark_companion_files(tmp_path, omitted):
    with pytest.raises(AssertionError, match="missing source payload"):
        ARTIFACTS.check_sdist(archive(tmp_path, omit=omitted), ROOT)


@pytest.mark.parametrize("extra", ["benchmarks/results.npz", "benchmarks/probe", "benchmarks/stray.rs"])
def test_source_archive_still_rejects_generated_or_unlisted_benchmarks(tmp_path, extra):
    with pytest.raises(AssertionError, match="unexpected benchmark result"):
        ARTIFACTS.check_sdist(archive(tmp_path, extra=extra), ROOT)
