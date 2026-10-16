"""Inspect built release archives without importing DenseCore or third-party code."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


def check_wheel(path, root):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata = next(n for n in names if n.endswith('.dist-info/METADATA'))
        header, body = archive.read(metadata).split(b'\n\n', 1)
        assert body.decode('utf-8').rstrip('\n') == (root / 'README.md').read_text().rstrip('\n'), 'wheel README metadata is stale'
        assert b'License-Expression: MIT' in header, 'missing MIT metadata'
        prefix = metadata.rsplit('/', 1)[0] + '/licenses/'
        expected = {'LICENSE', 'THIRD_PARTY_NOTICES.md'}
        for dependency in json.loads((root / 'docs/NATIVE_DEPENDENCIES.json').read_text()):
            expected.update(dependency['files'])
        actual = {n.removeprefix(prefix) for n in names if n.startswith(prefix) and not n.endswith('/')}
        assert actual == expected, f'license payload mismatch: {actual ^ expected}'
        for name in expected:
            assert archive.read(prefix + name) == (root / name).read_bytes(), f'changed license: {name}'
        assert not any(n.startswith(('benchmarks/', 'data/', 'models/', 'tests/')) for n in names), 'unexpected repository payload in wheel'
    return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'license_files': len(expected), 'readme_matches': True}


def check_sdist(path, root):
    with tarfile.open(path) as archive:
        members = archive.getnames()
        prefix = members[0].split('/', 1)[0] + '/'
        names = {n.removeprefix(prefix) for n in members}
        for name in names:
            assert not name.startswith(('data/', 'dist/', 'target/', '.venv/')), f'unexpected generated directory: {name}'
            assert not name.endswith('.prof') and '.before_' not in name, f'unexpected scratch file: {name}'
            if name.startswith('benchmarks/'):
                assert name.endswith(('.py', '.md')) or name == 'benchmarks/gemm_initialization_probe.rs', f'unexpected benchmark result: {name}'
        required = {'Cargo.toml', 'Cargo.lock', 'pyproject.toml', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
                    'python/densecore/__init__.py', 'src/graph_api.rs', 'tests/test_native_training.py',
                    'examples/predict_letter.py', 'models/letter128.safetensors', 'docs/DATASET_ATTRIBUTION.md',
                    'benchmarks/benchmark_native_training.py', 'scripts/check_release_artifacts.py',
                    'benchmarks/benchmark_gemm_initialization.py', 'benchmarks/gemm_initialization_probe.rs'}
        assert required <= names, f'missing source payload: {required - names}'
        assert archive.extractfile(prefix + 'README.md').read() == (root / 'README.md').read_bytes(), 'sdist README is stale'
    return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'members': len(names), 'scratch_excluded': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', nargs='*', type=Path, default=[])
    parser.add_argument('--sdist', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if not args.wheel and args.sdist is None:
        parser.error('provide --wheel or --sdist')
    results = [check_wheel(p, args.root) for p in args.wheel]
    if args.sdist is not None:
        results.append(check_sdist(args.sdist, args.root))
    print(json.dumps(results, indent=2))
