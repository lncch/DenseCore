# Local 0.9 release candidate

No remote push, GitHub release or PyPI upload occurred. Project source is MIT. Bundled native dependency notices and UCI dataset attribution are included separately.

## Verified platforms

- macOS ARM64, CPython 3.13: release wheel installed in a fresh environment without Torch; 154 tests passed; pretrained classifier predicted T.
- Linux ARM64, CPython 3.13.16, glibc 2.36: isolated Docker build, 52 release Rust tests passed, 6 manual benchmarks ignored; release clippy passed; 154 installed-wheel tests passed; compiled regression and pretrained classifier passed.
- Linux x86-64 under local emulation, CPython 3.13.16, glibc 2.36: same checks passed. This proves correctness/installability in this environment, not native x86 performance.

Linux wheels have manylinux_2_34 tags and use the reference GEMM backend. macOS uses Accelerate. Wheels target CPython 3.13 only; no cross-version ABI claim. CI prepares Python 3.11/3.12/3.13 on Ubuntu/macOS, but remote CI has not run. Other platforms remain unverified.

## Artifacts

Three local wheels live in dist/. Exact sizes and SHA256 hashes: RELEASE_ARTIFACTS.json. Each inspected macOS/ARM64 wheel includes 28 dependency license texts plus the project MIT license (29 files total) with License-File metadata. Model weights live in models/letter128.safetensors; their manifest carries dataset attribution and preprocessing. Dataset itself is excluded from distributions.

## Before public publication

1. Choose and check final repository/package name, owner and metadata.
2. Run prepared CI against the public repository and inspect resulting wheels.
3. Install a produced wheel in a clean supported environment and run the documented example.
4. Review MIT LICENSE, dependency notices and dataset attribution together.
5. Publish only with user authorization. This task prepares local artifacts only.
