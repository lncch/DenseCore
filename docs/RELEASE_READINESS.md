# DenseCore 0.10 release readiness

Status: validated local release candidate; public release remains gated by remote CI and refreshed cross-platform packaging checks. Reviewed HEAD7513bce plus the scoped corrections in this commit. No push, publication or unrelated performance refactor.

## Fixed

- Source archive accidentally included untracked JSON, profiles and source backup, producing101,573,694bytes. Explicit distribution exclusions now retain code/tests/models/scripts/reports and omit benchmark result data/backups. Reviewed benchmark evidence remains in Git, unchanged. Generated artifacts are ignored by default; promotion requires explicit review. Artifact manifests remain outside source archives to avoid self-referential hashes.
- Wheels lacked readable third-party notices. Current payload:28 dependency license texts, project MIT license and notices,30files. Historical wording corrected; native dependency package/version/license inventory matches Cargo.lock.
- Public compile docstring omitted LayerNorm. Fixed; functional module description now covers its dense operations.
- Frozen test previously failed on replaced-parameter identity before reaching native guard. New SGD/Adam cases freeze before optimizer creation and verify exact rejection plus unchanged model/optimizer state.
- Added full PyTorch-oracle CI job, archive inspection and fresh source-distribution install tests. Existing matrix still verifies installed no-Torch wheels; Rust job checks formatting/release tests/clippy. Workflow YAML parsed; remote execution remains unverified.
- Original0.10 wheels retain their hashes and earlier README metadata. Historical manifest now labels provenance. New candidates and hashes use a separate ignored work directory and manifest; no binaries committed.

## Verified

290 source Python tests,53 release Rust tests,6 manual benchmarks ignored; clippy with warnings denied, rustfmt and diff checks pass. Current macOS ARM64/Python3.13 wheel passes189 installed-package tests without PyTorch. All six standalone Python README examples rerun against the corrected wheel, including conversion in a Torch development environment.

Source installation and all five shell fences have prior successful literal execution with recorded block hashes in README_VERIFICATION.json. Those commands are unchanged. Current source-distribution installation is additionally checked from an extracted archive; benchmark timing is not repeated because execution logic and benchmark script are unchanged. Both retained benchmark archives and gate/summary calculations were audited:30pre-gates/150timedprocesses/150finalchecks pass, all six README rows match raw medians. Scope remains the measured M4 workload; eager remains faster than fused. No Linux/GPU/energy claims added.

MIT text retained. Cargo/Python versions agree at0.10.0; declared dependencies match installation requirements. Current wheel embeds current README metadata and exact license/notices bytes. Source archive keeps the required source/tests/examples/model attribution and excludes generated evidence. Full check details: RELEASE_READINESS_AUDIT.json. Artifact hashes: RELEASE_READINESS_ARTIFACTS.json.

## Remaining public-release gates

1. Run actual GitHub Actions matrix, including the new oracle/archive jobs. No remote run is represented as passing.
2. Rebuild/check current packaging on Linux ARM64/x86-64 and remaining Python combinations. Historical runtime checks passed; they do not prove the updated packaging matrix.
3. Finalize repository/package ownership and repository URLs before publication. Cargo's missing optional repository/homepage metadata remains disclosed; no URL invented.

Local corrections are committed independently. Historical artifacts and untracked experimental files remain preserved. No release/publish action authorized or performed.
