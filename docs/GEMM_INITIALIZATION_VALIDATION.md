# GEMM initialization gate

2026-10-15. Implementation a2fb278, baseline 17e95a5. Local Apple M4,
macOS arm64, Rust 1.99.0, float32 Accelerate, thread environment set to one.
Package metadata stays 0.10.0.

## Change and correctness

For nonempty macOS matrix multiplication without bias, BLAS receives beta=0
and overwrites the output. Skip the redundant explicit zero-fill. Rust Vec
storage remains initialized, including when resize grows the buffer. Bias keeps
initialization and beta=1; zero-inner-dimension products and the accumulating
reference backend keep zero-fill. No optimizer, ownership or BLAS ABI change.

Independent read-only review found no required issue. New regression tests poison
reused buffers with NaNs, shrink/grow them, and cover both transpose flags,
bias/no-bias and zero/nonzero inner dimensions. They passed before the optimization
as correctness baselines and after it. A scratch mutation removing required
reference/empty initialization fails the regression with exit 101. This is a
performance change preserving behavior, not a newly failing correctness bug.

Rust debug/release each pass 93 tests, six manual benchmarks ignored. Formatting
and Clippy all targets with warnings denied pass. Rebuilt release Python extension;
full suite passes 826 tests with two existing NumPy empty-mean reference warnings.
Four kernel tests also pass in a scratch build selecting the reference code branch
on this Mac. That is branch coverage, not validation on Linux hardware/toolchains.
Scratch source/results: work/gemm-reference-branch-51dy9eco relative to workspace
root. No public API or installation requirement changed.

## Isolated kernel evidence

The standalone versioned probe compares the exact Git baseline kernel with the
current source. It checks 64 small numerical cases before timing and validates
each measured case against an independent scalar reference. Timing uses initialized
reused output, 30 warmup calls, 500 calls per repetition and five repetitions,
alternating version order. Both versions compile into one optimized executable.
This removes Python/optimizer overhead but is not an independent-process study.

Versioned confirmation, median microseconds (baseline to candidate):

| M,K,N | Transpose A/B | No bias | Bias control |
| --- | --- | --- | --- |
| 1024,32,256 | yes/no | 32.67 to 21.01 | 47.78 to 47.61 |
| 32,256,1024 | no/yes | 38.68 to 32.78 | 40.58 to 40.89 |
| 256,32,1024 | yes/no | 52.73 to 41.30 | 72.31 to 72.28 |

All tested no-bias kernel ranges are separated in this confirmation. Initial
throwaway probe also has lower no-bias medians, but noisier unchanged bias controls.
Both datasets and ranges remain in the deliberately versioned compact JSON.
These are specific kernel results, not a broad framework advantage.

Executed command (choose a new output directory for another run):

```sh
.venv/bin/python benchmarks/benchmark_gemm_initialization.py --baseline-ref 17e95a5 --output-dir ../../work/gemm-initialization-confirmation-20261015
```

Rust compiler must be on PATH with the configured toolchain. macOS uses Accelerate;
other hosts execute their actual kernel backend and must be reported separately.
Generated source, executable and timing logs stay in the fresh output directory.
Initial probe used standalone rustc's default old edition and failed to compile;
rerun explicitly selected the repository's Rust 2024 edition. No failed run is
counted as validation.

## Full compiled-training checks

Both sweeps use the existing pre-gated comparator, model 256-1024-256, batches
1/32, three fresh processes per version, alternating order. Baseline is the installed
private-gradient-transfer wheel, not the older hook wheel, so the runtime delta is
GEMM initialization. Package/native identities are recorded in each artifact.
Both sweeps pass two pre-timing batch-pair checks and 12 timed-process comparisons,
including full initial/20-step arrays, weight/data hashes and final loss tolerance.

| Sweep/batch | Baseline median us (range) | Candidate median us (range) |
| --- | --- | --- |
| First/1 | 422.35 (420.84-425.21) | 473.91 (405.43-537.73) |
| First/32 | 730.60 (694.69-734.94) | 690.39 (666.13-700.44) |
| Confirmation/1 | 469.12 (454.50-491.32) | 426.12 (410.92-498.73) |
| Confirmation/32 | 693.86 (641.78-710.75) | 611.18 (586.27-635.86) |

Batch 1 changes direction across sweeps. Batch 32 medians are lower in both,
with overlapping ranges in the first sweep. Retain the bounded kernel optimization;
do not claim stable end-to-end benefit across workloads. No Torch competitive
comparison was rerun here. Historical Torch timings describe their recorded builds.

Raw JSON/NPZ and both timing sweeps stay outside Git in
work/gemm-initialization-confirmation-20261015. Initial probe lives in
work/gemm-initialization-n0x90xe_. Selected timing rows, ranges and source/native
identities are intentionally versioned as
benchmarks/gemm-initialization-2026-10-15-summary.json. No binary arrays or release
artifacts are committed. No push, version bump, publication or hardware claim.

Next gate: audit supported-operation shared/retained/disconnected/empty/strided
AD coverage and resolve concrete holes. Separate first-write scheduling/workspace
pooling work needs ownership analysis before implementation; avoid speculative
optimizer refactoring.
