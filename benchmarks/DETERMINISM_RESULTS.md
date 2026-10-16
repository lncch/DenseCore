# Determinism probe, 2026-10-16

Command: `PYTHONPATH=python python benchmarks/determinism_probe.py` (defaults: 50 Adam steps,
3 fresh processes per thread setting, requested threads 1/2/4 via VECLIB/OMP/OpenBLAS/MKL env vars).
Machine: Apple M4 MacBook, macOS 27, Python 3.13.7, DenseCore 0.10.0 at 0b6b9b8 + this probe.

| Workload | Model | Batch | Runs | Distinct weight hashes |
|---|---|---|---|---|
| small | 64-128-32 MLP | 32 | 9 | 1 |
| large | 256-1024-256 MLP | 32 | 9 | 1 |
| large-b256 | 256-1024-256 MLP | 256 | 9 | 1 |

Every run in a workload produced the same SHA-256 over all final float32 parameters.

What this shows: eager training is bit-repeatable on this machine across fresh processes and
the tested thread requests, for these shapes.

What it does not show:
- Same bits on another machine. GEMM goes through Apple Accelerate `cblas_sgemm`, which makes
  no published cross-chip promise. Linux would use a different BLAS entirely.
- That the thread env vars changed Accelerate's actual worker count. DenseCore does not report it.
- Compiled or fused training paths. Only eager Adam was probed.
- PyTorch's behavior on the same workloads (not measured here).

Raw JSON: kept outside Git at `../overnight-loop/t1-determinism.json`.
