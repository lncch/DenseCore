# Compiled backward: direct BLAS accumulation, 2026-10-16

Change (commit aa78d6f): compiled `Matmul` and `Linear` VJPs called `gemm_into` into a scratch
buffer, then added scratch into the parent adjoint in a second loop. They now call
`cpu_kernels::gemm_accumulate`, which runs Accelerate `cblas_sgemm` with beta=1 directly on the
adjoint. That removes one write, one read and one add pass per matrix gradient. LayerNorm still
uses scratch.

Correctness: new Rust test `accumulate_adds_into_existing_values_for_all_layouts` (all transpose
layouts, k=0, repeated accumulation, length mismatch). Rust debug/release 105 passed each,
Clippy -D warnings and fmt pass, Python 982 passed. The official benchmark below accepted all
32 numerical checks against PyTorch.

## Timing split (scratch script, one process per row, medians us, 256-1024-256, SGD, 1 thread)

| Batch | Path | Backward before | Backward after |
|---|---|---|---|
| 1 | compiled | 124.9, 124.5 | 29.2, 28.9, 28.8 |
| 32 | compiled | 128.3, 129.6 | 83.2, 80.9, 80.6 |
| 1 | eager (unchanged code) | 62.8, 62.5 | 64.2, 64.5, 62.6 |

## Official harness (accepted=True)

`PYTHONPATH="$PWD/python" .venv/bin/python benchmarks/benchmark_cpu.py --include-densecore-compile --features 256 1024 256 --batches 1 32 --output <new path>`

Training step medians per fresh process (us), three processes each:

| Batch | densecore | densecore_compiled | torch_eager | torch_optimized_eager |
|---|---|---|---|---|
| 1 | 155.6, 156.1, 159.1 | **127.7, 132.0, 132.4** | 163.0, 164.9, 166.1 | 156.2, 156.5, 159.4 |
| 32 | 215.6, 215.6, 215.8 | 232.8, 233.1, 234.0 | 212.6, 213.1, 215.4 | 202.9, 209.3, 210.6 |

Reading:
- Batch 1: compiled training is the fastest engine here, about 16% under optimized Torch, with
  non-overlapping process ranges. This is one model, one machine, one thread, SGD.
- Batch 32: compiled still loses to DenseCore eager (~8%) and optimized Torch (~11%). Retained.
- Absolute numbers are lower than the 2026-10-15 LARGE_MODEL_CHECKPOINT table for every engine,
  including unchanged Torch, so machine state differed. Compare only within one run.

Raw JSON and arrays: `../overnight-loop/t4-official/` (outside Git). Native .so SHA-256 prefix c5464037251b1829.
