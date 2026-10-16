# Stage5 public CPU API results

All pre-timing and post-timing numerical gates passed. Three fresh-process repetitions per engine/batch. Apple M4, CPU float32; see PROTOCOL.md and compressed raw artifact.

| Engine | Batch | Infer us | Train us | Import s | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| densecore | 1 | 25.10 | 94.24 | 0.072 | 40.1 |
| torch_eager | 1 | 12.79 | 110.38 | 0.821 | 276.8 |
| torch_optimized_eager | 1 | 7.12 | 98.49 | 0.823 | 207.7 |
| torch_compile_forward | 1 | 34.91 | 176.80 | 0.821 | 381.6 |
| densecore | 32 | 51.26 | 212.05 | 0.073 | 40.3 |
| torch_eager | 32 | 17.74 | 114.52 | 0.822 | 276.9 |
| torch_optimized_eager | 32 | 11.99 | 101.62 | 0.814 | 207.6 |
| torch_compile_forward | 32 | 38.72 | 181.04 | 0.814 | 383.9 |

Demonstrated advantage: about5.2x lower peak process RSS and11x faster imports versus optimized eager control in this scenario. Densecore is a smaller dense CPU subset; Torch has much broader functionality. RSS includes shared/imported code and is not unique per-worker memory, activation-only memory or a host-capacity multiplier.

Hot inference is3.5x/4.3x slower than optimized eager at batches1/32. Batch32 training is2.1x slower. Batch1 training differs by only about4%, which is not a robust broad speed claim with three timing repeats. Losses and parameters were checked, not just shapes.

Compiled-forward control records cold first-call work separately. This does not cover fully compiled optimizer steps, thread/batch tuning or every PyTorch execution/deployment alternative. No energy/monetary cost claim.

Raw archive: stage5-results.json.gz. Decode with Python gzip.open. Benchmark script writes results.json when rerun; that transient uncompressed file is git-ignored.

Fixed-seed repetitions measure timing variation, not accuracy-seed uncertainty. Process CPU times and all block timings remain in raw data; CPU affinity/power state were not pinned.
