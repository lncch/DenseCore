# Training step split vs PyTorch, 2026-10-16

Scope: eager SGD (lr 1e-3), MSE, Linear-ReLU-Linear, one requested thread, Apple M4, PyTorch 2.14.1.
300 steps per fresh process, first 50 dropped, medians in microseconds. One process per engine for
the comparison rows; five processes for the DenseCore before/after rows. Script kept outside Git
(`../overnight-loop`), raw lines in t3-before.txt / t3-after.txt there.

| Model | Batch | Engine | Forward | Backward | Step | Sum of medians |
|---|---|---|---|---|---|---|
| 64-128-32 | 1024 | DenseCore | 77.3 | 64.1 | 4.5 | 145.9 |
| 64-128-32 | 1024 | PyTorch | 66.8 | 75.6 | 11.7 | 154.0 |
| 256-1024-256 | 32 | DenseCore | 67.4 | 73.5 | 72.9 | 213.8 |
| 256-1024-256 | 32 | PyTorch | 72.9 | 84.1 | 60.6 | 217.6 |
| 256-1024-256 | 256 | DenseCore | 283.3 | 325.7 | 71.9 | 681.0 |
| 256-1024-256 | 256 | PyTorch | 257.6 | 326.4 | 59.1 | 643.2 |

Reading: with eager SGD at one thread these are close to parity. DenseCore forward is about 10%
slower at 256-1024-256 batch 256, and its SGD step is about 20% slower on the 525k-parameter model.
Single-process rows; differences under about 5% are not evidence.

Tried and reverted: skipping the output zero-fill before beta=0 Accelerate GEMM. Five-process medians
of the 256-1024-256 batch-256 step sum were 673 us before and 673 us after; backward medians moved
321 -> 312 us while small-model forward moved 75 -> 78 us. No clear gain, so the change was not kept.
Final weights were bit-identical with and without it.

Open leads, not yet measured in isolation: the SGD step on large parameters (immutable storage
likely allocates a new buffer per update) and the forward gap at batch 256.
