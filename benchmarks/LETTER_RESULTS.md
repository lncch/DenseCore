# Real classifier results

CPU float32 MLP: 16 → width → LayerNorm → ReLU → 26. Apple M4/macOS, CPython 3.13, PyTorch 2.14.1, development build preceding the 0.9 version bump. macOS uses Accelerate GEMM. Linux correctness was checked separately; no Linux speed claim.

## Accuracy

UCI Letter Recognition, 20,000 rows, 16 features, 26 classes. There are 18,668 distinct feature groups. Identical feature tuples stay in one partition: 15,999 training, 1,979 validation, 2,022 heldout rows. Seed 20261014; standardization uses training rows only. All engines use identical initial weights, architecture, minibatch order and Adam settings: learning rate .001, batch 128, 100 epochs. Select checkpoint by validation accuracy only.

Width 128: Densecore eager and compiled heldout accuracy 95.8457%, macro-F1 .958278; PyTorch 96.0930%, macro-F1 .960804. Accuracy difference .2473 percentage points, below the predeclared .5-point limit. Width 32: Densecore 90.2572%, PyTorch 90.3066%. Heldout was evaluated once. Cache binding was strengthened afterward and the original cache verified without repeating test inference. See letter-quality/heldout-test.json and dataset-verification.json.

## Steady-state performance

Each cell uses five fresh processes per engine. Five blocks per process: 300 inference calls or 100 complete zero_grad/forward/cross-entropy/backward/Adam steps. Warmups: 100 inference calls and 20 training steps. Garbage collection disabled only during timing. Requested single-thread environment; PyTorch intra/inter-op threads set to one. Setup, compilation and validation are outside steady-state timing. Peak RSS includes imports, setup, full prepared data, validation and timing, so it is process memory rather than tensor allocation alone.

Values below are median microseconds per inference / training step. RSS is MiB.

| Width / batch | Densecore eager | Densecore compiled | PyTorch eager | PyTorch optimized eager | PyTorch compile |
|---|---:|---:|---:|---:|---:|
| 32 / 32 | 15.47 / 67.94 | 26.07 / 79.17 | 19.45 / 209.77 | 13.44 / 205.39 | 36.14 / 274.77 |
| 32 / 128 | 32.97 / 156.93 | 44.27 / 178.83 | 26.76 / 237.75 | 20.68 / 228.82 | 37.46 / 287.40 |
| 128 / 32 | 30.36 / 121.36 | 38.69 / 135.13 | 24.42 / 226.04 | 17.55 / 221.09 | 36.55 / 279.76 |
| 128 / 128 | 85.23 / 322.14 | 95.75 / 344.28 | 43.10 / 269.46 | 35.09 / 267.10 | 58.93 / 336.56 |

Densecore eager peak RSS 41.83–42.78 MiB; compiled 43.14–43.95; PyTorch eager/optimized 283.14–283.86; PyTorch compile 393.83–394.92. Smallest Densecore training case is 3.02× faster than optimized PyTorch with 6.78× lower peak RSS. Largest Densecore case is 1.21× slower in training and 2.43× slower in inference. Compiled Densecore is slower than eager across these cells.

Optimized PyTorch uses preallocated addmm plus functional LayerNorm/ReLU for inference and foreach Adam for training. PyTorch compile compiles the model, with optimizer outside. Correctness gates compare initial outputs, first loss/gradients and 20-step parameters at rtol 1e-4 / atol 1e-5 before timing; final finite losses use rtol 1e-3 / atol 1e-5. Every gate passed. This cached-batch benchmark does not measure convergence time or full data-pipeline throughput. Lower RSS is not evidence of lower energy use.

## Numerical and optimization evidence

LayerNorm uses float32 storage with shifted two-pass float64 centered arithmetic. Ordinary inputs compare to PyTorch float32; large offsets compare against PyTorch float64 on the exact quantized float32 inputs and rounded epsilon. No tolerances were widened to accommodate cancellation. At offset 1e6, maximum Densecore input-gradient error was approximately 1e-7 versus the float64 reference, while PyTorch float32 error was 1.93 on this diagnostic. See layernorm-accuracy.json. This is a targeted stress case, not a general accuracy superiority claim.

Profiling isolated LayerNorm as the main cost. Replacing Welford statistics with shifted two-pass statistics reduced largest-case eager training from 569 to 322 microseconds, inference from 167 to 85. Existing parity gates remained fixed. New/old outputs matched exactly on the recorded 256 training/validation samples per width; heldout was not reopened. See layernorm-optimization-equivalence.json.

## Reproduce

Use examples/letter_data.py to download and verify the pinned official dataset, examples/train_letter.py for six quality runs, and benchmarks/benchmark_letter.py for performance. Each script has CLI help. examples/evaluate_letter.py refuses a second evaluation once its started marker exists; do not remove that marker to tune on test results.

Raw final records: letter-optimized-results.json.gz. Baseline: letter-results.json.gz. The uncompressed local originals are preserved. Raw records include process-level samples, setup timings and validation details. Source checksums and wheel validation are recorded in docs/LOCAL_RELEASE.md and docs/RELEASE_ARTIFACTS.json. CPU-specific findings do not establish GPU performance or broad PyTorch compatibility.
