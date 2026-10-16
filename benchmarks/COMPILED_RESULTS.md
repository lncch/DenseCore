# Public compiled API benchmark, version 0.7

Apple M4, float32 CPU Accelerate; MLP64→128→ReLU→32, mean MSE, SGD. Same input/target/weight hashes, cached native input tensors and stable owned outputs. Three fresh timed processes per engine/batch, ten pre-timing numerical cases, 20-step weight parity and post-timing final loss parity. Five inference blocks of 500 calls after 100 warmups; three complete-training blocks of 200 steps after 20 warmups. Thread controls request one. First calls and compile setup recorded separately from steady state. Full protocol: PROTOCOL.md and benchmark_cpu.py. Raw archive: compiled-python-final-results.json.gz.

| Batch | Engine | Inference µs | Complete training µs | Peak process RSS MiB |
| --- | --- | ---: | ---: | ---: |
| 1 | densecore | 3.130 | 18.313 | 39.766 |
| 1 | densecore_compiled | 5.703 | 24.534 | 40.781 |
| 1 | torch_eager | 7.122 | 59.761 | 281.453 |
| 1 | torch_optimized_eager | 4.164 | 50.606 | 211.125 |
| 1 | torch_compile_forward | 19.680 | 90.989 | 388.375 |
| 32 | densecore | 5.412 | 20.710 | 39.875 |
| 32 | densecore_compiled | 8.406 | 28.216 | 41.172 |
| 32 | torch_eager | 9.612 | 61.724 | 282.188 |
| 32 | torch_optimized_eager | 6.594 | 55.002 | 211.266 |
| 32 | torch_compile_forward | 22.517 | 91.911 | 390.812 |

All numerical gates passed. Densecore eager is faster than this initial Python compilation path. Capture guards and a separate workspace per live training forward add overhead; compilation remains opt-in. Complete native training has reusable workspaces internally, but Python compilation does not yet expose that complete-step path.

For this declared batch32 workload, public compiled training measured below30 µs and roughly1.95× faster than optimized eager PyTorch; Densecore eager measured roughly2.66× faster. Compiled inference is slower than optimized eager PyTorch. Compiled peak process RSS is about5.13× lower. These results do not establish a general framework speedup or best possible PyTorch performance.

The optimized eager control uses preallocated addmm/ReLU inference and post-accumulate SGD hooks. The torch.compile control compiles forward; optimizer remains outside. Peak RSS includes imports, setup, validation and timing; it is not unique resident memory, activation-only memory, or energy cost. Earlier runs are preserved separately. Absolute timings differ from historical stages with host conditions; comparisons use controls from the same protocol run.
