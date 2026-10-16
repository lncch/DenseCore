# Public CPU API benchmark protocol

Declared 2026-10-14 before benchmark execution. Test the current release-built Python API, not the earlier fixed research graph.

Apple M4, CPU float32, MLP64->128/ReLU/32, mean MSE, SGD lr0.001. Batch1 and32. Each subprocess creates identical seeded weights and cached inputs/targets (seed20261014). NumPy is present in every engine. Thread environment requests one CPU thread; PyTorch intra/inter-op counts are also set to one.

Compare densecore with standard eager PyTorch and an optimized eager configuration: preallocated addmm/ReLU inference returning a fresh output clone, plus post-accumulation SGD hooks for training. The hook configuration avoids optimizer wrapper overhead while preserving the tested mathematical SGD rule. Optional torch_compile case compiles the forward model, not the complete optimizer step; report compilation first-call cost separately if run.

Before timings, record initial weight/data hashes and initial inference. Run20 matched training updates and export resulting weights. Parent validates outputs/weights at rtol1e-4/atol1e-6. After equal timed update counts, validate finite final losses and cross-engine loss closeness at rtol1e-3/atol1e-5. A failed gate means no accepted comparison.

Per subprocess:100 inference warmups,5 blocks of500 cached-input inference calls;20 training warmups,3 blocks of200 training steps. Inference mode surrounds each block outside timing for both frameworks. Steps include zero_grad where required, forward, loss, backward and update. No per-step NumPy transfer or logging is included. GC is disabled only during each timed block for all engines. Imports/setup, correctness exports and any first compilation are excluded from steady timing and recorded separately.

Run3 fresh subprocess repetitions per engine/batch, alternating engine order across repetitions. Report medians and all raw repetitions. Cold import timing includes NumPy and framework imports, excludes interpreter launch. Peak process RSS includes imports, setup, correctness checks and timed operations. macOS ru_maxrss is bytes; Linux is KiB. It is not activation-only memory or proof of lower optimizer memory.

This compares a supported dense CPU subset with declared PyTorch configurations, not PyTorch's best possible performance or feature coverage. No GPU, deployment-energy or universal speed claim. Include slower cases. Linux performance/runtime remains untested here.
