# Native compiled training implementation plan

User mission is design authority. Execute inline; root owns source writes, independent reviewers read only. Preserve existing uncommitted Rust changes and all experimental records. Base308c7bd, feat/native-training-step. Local commits only.

## Design

Add dc.compiled_train_step(model,x,target,optimizer) for compiled built-in dense models and MSE. Cache a separate executor using existing capture/signature guards. Insert detached target at P+1 and shift intermediate references only. Reuse optimizer native state. Reject unsupported models, subclasses, frozen parameters, differentiable inputs/targets, mismatched optimizer order/identity and disabled gradients. Shared model parameters may use one deduplicated slot if native invariants pass. Native explicit gradients leave eager leaf gradients unchanged; detached loss cannot backward. Serialize fused calls through publication; ordinary concurrent model/optimizer mutation remains unsupported as in existing API.

## Units

1. Tests and Python integration: tests/test_native_training.py, python/densecore/native_training.py, __init__.py; preserve native graph_api/backward implementation. Ten-plus-step loss/weights/state parity for SGD, momentum, Adam. Guard mutation, optimizer mismatch, shape failure, state overflow, overlapping graphs, shared/frozen parameters, thread serialization. Tests first, native rebuild, regression suite and read-only lock review.
2. Benchmark: new benchmarks/benchmark_native_training.py, fresh processes and same64→128→32 MSE models, SGD/Adam, batches1/32. Compare eager/compiled/fused/optimizedTorch; validate first10steps before timing, warmup, repeated medians/RSS and setup. Preserve raw results and report. Commit only after correctness and measured improvement.
3. Profile successful fused baseline; choose next hotspot using measured boundaries. Implement separate improvement with regression coverage; repeat same benchmark and keep only measured win. Separate commit, full checks, engineering report, progress ledger.

Validation: cargo fmt/testrelease/clippywarningsdenied, diffcheck, maturindeveloprelease, fullPython suite, installedwheel smoke where source changed. No push or publication.
