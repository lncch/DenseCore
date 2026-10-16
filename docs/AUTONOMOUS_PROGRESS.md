# Autonomous development progress

## Current milestone

Version0.10 local candidate completed on feat/native-training-step: transactional fused MSE training, strict cached guards and bounded full-workspace reuse. Verified288Python/53Rust tests; eachmacOSARM64/Py3.13, LinuxARM64/Py3.11 andLinuxx86_64/Py3.13 cleanwheel passed187tests. Implementation commitsa94322c/0684351. Report:NATIVE_TRAINING_ENGINEERING_REPORT.md. No merge/push/publication.

## Repository baseline

Branch `feat/graph-execution`; graph benchmark infrastructure was already committed as `29b3140`. Forward workspace commit `14f8245`. Existing untracked benchmark JSON, profile and autograd backup preserved. Mission historical timings are workload-specific, not performance guarantees.

## Compiled backward

`src/graph_backward.rs` separates immutable BackwardPlan from mutable TrainingWorkspace. Compilation computes reverse integer-indexed traversal, requested gradient propagation and output reachability. Workspace retains forward activations, gradient buffers and reusable GEMM scratch. Forward borrows immutable input values; saved TensorHandles preserve version/provenance checks under the engine lock. Wrong-plan workspace, failed forward, invalid seed and stale saved input reject execution. Gradient access is disabled after failure.

This is an internal vector-Jacobian product boundary: `input_grad` explicitly selects differentiated arguments independently of TensorHandle.requires_grad. Nonleaf inputs are differentiation boundaries. Repeated backward resets workspace gradients; it does not accumulate eager `.grad`. Disconnected arguments return None. All saved arguments, including frozen/disconnected operands, receive conservative version checks. No optimizer or eager gradient state is modified.

## Verification

36 release Rust tests passed; 3 manual benchmarks ignored in the ordinary suite. 167 Python tests passed after `maturin develop --release` in repository `.venv` with source PYTHONPATH. Release clippy with warnings denied passed. Independent read-only architecture/code review found no blocking correctness issues and independently ran graph tests.

Covered nonzero branching/shared weights, eager gradient parity, several shapes, changing values, zero dimensions, repeated executions/allocation stability, non-scalar seeds, invalidation, frozen parameter mutation and disconnected NaN branches.

## Measurements, 2026-10-14

Apple M4, Accelerate float32, VECLIB_MAXIMUM_THREADS=1 and OMP_NUM_THREADS=1. Native Rust calls; no Python crossings or optimizer. Backward benchmark uses 200 warmups, five blocks of 2,000 executions per mode, rotating timing order, eager gradient parity before timing. Compilation/workspace setup reported separately. Same-process block medians are preliminary, not independent process confidence intervals.

| Batch / features | Backward µs | Forward + MSE + backward µs | Plan/workspace setup µs |
| --- | ---: | ---: | ---: |
| 1 / 64→128→32 | 9.112 | 10.872 | 32.208 |
| 32 / 64→128→32 | 14.721 | 21.482 | 20.916 |
| 8 / 17→23→7 | 2.401 | 4.504 | 5.583 |

Forward-only graph benchmark: batch1 allocating graph 3.201 µs, compiled 3.262 µs, workspace 1.084 µs; batch32 10.164 / 10.196 / 6.898 µs. One run with 500 warmups and 5,000 iterations; not full training.

Reproduce manual backward measurements with `cargo test --release compiled_backward_benchmark -- --ignored --nocapture --test-threads=1` and thread environment above.

## Limits and next priorities

- Internal Rust API only; single final output; fixed shapes require another plan.
- Dedicated activation/gradient buffers per node; liveness recycling not implemented.
- One temporary borrowed-reference vector and owned final output allocated per forward. No allocation-free claim.
- Workspace gradients are private results, not optimizer inputs yet.
- CPU float32 only. Other architectures, GPUs and compiled Python capture remain roadmap work.
- Public complete-step speed targets remain unverified for this compiled path.
- Next: share existing transactional optimizer update logic with explicit borrowed gradient slices, then integrate complete native training and validate SGD/momentum/Adam parity and failure atomicity.

## Milestone B: complete native training

BackwardPlan.train_step runs forward, loss, backward and optimizer under one engine lock. The optimizer shares its existing prepare-then-commit transaction between stored eager gradients and explicit borrowed compiled gradients. Successful steps replace caller parameter handles only after every replacement is prepared. Failed updates preserve parameter values, versions and optimizer state. Workspace contents are scratch and may change on failure.

SGD, SGD momentum and Adam validated against eager for 12 steps with changed inputs. Added gradient-length preflight, preserved eager gradient storage, later-parameter step-overflow atomicity, same-slot input/weight gradient accumulation, disconnected parameter decay/state skip, and rejection of logical parameter aliases across different input slots. Shared operations must reference one canonical graph input slot. Compiled steps preserve previous eager .grad; clear eager gradients before switching back to eager optimizer use.

Verification: 42 release Rust tests passed, 4 manual benchmarks ignored; 167 Python tests passed after release editable build; release clippy with warnings denied passed. Independent review caught the distinct-slot alias issue, now reproduced and rejected by regression test.

Five independent native Rust test processes, each five rotating timing blocks of 300 complete steps after 100 warmup/parity steps. Initial and final weights checked against native eager. Thread environment as above. Each result is median of process block medians. Raw records: benchmarks/compiled-training-results.json. Earlier pre-alias-check experiment preserved separately. Timing includes parameter replacement, gradient reset and optimizer; Python calls/imports/capture excluded. These timings do not establish public Python performance or a PyTorch speedup.

| Batch / shape | Optimizer | Compiled complete step µs | Native eager complete step µs |
| --- | --- | ---: | ---: |
| 1 / 64/128/32 | sgd | 9.740 | 12.375 |
| 1 / 64/128/32 | adam | 15.060 | 17.296 |
| 32 / 64/128/32 | sgd | 14.165 | 16.385 |
| 32 / 64/128/32 | adam | 19.597 | 22.274 |
| 8 / 17/23/7 | sgd | 3.112 | 5.436 |
| 8 / 17/23/7 | adam | 3.633 | 5.960 |

Native batch32 SGD measured below 20 µs on this workload. Public Python <30 µs target is still unverified. First A/B runs differed with host timing conditions, so compare paired modes within the repeated protocol rather than combining component medians from different runs.

Next: liveness intervals must include backward activation reads. Start with optional planned forward buffers and preserve existing dedicated workspaces as baseline. Count allocated capacity and verify branches, repeated execution and wrong-plan errors before enabling reuse in training.

## Milestone C: liveness and optional buffer pooling

ExecutionWorkspace now stores a private node-to-slot layout and unique plan identity. Inclusive forward lifetimes include every consumer; reuse requires previous lifetime to end strictly before a producer executes. Training pins all activations read by scheduled derivatives, including frozen operands needed by trainable arguments. Logical lengths stay separate from allocation capacities. Dedicated workspaces remain available and default; pooling is optional.

Tests cover eight-node ReLU reuse, output ownership, wrong-plan rejection, oversized storage hosting smaller values, frozen intermediate weights, shared branches, empty dimensions, changing inputs and complete pooled-training optimizer parity. Independent review found no blocking lifetime issues. Verification: 46 Rust release tests and 167 Python tests passed; 5 manual benchmarks ignored in normal suite; release editable build and clippy passed.

Five independent native Rust test processes, each five rotating 2,000-iteration blocks; raw records in benchmarks/planned-memory-results.json. Eight ReLU inference nodes on 32×128: float32 payload capacity 131,072→32,768 bytes, eight→two tensor payload allocations at workspace construction. These counts exclude metadata, borrowed-reference vectors, returned outputs, handles, inputs and allocator overhead; they are not whole-process allocation/RSS measurements. MLP training workspace remains 156,296 bytes in both modes because backward needs its intermediates.

- inference_relu8: dedicated 3.359 µs, pooled 1.430 µs.
- training_mlp: dedicated 11.737 µs, pooled 12.001 µs.

Pooling reduces memory on eligible graphs but adds small measured latency on this training MLP. It remains optional rather than replacing the baseline. Gradient slots are still dedicated; no gradient liveness recycling implemented. Adding derivatives requires extending activation pin rules.

Next design: Python compilation must preserve ordinary Tensor.backward, leaf accumulation, retained graphs, state_dict names, optimizer/load updates and shape specialization. Reuse mutable workspaces only when no live autograd graph still owns their activations. Unsupported Python modules must run eagerly. Keep plan/cache separate from per-live-forward activation ownership.

## Milestone D: public guarded Python compilation

`densecore.compile(model)` returns the same Module; `model.compile()` enables in place and returns None. Exact built-in Linear/ReLU/Sequential capture to native Graph. Child order/replacement, parameter identity/shape and Python method guards preserve fallback semantics. Tensor subclasses and changed Tensor.relu/_wrap methods run eagerly. Current native parameter handles are read each invocation; state_dict names and Parameter objects remain unchanged.

Native executor caches four shape/gradient-mask plans. No-grad inference reuses pooled buffers. Each live gradient forward owns a separate workspace and plan Arc, so overlapping forwards, retained backward and cache eviction cannot overwrite saved activations. Compiled VJP feeds existing eager autograd, including upstream nonleaf inputs and shared argument/parameter aliases. Engine lock precedes executor/workspace locks. This API allocates training workspaces per live forward; complete-step native reuse remains internal.

Independent reviews caught Tensor subclass dispatch bypass and stale immutable-snapshot rejection. Added red→green regressions for subclasses/method replacement, no-grad snapshots and frozen stale-input value reads. Forward matches eager immutable value reads; backward still validates saved versions/provenance before gradient or optimizer mutation. Failed shape/runtime/version errors propagate without retrying eager.

Verification: 46 Rust release tests, 182 Python tests (15 compilation tests), release clippy with warnings denied. Final macOS ARM64/CPython3.13 wheel installed into a clean environment without PyTorch: 121 tests passed, no oracle tests selected. Compiled regression example MSE1.4032024e-10 after100SGDsteps. Version0.7 Linux remains unvalidated; Docker daemon unavailable. Existing version0.5 Linux artifacts preserved.

Final public benchmark passed ten pre-timing cases and every post-timing gate with three fresh processes per engine/batch. Results/report: benchmarks/COMPILED_RESULTS.md, compiled-python-final-results.json.gz. Initial Python compile path is slower than Densecore eager but has architectural value and remains opt-in.

Important local commits: a338376 compiled backward; a8662d4 transactional native complete step; 72b2299 safe activation liveness planning. Public compilation commit follows this entry. No push/release/publication performed.

## Resume priorities

1. Extend internal graph IR with elementwise add/multiply for residual/gating graphs, preserving equal/scalar/row broadcasting semantics and precomputing index maps. Generalize parent enumeration and backward pin rules; compare pooled/dedicated/eager VJP across shapes and shared operands.
2. Reduce Python capture/shape-guard overhead only with mutation-safe guards. Consider returning dead training workspaces to a bounded pool, preserving independent live/retained graphs. Benchmark against current eager and compiled controls.
3. Broaden graph outputs/operators and real architecture workloads before claiming generality. GPU/backends/normalization/attention remain roadmap work.

Working tree contains preserved untracked earlier benchmark/profile/backup files; do not clean them. New raw public JSON files also preserved untracked, with compressed records committed. Repository .venv contains Torch oracle; work/compiled-wheel-clean intentionally lacks Torch. Use release-mode Rust checks and set source PYTHONPATH when running repository Python tests.

## Milestone E, first increment: residual/gating operators

Internal IR and private native graph binding now support Add/Mul with equal, scalar and row broadcasting. Shape inference shares eager validation without allocating maps. Compiled instructions cache validated index maps; common parent enumeration preserves duplicate edges. Backward scatters each contribution into indexed gradient slots and pins opposite Mul operands only for requested gradients. Public module capture remains the built-in dense subset; arbitrary Python function capture is not implemented.

TDD Rust tests cover both operand orders, scalar/row/empty shapes, residual diamonds/shared weights, duplicate edges, frozen intermediate pinning and non-unit seeds. Private binding integration covers direct x*x with seed[2,3], gradient[4,12], and empty row broadcast gradients. Independent review found no blocking issues. Source validation:49 release Rust tests,183 Python tests and release clippy passed before version bump0.8. Version0.8 release wheel validation follows below.

Native gating graph x*row_gate + scalar_bias → ReLU → meanMSE, width128, forward/backward only. Five fresh test processes, each five rotating1,000-call blocks after200warmups. Initial VJP checked against native eager. No Python or optimizer. Raw records:benchmarks/binary-graph-results.json.
- Batch1: compiled 0.794 µs, native eager 5.008 µs; workspace payload 4108 bytes, cached broadcast maps 4096 bytes.
- Batch32: compiled 25.172 µs, native eager 87.192 µs; workspace payload 115212 bytes, cached broadcast maps 131072 bytes.

The batch32 maps consume131,072bytes beyond115,212bytes workspace payload. Next hypothesis: supported broadcasting needs only equal/scalar/row descriptors. Replacing output-sized index maps with compact descriptors may cut plan memory without sacrificing speed. Measure before keeping; avoid per-element division by iterating row/column ranges.

## Milestone E, compact broadcast experiment

Hypothesis: supported equal/scalar/row broadcasts do not need output-sized index maps. CompiledBroadcast stores validated count plus layout. Forward and gradients use contiguous slice/chunk loops; zero-sized output returns before any zero-width chunk/division. Mul activation pins and immutable shape specialization remain unchanged. Scalar gradient reductions sum a contribution before adding existing gradients; floating-point grouping may differ within explicit tolerances.

First compact-index emitter prototype reduced storage but regressed batch32 latency to50.302µs versus25.172µs map baseline (single preliminary run). Rejected that production loop. Contiguous slice/chunk loops retain constant-size descriptors and improve latency. Root cause at compiler instruction level was not independently profiled; no SIMD/assembly claim.

Property validation checks169shape pairs, support/errors/overflow and production forward/add-gradient/mul-gradient against eager maps, with nonuniform seeds and prefilled gradients. Existing residual diamonds, shared parameters, scalar/row/empty cases and stale-version checks remain passing. Independent review found no blockers. Final validation:52Rust release tests,183Python tests, release clippy; six manual benchmarks ignored in ordinary suite and relevant experiments run separately.

Five fresh native test processes using the same gating fixture/protocol as map baseline. Raw:benchmarks/compact-broadcast-results.json; prior map data preserved in binary-graph-results.json. Descriptor sizes include layout enum padding; other plan metadata, workspace objects, inputs and allocator overhead excluded. Total-plan/RSS reductions are not claimed.
- Batch1: compiled 0.298µs, native eager 4.873µs; workspace payload 4108bytes, broadcast maps0bytes, descriptors48bytes.
- Batch32: compiled 7.282µs, native eager 80.682µs; workspace payload 115212bytes, broadcast maps0bytes, descriptors48bytes.

Batch32 compiled path improves25.172→7.282µs versus earlier map implementation; broadcast indexing storage131,072→48bytes. Native eager control is80.682µs in the new run. This graph measures forward+MSE+backward, without Python or optimizer; do not substitute it for the public complete-step MLP benchmark.

Version0.8 macOS ARM64/CPython3.13 wheel built and installed into clean no-Torch environment:122tests passed; compiled regression example passed. All earlier wheels and raw benchmark/profile/backup files preserved. No remote push, publication, paid service or destructive cleanup.

## Context checkpoint / exact resume state

Branch:feat/graph-execution. Completed local commits before this increment:a338376,A; a8662d4,B;72b2299,C;8f2ad07,D;e12df97,E add/mul. Current compact-broadcast commit follows this entry. Root is sole source writer; reviewers read-only. Current source/runtimes52Rust/183Python tests pass; cleanwheel122tests pass. Docker daemon unavailable, so Linux0.8 remains unvalidated; Linux0.5 ARM64 artifact is historical.

Resume with internal matmul and whole-tensor sum/mean, reusing checked GEMM and extending shape validation, parent enumeration, derivative schedules and activation pinning. Require branch/shared/frozen/zero-shape VJP parity, pooled/dedicated comparison and before/after benchmarks. General Python capture, multiple graph outputs, cached dead training-workspace pooling, kernel/thread research and GPU backends remain incomplete. Public compile subset still Linear/ReLU/Sequential with eager fallback.

Do not overwrite benchmark archives or untracked backup/profile files. Source venv:outputs/framework/.venv; clean wheel venv:work/compiled-wheel-clean; task Rust homes:work/rust-cargo andwork/rust-toolchains. Rust checks release mode. Set PYTHONPATH tosourcepython fororacle tests. Detailed public MLP timings are version0.7 report benchmarks/COMPILED_RESULTS.md; internal gating figures are version0.8 and a different workload.

## Version 0.9, real classifier and local release preparation

Completed compiled matmul/sum/mean and native eager/compiled LayerNorm with guarded capture. Float32 tensor storage; shifted two-pass float64 centered statistics and gradients. Ordinary PyTorch parity and large-offset float64-reference checks remain separate with fixed tolerances.

UCI Letter Recognition duplicate-group-safe split: 15999/1979/2022 rows. Validation-selected width128 checkpoint: Densecore heldout95.8457%, PyTorch96.0930%; fixed accuracy/F1 gates passed. Six training arms used identical initial arrays/schedules. Test evaluated once. Review found evaluator cache-binding weakness; fixed with recomputation, cross-manifest checks, started marker and mutation regressions. Original cache verified afterward without repeating test inference.

Five fresh processes per engine/cell benchmark: smallest eager training67.94µs vs optimized PyTorch205.39µs; largest322.14µs vs267.10µs. ProcessRSS roughly42MiB vs284MiB. Compiled execution remains opt-in and slower here. Full protocol/results: benchmarks/LETTER_RESULTS.md. Baseline/final raw archives preserved. Optimization checked on training/validation data only.

Source:255 Python tests;52 Rust release tests with6 manual benches ignored; release clippy passed. Local0.9 wheels: macOSARM64, LinuxARM64, Linuxx86_64 (emulated), CPython3.13. Each clean no-Torch install passed154tests and pretrained T inference. Linux reference backend; no Linux performance claim. CI matrix prepared, not remotely executed. Dependency notices,29 bundled license files, dataset attribution, pretrainedweights, README and local release checklist included. No publication or remote configured.

Remaining broader roadmap: general Python capture, multiple outputs, broader broadcasting/dtypes, optimized Linux kernels and GPU backends. This release supports dense CPU subset only. Preserve old raw benchmarks/profile/backup files. Root sole writer, final read-only review follows; local commit records completion.

## Optimization 5, fused native Python training

Base308c7bd. Continued existing graph_api.rs/graph_backward.rs changes, without resetting or reimplementing native transaction. Added native publication under held GIL with slot descriptor preflight; exclusive fused wrapper lock. Cached strict runner shifts only intermediate references for added targetslot, reuses optimizer state and existing signatureguards. Exact registered trainable Parameters, detached x/target, builtinLinear/ReLU/Sequential, meanMSE, orderedSGD/Adam. SharedParameterdedup accepted; separate-nativealias objects rejected. Existing eager gradients preserved, lossdetached, ordinarytrainingunchanged.

27targettests include12stepSGD/momentum/Adam loss/weights/stateparity, continuation throughordinarytraining, shared/frozen/replacedparameters, overflowatomicfailure, shapes/dispatch/registrymutations, stalegraph, fusedthreads and tracepublication. Source283passed before lasttracecase, target27passed aftertracecase. Rust52release,6manualignored; clippywarningsdenied/fmt/diffchecks pass.

Fiveprocesspercell baseline initially slowerthanordinarycompiled. Profilefound duplicateparameterwalk; combinedstrictregistryguard removes secondwalk. Guardedbatch1 fused8–14% fastercompiled, batch32tie/smallwin/regression; eagerstillfaster. Controlsdrift acrossruns, noabsolutecausalclaim. All24first12stepgates and120timedprocessgates pass; post-run final/hash audits archived. Fullreport benchmarks/NATIVE_TRAINING_RESULTS.md.

Measurednewbottleneck workspacecreation1.625/2.333µs(batch1/32) usingtemporarynative timers, nowremoved. Nextseparatechange: cachefullworkspace onlydetachedfusedpath underexecutorlock. This differsfromdiscardedlayoutmetadata experiment; no liveautogradoutputsharesworkspace. NativeAdamstatecopy sampled aslatercandidate, nooptimizer mathchange yet. Originaluntrackedarchives preserved, no remote/publication.

Publicationrefinement: prepare/commit optimizertransaction split stagesnewparameters/state, preallocatesallPythonreturn/handle/setterobjects beforecommit. DiscardPreparedUpdate regression checks no mutation onabort. GCPauseRAII disable/restore whilelocksheld, requiredbyCPython3.11 synchronousGCallocation;3.13pilot alonemissedversiondifference. Newtests restoreGCmode success/failure. Staged benchmark120freshprocesses/24pre-gates pass; useroriginalarchives untouched.

Optimization5committed a94322c. Optimization6fullworkspacecache tested287Python/53Rust, reviewno blockers. Same-build5engine benchmark150freshprocesses/30pre-gates/final-loss gatespass. Cachedfused~5–15%fasteruncached,~10–25%fasterordinarycompiled, eagerstillfaster. PeakRSScomparable38–39MiB, Adamprocessvariance disclosed. NewGCcallbacktimeouttest added; finalcrossplatformverification/versioncandidate follows.

Finallocal0.10candidate: Optimization6committed0684351. Source288Pythonpassed;Rust53passed/6manualignored;clippy/fmt/diffchecks pass. macOSARM64/Py3.13, LinuxARM64/Py3.11.17, Linuxx86_64emulated/Py3.13.16 eachcleanwheel187tests pass;LinuxRust53/clippy/examplespass. GCcallbackregression verifiedon3.11. Historical0.9artifacts preserved. Detailedengineeringreport andwheelSHA manifests saved. MITretained;no merge/push/publication. NextprofilecachedfusedAdamstatecopy; no optimizer-mathchange thismission.

Release-readiness audit after7513bce: source290Python/53Rust checks pass; correctedcleanwheel andextractedsourceinstall189tests each. Addedmatching-optimizerfrozenSGD/Adam regressions, publicdocstringcorrections, noticespayload, sourceartifactexclusions, explicitpromotionignorepolicy, archivechecks/fulloracleCI. Historicaldistwheels preserved; currentcandidates outsideGit. RemoteCI/currentLinuxpackaging remainpublicreleasegates. Report:RELEASE_READINESS.md. No performance/runtime refactor, push or publication.
