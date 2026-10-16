# DenseCore milestone ledger

Updated 2026-10-15. Working branch: feat/v0.12-storage. Baseline for this ledger:
92b3744. Package metadata remains 0.10.0; roadmap version labels are not completion
claims. Mission: the user's 8206c89c attachment plus the original 3c52362a mission.

## Current continuation checkpoint

2026-10-16 overnight session (Claude), branch overnight/2026-10-16 from 0b6b9b8, not merged:
- Determinism probe: eager Adam weights bit-identical across processes and requested threads
  1/2/4 on this M4 (DETERMINISM_RESULTS.md). Cross-machine untested.
- Device/dtype audit: storage is host `Arc<Vec<T>>`, autograd fixed to f32; storage is the GPU
  blocker (DEVICE_DTYPE_READINESS_AUDIT.md). No storage change made.
- perf aa78d6f: compiled Matmul/Linear VJPs accumulate with BLAS beta=1. Compiled training on
  256-1024-256 batch 1 now 128-132 us vs optimized Torch 156-159 us (accepted official harness);
  batch 32 still loses (COMPILED_BACKWARD_ACCUMULATE.md).
- New ops/modules with gradient and PyTorch parity tests: GELU (erf, tanh), Embedding, rank>2
  Linear/LayerNorm/matmul, Dropout, masked_fill, scaled_dot_product_attention.
- examples/tiny_gpt.py: first 3 Adam steps match PyTorch; trains below unigram baseline.
  DenseCore ~12.3 ms/step vs PyTorch 2.9 ms (TINY_GPT_RESULTS.md). GELU array kernel cut 13.7->12.3.
- Gate at 7748590: Rust debug/release 108 passed each (6 ignored), Clippy/fmt pass, Python 1028
  passed + 1 skip with PYTHONPATH=python (macOS re-hides .venv .pth files; Python 3.13 skips them).
- Next task: native-level profile of tiny GPT backward (7.3 ms of the step).

2026-10-15: CPU thread benchmark 00db6af; runtime remains a2fb278. Full Python
978 passed, two existing NumPy reference warnings. No native runtime change;
preceding native debug/release 93-test, fmt/Clippy and installed-wheel gates remain
the runtime baseline. Thread controls passed independent read-only review.

Six serial configurations complete: small/larger MLPs, batches 1/32, requested
1/2/4 threads, four engines, three fresh-process repetitions. All 48 pre-timing,
144 final and 96 post-sweep cross-setting numerical checks pass. Small-model
training wins persist; larger batch-32 training loses to both Torch controls at
all settings. DenseCore does not expose observed Accelerate worker counts; claims
are limited to requested settings. See benchmarks/THREAD_SCALING_RESULTS.md and
thread-sweep-2026-10-15-summary.json. Raw scripts/logs/JSON/NPZ and source/native
hashes stay outside Git under work/cpu-thread-sweep-lutx1bwa. Session 66940 finished;
no pending benchmark/review processes. Updated archives pass README/license/
scratch checks; unpacked-source benchmark/archive tests: 24 passed, one optional
Torch skip. Exact artifact hashes are in that directory's package-receipt.json.

Workspace and tensor capacity studies complete within bounded scope; see
WORKSPACE_MEMORY_VALIDATION.md. Training study measures workspace capacities,
unique shared TensorData, returned outputs, moved VJP storage and retained leaf
snapshots. Large MLP batch 1 subtotal is 10,546,200 bytes for two live workspaces
plus counted input/parameter, output and moved-gradient payloads. Retaining an old
2,102,272-byte gradient snapshot while publishing the cumulative gradient doubles
those two storages to 4,204,544 bytes. Neither is peak memory or RSS.

Inference lifetime study commit 627285a measures 32x128 float32 schedules. An
8-ReLU chain uses 131,072 B dedicated versus 32,768 B planned workspace capacity
(75% lower); a ReLU diamond plus four-ReLU tail uses 147,456 B versus 49,152 B
(66.7% lower). Outputs match on repeated executions, and the first output remains
valid after workspace reuse. Duplicate-parent merge study commit e3328cf measures
three simultaneous gradient Vec capacities; at 525,568 elements this boundary is
6,306,816 B. It excludes other derivative temporaries and total backward peak. No
runtime API change. Follow-up wide-branch merge study commit 3391493 builds four
`x*x` branches, with seven contributions reaching one leaf. At the largest shape,
the merge boundary holds 8,409,088 B of unique gradient Vec capacities: three
leaf-merge buffers and one shared pending branch seed. Other backward temporaries
remain excluded. Backward TensorData high-water study commit f3ffbd7 tracks all
unique Arc-backed Vec capacities created on the test thread. On the same graph at
525,568 elements, live backing capacity peaks at 29,431,812 B during backward,
from 16,818,180 B after forward. This counts graph values, gradient values and
derivative tensors wrapped as TensorData. It excludes raw Vec scratch, metadata,
allocator behavior, cross-thread allocations and RSS. Follow-up commit cbdb304
extends tracker to matrix multiplication and LayerNorm. At matrix dimensions
`32x128` by `128x64` and LayerNorm shape `32x128`, forward/backward capacities are
57,348/114,692 B and 33,796/67,588 B; gradient checks pass. Commit 6f3f84e adds
broadcast addition and strided-slice backward: peaks are 66,564 B and 81,924 B
for tested shapes, with exact broadcast and slice gradient checks. Commit b639a4e
measures resumable callback session boundaries: hook path peaks at 114,688 B,
custom VJP at 49,152 B. These Rust tests model callback results as TensorData and
exclude Python wrapper heap.

Final code gate: Rust debug/release 104 passed each, six ignored; fmt and Clippy
with warnings denied pass; Python 978 passed with two existing NumPy warnings.
All TensorData high-water rows match across debug/release. Leaf and inference
workspace rows also match across profiles. Logs remain under
work/workspace-payload-final-zfgb1sh2, work/workspace-lifetime-2026-10-15 and
work/workspace-memory-phases-2026-10-15. Fresh wheel/sdist match README, license and
scratch checks; clean Python 3.13.7 source install passes 24 targeted tests with one
optional Torch skip. Receipt: work/backward-live-tensor-2026-10-15/package-receipt.json.
Linux/remote CI and broader CPU foundation gates remain open. No push, tag, version
bump or publication.

Next memory task: measure Python wrapper allocations around hook/custom-VJP
callbacks beside native TensorData peaks; report their scopes separately.

## Milestone 1: Reliable CPU and autograd foundation, incomplete

Implemented and locally validated: immutable typed float32/bool storage, checked
layouts/views and logical materialization, explicit CPU dtype dispatch, supported
binary/unary/reduction/loss/NN kernels, eager first-order AD, saved parameter
versions, retained/shared graphs, atomic backward publication, custom Function,
no_grad, full-Jacobian float32 gradcheck, guarded module compilation, compiled
backward, failure-atomic SGD/Adam and fused training. Boolean AD is rejected.

Initial pre-hook checkpoint: 793 Python tests passed; 87 Rust debug tests passed, 6 existing
manual benchmarks ignored. Rust formatting/Clippy passed. These are local macOS
arm64 results recorded in SINGLETON_DIMENSIONS_VALIDATION.md, not new runs by
this audit. Fresh-package targeted checks passed at that checkpoint. Historical
release-readiness counts refer to older commits and must not replace current
validation evidence. Linux/new-version remote CI remains unverified here.

Acceptance audit (updated after hook implementation):

- Completed local hook gate: late leaf/nonleaf registration, ordered callbacks,
  removal, parameter replacement, GC-visible ownership and failure atomicity.
  Validate compiled outputs and reject unsupported fused updates before mutation.
- Completed independent callback ownership/session review. Two concrete native
  boundary findings were reproduced and fixed; final review found no code blocker.
- Completed inference/checkpoint investigation and documented reasoned deferrals
  in INFERENCE_CHECKPOINT_INVESTIGATION.md; no placeholder API added.
- Audit supported operation coverage across shared/retained/disconnected/empty/
  strided graphs and version mutation. Resolve concrete findings before declaring
  the foundation complete. Higher-order and forward AD stay unsupported.
- Reconcile README, compatibility matrix, public API docs and CI with final scope;
  run full relevant gates and fresh installation checks on the final milestone.

Resolved hook ownership decision: mutable slots propagate across differentiable
wrappers, and native Weak<Node> identities avoid retaining full graphs. Independent
review completed. Python callbacks stay outside Rust Arc graphs and native locks.

Hook gate update: implemented real hooks, independent review and two reproduced
native-boundary fixes. Hook tests, Weak lifetime proof and allocation-cost probe
are recorded in HOOK_VALIDATION.md. Current callback objects remain outside native
graphs. Inference/checkpoint investigations are recorded in
INFERENCE_CHECKPOINT_INVESTIGATION.md with reasoned deferrals and prerequisites.
The earlier external-provider blocker is resolved by available read-only review.

Current hook checkpoint: implementation f16fbf7, documentation 221da20. Full Python
819 passed; Rust debug/release 89 passed each, 6 manual benchmarks ignored;
format/Clippy and fresh source/wheel checks passed. README/API compatibility
updated. Local CPU supported-subset hook gate is complete. Whole foundation and
later versions remain open; Linux/new remote CI unverified.

Private compiled-gradient copy gate: implementation 593d40d, comparator da3a221.
Implemented after independent design/code review: move only privately owned adjoints, restore lengths before retained/retry
seeding. Pointer/capacity, identity-plan, empty/frozen, duplicate-parent and parent-
hook retry regressions pass. Full Python 826; Rust debug/release 92 each, six manual
benchmarks ignored; fmt/Clippy and fresh source/wheel checks pass. Three paired
sweeps are variable and include a batch-32 regression. Structural gradient payload
copy reduction is established; reliable latency gain is not. See
COMPILED_GRADIENT_TRANSFER.md for all measurements and artifact provenance.

Post-transfer native sampling completed with workload/sample exit zero; see
COMPILED_BACKWARD_PROFILE.md. Zeroing, scratch accumulation and fresh workspace
construction remain visible. No causal speedup claim or kernel change follows.
GEMM initialization gate completed: implementation a2fb278. NaN reuse and
empty/transposed/bias tests pass; debug/release 93 Rust tests, Python 826,
fmt/Clippy and release extension build pass. Independent review found no required
issue. Standalone no-bias kernels improve on tested shapes; full training has
mixed batch-1 direction and lower batch-32 medians in both sweeps. All numerical
gates pass. See GEMM_INITIALIZATION_VALIDATION.md; no universal speed claim.
Supported-operation lifecycle audit is progressing; see the current continuation
checkpoint and AUTOGRAD_COVERAGE_AUDIT.md. Thread and isolated-memory gates remain open. Startup and declared-dependency
payload measurements are recorded in the current continuation checkpoint.

## Milestone 2: Measurable advantage, incomplete

Existing benchmark_cpu.py and benchmark_native_training.py use subprocesses,
thread controls, shared weight/data hashes and matched numerical validation.
Historical result files exist. They establish only their recorded revisions,
workloads and machines; no current broad advantage follows from their presence.
Current rerun: CURRENT_CPU_CHECKPOINT.md and intentionally selected
current-cpu-2026-10-15-summary.json record all 12 pre-timing and 36 final gates
passing across batches 1/32/1024. Scope is one 64-128-32 MLP on Apple M4, three
fresh processes per configuration, one thread. Small-batch wins and batch-1024
losses are both retained. Complete raw artifacts/provenance stay outside Git at
work/current-cpu-bench-c5y49x relative to workspace root. This is not whole
Milestone 2 completion. Current SGD/momentum/Adam/fused sweep passes all 30 pre-timing and 90 final
gates. Compact process-level timing records are intentionally versioned; full raw
artifacts remain outside Git. Fused wins this small-model sweep against foreach
Torch controls, while the larger-batch CPU losses remain. Control protocols differ
and are labeled explicitly in CURRENT_CPU_CHECKPOINT.md.

Larger-model gate update: LARGE_MODEL_CHECKPOINT.md records 525,568-parameter
256-1024-256 MLP at batches 1/32, eight pre-timing and 24 final checks passing.
Full NPZ arrays are checked for keys, shapes, finite values, allclose and integrity;
raw JSON/arrays/provenance remain outside Git under work/large-model-nwBwA7.
Binary validation changes cold conversion and whole-process RSS, not steady loop
boundaries. Independent review found a matching-infinity gate regression, which
was reproduced and fixed. Final review found no remaining gate blocker.
Compiled training regresses substantially; batch-32 compiled inference improves
its median. Mixed results are retained in intentionally versioned compact data.
Native eager/compiled sampling profiles completed; copy paths are investigation
targets, not proven sole causes. No speculative kernel/optimizer change made.

Acceptance: audit measurement boundaries and numerical gates, rerun on current
code with small/large workloads and multiple repetitions, report latency/
throughput, peak memory, cold import/startup, footprint and compile/steady-state
costs. Preserve losing results and variability. Profile before optimizing. A
useful measured win is required before making a competitive claim; otherwise
record failure and revise direction. Keep generated artifacts outside Git unless
selected deliberately for provenance. No universal PyTorch-beating claim.

Startup/dependency gate update: benchmarks/STARTUP_PAYLOAD_RESULTS.md and
startup-payload-2026-10-15-summary.json record fresh-process controls, matched
first-forward checks and dependency-closure payload. This closes the scoped
startup/recorded-payload investigation, not whole Milestone 2. Full environment
size, isolated tensor RAM and multi-thread evidence are separate measurements.

## Milestone 3: Useful end-to-end models, incomplete

Dense modules, safetensors/checkpoint helpers and existing tabular examples are
foundations. Audit actual supported architectures and examples before broadening
claims. Acceptance: complete reproducible training/inference workflows, explicit
weight-mapping subset with reference parity, data provenance and runnable docs.
Embeddings/attention depend on real operator/backend coverage.

## Milestone 4: Native GPU backends, incomplete

No Metal/CUDA execution validated. Acceptance: real device storage/transfers,
dispatch, synchronization, native kernels and AD/compiler integration, tested on
claimed hardware. CPU-only builds remain supported. Hardware unavailable means
an outstanding gate, never a passing result or disguised CPU fallback.

## Milestone 5: Broader framework, incomplete

Mixed precision, expanded compilation and distributed execution require earlier
foundations and demonstrated workflows. Define supported subsets, failure
semantics and validation before implementation. Avoid speculative infrastructure.

## Continuation contract

Continue through dependency-ready tasks without a user 'continue'. Report completed
milestones or blockers, not each small API. Commit coherent implementation/tests
and documentation separately. No pushes, merges, tags, publication, history
rewrites or user-data deletion. Preserve parallel work. Record precise processes,
files, pending checks and next command if execution limits interrupt progress.

## Prior checkpoint provenance

Private-gradient-transfer: runtime 593d40d, comparator da3a221, evidence a4e3320.
Post-transfer native profile: work/compiled-post-transfer-profile-jcrbgu20.
GEMM runtime a2fb278; raw probe/training: work/gemm-initialization-confirmation-20261015;
reference branch/mutation: work/gemm-reference-branch-51dy9eco.
Unary lifecycle tests 3fc37db passed the earlier 844-test Python checkpoint.
Current results and next task are recorded at the top of this ledger.
