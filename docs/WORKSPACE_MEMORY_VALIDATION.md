# Training and tensor payload accounting

2026-10-15. Test-only measurement on Apple M4, macOS 27 arm64, Rust 1.99.0.
Runtime source remains a2fb278; package metadata remains 0.10.0. No inference,
runtime API, allocator or kernel changes.

## Counted payload

Workspace totals equal sum of `Vec<f32>::capacity() * 4` across forward
`ExecutionWorkspace` buffers, backward adjoint buffers and scratch. New `cfg(test)`
helpers expose backing `TensorData` Vec capacity and Arc storage identity. Study
deduplicates shared input/parameter TensorData by storage identity, then counts two
returned scalar outputs and VJP gradient TensorData created by moving the extracted
Vecs into `TensorData`.

The tracked subtotal after retry is:

```text
2 live TrainingWorkspace payloads
+ unique borrowed input/parameter TensorData payloads
+ two returned output TensorData payloads
+ transferred VJP gradient TensorData payloads
```

All four terms count float32 Vec capacities; cloned handles sharing one storage
count once. Dedicated validation workspace is dropped before this subtotal.

The workspace subtotal excludes Vec/Arc headers, plan/graph and Python metadata,
allocator rounding, BLAS internals, the control-forward TensorData, invalid-seed
test tensor, temporary output comparison, leaf `.grad` publication/accumulation,
allocator cache and RSS. It estimates selected live numeric payload, not process
peak memory.

## Workload and lifecycle

Existing Linear-ReLU-Linear-MSE graph, shapes 64-128-32 and 256-1024-256, batches
1/32. Dedicated and planned workspace capacities compared explicitly. Their
forward values and adjoints match. A second planned workspace runs on changed input
while first output and activation state remain live; positive-capacity adjoint
buffers have distinct pointers. Empty buffers use Rust’s shared dangling pointer,
so pointer checks skip zero-capacity entries.

After backward, four input/parameter adjoints move from first workspace into
TensorData without copying the Vec. Measured capacity reduction must equal transferred
TensorData storage capacity. Invalid scalar seed shape fails before refill. Retry
restores adjoint capacity and values. Second workspace then completes backward;
earlier transferred TensorData snapshots remain intact.

Dedicated and planned workspace capacities match exactly in all four measured cases.
This training graph needs saved activations through backward, so planned inference
buffer reuse does not reduce payload here. Result is specific to these graph shapes.

| Model / batch | One workspace | Two live workspaces | Borrowed inputs/parameters | Returned outputs | VJP gradients | Tracked total after retry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64-128-32 / 1 | 84,872 B | 169,744 B | 50,432 B | 8 B | 49,792 B | 269,976 B |
| 64-128-32 / 32 | 156,296 B | 312,592 B | 70,272 B | 8 B | 49,792 B | 432,664 B |
| 256-1024-256 / 1 | 3,169,288 B | 6,338,576 B | 2,105,344 B | 8 B | 2,102,272 B | 10,546,200 B |
| 256-1024-256 / 32 | 3,740,680 B | 7,481,360 B | 2,200,576 B | 8 B | 2,102,272 B | 11,784,216 B |

## Short-lived inference activations

A separate test measures inference `ExecutionWorkspace` capacities for two
`32x128` float32 graphs. The dedicated baseline reserves one buffer per non-input
instruction. Planned workspaces reuse buffers after each value's last consumer;
outputs are returned as owned `TensorData` and excluded from workspace capacity.
Each graph runs twice with different inputs, values match the dedicated baseline,
and the first returned output remains intact after reuse.

| Graph | Dedicated buffers | Planned buffers | Dedicated capacity | Planned capacity | Capacity reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eight-ReLU chain | 8 | 2 | 131,072 B | 32,768 B | 75.0% |
| ReLU diamond plus four-ReLU tail | 9 | 3 | 147,456 B | 49,152 B | 66.7% |

These results show workspace payload reduction for the two tested inference
schedules. They do not measure total process memory, returned-output storage,
training activations retained for backward, or runtime speed.

## Leaf-gradient snapshot storage

A separate Rust test observes leaf `.grad` storage for `x*x` followed by sum,
with `x=2`. First retained backward publishes gradient values 4. A second
backward accumulates them to 8 by replacing the leaf gradient storage. Holding
the first `TensorData` snapshot keeps its old Arc-backed Vec alive, so the test
counts both the old snapshot and current gradient. Dropping the snapshot leaves
one gradient capacity. This counts backing Vec capacity; it does not measure
temporary gradient materialization or when the allocator returns pages to the OS.

| Elements | One gradient storage | Old snapshot + current after second backward |
| ---: | ---: | ---: |
| 64 | 256 B | 512 B |
| 12,448 | 49,792 B | 99,584 B |
| 525,568 | 2,102,272 B | 4,204,544 B |

The doubling is specific to retaining an earlier `grad()` snapshot across
cumulative backward. It is not a measure of ordinary training with no retained
snapshot, nor a peak-memory or RSS result.

The same `x*x` graph sends two contributions to the same leaf. Test-only
thread-local instrumentation records the live Vec capacities at the duplicate-
parent merge: existing accumulated gradient, incoming contribution and newly
materialized sum. Each observation equals three gradient capacities. Other
derivative temporaries and session state are outside this boundary, so this is a
specific merge payload, not total backward peak.

| Elements | Duplicate-parent merge live payload |
| ---: | ---: |
| 64 | 768 B |
| 12,448 | 149,376 B |
| 525,568 | 6,306,816 B |

A wider graph adds four `x*x` branches, adds branch outputs, then reduces to a
scalar. It sends seven contributions to the same leaf. The counter deduplicates
Arc-backed gradient storage by identity across the whole pending-gradient map,
incoming value and new sum. At the largest shape, one shared pending branch seed
is live beside the three unique leaf-gradient buffers, bringing this merge boundary
to four gradient capacities. It still excludes derivative temporaries and all
other saved forward values.

| Elements | Branches | Merge count | Unique live capacity at merge boundary |
| ---: | ---: | ---: | ---: |
| 64 | 4 | 7 | 1,024 B |
| 12,448 | 4 | 7 | 199,168 B |
| 525,568 | 4 | 7 | 8,409,088 B |

## Live TensorData storage during backward

A test-only tracker starts before graph construction and records weak references
to each new Arc-backed `TensorData` Vec on the test thread. After each allocation,
it deduplicates live storage identities and updates a capacity high-water mark.
The forward baseline is current live capacity after graph construction. Backward
high-water starts at that baseline and includes live graph values, gradient values
and derivative intermediates wrapped as `TensorData`. It measures Vec capacities,
not RSS or all process allocations.

The workload is the same four-branch `x*x` graph, with branches added then reduced
to scalar. Values and gradients validate against `x=2`. Debug/release high-water
rows match exactly.

| Elements | Forward live TensorData | Backward live TensorData peak | Peak / forward |
| ---: | ---: | ---: | ---: |
| 64 | 2,052 B | 3,588 B | 1.75x |
| 12,448 | 398,340 B | 697,092 B | 1.75x |
| 525,568 | 16,818,180 B | 29,431,812 B | 1.75x |

Additional derivative families use the same tracker. Matrix multiplication
uses `(32x128) @ (128x64)` with all-one left values and 0.5 right values, then
sums output; both gradients equal 32. LayerNorm uses rows of values 0 through 127,
unit weights, zero bias, then sums output; gradients are finite and bias gradient
equals 32. Broadcast uses `[32,128]` ones plus a `[128]` bias of 0.5; input and
bias gradients equal 1 and 32. Strided slicing sums a `[32,128]` view selecting
every second column from `[32,256]`; source gradients equal 1 at selected columns
and 0 elsewhere.

| Workload | Forward live TensorData | Backward live TensorData peak |
| --- | ---: | ---: |
| Matmul `(32x128) @ (128x64)` | 57,348 B | 114,692 B |
| LayerNorm `(32x128)` | 33,796 B | 67,588 B |
| Broadcast add `[32,128] + [128]` | 33,284 B | 66,564 B |
| Strided slice `[32,256]` to `[32,128]` | 32,772 B | 81,924 B |

Resumable callback sessions use same tracker at native hook/custom-VJP pause and
resume boundaries. Hook case applies output-hook gradient 2, then leaf-hook
gradient 9; final leaf gradient equals 9. Custom VJP case resumes with gradient
2 and commits gradient 2 to parent. These tests exercise Rust session boundaries
with callback results represented as `TensorData`; they do not execute Python
callback wrappers or count Python heap objects.

| Session path | Live TensorData before callback | Backward live TensorData peak |
| --- | ---: | ---: |
| Hook pause/resume | 49,152 B | 114,688 B |
| Custom VJP pause/resume | 32,768 B | 49,152 B |

Tracker scope excludes raw Vecs outside `TensorData`, storage created on other
threads or before tracking starts, Vec/Arc headers, graph metadata, allocator
rounding, allocator caches and RSS. The high-water mark observes each new
`TensorData` backing allocation; it is not an operating-system peak-memory measure.

## Verification and provenance

A first pointer check counted zero-capacity Vecs as distinct; review identified the
shared sentinel pointer. Test now compares only allocated adjoints. Review also
required explicit equality assertion for dedicated/planned counts; assertion is in
the test. Follow-up review found no issue in TensorData storage deduplication, moved
VJP ownership or scope limits.

Rust debug/release each pass 104 tests, six manual benchmarks ignored. Fmt and
Clippy all targets with warnings denied pass. Python suite: 978 passed, two existing
NumPy empty-mean warnings. Fresh candidate wheel/source pass README metadata,
30 license-file and scratch-exclusion checks. From unpacked source, benchmark/archive
tests pass: 24 passed, one optional Torch case skipped. No public API changed.

Exact source hashes, compiler, machine and identical debug/release rows are recorded
in benchmarks/workspace-payload-2026-10-15-summary.json,
benchmarks/workspace-lifetime-2026-10-15-summary.json and
benchmarks/backward-live-tensor-2026-10-15-summary.json. Raw logs and provenance stay
outside Git under work/workspace-payload-final-zfgb1sh2 and
work/workspace-lifetime-2026-10-15 and
work/workspace-memory-phases-2026-10-15. Candidate wheel/sdist built from the final
README pass matching metadata, 30 license-file and 225-member scratch checks. A
fresh Python 3.13.7 environment installs the sdist with its venv `bin` on `PATH`;
24 targeted archive/startup tests pass with one optional Torch test skipped, and a
tensor/gradient smoke check passes. Exact artifact hashes and environment details:
`work/backward-live-tensor-2026-10-15/package-receipt.json`.

Reproduce focused capacity rows with the repository Rust toolchain configured:

```sh
cargo test -q workspace_payload_lifecycle_study -- --nocapture
cargo test --release -q workspace_payload_lifecycle_study -- --nocapture
cargo test -q retained_snapshot_and_cumulative_leaf_grad_payload -- --nocapture
cargo test --release -q retained_snapshot_and_cumulative_leaf_grad_payload -- --nocapture
cargo test -q wide_branch_merge_payload_deduplicates_shared_seed_storage -- --nocapture
cargo test --release -q wide_branch_merge_payload_deduplicates_shared_seed_storage -- --nocapture
cargo test -q backward_family_payload_study -- --nocapture
cargo test --release -q backward_family_payload_study -- --nocapture
```

Next memory gate: compare native session high-water with Python callback wrapper
overhead in a process-level probe. Keep TensorData payload separate from Python
heap, allocator and RSS.
