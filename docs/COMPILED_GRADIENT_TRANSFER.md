# Private compiled-gradient transfer

2026-10-15. Base b652323, feat/v0.12-storage. Runtime change is confined to
compiled VJP extraction and adjoint-buffer preparation, not optimizer kernels.

## Evidence and design

Larger-model native samples include compiled backward and gradient extraction
memmove frames. Existing CompiledVjp copied each input adjoint with to_vec before
constructing immutable TensorData. Independent review supports moving those Vec
allocations only from the privately owned workspace. Public executor and fused
optimizer consumers still borrow input_gradient slices.

TrainingWorkspace.take_input_gradient is crate-private and follows the same
readiness/reachability guards as borrowed access. It transfers each input slot once
per successful private backward. On the next retained or retried traversal, exact
reachable input lengths are restored with fallible allocation before seeding/VJP.
Implicit scalar seeds use immutable output shape rather than possibly drained
buffer length. All gradient buffers are then zeroed as before.

A pointer test first failed for missing extraction, then exposed the identity-plan
scalar seed trap after the move helper alone. It passes after refill/spec checks.
Tests cover pointer transfer, immutable old gradients, identity/empty/frozen slots,
duplicate native parents and hook failure after extraction followed by retry and
retained backward. Existing public workspace/fused tests remain passing.
Independent read-only implementation review found no required correctness defect.

## Validation and paired measurements

Rust debug/release: 92 passed each, six existing manual benchmarks ignored. Fmt
and Clippy all targets with warnings denied pass. Full Python suite: 826 passed,
two existing NumPy empty-mean reference warnings. Release extension rebuilt.
An initial targeted command named a nonexistent compiled-backward test file;
corrected by running the actual complete suite, without dropping tests.

Paired runs use the same current binary-gated benchmark child for baseline and
candidate interpreters. Model 256-1024-256, batches 1/32, one thread, three fresh
processes per label, alternating order. Baseline is the installed pre-transfer
hook wheel under work/tensor-hooks-bJwIdN/wheel-env relative to workspace root;
candidate is the working release extension with current Python sources. Package
metadata is 0.10.0 for both; native hashes identify builds. Gates check identical
weight/data hashes, full initial/20-step payload arrays and final loss tolerances.

Initial paired sweep (median us, process range):

| Batch | Baseline | Candidate |
| --- | --- | --- |
| 1 | 483.69 (483.14-487.99) | 448.76 (434.18-546.74) |
| 32 | 705.66 (699.50-783.37) | 633.35 (611.25-637.65) |

Initial medians are approximately 7%/10% lower; batch-1 ranges overlap. A separate
confirmation sweep with the versioned generic comparator gives:

| Batch | Baseline | Candidate |
| --- | --- | --- |
| 1 | 522.92 (488.72-747.68) | 513.75 (480.44-632.05) |
| 32 | 719.96 (691.53-894.61) | 711.44 (597.51-728.51) |

Confirmation medians are only about 1-2% lower and ranges overlap. No reliable
latency speedup is established. Both sweeps are retained; do not select only the
larger initial improvement or compare with unrelated historical timings. Three
processes per sweep are descriptive, not confidence intervals.

The retained benefit is structural copy/storage elimination. Pointer identity and
workspace payload-capacity accounting prove that the gradient allocation moves
into immutable result storage. For the measured model, copying 525,568 parameter
gradient float32 values (2,102,272 payload bytes) is avoided per extraction.
This is not a claim about allocator RSS, total peak memory or end-to-end speed.
Retained/retry paths must refill drained buffers on their next traversal. The
whole compiled-training slowdown remains open.

Raw first-sweep JSON/NPZ artifacts remain outside Git under
work/large-model-nwBwA7/transfer-paired relative to workspace root. Its scratch
orchestrator is preserved under work/milestone-audit/paired_compiled_transfer.py.
The generic versioned comparator uses the same benchmark child protocol:

```sh
.venv/bin/python benchmarks/compare_compiled_training.py --baseline-python /path/to/baseline-env/bin/python --candidate-python "$PWD/.venv/bin/python" --candidate-pythonpath "$PWD/python" --output /tmp/densecore-transfer-pair-new.json
```

Select a real baseline environment and a fresh output path. Preserve generated NPZ
directories with JSON results. The actual local baseline path is recorded above;
this generic command is a template, not an additional claimed execution.

No public API, dtype/backend, compiler capture, version bump or GPU claim. No
push, merge, tag, publication or user-data deletion. Generated binary payloads
stay outside Git; selected compact timing/provenance records are intentional.


Both paired sweeps passed all 12 process comparison gates each. Selected records
in benchmarks/compiled-transfer-2026-10-15-summary.json include both complete sets
of timing intervals, payload checksums and native identities. Confirmation raw
files remain beside the primary sweep as transfer-confirmation.json and its owned
NPZ directory. No binary payload is committed. Installation results follow below.


The final versioned comparator additionally validates baseline/candidate snapshots
for every batch before starting timed children. Its pre-gated confirmation passes
both batch-pair gates and all 12 timed-process checks:

| Batch | Baseline median us (range) | Candidate median us (range) |
| --- | --- | --- |
| 1 | 579.37 (481.70-602.05) | 503.75 (490.82-556.43) |
| 32 | 662.76 (654.68-678.60) | 677.24 (652.59-683.69) |

Batch 32 is slightly slower in this confirmation, reinforcing that latency benefit
is inconclusive. Earlier paired sweeps checked equality after each timed process;
this final sweep adds explicit pre-timing comparisons. All three datasets remain
visible in the compact evidence file. Do not cherry-pick an apparent speedup.

Fresh activated source installation: 682 passed, 35 skipped without optional
Torch; three Torch-only modules excluded. Fresh wheel compiled/hook/custom tests:
83 passed, 1 optional Torch case skipped. Two existing NumPy reference warnings
remain in the source suite. Archive checker passes README, 30 license files and
scratch exclusions. Artifacts outside Git: work/compiled-transfer-78GbHf relative
to workspace root. Wheel SHA256: 4f0fe0cc4ea04a44781a1a962bfafc805426794c3324790b29124d9d523d79b1.
Source SHA256: e13558cd923aabdabe2b2662a008e92b3a75ff96e20cac796c648a30a3692f78.
Archives precede final comparator pre-gates and validation text; runtime source is
the validated ownership-transfer implementation, not a published release.
