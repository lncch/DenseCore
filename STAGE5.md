# Stage5 verification: inference modes and CPU backend

Date:2026-10-14. Native no_grad/enable_grad use Rust thread-local mode. Synchronous contexts support nesting/reuse/exceptions; async/generator decorators reject. Tensor.detach copies. macOS matmul uses checked system Accelerate LP32 SGEMM; Linux retains reference arithmetic. Empty dimensions bypass BLAS and transpose dimension loops.

## Verified

158 Python cases,13 Rust tests and clippy pass on macOS. Clean installed release wheel executes100 non-oracle cases without Torch. Independent whole-stage review approved334a558 with no material findings, reran158/13/clippy and additional shared-context/thread/detach probes.

Linux ARM64 Docker passed13 Rust tests,clippy,100 Python non-oracle cases and training/checkpoint examples. Built CPython3.13 manylinux2.34 ARM64 wheel. Docker image index digest a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641; Python3.13.16,glibc2.36,NumPy2.5.4,safetensors0.8.0. Linuxx86_64 compiled an editable extension but validation was interrupted by disk exhaustion/Docker shutdown; no x86 validation or release wheel is claimed. No macOS11 runtime test.

## Performance

See benchmarks/STAGE5_RESULTS.md and stage5-results.json.gz. Three fresh-process repetitions; all8 pre-timing numerical gates and24 timed-case final-loss gates passed. MLP64->128->32 on M4, one requested CPU thread. Peak process RSS about40MiB versus208MiB for optimized eager control; imports about0.073s versus0.82s. About5.2x lower RSS and11x faster imports for this supported small workload.

Hot inference remains3.5x/4.3x slower than optimized eager at batches1/32; batch32 training2.1x slower. Batch1 training differs4%, not a robust broad speed claim. Compiled-forward Torch control is included; fully compiled optimizer steps/all tuning are not. Smaller feature scope and RSS shared-page accounting limit interpretation.

## Decisions and next work

LP32 dimensions/storage products/addressable bytes checked before pointer use; empty outputs bypass FFI. Cost: nonempty axes beyond int32 rejected. Owned contiguous inputs and separate output guarantee synchronous lifetime/non-aliasing. Grad-mode gate precedes provenance recording; disabled mode returns an independent stable snapshot. Cost: synchronous thread-local behavior, not async task-local mode.

Profile of2000 batch32 forwards attributes65ms to bias add and26ms to transpose, out of136ms total under cProfile. Supports native fused Linear next. Archived benchmark preserves this pre-optimization result.

Resource incident: Linux validation filled local disk and Docker stopped. Removed only generated Linux toolchain caches, Rust debug target and superseded task environments. Source, wheel artifacts and numerical/benchmark evidence preserved. Further heavy Linux builds require disk headroom; do not prune unrelated user data.
