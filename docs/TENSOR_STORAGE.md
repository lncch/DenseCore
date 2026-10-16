# Tensor storage and layouts (unreleased v0.12 work)

The baseline is commit 4044e5c on feat/v0.11. This development branch does not change release metadata or publish a release.

## Ownership and invariants

TensorStorage owns an immutable Arc<Vec<f32>>. TensorLayout owns shape, element strides, storage offset and checked element count. TensorData combines them with a logical-values cache. Constructors reject unrepresentable nonempty products, strides and reachable addresses. Empty layouts saturate irrelevant strides rather than reject shapes with no reachable elements. Contiguity is computed once when metadata is built. Scalars have one element and no axes; empty tensors have zero accessible elements. Singleton strides do not affect contiguity. All exported NumPy values copy.

Tensor identity belongs to the autograd Node, independently of storage identity. Reshape and permutation create graph nodes with separate gradient metadata while sharing the allocation. Identical TensorData clones share the logical-values cache. Storage identity is used only for shares_storage(), never for graph identity or optimizer state.

## Kernel boundary

Existing CPU kernels, graph workspaces, optimizers and serialization consume values() as logical row-major values. Contiguous tensors borrow their storage interval directly. A strided tensor lazily gathers logical values once into a cache. Repeated reads reuse that cache. Views remain views after materialization; shares_storage() reports the backing allocation rather than the cache.

This preserves correctness across all existing consumers without rewriting each kernel. It also retains an extra buffer after a strided tensor is consumed and can keep an entire source allocation alive through a small view. Future stride-aware kernels must be measured before replacing this boundary. No native strided BLAS or GPU kernel is claimed.

## Public Python subset

Tensor.stride() returns element strides; storage_offset() returns element offset. numel() does not read values. is_contiguous() ignores singleton strides and returns true for empty tensors. shares_storage(other) checks immutable allocation identity. contiguous() shares contiguous storage and copies strided logical values while retaining the autograd path.

reshape shares storage for contiguous inputs and copies logical row-major values for non-contiguous inputs. Some strided layouts could be reshaped without copying, but that optimization is not implemented. transpose and permute create zero-copy metadata views.

## Mutation and saved tensors

No writable storage or in-place mutation API exists. Optimizers and state loading replace leaf handles with new immutable snapshots, preserving their logical parameter version token. Existing views keep old values, and saved autograd computations reject a changed source version. This snapshot behavior differs from PyTorch mutable alias behavior. detach creates an independent detached snapshot that may share immutable storage; it does not grant writable access.

## Validation and remaining work

Shared views, scalar/empty/singleton layouts, bounds/overflow, cached clones across threads, parent lifetime, non-contiguous gradients, finite differences, shared gradient paths, compiler input layouts and saved parameter versions have tests. The full suite also covers checkpoints and fused native training. Basic integer and positive-step slice views support backward scatter into logical parent order. Negative integer indices and one ellipsis are supported; new-axis, boolean, advanced indexing and non-positive slice steps are rejected. Empty slices keep a valid canonical offset; their offset is not intended to match NumPy pointer arithmetic. Arbitrary as_strided, overlapping writable views, negative strides, GPU storage and multiple dtypes are unsupported. v0.12 is not complete.

## Local view-construction measurements

Same macOS arm64 host, release build, 1024 by 1024 float32 input, 100 view constructions per sample, seven samples, median nanoseconds per construction. Baseline 4044e5c: reshape 103652.50 ns, transpose 6280717.92 ns, permute 2725501.25 ns. Shared-storage candidate: reshape 3887.50 ns, transpose 1931.25 ns, permute 3910.41 ns. Timing excludes downstream materialization and is not an end-to-end training speed claim. Results describe one local run, not a cross-platform guarantee. Raw measurements remain outside the repository under work/v012-view-comparison.json.


Run the current layout benchmark with `PYTHONPATH=python .venv/bin/python benchmarks/benchmark_layout.py`. This measures creation, views, positive-step slicing and cold transpose materialization separately. Each memory mode runs in a fresh process and retains 32 tensors. One local run recorded no increase in peak RSS while retaining views, versus 102350848 additional peak bytes while retaining materialized tensors. Initial tensor construction already raises peak RSS; these differences are not live allocation totals. Raw output is outside Git at work/v012-layout-benchmark.json. No memory or performance result is a release gate by itself.


## Training regression review

Sequential fresh-process comparison against a separately built 4044e5c wheel used SGD/Adam, batches 1/32 and eager/compiled/fused execution, three process repeats with candidate order alternated. Initial weights and data hashes matched, and 12-step losses and final 1562-step losses matched within rtol=1e-6, atol=1e-7. Median candidate/baseline latency ratios ranged from 0.791 to 1.081. Results are mixed and limited to this local dense MLP workload; no general training speed improvement is claimed. The separate five-engine PyTorch-reference benchmark passed all numerical gates with one process repeat. Its timings overlapped an exploratory comparison and are excluded from performance claims. Sequential reviewed data remains outside Git at work/v012-training-regression-reviewed.json.

Validation checkpoint: 422 Python tests, 59 Rust tests, six ignored manual benchmarks, formatting and Clippy pass. The macOS arm64 wheel and source distribution built and passed artifact/license checks before the final empty-softmax guard; a fresh source installation passed 310 tests with three PyTorch-only skips. These local artifacts are not published and Linux CI has not run for this branch. Cross-platform release acceptance remains open.


## Migration alternatives

Keeping owned Vec copies would preserve the previous implementation but prevent zero-copy views. Mutable Arc/Mutex buffers would need storage-scoped mutation/version checks across every alias and saved tensor; adding that model without a public in-place API would create unnecessary risk now. The chosen immutable allocations retain existing parameter replacement semantics. A future in-place API requires a new storage mutation/version design, not removal of the current safeguards.

A reshape-only intermediate was considered in independent review. Checked strides were retained because transpose, permutation and slicing views are explicit foundation requirements. Rewriting all kernels for strided storage immediately would increase migration scope and could discard existing optimized CPU paths. The logical-values materialization boundary allows tested incremental migration, with its retained-cache memory cost disclosed above.
