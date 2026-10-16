# Changelog

## Unreleased, workspace and tensor payload study

- Add test-only capacity accounting for native workspaces, shared TensorData, transferred VJP gradients, retained leaf-gradient snapshots, backward TensorData high-water, resumable callback sessions, duplicate-parent/wide-branch merge payloads and short-lived inference graphs. Record bounded scope, exclusions and exact source/build provenance.

## Unreleased, workspace payload accounting

- Add test-only capacity study for dedicated/planned training workspaces, concurrent live workspaces, moved gradients and retry refill. Record exact float32 capacity scope and excluded memory categories.

## Unreleased, requested CPU thread controls

- Add positive --threads control to the CPU benchmark, propagate it to fresh children and validate reported environment/Torch settings. Preserve default-one behavior and reject artifact overwrites. Record small/larger-model results and cross-setting numerical checks.

## Unreleased, startup and dependency measurements

- Add isolated fresh-process startup controls and a fixed numerically checked first-forward workload. Record declared dependency payload using raw wheel RECORD entries, rejecting incomplete/editable installations. Preserve all local samples and measurement limits.

## Unreleased, lifecycle and source archive validation

- Extend binary, matrix and empty-reduction lifecycle coverage, including frozen operands, retained retry and saved-version rejection. Correct stale tensor-hook compatibility text.
- Accept the standalone GEMM Rust probe source in source-archive checks and require its Python companion; keep generated results and executable payloads rejected. Add archive regression tests.

## Unreleased, CPU GEMM initialization

- Skip redundant zero-fill for nonempty Accelerate GEMM without bias, preserving initialized storage and empty/reference/bias behavior. Add NaN-reuse regressions and isolated kernel comparisons. End-to-end training results remain workload dependent.

## Unreleased, compiled gradient ownership

- Move private compiled-VJP input gradients into immutable tensors instead of copying. Restore drained buffers before retained/retried backward, preserving public/fused borrowed paths. Pointer tests prove transfer; paired timing sweeps do not establish a reliable latency win.

## Unreleased, gradient hooks

- Add ordered removable first-order tensor hooks with late registration, weak native identity, GC-visible Python ownership and per-callback version/shape validation. Preserve hooks across parameter replacement and reject unsupported fused updates at public/native boundaries. Record no-hook ownership allocation and construction regressions.

## Unreleased, singleton dimensions

- Add Tensor.squeeze/unsqueeze through checked reshape, with negative axes, scalar/empty/bool support and first-order gradients. Confirm new eager modules reject fused training without changing parameters, gradients or optimizer state.

## Unreleased, flattening

- Add Tensor.flatten and nn.Flatten with checked inclusive axes, scalar/empty/bool support, reshape storage semantics and first-order gradients. Flatten modules use compiler eager fallback.

## Unreleased, activation modules

- Add parameter-free nn.Sigmoid/nn.Tanh and functional.sigmoid/tanh wrappers. Sequential training and guarded compiler eager fallback preserve gradients and optimizer behavior.

## Unreleased, dense activations

- Register eager CPU float32 sigmoid/tanh with strided inputs and first-order gradients. Sigmoid uses sign-based exponent evaluation to avoid exponential overflow. Float32 saturation and IEEE nonfinite behavior are documented.

## Unreleased, arithmetic coverage

- Register CPU float32 subtraction/division with scalar/reflected arithmetic, checked broadcasting, strided inputs and first-order gradients. Extend shared native binary workspace formulas and division operand pinning; Python compiler capture remains unchanged.

## Unreleased, gradcheck validation

- Reject boolean gradcheck outputs before evaluating numerical Jacobians, including empty results and dtype changes during perturbation. Invalid outputs raise in both mismatch modes.

## Unreleased, custom autograd

- Add single-output float32 `autograd.Function.apply`, saved version-checked snapshots, detached first-order VJPs and Python-visible callback ownership. Callback failures preserve native leaf gradients and graph readiness. Hooks and higher-order differentiation remain unsupported.

## Unreleased, continuing operator coverage

- Stage eager backward leaf updates and graph consumption until every fallible derivative/accumulation/lock step succeeds. Reject VJP contributions whose shapes differ from their parents.

- Register bool-input comparisons and add boolean `&`, `|`, `^` with reflected Python/NumPy boolean scalars, checked broadcasting and nondifferentiable results. Preserve same-dtype tensor rules.

- Add float32 Python comparison operators, including reflected real scalars, while preserving identity-based hashing and unsupported-object fallback. Comparisons return nondifferentiable bool masks.

- Fix compact compiled broadcasting to recognize empty results before unreachable stride/product arithmetic, preserving zero scalar VJPs for add/multiply.

- Add registered eager float32 named comparisons with real bool outputs and registered bool logical-and/or/xor, sharing checked broadcasting. Fix eager broadcast shape/index handling for empty tensors with huge unreachable dimensions.

- Add real CPU boolean storage through shared typed layouts, explicit bool construction/views/export and native logical-not. Preserve float32-only autograd; reject unsupported boolean math and boolean Parameter/state loading. Require one element for Python tensor truth testing.

- Propagate checked tensor shape/device/dtype specifications through compiled plans and specialization caches. Preserve CPU float32 lowering and validate runtime input/seed contracts. Fix inference/training workspace counts for empty tensors with huge unreachable dimensions.

- Add eager `cat` and `stack` with checked concatenation, negative-axis normalization, empty-tensor support and first-order gradients through shared parents.

- Add eager CPU float32 `bmm` and `Tensor.bmm` with strict equal-batch rank-3 shapes, checked allocation, strided operands and first-order gradients.

- Register ReLU/softmax/log-softmax CPU float32 forwards with shared checked softmax dimensions; preserve stable arithmetic and gradients.

- Register Linear eager kernels and share checked shape inference with compiler validation, preserving GEMM and degenerate-dimension behavior.

## Unreleased, v0.13 operator dispatch foundation

- Add static CPU float32 negation/exp/log/sqrt/sin/cos registration with first-order gradients, strided-view coverage and native dtype/device metadata.
- Route eager add/multiply/matmul through registered kernels; share registered binary/matmul shape inference with graph validation.
- Register sum/mean reductions with checked axis/output-shape/divisor semantics shared with compiler validation; preserve whole-reduction CPU arithmetic and axis-gradient maps.
- Register existing LayerNorm forward kernel and share checked rank/feature/affine/epsilon semantics with compiler inference; preserve forward/backward arithmetic and fused workspace execution.
- Register MSE/cross-entropy forward kernels with shared checked schemas and explicit empty-batch policies. Fix compiled cross-entropy cancellation for large finite logits using shared shifted-logit arithmetic.

## Unreleased, autograd validation

- Add full-Jacobian float32 numerical gradcheck with independent input copies, mismatch diagnostics and explicit precision/determinism limits.

## Unreleased, v0.12 storage foundation

- Add immutable shared CPU float32 storage, checked strides and offsets, zero-copy reshape for contiguous inputs, transpose/permute and basic positive-step slicing views.
- Add layout inspection and contiguous materialization; preserve logical row-major kernel inputs, autograd identities and version-checked parameter snapshots.
- Add slicing backward scatter and view safety/parity tests. GPU, mutable aliasing, negative strides and multiple dtypes remain unsupported. Package metadata remains 0.10.0.

## Unreleased, v0.11 foundation

- Add generalized broadcasting, axis reductions, reshape/permutation, stable softmax/log-softmax and fused cross-entropy/LayerNorm training. Baseline commit 4044e5c passed local tests and CI run 38648968163.

## 0.10.0, local release candidate

Release-readiness corrections: document LayerNorm capture, test native frozen-parameter rejection with matching optimizers, include readable dependency notices, exclude experimental data/backups from source distributions, and add full PyTorch-oracle CI coverage. Historical wheels retain their original provenance; current audited artifacts are recorded separately.

- Add opt-in compiled_train_step for built-in dense MSE training with stateful SGD/Adam.
- Stage optimizer transactions and Python publication allocations before commit; preserve Parameter identity and existing eager gradients.
- Combine strict training registry validation with graph signature guards.
- Reuse bounded serialized fused workspaces with same-build cached/uncached benchmarks and retained-loss/failure-recovery coverage.
- Ordinary autograd training remains unchanged. Public publication has not occurred.

## 0.9.0, local release candidate

- Add native eager/compiled rank-2 LayerNorm with affine gradients and guarded Python capture.
- Add compiled rank-2 matmul and whole-tensor sum/mean, including backward execution.
- Add duplicate-group-safe UCI Letter Recognition training, one-time heldout evaluation, pretrained safetensors weights and a no-Torch inference example.
- Add matched multi-process training/inference/RSS benchmarks, numerical stress diagnostics and preserved optimization evidence.
- Prepare Linux/macOS wheel CI, third-party license files and dataset attribution. No public release or registry upload has occurred.
