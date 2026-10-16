# Unary dispatch foundation (unreleased)

The first registered operator family is eager unary mathematics: negation, exp, log, sqrt, sin, cos, sigmoid and tanh. This is a partial v0.13 foundation, not a completed general dispatcher or released version. Registered families now include eager unary math, add/subtract/multiply/divide/matmul and sum/mean. LayerNorm now uses registered eager forward dispatch and shared shape validation. MSE and cross-entropy now use registered eager forward dispatch and shared loss schemas.

Each unary registration has an operator identity, public name, actual device/dtype key and scalar kernel. The current immutable storage physically contains CPU float32 values. Native Device::Cpu and DType::Float32 metadata describe that storage; Python tensor properties now query native metadata. No other device/dtype variants or GPU kernels are claimed. Unsupported Python construction requests remain errors.

The unary schema has one input and one fresh output, preserving shape, dtype and device, without mutation or broadcasting. Registration is static and checked for duplicate names/identities. A missing name or missing kernel is an error. Strided views pass through the established logical-values boundary, so registered scalar kernels do not assume raw backing storage is contiguous. Scalar and empty tensors use the same schema.

```text
Python Tensor method
    |
TensorHandle + autograd recipe
    |
Unary operation -> registry (operator, CPU, float32) -> CPU kernel
    |
New immutable TensorData

Backward recipe -> shared mathematical formula -> parent gradient
```

Autograd formulas remain in autograd.rs, outside backend kernel registration. Each formula is tested with a full numerical Jacobian and PyTorch VJP parity on offset, strided slices. Special-value forward tests compare NumPy NaN/infinity behavior. Kernels use Rust float32 math; exact transcendental results can differ across platforms, so comparisons use explicit tolerances.

```python
import densecore as d

x = d.tensor([[.2, .7], [.5, .3]], requires_grad=True)
y = x.T[::2].exp().log().sum()
y.backward()
assert d.autograd.gradcheck(lambda a: a.T[::2].sin(), x)
```

Only Tensor methods are added; there are no corresponding top-level functions yet. New math operators have not been added to Python compiler capture or graph IR. Existing compiler operations and eager strided graph inputs retain their previous behavior. This registry does not imply general dispatch coverage or GPU support.

Next operator-system task: move shared shape/device/dtype inference and additional operator families into explicit schemas, then add compiler lowering with eager-versus-compiled tests. Preserve existing optimized graph kernels and autograd semantics during migration. Batched matmul, concatenation, stacking, comparisons and additional dtypes remain open roadmap work.

Sqrt backward computes seed / (2 * sqrt(x)) to avoid prematurely underflowing a subnormal seed when the final gradient is representable. An exact-power-of-two regression test reproduced the earlier zero gradient before correction. Special-value forward tolerances match the other float32 parity checks because system math libraries may differ by a few last bits. Final local suite for this unit: 464 Python tests and 60 Rust tests pass, with six manual Rust benchmarks ignored. Formatting and Clippy pass.


## Two-input operator migration

Existing eager add, multiply and rank-2 matmul now select registered CPU float32 kernels using both operands' device/dtype keys. The existing scalar/broadcast and optimized matrix kernels remain in ops.rs/cpu_kernels.rs. The registry records each two-input family's pure shape inference function. Graph shape inference calls those same registered schemas; eager matmul uses the same rank/inner-dimension and output-overflow checks. Autograd formulas remain unchanged.

Optimized compiled execution still has specialized CPU instructions and workspace kernels. This change unifies shape rules and eager kernel selection; it does not claim full compiler backend dispatch. Loss forward dispatch is now registered. Mixed dtypes and devices are unsupported, and no promotion or fallback policy is implied.

The additional registry tests cover strided two-input values, incompatible ranks/dimensions, empty shapes and output-size overflow. This unit increases passing Rust tests to 62; all 464 Python tests remain passing.


## Reduction migration

Sum and mean now use static CPU float32 registrations. A shared checked ReductionSpec normalizes whole-tensor versus explicit axes, validates axis bounds/duplicates and infers output shape. Mean divisors are computed lazily with checked arithmetic, so sum does not reject an otherwise valid empty result merely because an unused divisor would overflow. A selected zero dimension yields divisor zero before multiplication, matching empty-mean NaN behavior.

Eager whole/axis reductions select kernels by input device/dtype. Axis kernels preserve logical row-major accumulation and backward index maps, including offset strided views. The eager axis VJP uses the same divisor rules. Output allocation capacity failures return errors.

Compiler shape validation uses the same schema for its existing whole-tensor sum/mean instructions. Specialized CPU workspace execution reuses the same whole-reduction arithmetic; its existing backward formula is preserved. This migration does not add axis reductions to graph capture or make compiled execution backend independent.

Validation for this unit: 489 Python tests pass, including numerical full-Jacobian checks and PyTorch VJP parity; 65 Rust tests pass in debug and release, with six manual benchmarks ignored. Formatting and Clippy pass. Two existing NumPy empty-mean reference warnings remain. No new performance claim is made.

Empty output means and empty backward maps skip unused divisor calculation, so huge reduced dimensions do not reject an empty result solely because their unused product exceeds usize. A failing native-backed Python regression reproduced this case before correction. The schema still reports divisor overflow if explicitly queried; it does not invent a zero divisor for nonempty reduction axes.


## LayerNorm migration

The existing rank-2 last-dimension LayerNorm forward operation now selects a static CPU float32 registration. The lookup checks input and every supplied weight/bias device and dtype. No implicit fallback is added.

LayerNormSpec checks input rank, positive feature width, checked element count, affine tensor shapes and positive finite float32 epsilon. Eager dimensions and compiler shape inference call the same schema. The compiler supplies an optional expected feature count when it is part of the captured module contract. Eager functional calls retain their Python normalized_shape contract check; the Rust eager operation infers actual feature width from its input and passes no separate expected width.

Statistics, forward arithmetic and shared first-order backward formulas are unchanged. Specialized compiled/fused workspace execution still calls existing CPU buffer kernels directly after schema validation. Buffer-length checks remain in those kernels. This is not general backend dispatch or broader LayerNorm rank support.

New coverage checks strided input and optional affine tensors, bias-only operation, empty-batch affine gradients, invalid rank/width/affine/epsilon metadata, normalized-shape contracts and shared input/weight storage. Numerical full Jacobians and PyTorch VJP comparisons cover gradients. No new performance claim is made.


## Mean-loss migration

MSE and cross-entropy eager forward operations now select registered CPU float32 kernels. MSE checks both operands' device/dtype; cross-entropy receives integer class-index metadata alongside logits. MseSpec enforces identical nonempty operand shapes and checked element count. CrossEntropySpec checks rank-2 logits, positive class count, rank-1 label shape and checked dimensions. Eager forward/gradient validation and compiler inference use these schemas.

Cross-entropy retains an explicit execution-policy difference: eager integer-metadata labels permit an empty batch, returning NaN with an empty logits gradient; compiled tensor-label execution requires a nonempty batch. The internal compiled label representation remains detached float32 integer-valued metadata, not a new public integer tensor dtype. Shared decoding rejects nonfinite, negative, fractional and out-of-range indices.

Specialized CPU workspace forward now reuses shared loss arithmetic. Cross-entropy evaluates log(sum(exp(logit - max))) - (target_logit - max), preserving small losses beside large finite logits. A failing regression showed the old compiled expression returned zero for equal logits at positive or negative 1e30; corrected eager and compiled results both equal log(3). No clamping or hardcoded loss values are used.

MSE arithmetic and gradient formulas are preserved. Eager cross-entropy forward uses the same shifted-logit result as before, without requiring a temporary shifted row. Compiled backward keeps its existing softmax VJP calculation and now shares label decoding; backward arithmetic has not been unified into the forward registration table. Existing fused training and failure-atomic optimizer publication remain supported.

Invalid empty MSE is now rejected during compiler shape inference rather than waiting for execution. This changes validation timing, not supported inputs. New tests cover strided operands/labels, both MSE gradients, shared-storage Jacobians, invalid-label workspace recovery, special values and explicit empty-batch policies. No new performance claim is made. Activation registration is described below; specialized compiled workspaces still execute CPU kernels directly.


## Linear migration

Linear eager execution selects a CPU float32 registration that checks input, weight and optional bias metadata. LinearSpec validates compatible rank-2 operands, bias shape and checked input/weight/output counts. Graph inference and the eager CPU entry use that same schema. Zero batch, zero input features and zero output features remain supported. Existing GEMM and backward arithmetic are unchanged.

Validation at this unit: full Python suite 548 passed; Rust tests and Clippy/formatting pass. New tests cover numerical Jacobians through transposed operands, optional bias, shared input/weight storage and zero-dimension gradients. Source/wheel fresh-install checks are deferred to the combined session checkpoint. No performance or release claim is made.


## Activation migration

ReLU, softmax and log-softmax eager forwards now select static CPU float32 registrations. SoftmaxSpec checks shape element count, axis bounds and nonempty reduction width. The existing empty-output fast path, stable arithmetic and backward formulas remain unchanged. ReLU compiler shape inference shares checked shape validation; specialized compiled workspace execution remains CPU-specific.

New tests cover strided ReLU values and gradients, including NaN propagation, through eager and graph execution. Numerical full Jacobians cover softmax and log-softmax on transposed input; empty-batch backward remains supported. Combined suite: 552 Python tests passed and 70 Rust tests passed in debug and release, with six manual benchmarks ignored. Formatting and Clippy passed. Two existing NumPy empty-mean reference warnings remain. No new speed or release claim is made.


## Batched matrix multiplication

The public bmm function and Tensor.bmm method select a static CPU float32 registration. Both operands must have rank three, equal batch counts and matching inner dimensions. Batch broadcasting, output arguments, additional dtypes and compiler capture are unsupported. Rank-2 matmul behavior is unchanged.

A checked schema validates input and output element counts. Output and reusable per-matrix workspace allocation use fallible reservation. Empty outputs return before unreachable matrix products are computed; zero inner width returns zeros. Nonempty products call the existing optimized GEMM boundary once per batch. Logical strided values materialize through the existing tensor cache. No native batched BLAS or speed claim is made.

Backward multiplies upstream gradients by batch-transposed operands and accumulates by autograd node identity. Tests cover full numerical Jacobians, duplicate parents, view aliases, arbitrary seeded PyTorch VJP parity, frozen operands, no_grad, stale parameter versions, shape/type errors and zero dimensions. Rust checks include output overflow and unreachable huge dimensions in empty batches. Final gates: 565 Python tests, 72 Rust tests in debug/release, formatting and Clippy pass. Six manual Rust benchmarks remain ignored; two existing NumPy empty-mean reference warnings remain.


## Concatenation and stacking

The public cat function selects a static CPU float32 registration after Python validates input types and normalizes an integer axis. A shared native schema checks nonempty inputs, positive rank, matching non-axis dimensions, checked axis-size addition and checked output count. Allocation uses fallible reservation; empty output returns before irrelevant products or data materialization. Concatenation copies logical values, including strided input.

Stack requires identical shapes, normalizes its insertion axis against input rank plus one, and composes differentiable singleton reshape with cat. Scalar stacking is supported. Neither operation extends compiler capture, batch broadcasting, dtypes or device support. Concatenation backward returns immutable upstream slices; small leaf gradients may therefore retain the whole upstream allocation. Existing accumulation uses node identity and preserves duplicate-parent and view-alias contributions.

Tests cover numerical Jacobians, negative insertion/joining axes, scalar stacking, empty dimensions, no_grad, stale parameter versions, shape/type errors and seeded PyTorch VJP parity. A Rust regression checks unreachable huge dimensions and axis-addition overflow. Combined gates pass: 582 Python tests and 73 Rust tests in debug/release; formatting and Clippy pass. No performance claim is made.

Independent read-only review found that accepting a bare Tensor as an iterable silently joined its rows. A failing regression reproduced both cat and stack accepting that call; Python now rejects it explicitly before iteration. Review reported no other actionable defects in the new batched/joining kernels and autograd paths.


## Sigmoid and tanh

Eager `Tensor.sigmoid()` and `Tensor.tanh()` are registered CPU float32 unary
operations. Sigmoid evaluates a sign-dependent formula so its exponential never
receives a positive finite argument. Negative finite inputs can retain subnormal
positive outputs even where other implementations flush them to zero. Tanh uses
Rust f32::tanh and preserves signed zero. Both saturate at infinities and propagate
NaNs. VJPs use rounded float32 activation values, so saturated outputs can produce
exact zero gradients. Numerical equivalence uses tolerances, not bitwise equality.

Matching nn.Sigmoid/nn.Tanh modules and functional.sigmoid/tanh wrappers now
call these eager tensor methods. Unsupported modules use compiler eager fallback.
No compiler capture tag, NumPy activation ufunc bridge or higher-order derivative
was added. Existing
custom Function owner propagation and parameter-version checks apply.
