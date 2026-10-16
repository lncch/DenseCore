# Tiny GPT example, 2026-10-16

`examples/tiny_gpt.py`: character-level GPT (2 blocks, width 64, 4 heads, context 32,
114,270 parameters, vocab 94) trained on this repository's README.md (39,796 characters,
first 90% train, last 10% held out). Adam lr 3e-3, batch 16. Apple M4, one requested thread.

## Correctness

`--parity` (width 32, 2 heads, context 16, batch 4, Adam lr 1e-3) copies DenseCore's initial
weights into an identical PyTorch 2.14.1 model and runs both on the same batches:

| Step | DenseCore loss | PyTorch loss |
|---|---|---|
| 1 | 4.8250556 | 4.8250546 |
| 2 | 4.6812253 | 4.6812248 |
| 3 | 4.8039074 | 4.8039074 |

Largest weight difference after three steps: 4.8e-5 (Adam moves each weight by about 1e-3 per step).
Covered by tests/test_tiny_gpt_example.py.

## Training

600 steps, 8.28 s. Held-out cross-entropy 4.716 -> 2.298, below the unigram baseline 3.459
and the uniform baseline 4.543. Greedy samples repeat short words; this is a small model and
greedy decoding, not a quality claim.

## Speed (loss: DenseCore is about 4.7x slower)

Same model and weights, 300 steps each, two alternating passes, one process:

| Engine | ms/step |
|---|---|
| DenseCore eager | 13.7, 13.7 |
| PyTorch eager | 2.94, 2.86 |

Not profiled yet. Candidate causes: attention is composed in Python from many small ops,
strided views materialize copies, and masks/indices cross into Rust as Python lists.

## Profile and first fix (same day)

cProfile over 100 steps: native backward 7.3 ms/step, GELU forward (`unary_math`) 1.6 ms/step,
then reshape, softmax, masked_fill and add at 0.3-0.7 ms each. Swapping GELU for ReLU cut the
step from 13.78 to 10.35 ms, so GELU was about 25% of the step.

Fix: GELU forward and backward now run the same f64 formulas over whole arrays, with the
exponentials batched through Accelerate vForce `vvexp` (scalar `exp` elsewhere). A Rust test
checks the array kernels against the scalar ones within 1 float32 ulp on 8,000+ points
including +-inf, NaN and +-0.

| Measurement | Before | After |
|---|---|---|
| GELU forward, 512x256 | 839 us | 320 us |
| Tiny GPT step, fresh processes | 13.7, 13.7, 13.78 ms | 11.97, 12.29, 12.37, 12.66 ms |
| GELU share (ReLU swap) | 3.4 ms | 1.5 ms |

PyTorch on the same step: 2.9 ms. DenseCore is still about 4.2x slower. The remaining gap is
mostly native backward and many small composed ops; not yet profiled at the native level.

## Native profile and two overhead fixes (2026-10-16 morning)

macOS `sample` over 15 s of training (about 10,200 samples). BLAS was only about 2%.

| Share | Where |
|---|---|
| 22% | `broadcast_plan` + indexed binary loop, mostly for same-shape operands |
| 17% | `values()` on strided views, one divide/modulo per axis per element |
| 15% | `exp` (vForce in GELU, scalar `expf` in softmax) |
| 9% | LayerNorm forward + gradient |
| 8% | softmax forward + gradient |

Fixes: c00db24 zips same-shape binary ops directly (forward and backward, signed zeros kept
as before). c2ed971 gathers strided views with a stride walk plus an odometer.

The machine was loaded (iCloud and a storage scan each held a core), so builds alternated in
fresh processes, 5 rounds each, 100 timed steps after 100 warm-up steps:

| Build | Median ms/step |
|---|---|
| before (d05e54c) | 22.4 |
| same-shape fix (c00db24) | 19.6 |
| both fixes (c2ed971) | 16.3 |

Loss after 200 Adam steps was 2.726423740386963 in all 15 runs. PyTorch under the same load:
5.24, 5.08 ms, so DenseCore is now about 3.2x slower on this step. Next in the profile: softmax
`expf`, LayerNorm gradient, and the remaining broadcast cases (bias add, position add).
