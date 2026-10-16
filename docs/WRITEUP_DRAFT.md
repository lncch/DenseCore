# Training nanoGPT bit-for-bit identical to PyTorch, in a 2 MB Rust framework

DRAFT for review. Not published. Every number below comes from runs on 2026-10-16/17
(MacBook, Apple silicon, macOS, PyTorch 2.14.1 CPU, 4 threads).

## The result

DenseCore is a small deep-learning framework with a Rust core and a PyTorch-style Python
API. Point `import torch` at it, run Karpathy's nanoGPT unchanged, and you get **the same
bits as PyTorch**: the same initial weights from `torch.manual_seed(1337)`, the same loss,
every gradient and every weight, after every training step.

```
nanoGPT (Shakespeare-char, 4 layers, 128 wide, AdamW, grad clipping), 50 steps

                         dropout 0.0          dropout 0.2
initial weights          identical            identical
loss at step 49          2.5679025650024414   2.6270534992218018
                         (same in both)       (same in both)
gradients and weights    identical, all 50    identical, all 50
```

Reproduce: `PYTHONPATH=python python scripts/demo_bit_exact.py` downloads nanoGPT's
`model.py` (pinned commit) and the Shakespeare text, builds the model in both frameworks,
trains both on the same batches, and stops at the first step where any bit differs.

Size and start-up, measured on the same machine:

```
                 installed size     import time
PyTorch 2.14.1   583 MB             0.8 - 1.1 s
DenseCore        2.3 MB             0.08 - 0.09 s   (wheel: 0.9 MB)
```

DenseCore is not faster than PyTorch at the math. On a small nanoGPT benchmark it took
about 1.5x to 2x the time per step of single-threaded PyTorch (noisy, loaded machine).
The point here is exactness, not speed.

## Why "close enough" was not the goal

Most frameworks that copy PyTorch's API check themselves with a tolerance, such as "within
1e-5". That hides a lot: a different summation order, a different exp function, a fused
multiply-add where PyTorch has none. Each difference is tiny, but training amplifies them,
and two runs drift apart within a few hundred steps. Bit-for-bit equality is the only check
that proves you compute what PyTorch computes.

It also turned out to be the only check that works. A 1-ulp change in the AdamW update
moves 5 to 21 weights per step, yet the printed nanoGPT loss stayed identical to all 17
digits after 25 steps. A loss comparison would have called the wrong code correct.

## What it took

The matching parts are ports of PyTorch's own CPU code paths (BSD-licensed), not
re-derivations. Some details that decide the last bit:

1. **Where PyTorch fuses multiply-add.** PyTorch's vector loops do `a * b + c` as two
   rounded steps, but the scalar loop that handles the last few elements gets compiled into
   one fused instruction. So `addcmul` in Adam, the attention mask kernel and others round
   differently at the end of each thread's chunk. A port has to know where each chunk ends,
   which depends on the thread count.
2. **Summation order.** `sum` uses a cascade over 4-lane vectors, split across threads the
   way PyTorch's TensorIterator splits it. Softmax and layer norm use their own reductions
   with their own orders.
3. **Vector math.** exp, log, tanh, erf and friends come from SLEEF and PyTorch's own NEON
   routines, including a branch where a whole group of 4 lanes switches to a slower formula
   if any lane is out of range.
4. **Attention is not the formula in the docs.** On CPU, `scaled_dot_product_attention`
   runs a flash-attention kernel with blocks of 32/64/256 queries and 512 keys, online
   softmax and its own exp. With dropout it falls back to a "math" path that scales q and
   k by the square root of the scale, each, before the matmul.
5. **Memory layout changes results.** For small batched matmuls, PyTorch uses a compiled
   loop whose rounding depends on whether the operands are contiguous. Scaling `k^T` as a
   transposed view keeps a layout that a plain copy would not.
6. **Python-level surprises.** `max_norm / (total_norm + 1e-6)` in `clip_grad_norm_` runs
   as `reciprocal() * max_norm`. Adam's bias correction uses Python's `x ** 0.5`, which is C
   `pow`, not `sqrt`.
7. **Random numbers.** PyTorch's generator is a Mersenne Twister, but `randn` turns
   uniforms into normals 16 at a time with Box-Muller, and refills the last 16 when the size
   is not a multiple of 16. `nn.Linear` init, dropout masks (drawn in double precision),
   `randperm` and `multinomial` each have their own transform. A DataLoader draws a seed
   when its iterator is created, before the shuffling sampler draws its own.

## PyTorch does not match itself across platforms

A side finding. I ran 26 common ops on fixed inputs in PyTorch 2.14.1 on three platforms
and compared the bits with PyTorch on the Mac:

```
                         same bits as Mac PyTorch
Linux arm64 (Docker)     21 / 26
Linux x86-64 (emulated)   1 / 26
```

Different math libraries, compilers and vector widths give different last bits. So
"bit-identical to PyTorch" always means "on the same platform". DenseCore's claim today is
for Apple-silicon macOS. The x86 run was emulated without wide vector units, so real x86
servers would differ again.

## Scope and limits

- CPU, float32, macOS on Apple silicon, PyTorch 2.14.1, same thread count.
- 420 of 676 ops in PyTorch's own OpInfo test suite match bit for bit; 639 match within
  tolerance. Convolution is not exact yet.
- No GPU. No speedup over PyTorch.
- Covered exactly: the ops nanoGPT uses, SGD/Adam/AdamW, all 15 LR schedulers, the
  generator and its samplers, and `torch.utils.data` (14 DataLoader setups compared against
  real PyTorch worker processes).

## Try it

(To fill in at publish time: install command, link, the nanoGPT command above.)
