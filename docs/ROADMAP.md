# DenseCore roadmap (set by the owner, 2026-10-16)

Order of work. Do not start a phase before the one above it is done.

1. **Copy PyTorch exactly.** Every PyTorch feature, on CPU, Apple GPU (MPS) and NVIDIA CUDA,
   giving **bit-for-bit** the same results as PyTorch on the same machine.
2. **Then be at least 2x faster than PyTorch.**
3. **Only then** add features PyTorch does not have.

When DenseCore and PyTorch differ, PyTorch is the spec, even where DenseCore's answer is more
accurate. Tests that pin DenseCore-only behaviour are repointed to PyTorch's behaviour, and each
repoint is logged.

Reference version: PyTorch 2.14.1 (git 5c4886908584029761b579af026dcfb627c84070). PyTorch is
BSD-3 licensed; ported kernels keep its copyright notice (THIRD_PARTY_NOTICES.md, THIRD_PARTY_LICENSES/pytorch-2.14.1).

## How "exact" is measured

`scripts/conformance.py --exact` runs PyTorch's own OpInfo samples and compares raw bits
(any NaN equals any NaN), forward values and gradients, float32 CPU.

| Date | Tolerance mode (`conformance.py`) | Bit-for-bit (`--exact`) |
|---|---|---|
| 2026-10-16 | 639 / 676 | 349 / 676 |
| 2026-10-16 (PyTorch sum order, div gradient) | 639 / 676 | 389 / 676 |
| 2026-10-16 (SLEEF + PyTorch vector math: 25 unary ops, GELU, sigmoid, atan2, their gradients) | 639 / 676 | 408 / 676 |
| 2026-10-16 (PyTorch's sgemm_ calls for mm, matmul, linear and their gradients) | 639 / 676 | 415 / 676 |
| 2026-10-16 (bmm: PyTorch's small-product loop as compiled, per-batch sgemm_ above) | 639 / 676 | 416 / 676 |

## Phase 1 stages

| Stage | Scope | Done when |
|---|---|---|
| 1.1 | CPU float32, every OpInfo op bit-for-bit | `--exact` passes every op that tolerance mode passes |
| 1.2 | Every dtype with autograd (float64, complex, float16, bfloat16, integer ops) | `--exact` over all OpInfo dtypes |
| 1.3 | The rest of the API: all of `torch.nn`, `optim` + `lr_scheduler`, `utils.data`, `autograd.functional`, `torch.func`, `torch.save` file format, RNG streams that match `torch.manual_seed` | API audit against `torch.__all__` and submodules shows no gaps |
| 1.4 | MPS device | `--exact --device mps` matches PyTorch's MPS results |
| 1.5 | CUDA device | same, on an NVIDIA machine (Kaggle T4/P100 or KAUST) |
| 1.6 | Distributed, compile/export | to be planned when 1.5 is done |

### Stage 1.1 ingredients (in order of reach)

1. PyTorch's summation order (cascade sum, `aten/src/ATen/native/cpu/SumKernel.cpp`): sum, mean,
   broadcast gradients, var/std, norms, softmax, layer norm, losses.
2. PyTorch's vectorized math (`Vectorized<float>` / SLEEF on aarch64): exp, log, trig, erf, GELU.
3. The same BLAS calls PyTorch makes (same library, transposes and leading dimensions).
4. Per-op formulas from ATen kernels and `tools/autograd/derivatives.yaml`.
5. Fused kernels PyTorch picks at runtime: `scaled_dot_product_attention` (4-D float32, no mask,
   no dropout) runs PyTorch's CPU flash attention port, `src/torch_attention.rs`, done 2026-10-16.
   Grouped-query calls (fewer key heads) still use the composed path.
6. Optimizers and clipping: SGD/Adam/AdamW follow torch.optim's single-tensor CPU path op by op
   (fused lerp and add-with-alpha, C `pow`, fused addcmul tails), and `clip_grad_norm_` with
   norm_type 2 uses PyTorch's float32 norm kernel. Done 2026-10-16.

7. `torch.optim.lr_scheduler`: PyTorch's file vendored (imports adapted) in
   `python/densecore/optim/lr_scheduler.py`; `param_groups` are now persistent dicts and any
   hyperparameter key applies from the next step. All 15 schedulers match torch per step.

8. Random numbers: PyTorch's CPU generator (mt19937, `src/torch_random.rs`) behind
   `manual_seed`, `Generator`, `default_generator`. rand/randn (float32 and float64),
   randint, randperm, normal_/uniform_/bernoulli_ (scalar and tensor p), torch.normal, nn
   init (Linear, Conv, Embedding, init.*), dropout/dropout1d/2d/3d/alpha_dropout and
   attention with dropout (PyTorch's math path) give PyTorch's values for the same seed.
   Not ported (NumPy seeded from the stream): exponential_, geometric_, cauchy_,
   log_normal_, multinomial; trunc_normal_ draws match but erfinv is DenseCore's;
   get_state bytes use DenseCore's layout.
9. Attention masks: 2-D/4-D bool or float32 masks without dropout run the flash port with
   PyTorch's mask kernel; with dropout, the math path. All 144 sweep cases (causal or not,
   masks, scales, dropout) match bit for bit.

10. `torch.utils.data`: datasets, samplers, collation and DistributedSampler vendored from
    PyTorch; a DataLoader that keeps PyTorch's draw order and runs workers in-process with
    PyTorch's per-worker seeds. 14 loader cases match real PyTorch workers
    (tests/test_dataloader_exact.py). exponential_/geometric_/cauchy_/log_normal_ and
    multinomial now follow PyTorch's kernels too.

Milestone 2026-10-16: nanoGPT (Shakespeare-char, 4 layers, dropout 0) trains bit-identically to
PyTorch 2.14.1 on CPU from the same weights: loss, every gradient and every weight, 50 of 50
steps. Check: `scripts/nanogpt_exact.py <nanoGPT dir> 50`. Since item 8, `manual_seed(1337)`
builds the same initial weights (no copying), and dropout 0.2 trains identically too
(`scripts/nanogpt_exact.py <dir> 20 0.2`).

## Known tension for phase 2

Bit-for-bit results need PyTorch's algorithms, which cost what PyTorch's cost. Being 2x faster
will likely need a fast mode whose results are not bit-identical, beside the exact mode.
Decide this at the start of phase 2.

## Repointed tests (DenseCore-only behaviour -> PyTorch behaviour)

| Date | Test | Before | After |
|---|---|---|---|
| 2026-10-16 | `tests/test_binary_same_shape.py` repeat-broadcast test | reference summed repeats in order from +0.0 | PyTorch's own forward and gradients, bit for bit (NaN sign excepted, as before) |
| 2026-10-16 | `tests/test_binary_same_shape.py` same-shape test | same-shape and (1, ...)-shaped paths had to agree with each other | each must match PyTorch bit for bit; PyTorch itself turns -0.0 into +0.0 through a size-1 axis |
| 2026-10-16 | `tests/test_gelu.py` float64-reference test (approximate='none') | within 2e-6 of float64 GELU, incl. the far negative tail | PyTorch's bits; PyTorch's float32 erf gives -0.0 in the far tail (owner approved) |
| 2026-10-16 | `tests/test_sigmoid_tanh.py` extreme-input test | sigmoid(-90) a positive subnormal with a positive gradient | PyTorch's 0.0 and 0.0 (owner approved) |
| 2026-10-16 | `src/dispatch.rs` GELU kernel test | array kernels within 1 ulp of a float64 GELU | fused value/slope equal the separate kernels; PyTorch bits checked in tests/test_unary_exact.py (owner approved) |

Note (superseded by scripts/nanogpt_exact.py, which compares every bit): the nanoGPT 30-step loss cannot see last-bit kernel changes (a 1-ulp GELU change moves
the loss by ~1e-8, below one ulp of 4.2), so it is a regression check, not an exactness check.
| 2026-10-16 | `tests/test_layer_norm_parity.py` near-constant rows | within 1e-4 of float64 PyTorch | float32 PyTorch's bits (owner approved such repoints) |
| 2026-10-16 | `tests/test_autograd_lifecycle_matrix.py` layer_norm expectations | float64 formula, 1e-7 tolerance | PyTorch float32 values; lifecycle checks unchanged |
| 2026-10-17 | 27 bit-exact test functions (`tests/conftest.py`, marker `mac_exact`) | ran on every platform | Apple-silicon macOS only: PyTorch's own bits differ on Linux and x86 (owner approved "make them all pass") |
| 2026-10-17 | `tests/test_determinism_probe.py` | weights identical across BLAS thread counts 1 and 4 | identical across fresh processes at each thread count; Accelerate (as called by PyTorch) changes bits with its thread count on GitHub's Mac VMs |
| 2026-10-17 | `tests/test_layer_norm_parity.py` import | `import torch` at module level: collection error without PyTorch | `pytest.importorskip('torch')` like the other parity files (owner approved); assertions unchanged |
| 2026-10-17 | `tests/test_torch_compat_b.py` multinomial test | negative weights raised ValueError | PyTorch's RuntimeError and message; draws now follow PyTorch's multinomial kernel |
| 2026-10-16 | `tests/test_torch_compat.py` param_group test | setting `betas` raised NotImplementedError | the change applies from the next step and matches torch; setting `params` still raises |

