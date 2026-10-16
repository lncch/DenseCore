# nanoGPT on DenseCore, unchanged (2026-10-16)

Goal: Karpathy's nanoGPT (`model.py`, `train.py`, `sample.py`, master as of 2026-10-16,
`model.py` sha1 291c6ed8) runs with `import torch` resolved to DenseCore, no edits.

`scripts/run_as_torch.py` maps `torch`, `torch.nn`, `torch.nn.functional` and `torch.optim`
to DenseCore and stubs `torch.nn.parallel` / `torch.distributed` (imported at the top of
train.py, used only for multi-GPU). It is a prototype of a future `densecore run`.

## Commands (README "CPU" settings, Shakespeare-char)

```
python data/shakespeare_char/prepare.py   # needs `requests` only to download input.txt
PYTHONPATH=<repo>/python python <repo>/scripts/run_as_torch.py train.py \
  config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=5 \
  --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 \
  --max_iters=300 --lr_decay_iters=300 --dropout=0.0 --eval_interval=100
python <repo>/scripts/run_as_torch.py sample.py --out_dir=out-shakespeare-char \
  --device=cpu --compile=False --num_samples=2 --max_new_tokens=150 --start=ROMEO:
```

sample.py imports `tiktoken` at the top but uses it only without meta.pkl; an empty stub
module was enough.

## Results

| Run | DenseCore | PyTorch 2.14.1 |
|---|---|---|
| Parameters | 0.80M | 0.80M |
| 20 steps, val loss at step 20 | 3.5735 | 3.5763 |
| 300 steps, val loss | 4.2182 -> 2.4003 | not run |
| Step time, 20-step run (Mac load 7.8, rough) | ~200-330 ms | ~70-165 ms |

Initial weights differ (each library's own random stream), so loss curves are close, not
identical. Exercised: AdamW with two parameter groups, live learning-rate schedule, gradient
clipping, GradScaler disabling itself, evaluation, checkpoint save, checkpoint load and
generation (topk, multinomial, masked fill, int64 cat).

## Not covered

`from_pretrained` (needs Hugging Face transformers), `torch.compile=True`, DDP, CUDA, and
enabled autocast. Speed was measured on a loaded machine and is not a benchmark claim.

## Bit-for-bit check (2026-10-16)

`scripts/nanogpt_exact.py <nanoGPT dir> [steps]` builds nanoGPT's GPT in both frameworks, copies
PyTorch's initial weights into DenseCore, runs the same batches with AdamW and grad clipping, and
compares the loss, every gradient and every weight each step. Result: 50 of 50 steps identical.
Since the generator port, both frameworks start from `manual_seed(1337)` with identical weights;
an optional third argument sets dropout (20 of 20 steps identical at 0.2).

