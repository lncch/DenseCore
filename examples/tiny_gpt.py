"""Character-level GPT trained end to end on this repository's README.

Exercises Embedding, rank-3 Linear/LayerNorm, causal scaled dot-product attention, GELU,
residual connections, cross-entropy and Adam. `--parity` copies the initial weights into an
identical PyTorch model and compares the first Adam steps (PyTorch is optional otherwise).

    PYTHONPATH=python python examples/tiny_gpt.py --steps 300
    PYTHONPATH=python python examples/tiny_gpt.py --parity
"""
import argparse
import json
import math
import pathlib
import time

import numpy as np
import densecore as d

TEXT = pathlib.Path(__file__).resolve().parents[1] / "README.md"


class Block(d.nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = d.nn.LayerNorm(width), d.nn.LayerNorm(width)
        self.query, self.key, self.value = (d.nn.Linear(width, width) for _ in range(3))
        self.project = d.nn.Linear(width, width)
        self.expand, self.act, self.contract = d.nn.Linear(width, 4 * width), d.nn.GELU(), d.nn.Linear(4 * width, width)

    def _split(self, x):
        batch, length, width = x.shape
        return x.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)

    def forward(self, x):
        batch, length, width = x.shape
        h = self.norm1(x)
        q, k, v = self._split(self.query(h)), self._split(self.key(h)), self._split(self.value(h))
        attended = d.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.project(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.contract(self.act(self.expand(self.norm2(x))))


class TinyGPT(d.nn.Module):
    def __init__(self, vocab, context, width=64, heads=4, layers=2):
        super().__init__()
        self.context = context
        self.tokens, self.positions = d.nn.Embedding(vocab, width), d.nn.Embedding(context, width)
        self.blocks = d.nn.Sequential(*(Block(width, heads) for _ in range(layers)))
        self.norm, self.head = d.nn.LayerNorm(width), d.nn.Linear(width, vocab)

    def forward(self, idx):
        length = idx.shape[1]
        x = self.tokens(idx) + self.positions(np.arange(length))
        return self.head(self.norm(self.blocks(x)))


def load_text():
    text = TEXT.read_text(encoding="utf-8")
    chars = sorted(set(text))
    lookup = {c: i for i, c in enumerate(chars)}
    return text, chars, np.array([lookup[c] for c in text], dtype=np.int64)


def batches(data, context, batch, rng):
    while True:
        starts = rng.integers(0, len(data) - context - 1, size=batch)
        x = np.stack([data[s:s + context] for s in starts])
        y = np.stack([data[s + 1:s + context + 1] for s in starts])
        yield x, y


def loss_of(model, x, y):
    logits = model(x)
    return d.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def train(steps, context=32, batch=16, lr=3e-3, seed=0):
    text, chars, data = load_text()
    split = int(len(data) * .9)
    rng = np.random.default_rng(seed)
    d.manual_seed(seed)
    model = TinyGPT(len(chars), context)
    optimizer = d.optim.Adam(model.parameters(), lr=lr)
    stream = batches(data[:split], context, batch, rng)
    held_x, held_y = next(batches(data[split:], context, 64, np.random.default_rng(seed + 1)))
    counts = np.bincount(data[:split], minlength=len(chars)) + 1
    unigram = float(-(counts / counts.sum() * np.log(counts / counts.sum())).sum())
    with d.no_grad():
        start = float(loss_of(model, held_x, held_y).numpy())
    began = time.perf_counter()
    for _ in range(steps):
        x, y = next(stream)
        optimizer.zero_grad()
        loss = loss_of(model, x, y)
        loss.backward()
        optimizer.step()
    seconds = time.perf_counter() - began
    with d.no_grad():
        held = float(loss_of(model, held_x, held_y).numpy())
    return {"characters": len(text), "vocab": len(chars), "parameters": sum(int(np.prod(p.shape)) for p in model.parameters()),
            "steps": steps, "seconds": round(seconds, 2), "uniform_loss": round(math.log(len(chars)), 4),
            "unigram_loss": round(unigram, 4), "heldout_loss_start": round(start, 4), "heldout_loss_end": round(held, 4),
            "sample": sample(model, chars, data[:context], 120)}


def sample(model, chars, prompt, count):
    idx = list(prompt)
    with d.no_grad():
        for _ in range(count):
            logits = model(np.array([idx[-model.context:]])).numpy()[0, -1]
            idx.append(int(logits.argmax()))
    return "".join(chars[i] for i in idx[len(prompt):])


def torch_twin(model, vocab, context):
    """Build the same network in PyTorch with DenseCore's initial weights."""
    import torch
    width = model.tokens.weight.shape[1]
    heads = model.blocks[0].heads

    class TBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1, self.norm2 = torch.nn.LayerNorm(width), torch.nn.LayerNorm(width)
            self.query, self.key, self.value, self.project = (torch.nn.Linear(width, width) for _ in range(4))
            self.expand, self.contract = torch.nn.Linear(width, 4 * width), torch.nn.Linear(4 * width, width)

        def split(self, x):
            b, t, c = x.shape
            return x.reshape(b, t, heads, c // heads).transpose(1, 2)

        def forward(self, x):
            b, t, c = x.shape
            h = self.norm1(x)
            a = torch.nn.functional.scaled_dot_product_attention(self.split(self.query(h)), self.split(self.key(h)),
                                                                 self.split(self.value(h)), is_causal=True)
            x = x + self.project(a.transpose(1, 2).reshape(b, t, c))
            return x + self.contract(torch.nn.functional.gelu(self.expand(self.norm2(x))))

    class TGPT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens, self.positions = torch.nn.Embedding(vocab, width), torch.nn.Embedding(context, width)
            self.blocks = torch.nn.Sequential(*(TBlock() for _ in range(len(model.blocks._modules))))
            self.norm, self.head = torch.nn.LayerNorm(width), torch.nn.Linear(width, vocab)

        def forward(self, idx):
            x = self.tokens(idx) + self.positions(torch.arange(idx.shape[1]))
            return self.head(self.norm(self.blocks(x)))

    twin = TGPT()
    mine = dict(model.named_parameters())
    with torch.no_grad():
        for name, parameter in twin.named_parameters():
            parameter.copy_(torch.from_numpy(mine[name].numpy()))
    return twin


def parity(steps=3, context=16, batch=4, lr=1e-3, seed=0):
    import torch
    torch.set_num_threads(1)
    _, chars, data = load_text()
    d.manual_seed(seed)
    model = TinyGPT(len(chars), context, width=32, heads=2, layers=2)
    twin = torch_twin(model, len(chars), context)
    optimizer = d.optim.Adam(model.parameters(), lr=lr)
    t_optimizer = torch.optim.Adam(twin.parameters(), lr=lr)
    stream = batches(data, context, batch, np.random.default_rng(seed))
    rows = []
    for _ in range(steps):
        x, y = next(stream)
        optimizer.zero_grad()
        loss = loss_of(model, x, y)
        loss.backward()
        optimizer.step()
        t_optimizer.zero_grad()
        logits = twin(torch.from_numpy(x))
        t_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.from_numpy(y.reshape(-1)))
        t_loss.backward()
        t_optimizer.step()
        rows.append((float(loss.numpy()), t_loss.item()))
    t_params = dict(twin.named_parameters())
    max_weight_diff = max(float(np.abs(p.numpy() - t_params[n].detach().numpy()).max()) for n, p in model.named_parameters())
    return {"losses_densecore_vs_torch": rows, "max_weight_abs_diff": max_weight_diff}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()
    print(json.dumps(parity() if args.parity else train(args.steps), indent=1))
