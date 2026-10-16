"""Train nanoGPT in torch and DenseCore from the same weights; compare bits every step.

Usage: PYTHONPATH=python python scripts/nanogpt_exact.py <nanoGPT dir> [steps] [dropout]
Exits 1 at the first step where the loss, a gradient or a weight differs."""
import sys, os, importlib.util, numpy as np, torch, densecore
S = sys.argv[1]  # nanoGPT checkout with data/shakespeare_char/train.bin prepared
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 50

def load(name, fake):
    saved = {k: sys.modules.get(k) for k in ("torch", "torch.nn", "torch.nn.functional")}
    if fake:
        sys.modules.update({"torch": densecore, "torch.nn": densecore.nn, "torch.nn.functional": densecore.functional})
    spec = importlib.util.spec_from_file_location(name, os.path.join(S, "model.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for k, v in saved.items():
        if v is not None: sys.modules[k] = v
    return mod

T, D = load("model_t", False), load("model_d", True)
DROPOUT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
cfg = dict(block_size=64, vocab_size=65, n_layer=4, n_head=4, n_embd=128, dropout=DROPOUT, bias=False)
torch.manual_seed(1337); densecore.manual_seed(1337)
mt = T.GPT(T.GPTConfig(**cfg))
md = D.GPT(D.GPTConfig(**cfg))
tparams = dict(mt.named_parameters()); dparams = dict(md.named_parameters())
assert tparams.keys() == dparams.keys(), (tparams.keys() ^ dparams.keys())
init_off = [k for k in tparams if tparams[k].detach().numpy().tobytes() != dparams[k].numpy().tobytes()]
print("initial weights from manual_seed(1337):", "identical" if not init_off else f"{len(init_off)} differ: {init_off[:4]}")
if init_off:  # still compare training from the same start
    with densecore.no_grad():
        for k, p in dparams.items():
            p.copy_(densecore.tensor(tparams[k].detach().numpy()))
torch.set_num_threads(densecore.get_num_threads())
ot = mt.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")
od = md.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")
data = np.memmap(os.path.join(S, "data/shakespeare_char/train.bin"), dtype=np.uint16, mode="r")
rng = np.random.default_rng(0)

def bits(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return int((a.view(np.uint32) != b.view(np.uint32)).sum())

for step in range(STEPS):
    ix = rng.integers(0, len(data) - 65, 12)
    x = np.stack([data[i:i + 64] for i in ix]).astype(np.int64); y = np.stack([data[i + 1:i + 65] for i in ix]).astype(np.int64)
    _, lt = mt(torch.from_numpy(x), torch.from_numpy(y)); lt.backward()
    _, ld = md(densecore.tensor(x), densecore.tensor(y)); ld.backward()
    gdiff = {k: bits(tparams[k].grad.numpy(), dparams[k].grad.numpy()) for k in tparams}
    nt = torch.nn.utils.clip_grad_norm_(mt.parameters(), 1.0); nd = densecore.nn.utils.clip_grad_norm_(md.parameters(), 1.0)
    ot.step(); od.step(); ot.zero_grad(set_to_none=True); od.zero_grad(set_to_none=True)
    pdiff = {k: bits(tparams[k].detach().numpy(), dparams[k].numpy()) for k in tparams}
    bad_g = {k: v for k, v in gdiff.items() if v}; bad_p = {k: v for k, v in pdiff.items() if v}
    print(f"step {step}: loss {lt.item()!r} vs {ld.item()!r} lossdiff={bits(lt.item(), ld.item())} norm {bits(nt.item(), nd.item())} grads_off={len(bad_g)} params_off={len(bad_p)}")
    if bad_g or bad_p:
        print("  grads:", bad_g, "ok:", [k for k in gdiff if not gdiff[k]]); print("  params:", dict(list(bad_p.items())[:6]))
        sys.exit(1)
print(f"all {STEPS} steps bit-identical")
