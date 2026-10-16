"""DataLoader cases run under torch or densecore (python _dataloader_cases.py <lib> [case...]);
tests/test_dataloader_exact.py compares the JSON. Workers need module-level classes (spawn)."""
import importlib, random, sys
import numpy as np


class RandMap:
    """Map dataset whose items mix the index with torch/np/python random draws and worker info."""
    def __init__(self, lib, n=23):
        self.lib, self.n = lib, n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        T = importlib.import_module(self.lib)
        w = T.utils.data.get_worker_info()
        return {"i": i, "t": T.rand(3), "np": float(np.random.rand()), "py": random.random(),
                "w": -1 if w is None else w.id, "seed": -1 if w is None else w.seed}


class RandIter:
    def __init__(self, lib, n=17):
        self.lib, self.n = lib, n
    def __iter__(self):
        T = importlib.import_module(self.lib)
        w = T.utils.data.get_worker_info()
        lo, step = (0, 1) if w is None else (w.id, w.num_workers)
        for i in range(lo, self.n, step):
            yield (i, T.randn(2))


import torch.utils.data as _TD
import densecore.utils.data as _DD


class RandIterT(RandIter, _TD.IterableDataset):
    pass


class RandIterD(RandIter, _DD.IterableDataset):
    pass


def init_fn(worker_id):
    random.seed(1000 + worker_id)


def flatten(x):
    if hasattr(x, "numpy"):
        return [x.numpy().dtype.str, x.numpy().tobytes().hex()]
    if isinstance(x, dict):
        return {k: flatten(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [flatten(v) for v in x]
    return x


def run(lib, case):
    T = importlib.import_module(lib)
    D = T.utils.data
    T.manual_seed(42); random.seed(7); np.random.seed(7)
    kind, kw, epochs = case
    if kind == "map":
        ds = RandMap(lib)
    elif kind == "iter":
        ds = (RandIterT if lib == "torch" else RandIterD)(lib)
    elif kind == "split":
        a, b = D.random_split(D.TensorDataset(T.arange(20)), [0.7, 0.3])
        ds = a
    kw = dict(kw)
    if kw.pop("gen", False):
        kw["generator"] = T.Generator().manual_seed(3)
    if kw.pop("subset_sampler", False):
        kw["sampler"] = D.SubsetRandomSampler(list(range(0, 23, 2)))
    if kw.pop("weighted", False):
        kw["sampler"] = D.WeightedRandomSampler([0.1, 0.5, 2.0, 1.0] * 5, 12)
    loader = D.DataLoader(ds, **kw)
    out = []
    for _ in range(epochs):
        out.append([flatten(b) for b in loader])
    out.append(["after", T.rand(2).numpy().tobytes().hex(), random.random(), float(np.random.rand())])
    return out


CASES = {
    "plain": ("map", dict(batch_size=4), 1),
    "shuffle": ("map", dict(batch_size=5, shuffle=True), 2),
    "shuffle_gen": ("map", dict(batch_size=5, shuffle=True, gen=True), 2),
    "drop_last_none": ("map", dict(batch_size=None, shuffle=True), 1),
    "workers2": ("map", dict(batch_size=4, shuffle=True, num_workers=2), 2),
    "workers3_init": ("map", dict(batch_size=3, num_workers=3, worker_init_fn=init_fn, drop_last=True), 1),
    "persistent": ("map", dict(batch_size=4, shuffle=True, num_workers=2, persistent_workers=True), 3),
    "prefetch5": ("map", dict(batch_size=2, num_workers=2, prefetch_factor=5), 1),
    "iter0": ("iter", dict(batch_size=4), 1),
    "iter_workers": ("iter", dict(batch_size=3, num_workers=2), 2),
    "iter_workers_none": ("iter", dict(batch_size=None, num_workers=3), 1),
    "split": ("split", dict(batch_size=4, shuffle=True), 1),
    "subset_sampler": ("map", dict(batch_size=4, subset_sampler=True), 1),
    "weighted": ("map", dict(batch_size=4, weighted=True), 1),
}

if __name__ == "__main__":
    import json
    lib = sys.argv[1]
    names = sys.argv[2:] or list(CASES)
    print(json.dumps({n: run(lib, CASES[n]) for n in names}))
