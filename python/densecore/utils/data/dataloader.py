"""torch.utils.data.DataLoader for DenseCore, following PyTorch 2.14.1's dataloader.py.

The constructor checks, default samplers, collation, `len()` and every random draw (the
iterator's `_base_seed`, then RandomSampler's seed on the first batch) follow PyTorch, so the
same seed gives the same batches. Workers (`num_workers > 0`) run in this process, one after
another, but each keeps PyTorch's per-worker state: the batch-to-worker assignment, seeds
`base_seed + worker_id` for torch/random and `_generate_state` for NumPy, one intra-op
thread, `get_worker_info()`, `worker_init_fn`, persistent workers. Results arrive in task
order. Differences: no processes, so no speedup, no `timeout`, and the dataset is shared with
the workers instead of copied into them; `pin_memory` warns and does nothing (no accelerator).
Based on PyTorch's code (BSD-3-Clause); see THIRD_PARTY_LICENSES/pytorch-2.14.1.
"""
import contextlib
import itertools
import math
import os
import random
import warnings

import numpy as np

import densecore as torch

from . import _utils
from .dataset import IterableDataset
from .sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler
from ._utils.collate import default_collate, default_convert

__all__ = ["DataLoader", "default_collate", "default_convert", "get_worker_info"]


class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class _InfiniteConstantSampler(Sampler):
    """Analogous to ``itertools.repeat(None, None)``; the sampler for an IterableDataset."""

    def __iter__(self):
        while True:
            yield None


class WorkerInfo:
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError(f"Cannot assign attributes to {self.__class__.__name__} objects")
        return super().__setattr__(key, val)

    def __repr__(self):
        items = [f"{k}={getattr(self, k)}" for k in self.__keys]
        return f"{self.__class__.__name__}({', '.join(items)})"


_worker_info = None


def get_worker_info():
    """The current worker's WorkerInfo inside a DataLoader worker, else None."""
    return _worker_info


def _generate_state(base_seed, worker_id):
    """PyTorch's NumPy seed for a worker (torch/utils/data/_utils/worker.py)."""
    INIT_A, MULT_A, INIT_B, MULT_B = 0x43B0D7E5, 0x931E8875, 0x8B51F9DD, 0x58F38DED
    MIX_MULT_L, MIX_MULT_R, XSHIFT, MASK32 = 0xCA01F9DD, 0x4973F715, 4 * 8 // 2, 0xFFFFFFFF
    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4
    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        return (value ^ (value >> XSHIFT)) & MASK32

    def mix(x, y):
        result_x, result_y = (MIX_MULT_L * x) & MASK32, (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        return (result ^ (result >> XSHIFT)) & MASK32

    for i in range(len(pool)):
        pool[i] = hash(entropy[i])
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))
    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        state.append((data_val ^ (data_val >> XSHIFT)) & MASK32)
    return state


class _IterableDatasetStopIteration:
    def __init__(self, worker_id):
        self.worker_id = worker_id


class _Worker:
    """One worker's state, entered around everything the worker would run."""

    def __init__(self, loader, worker_id, base_seed):
        self.id, self.loader = worker_id, loader
        seed = base_seed + worker_id
        saved = self._save()
        try:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(_generate_state(base_seed, worker_id))
            self.state = self._save()
        finally:
            self._restore(saved)
        self.info = WorkerInfo(id=worker_id, num_workers=loader.num_workers, seed=seed, dataset=loader.dataset)
        with self.context():
            if loader.worker_init_fn is not None:
                loader.worker_init_fn(worker_id)
            self.reset()

    @staticmethod
    def _save():
        return (torch.default_generator.get_state(), random.getstate(), np.random.get_state(), torch.get_num_threads())

    @staticmethod
    def _restore(state):
        torch.default_generator.set_state(state[0])
        random.setstate(state[1])
        np.random.set_state(state[2])
        torch.set_num_threads(state[3])

    @contextlib.contextmanager
    def context(self):
        global _worker_info
        saved, saved_info = self._save(), _worker_info
        self._restore(self.state)
        torch.set_num_threads(1)
        _worker_info = self.info
        try:
            yield
        finally:
            self.state = self._save()
            _worker_info = saved_info
            self._restore(saved)

    def reset(self):
        """Start (or resume, for persistent workers) one pass over the dataset."""
        loader = self.loader
        self.iteration_end = False
        self.fetcher = _DatasetKind.create_fetcher(loader._dataset_kind, loader.dataset, loader._auto_collation,
                                                   loader.collate_fn, loader.drop_last)

    def run(self, index):
        """The worker loop body for one task: None when the worker ignores it (after the end)."""
        if self.iteration_end:
            return None
        with self.context():
            try:
                return (self.fetcher.fetch(index),)
            except StopIteration:
                if self.loader._dataset_kind == _DatasetKind.Iterable:
                    self.iteration_end = True
                    return (_IterableDatasetStopIteration(self.id),)
                raise


class _BaseDataLoaderIter:
    def __init__(self, loader):
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        if loader.pin_memory:
            warnings.warn("'pin_memory' argument is set as true but no accelerator is found, "
                          "then device pinned memory won't be used.", stacklevel=2)
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0

    def __iter__(self):
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)

    def _next_data(self):
        raise NotImplementedError

    def __next__(self):
        data = self._next_data()
        self._num_yielded += 1
        if (self._dataset_kind == _DatasetKind.Iterable and self._IterableDataset_len_called is not None
                and self._num_yielded > self._IterableDataset_len_called):
            warn_msg = (f"Length of IterableDataset {self._dataset} was reported to be "
                        f"{self._IterableDataset_len_called}(when accessing len(dataloader)), but "
                        f"{self._num_yielded} samples have been fetched. ")
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                             "IterableDataset replica at each worker. Please see "
                             "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg, stacklevel=2)
        return data

    def __len__(self):
        return len(self._index_sampler)

    def __getstate__(self):
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()
        return self._dataset_fetcher.fetch(index)


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    """PyTorch's multiprocess iterator with the workers run in place, results in task order."""

    def __init__(self, loader):
        super().__init__(loader)
        self._prefetch_factor = loader.prefetch_factor
        self._in_order = loader.in_order
        self._workers = [_Worker(loader, i, self._base_seed) for i in range(self._num_workers)]
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0
        self._rcvd_idx = 0
        self._task_info = {}
        self._tasks_outstanding = 0
        self._arrived = []  # task ids whose results have "arrived", in arrival order
        self._results = {}
        self._workers_status = [True] * self._num_workers
        self._workers_num_tasks = [0] * self._num_workers
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        if not first_iter:
            for worker in self._workers:
                with worker.context():
                    worker.reset()
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _get_data(self):
        idx = self._arrived.pop(0)
        return idx, self._results.pop(idx)

    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info.get(self._rcvd_idx, None)
                if info:
                    worker_id = info[0]
                    if len(info) == 2 or self._workers_status[worker_id]:
                        break
                    del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                raise StopIteration
            if len(self._task_info[self._rcvd_idx]) == 2:
                worker_id, data = self._task_info.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                return self._process_data(data, worker_id)
            if not self._arrived:
                raise RuntimeError("DataLoader worker emulation lost a task")
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if isinstance(data, _IterableDatasetStopIteration):
                self._workers_status[data.worker_id] = False
                self._try_put_index()
                continue
            if idx != self._rcvd_idx:
                self._task_info[idx] += (data,)
            else:
                worker_id = self._task_info.pop(idx)[0]
                self._rcvd_idx += 1
                return self._process_data(data, worker_id)

    def _try_put_index(self):
        max_tasks = self._prefetch_factor * self._num_workers
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                if self._in_order:
                    break
                if self._workers_num_tasks[worker_queue_idx] < max_tasks // sum(self._workers_status):
                    break
        else:
            return
        # The worker runs the task now; its result "arrives" in task order.
        result = self._workers[worker_queue_idx].run(index)
        if result is not None:
            self._results[self._send_idx] = result[0]
            self._arrived.append(self._send_idx)
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._workers_num_tasks[worker_queue_idx] += 1
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data, worker_idx):
        self._workers_num_tasks[worker_idx] -= 1
        self._try_put_index()
        return data


class DataLoader:
    """torch.utils.data.DataLoader (see the module docstring for what differs)."""

    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor=None, persistent_workers=False,
                 pin_memory_device="", in_order=True):
        if num_workers < 0:
            raise ValueError("num_workers option should be non-negative; "
                             "use num_workers=0 to disable multiprocessing.")
        if timeout < 0:
            raise ValueError("timeout option should be non-negative")
        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError("prefetch_factor option could only be specified in multiprocessing."
                             "let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.")
        elif num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError("prefetch_factor option should be non-negative")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs num_workers > 0")
        if multiprocessing_context is not None and num_workers == 0:
            raise ValueError("multiprocessing_context can only be used with multi-process loading "
                             f"(num_workers > 0), but got num_workers={num_workers}")

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.in_order = in_order

        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            if shuffle not in {False, None}:
                raise ValueError("DataLoader with IterableDataset: expected unspecified shuffle option, "
                                 f"but got shuffle={shuffle}")
            if sampler is not None:
                raise ValueError("DataLoader with IterableDataset: expected unspecified sampler option, "
                                 f"but got sampler={sampler}")
            elif batch_sampler is not None:
                raise ValueError("DataLoader with IterableDataset: expected unspecified "
                                 f"batch_sampler option, but got batch_sampler={batch_sampler}")
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError("batch_sampler option is mutually exclusive with batch_size, shuffle, "
                                 "sampler, and drop_last")
            batch_size = None
            drop_last = False
        elif batch_size is None:
            if drop_last:
                raise ValueError("batch_size=None option disables auto-batching and is mutually exclusive "
                                 "with drop_last")

        if sampler is None:
            if self._dataset_kind == _DatasetKind.Iterable:
                sampler = _InfiniteConstantSampler()
            elif shuffle:
                sampler = RandomSampler(dataset, generator=generator)
            else:
                sampler = SequentialSampler(dataset)
        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator
        if collate_fn is None:
            collate_fn = default_collate if self._auto_collation else default_convert
        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers
        self.__initialized = True
        self._IterableDataset_len_called = None
        self._iterator = None
        self.check_worker_number_rationality()

    def _get_iterator(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        return _MultiProcessingDataLoaderIter(self)

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ("batch_size", "batch_sampler", "sampler", "drop_last", "dataset",
                                           "persistent_workers"):
            raise ValueError(f"{attr} attribute should not be set after {self.__class__.__name__} is initialized")
        super().__setattr__(attr, val)

    def __iter__(self):
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        return self.batch_sampler if self._auto_collation else self.sampler

    def __len__(self):
        if self._dataset_kind == _DatasetKind.Iterable:
            length = self._IterableDataset_len_called = len(self.dataset)
            if self.batch_size is not None:
                length = length // self.batch_size if self.drop_last else math.ceil(length / self.batch_size)
            return length
        return len(self._index_sampler)

    def check_worker_number_rationality(self):
        if not self.num_workers:
            return
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        if cpus is not None and self.num_workers > cpus:
            warnings.warn(f"This DataLoader will create {self.num_workers} worker processes in total. Our "
                          f"suggested max number of worker in current system is {cpus}, which is smaller than "
                          "what this DataLoader is going to create. Please be aware that excessive worker "
                          "creation might get DataLoader running slow or even freeze, lower the worker number "
                          "to avoid potential slowness/freeze if necessary.", stacklevel=2)
