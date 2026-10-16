"""torch.Generator, manual_seed and the default CPU stream, bit for bit with PyTorch's.

The engine is PyTorch's mt19937 with its sampling transforms (src/torch_random.rs). float32 and
float64 rand/randn/randint/randperm/normal_/uniform_/bernoulli_/exponential_/geometric_/
cauchy_/log_normal_/multinomial and dropout masks match PyTorch's values for the same seed and
the same call order. Older call sites that ask for other NumPy-style draws (choice, ...) get a
NumPy generator seeded from this stream: deterministic, but not PyTorch's values.
"""
import operator
import os

import numpy as np

from ._native import _Generator


def _seed_value(seed):
    """torch.manual_seed's accepted range; negatives wrap to 0xffff_ffff_ffff_ffff + seed + 1."""
    if isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be an integer")
    seed = operator.index(seed)
    if not -0x8000_0000_0000_0000 <= seed <= 0xFFFF_FFFF_FFFF_FFFF:
        raise RuntimeError("Overflow when unpacking long long")
    return seed & 0xFFFF_FFFF_FFFF_FFFF


def _count(shape):
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


class Generator:
    """CPU generator with PyTorch's engine; a new one starts from PyTorch's default seed."""

    def __init__(self, device="cpu"):
        if str(device) != "cpu":
            raise NotImplementedError("only CPU generators are supported")
        self.device = "cpu"
        self._native = _Generator()

    def manual_seed(self, seed):
        self._native.manual_seed(_seed_value(seed))
        return self

    def initial_seed(self):
        return self._native.initial_seed()

    def seed(self):
        """Reseed from the operating system and return the new seed (like torch)."""
        seed = int.from_bytes(os.urandom(8), "little") & 0x7FFF_FFFF_FFFF_FFFF
        self._native.manual_seed(seed)
        return seed

    def get_state(self):
        """Engine state as a uint8 Tensor (DenseCore's layout, not PyTorch's byte format)."""
        from .tensor import Tensor
        return Tensor(np.array(self._native.get_state(), np.uint64).view(np.uint8), dtype="uint8")

    def set_state(self, state):
        array = state.numpy() if hasattr(state, "numpy") else np.asarray(state)
        array = np.ascontiguousarray(array, np.uint8)
        if array.size % 8:
            raise ValueError("invalid generator state")
        self._native.set_state(array.view(np.uint64).tolist())
        return self

    # ---- samplers used by DenseCore's ops (values in PyTorch's order) ----

    def _uniform(self, shape, low=0.0, high=1.0, dtype="float32"):
        double = dtype == "float64"
        values = np.frombuffer(self._native.uniform(_count(shape), float(low), float(high), double),
                               np.float64 if double else np.float32).copy()
        return values.reshape(shape)

    def _normal(self, shape, mean=0.0, std=1.0, dtype="float32"):
        double = dtype == "float64"
        values = np.frombuffer(self._native.normal(_count(shape), float(mean), float(std), double),
                               np.float64 if double else np.float32).copy()
        return values.reshape(shape)

    def _randint(self, shape, low, high):
        return np.frombuffer(self._native.randint(_count(shape), int(low), int(high)), np.int64).copy().reshape(shape)

    def _randperm(self, n):
        return np.frombuffer(self._native.randperm(int(n)), np.int64).copy()

    def _bernoulli(self, shape, p):
        return np.frombuffer(self._native.bernoulli(_count(shape), float(p)), np.float32).copy().reshape(shape)

    def _sample(self, kind, shape, a=0.0, b=0.0, dtype="float32"):
        """exponential / geometric / cauchy / log_normal: drawn in double, cast like PyTorch."""
        values = np.frombuffer(self._native.sample(kind, _count(shape), float(a), float(b)), np.float64)
        return values.astype(np.dtype(dtype)).reshape(shape)

    def _uniform_doubles(self, n):
        return np.frombuffer(self._native.uniform_doubles(int(n)), np.float64)

    def _numpy(self):
        """A NumPy generator seeded from this stream, for distributions without a port."""
        return np.random.default_rng(self._native.random64())


default_generator = Generator()


def _gen(generator=None):
    if generator is None:
        return default_generator
    if not isinstance(generator, Generator):
        raise TypeError("generator must be a densecore Generator")
    return generator


def manual_seed(seed):
    """Seed the default generator; the same seed gives PyTorch's CPU values."""
    return default_generator.manual_seed(seed)


def initial_seed():
    return default_generator.initial_seed()


def seed():
    return default_generator.seed()


def get_rng_state():
    return default_generator.get_state()


def set_rng_state(new_state):
    default_generator.set_state(new_state)


class _NumpyLike:
    """The NumPy Generator methods DenseCore's older call sites use, drawn from a Generator.
    Float results are float32, as PyTorch fills float32 tensors."""

    def __init__(self, generator=None):
        self._generator = generator

    @property
    def _g(self):
        return _gen(self._generator)

    @staticmethod
    def _shape(size):
        if size is None:
            return ()
        return tuple(size) if isinstance(size, (tuple, list)) else (operator.index(size),)

    @staticmethod
    def _scalar(values, size):
        return values[()] if size is None else values

    def random(self, size=None):
        return self._scalar(self._g._uniform(self._shape(size)), size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._scalar(self._g._uniform(self._shape(size), low, high), size)

    def standard_normal(self, size=None):
        return self._scalar(self._g._normal(self._shape(size)), size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if np.ndim(loc) or np.ndim(scale):
            # torch.normal(Tensor, Tensor): normal_(0, 1), then mul_(std).add_(mean) in float32.
            shape = self._shape(size) if size is not None else np.broadcast_shapes(np.shape(loc), np.shape(scale))
            z = self._g._normal(shape)
            return z * np.asarray(scale, np.float32) + np.asarray(loc, np.float32)
        return self._scalar(self._g._normal(self._shape(size), loc, scale), size)

    def integers(self, low, high=None, size=None):
        low, high = (0, low) if high is None else (low, high)
        return self._scalar(self._g._randint(self._shape(size), low, high), size)

    def permutation(self, n):
        if isinstance(n, (int, np.integer)):
            return self._g._randperm(n)
        array = np.asarray(n)
        return array[self._g._randperm(len(array))]

    def __getattr__(self, name):
        # shortcut: unported distributions (exponential, geometric, choice, ...) use NumPy
        # seeded from this stream; upgrade when a test needs PyTorch's exact values.
        return getattr(self._g._numpy(), name)


def rng(generator=None):
    return _NumpyLike(generator)
