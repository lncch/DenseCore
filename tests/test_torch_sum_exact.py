import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')

CASES = [((7,), None), ((100000,), None), ((300001,), None), ((5, 40000), (0,)), ((100, 1000), (0,)),
         ((768, 128), (0,)), ((768, 128), (1,)), ((12, 64, 128), (0, 1)), ((12, 4, 64, 32), (0, 2)),
         ((2, 3, 4, 5, 6), (0, 2, 4)), ((5, 1, 7), (0,)), ((40000, 3), (0,)), ((33, 1025), (1,))]


@pytest.mark.parametrize('threads', [1, 4, 7])
@pytest.mark.parametrize('shape,axes', CASES)
def test_sum_and_mean_match_torch_bits(shape, axes, threads):
    rng = np.random.default_rng(len(shape) * 31 + threads)
    x = (rng.normal(size=shape) * rng.choice([1, 1e3, 1e-3], size=shape)).astype(np.float32)
    before = (d.get_num_threads(), torch.get_num_threads())
    d.set_num_threads(threads)
    torch.set_num_threads(threads)
    try:
        t, ours = torch.from_numpy(x), d.tensor(x)
        for f in ('sum', 'mean'):
            want = getattr(t, f)() if axes is None else getattr(t, f)(axes)
            got = getattr(ours, f)() if axes is None else getattr(ours, f)(axes)
            assert got.numpy().tobytes() == want.numpy().tobytes(), f
    finally:
        d.set_num_threads(before[0])
        torch.set_num_threads(before[1])


def test_thread_count_api():
    with pytest.raises(ValueError):
        d.set_num_threads(0)
    assert d.get_num_threads() >= 1
