import numpy as np
import pytest
import densecore as d


def test_embedding_gathers_rows_for_any_index_rank():
    weight = d.tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
    np.testing.assert_array_equal(d.functional.embedding([3, 0], weight).numpy(), [[9, 10, 11], [0, 1, 2]])
    grid = np.array([[1, 2], [2, 3], [0, 0]])
    out = d.functional.embedding(grid, weight)
    assert out.shape == (3, 2, 3)
    np.testing.assert_array_equal(out.numpy(), weight.numpy()[grid])
    assert d.functional.embedding(np.int64(2), weight).shape == (3,)
    assert d.functional.embedding(np.empty((0, 5), np.int64), weight).shape == (0, 5, 3)


def test_embedding_repeated_indices_scatter_add_gradients():
    weight = d.tensor(np.zeros((5, 2), np.float32), requires_grad=True)
    out = d.functional.embedding([[4, 1], [4, 4]], weight)
    (out * d.tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2))).sum().backward()
    expected = np.zeros((5, 2), np.float32)
    expected[4] = [0 + 4 + 6, 1 + 5 + 7]
    expected[1] = [2, 3]
    np.testing.assert_array_equal(weight.grad.numpy(), expected)


def test_embedding_full_jacobian_on_strided_weight():
    base = d.tensor(np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32), requires_grad=True)
    assert d.autograd.gradcheck(lambda w: d.functional.embedding([[2, 0], [2, 1]], w.T), base)


def test_embedding_matches_torch_values_and_gradients():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(5)
    table = rng.normal(size=(7, 4)).astype(np.float32)
    idx = rng.integers(0, 7, size=(3, 5))
    seed = rng.normal(size=(3, 5, 4)).astype(np.float32)
    w = d.tensor(table, requires_grad=True)
    tw = torch.tensor(table, requires_grad=True)
    out = d.functional.embedding(idx, w)
    ref = torch.nn.functional.embedding(torch.from_numpy(idx), tw)
    np.testing.assert_array_equal(out.numpy(), ref.detach().numpy())
    out.backward(d.tensor(seed))
    ref.backward(torch.from_numpy(seed))
    np.testing.assert_allclose(w.grad.numpy(), tw.grad.numpy(), rtol=1e-6, atol=1e-6)


def test_embedding_module_trains_in_a_model():
    d.manual_seed(2)
    model = d.nn.Sequential(d.nn.Linear(3, 1))
    emb = d.nn.Embedding(6, 3)
    assert emb.weight.shape == (6, 3) and emb.weight.requires_grad
    params = list(emb.parameters()) + list(model.parameters())
    optimizer = d.optim.Adam(params, lr=.05)
    idx = np.array([0, 1, 2, 3, 4, 5])
    target = d.tensor(np.linspace(-1, 1, 6, dtype=np.float32).reshape(6, 1))
    loss_fn = d.nn.MSELoss()
    first = float(loss_fn(model(emb(idx)), target).numpy())
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(emb(idx)), target)
        loss.backward()
        optimizer.step()
    assert float(loss.numpy()) < first * .1
    state = emb.state_dict()
    assert set(state) == {"weight"}


def test_embedding_rejects_bad_indices_and_weights():
    w = d.tensor(np.zeros((3, 2), np.float32))
    for bad, error in [([3], ValueError), ([-1], ValueError), ([1.0], TypeError),
                       ([True], TypeError), (d.tensor([1.]), TypeError)]:
        with pytest.raises(error):
            d.functional.embedding(bad, w)
    with pytest.raises(ValueError, match='shape'):
        d.functional.embedding([0], d.tensor(np.zeros(3, np.float32)))
    with pytest.raises(TypeError):
        d.functional.embedding([0], np.zeros((3, 2)))
    with pytest.raises(ValueError):
        d.nn.Embedding(0, 2)
    with pytest.raises(TypeError):
        d.nn.Embedding(True, 2)
    with pytest.raises(TypeError, match='float32'):
        d.functional.embedding([0], d.tensor(np.zeros((3, 2), bool), dtype='bool'))


def test_embedding_no_grad_and_version_check():
    w = d.nn.Parameter(np.ones((3, 2), np.float32))
    with d.no_grad():
        assert not d.functional.embedding([1], w).requires_grad
    out = d.functional.embedding([1, 2], w).sum()
    w.backward(d.tensor(np.zeros((3, 2), np.float32)))
    d.optim.SGD([w], lr=.1).step()
    with pytest.raises(RuntimeError, match='modified'):
        out.backward()
