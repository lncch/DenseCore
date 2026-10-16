import numpy as np
import pytest
import densecore as dc


def pair():
    dc.manual_seed(881)
    a = dc.nn.Sequential(dc.nn.Linear(3, 5), dc.nn.ReLU(), dc.nn.Linear(5, 4))
    dc.manual_seed(881)
    b = dc.compile(dc.nn.Sequential(dc.nn.Linear(3, 5), dc.nn.ReLU(), dc.nn.Linear(5, 4)))
    return a, b


@pytest.mark.parametrize('optimizer', ['sgd', 'momentum', 'adam'])
def test_cross_entropy_fused_matches_eager(optimizer):
    a, b = pair()
    def make(model):
        if optimizer == 'adam':
            return dc.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.01)
        return dc.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9 if optimizer == 'momentum' else 0.0,
                            weight_decay=0.01)
    oa, ob = make(a), make(b)
    for step in range(8):
        x = dc.tensor(np.random.default_rng(step).normal(size=(4, 3)), dtype="float32")
        labels = np.array([0, 1, 3, step % 4], dtype=np.int64)
        oa.zero_grad()
        expected = dc.functional.cross_entropy(a(x), labels)
        expected.backward()
        oa.step()
        actual = dc.compiled_train_step(b, x, labels, ob, loss='cross_entropy')
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-6)
        for p, q in zip(a.parameters(), b.parameters()):
            np.testing.assert_allclose(p.numpy(), q.numpy(), rtol=1e-4, atol=1e-6)


def test_invalid_labels_preserve_parameters():
    _, model = pair()
    optimizer = dc.optim.Adam(model.parameters(), lr=0.002)
    x = dc.tensor(np.ones((3, 3), dtype=np.float32), dtype="float32")
    before = [p.numpy().copy() for p in model.parameters()]
    for labels in [[0, 1, 4], [-1, 0, 1], [0.5, 1, 2], [0, 1], [2**24, 0, 1]]:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            dc.compiled_train_step(model, x, labels, optimizer, loss='cross_entropy')
        for old, p in zip(before, model.parameters()):
            np.testing.assert_array_equal(old, p.numpy())


def test_mse_default_unchanged():
    _, model = pair()
    optimizer = dc.optim.SGD(model.parameters(), lr=0.01)
    x = dc.tensor(np.ones((2, 3), dtype=np.float32), dtype="float32")
    target = dc.tensor(np.zeros((2, 4), dtype=np.float32), dtype="float32")
    loss = dc.compiled_train_step(model, x, target, optimizer)
    assert loss.shape == ()
