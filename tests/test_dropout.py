import numpy as np
import pytest
import densecore as d


def test_dropout_scales_survivors_and_rate_is_close_to_p():
    d.manual_seed(0)
    x = d.tensor(np.ones((200, 100), np.float32))
    out = d.functional.dropout(x, .3).numpy()
    kept = out != 0
    np.testing.assert_allclose(out[kept], 1 / .7, rtol=1e-6)
    assert abs(1 - kept.mean() - .3) < .01


def test_dropout_gradient_uses_the_same_mask():
    d.manual_seed(1)
    x = d.tensor(np.full((8, 8), 2., np.float32), requires_grad=True)
    out = d.functional.dropout(x, .5)
    out.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), out.numpy() / 2.)


def test_dropout_is_reproducible_under_manual_seed():
    x = d.tensor(np.ones((4, 5), np.float32))
    d.manual_seed(7)
    first = d.functional.dropout(x, .5).numpy()
    d.manual_seed(7)
    np.testing.assert_array_equal(d.functional.dropout(x, .5).numpy(), first)


def test_dropout_module_eval_identity_and_edge_probabilities():
    layer = d.nn.Dropout(.4)
    x = d.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
    layer.eval()
    assert layer(x) is x
    layer.train()
    assert layer.training
    assert d.functional.dropout(x, 0.) is x
    zeros = d.functional.dropout(x, 1.)
    np.testing.assert_array_equal(zeros.numpy(), np.zeros((2, 3)))
    zeros.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.zeros((2, 3)))
    model = d.nn.Sequential(d.nn.Linear(3, 3), d.nn.Dropout(.5))
    model.eval()
    assert not model[1].training


def test_dropout_rejects_bad_arguments():
    x = d.tensor([1.])
    for bad in (-.1, 1.5):
        with pytest.raises(ValueError):
            d.nn.Dropout(bad)
    with pytest.raises(TypeError):
        d.nn.Dropout(True)
    with pytest.raises(TypeError):
        d.functional.dropout(x, .5, training=1)
    with pytest.raises(TypeError):
        d.functional.dropout([1.], .5)


def test_compiled_model_with_dropout_runs_eagerly():
    d.manual_seed(3)
    model = d.compile(d.nn.Sequential(d.nn.Linear(4, 4), d.nn.Dropout(.5)))
    x = d.tensor(np.ones((3, 4), np.float32))
    model.eval()
    expected = model[0](x).numpy()
    np.testing.assert_array_equal(model(x).numpy(), expected)
