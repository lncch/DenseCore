"""Fused LayerNorm training parity and validation."""
import numpy as np
import pytest
import densecore as dc


def _models(*, affine=True, bias=True, hidden=False):
    def make():
        modules = [dc.nn.Linear(4, 4)]
        modules += [dc.nn.LayerNorm(4, eps=1e-5,
                                    elementwise_affine=affine, bias=bias)]
        if hidden:
            modules += [dc.nn.ReLU(), dc.nn.Linear(4, 3)]
        else:
            modules += [dc.nn.ReLU(), dc.nn.Linear(4, 4)]
        return dc.nn.Sequential(*modules)
    dc.manual_seed(37)
    eager = make()
    dc.manual_seed(37)
    fused = dc.compile(make())
    return eager, fused


def _compare(a, b):
    if isinstance(a, dc.Tensor):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=3e-4, atol=3e-6)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _compare(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _compare(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('optimizer_kind', ['sgd', 'momentum', 'adam'])
@pytest.mark.parametrize('affine,bias', [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize('loss_name', ['mse', 'cross_entropy'])
def test_layernorm_training_matches_eager(optimizer_kind, affine, bias, loss_name):
    eager, fused = _models(affine=affine, bias=bias,
                            hidden=loss_name == 'cross_entropy')
    def optimizer(model):
        if optimizer_kind == 'adam':
            return dc.optim.Adam(model.parameters(), lr=.001, weight_decay=.01)
        return dc.optim.SGD(model.parameters(), lr=.005,
                            momentum=.8 if optimizer_kind == 'momentum' else 0,
                            weight_decay=.01)
    eo, fo = optimizer(eager), optimizer(fused)
    initial_ids = tuple(map(id, fused.parameters()))
    for step in range(6):
        rng = np.random.default_rng(500 + step)
        x = dc.tensor(rng.normal(size=(3 + step % 2, 4)).astype('float32'))
        if loss_name == 'mse':
            target = dc.tensor(rng.normal(size=(x.shape[0], 4)).astype('float32'))
            eo.zero_grad()
            expected = dc.functional.mse_loss(eager(x), target)
            expected.backward()
            eo.step()
        else:
            labels = rng.integers(0, 3, size=x.shape[0], dtype=np.int64)
            # Match the existing eager cross-entropy API used elsewhere.
            eo.zero_grad()
            expected = dc.functional.cross_entropy(eager(x), labels)
            expected.backward()
            eo.step()
            target = labels
        actual = dc.compiled_train_step(fused, x, target, fo, loss=loss_name)
        _compare(expected, actual)
        _compare(eager.state_dict(), fused.state_dict())
        _compare(eo.state_dict(), fo.state_dict())
        assert tuple(map(id, fused.parameters())) == initial_ids


@pytest.mark.parametrize('mutation', ['extra_parameter', 'extra_child',
                                     'registration_order', 'patched_forward'])
def test_layernorm_rejection_does_not_update_parameters(mutation):
    _, model = _models()
    opt = dc.optim.Adam(model.parameters())
    x = dc.tensor(np.arange(8, dtype=np.float32).reshape(2, 4))
    target = dc.tensor(np.zeros((2, 4), dtype=np.float32))
    dc.compiled_train_step(model, x, target, opt)
    norm = model._modules['1']
    if mutation == 'extra_parameter':
        norm.extra = dc.nn.Parameter([1.0])
    elif mutation == 'extra_child':
        norm.extra = dc.nn.ReLU()
    elif mutation == 'registration_order':
        norm._parameters = {'bias': norm.bias, 'weight': norm.weight}
    else:
        norm.forward = lambda x: x
    before = model.state_dict()
    state = opt.state_dict()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        dc.compiled_train_step(model, x, target, opt)
    _compare(before, model.state_dict())
    _compare(state, opt.state_dict())


def test_layernorm_optimizer_failure_is_atomic():
    _, model = _models()
    opt = dc.optim.Adam(model.parameters())
    x = dc.tensor(np.arange(8, dtype=np.float32).reshape(2, 4))
    target = dc.tensor(np.zeros((2, 4), dtype=np.float32))
    dc.compiled_train_step(model, x, target, opt)
    state = opt.state_dict()
    state['state'][-1]['step'] = 2**64 - 1
    opt.load_state_dict(state)
    before = model.state_dict()
    state = opt.state_dict()
    with pytest.raises(RuntimeError, match='overflow'):
        dc.compiled_train_step(model, x, target, opt)
    _compare(before, model.state_dict())
    _compare(state, opt.state_dict())
