import numpy as np
import pytest
import densecore as d
from densecore._native import _GraphExecutor


def _execute(x, weight, bias, *, compiled, eps=1e-4, features=3):
    if not compiled:
        return d.functional.layer_norm(x, features, weight=weight, bias=bias, eps=eps)
    operands = [x]
    weight_slot = None
    bias_slot = None
    if weight is not None:
        weight_slot = len(operands)
        operands.append(weight)
    if bias is not None:
        bias_slot = len(operands)
        operands.append(bias)
    executor = _GraphExecutor(len(operands), [("layer_norm", 0, weight_slot, bias_slot)],
                              normalization_eps=[eps], normalized_features=[features])
    return d.Tensor._wrap(executor.execute([t._native for t in operands]))


@pytest.mark.parametrize("with_weight,with_bias", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("compiled", [False, True])
def test_registered_layer_norm_strided_affine_vjp_matches_torch(with_weight, with_bias, compiled):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(401)
    arrays = [rng.normal(size=(6, 6)).astype(np.float32),
              rng.normal(size=6).astype(np.float32), rng.normal(size=6).astype(np.float32)]
    x, w, b = [d.tensor(a, requires_grad=True) for a in arrays]
    rx, rw, rb = [torch.tensor(a, requires_grad=True) for a in arrays]
    weight = w[1::2] if with_weight else None
    bias = b[::2] if with_bias else None
    got = _execute(x[1::2, ::2].T, weight, bias, compiled=compiled)
    want = torch.nn.functional.layer_norm(rx[1::2, ::2].T, (3,),
                                         rw[1::2] if with_weight else None,
                                         rb[::2] if with_bias else None, 1e-4)
    seed = rng.normal(size=got.shape).astype(np.float32)
    got.backward(d.tensor(seed))
    want.backward(torch.tensor(seed))
    np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=3e-5, atol=2e-6)
    assert got.dtype == "float32" and got.device == "cpu"
    assert not got.shares_storage(x)
    for actual, reference in [(x, rx), (w, rw), (b, rb)]:
        if reference.grad is None:
            assert actual.grad is None
        else:
            np.testing.assert_allclose(actual.grad.numpy(), reference.grad.numpy(), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("with_weight,with_bias", [(False, False), (True, False), (False, True), (True, True)])
def test_registered_layer_norm_full_jacobian(with_weight, with_bias):
    rng = np.random.default_rng(87)
    x = d.tensor(rng.normal(size=(3, 3)).astype(np.float32), requires_grad=True)
    w = d.tensor([.7, .9, 1.2], requires_grad=True)
    b = d.tensor([.1, -.2, .3], requires_grad=True)
    def fn(a, weight, bias):
        return _execute(a.T, weight if with_weight else None, bias if with_bias else None, compiled=False)
    assert d.autograd.gradcheck(fn, (x, w, b))


@pytest.mark.parametrize("compiled", [False, True])
def test_empty_batch_preserves_affine_gradient_shapes(compiled):
    x = d.tensor(np.empty((0, 3), dtype=np.float32), requires_grad=True)
    w = d.tensor([1., 2., 3.], requires_grad=True)
    b = d.tensor([.1, .2, .3], requires_grad=True)
    y = _execute(x, w, b, compiled=compiled)
    assert y.shape == (0, 3)
    y.backward(d.tensor(np.empty((0, 3), dtype=np.float32)))
    assert x.grad.shape == x.shape
    np.testing.assert_array_equal(w.grad.numpy(), np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(b.grad.numpy(), np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("input_shape,weight_shape,bias_shape", [
    ((3,), None, None), ((1, 2, 3), None, None), ((2, 0), None, None),
    ((2, 3), (1, 3), None), ((2, 3), None, (2,)), ((2, 3), (0,), None),
])
def test_eager_and_compiler_reject_invalid_metadata(input_shape, weight_shape, bias_shape):
    x = d.tensor(np.zeros(input_shape, dtype=np.float32))
    w = None if weight_shape is None else d.tensor(np.zeros(weight_shape, dtype=np.float32))
    b = None if bias_shape is None else d.tensor(np.zeros(bias_shape, dtype=np.float32))
    with pytest.raises(ValueError):
        x._native.layer_norm(None if w is None else w._native, None if b is None else b._native, 1e-4)
    with pytest.raises(ValueError):
        _execute(x, w, b, compiled=True)


@pytest.mark.parametrize("eps", [0., -1., float("nan"), float("inf"), 1e-60, 1e308])
def test_eager_and_compiler_reject_invalid_float32_epsilon(eps):
    x = d.tensor([[1., 2., 3.]])
    with pytest.raises(ValueError):
        x._native.layer_norm(None, None, eps)
    with pytest.raises(ValueError):
        _execute(x, None, None, compiled=True, eps=eps)


def test_normalized_feature_contract_matches_eager_and_compiler():
    x = d.tensor([[1., 2., 3.]])
    for compiled in [False, True]:
        with pytest.raises(ValueError):
            _execute(x, None, None, compiled=compiled, features=2)


@pytest.mark.parametrize("compiled", [False, True])
def test_shared_input_weight_storage_keeps_gradient_identity(compiled):
    x = d.tensor([[.2, .4, .9]], requires_grad=True)
    def fn(a):
        return _execute(a, a.reshape(3), None, compiled=compiled)
    assert x.shares_storage(x.reshape(3))
    assert d.autograd.gradcheck(fn, x)
