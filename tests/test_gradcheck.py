import numpy as np
import pytest
import densecore as d
from densecore.autograd import gradcheck, GradcheckError


def test_full_jacobian_through_chained_views_and_broadcast():
    x = d.tensor(np.arange(12, dtype=np.float32).reshape(3, 4) / 10, requires_grad=True, dtype="float32")
    y = d.tensor([[.2], [.4]], requires_grad=True)
    assert gradcheck(lambda a, b: (a.T[1:4:2, :] * b).softmax(1), (x, y))


def test_scalar_empty_and_frozen_inputs():
    x = d.tensor(.3, requires_grad=True)
    scale = d.tensor(2.)
    assert gradcheck(lambda a, b: a * a * b, (x, scale))
    empty = d.tensor(np.empty((0, 3), dtype=np.float32), requires_grad=True, dtype="float32")
    assert gradcheck(lambda a: a.T, (empty,))


def test_wrong_detached_gradient_is_detected():
    x = d.tensor([1., 2.], requires_grad=True)
    with pytest.raises(GradcheckError, match="Jacobian mismatch"):
        gradcheck(lambda a: a.detach() * a, (x,))
    assert not gradcheck(lambda a: a.detach() * a, (x,), raise_exception=False)


def test_original_inputs_and_gradients_are_untouched():
    x = d.tensor([.3, .7], requires_grad=True)
    (x * 3).sum().backward()
    assert gradcheck(lambda a: a * a, x)
    np.testing.assert_array_equal(x.grad.numpy(), [3., 3.])
    np.testing.assert_array_equal(x.numpy(), np.array([.3, .7], dtype=np.float32))


def test_disconnected_and_constant_outputs():
    x = d.tensor([.3, .7], requires_grad=True)
    y = d.tensor([.2], requires_grad=True)
    assert gradcheck(lambda a, b: a * a, (x, y))
    assert gradcheck(lambda a: d.tensor(2.), (x,))


@pytest.mark.parametrize("kwargs", [{"eps": 0}, {"eps": True}, {"atol": -1}, {"rtol": float("nan")}, {"raise_exception": 1}])
def test_bad_options(kwargs):
    with pytest.raises((TypeError, ValueError)):
        gradcheck(lambda a: a * a, d.tensor(.2, requires_grad=True), **kwargs)


def test_invalid_inputs_outputs_and_unresolvable_steps():
    with pytest.raises(ValueError): gradcheck(lambda a: a, d.tensor(1.))
    with pytest.raises(TypeError): gradcheck(lambda a: a, [1.])
    x = d.tensor(.2, requires_grad=True)
    with pytest.raises(TypeError): gradcheck(lambda a: a.numpy(), x)
    with pytest.raises(ValueError, match="finite"): gradcheck(lambda a: a, d.tensor(float("inf"), requires_grad=True, dtype="float32"))
    with pytest.raises(ValueError, match="resolve"): gradcheck(lambda a: a, d.tensor(1e20, requires_grad=True))


def test_full_jacobian_does_not_hide_cancelling_errors():
    x = d.tensor([.4, .7], requires_grad=True)
    # Sum-only checks could hide opposite output errors. Full seeds must catch them.
    def wrong(a):
        return a * d.tensor([1., -1.]) + a.detach() * d.tensor([-1., 1.])
    assert not gradcheck(wrong, x, raise_exception=False)


@pytest.mark.parametrize("kind", ["matmul", "linear", "layer_norm", "mse", "cross_entropy"])
def test_existing_kernel_jacobians(kind):
    x = d.tensor([[.2, .7], [.5, -.3]], requires_grad=True)
    weight = d.tensor([[.4, -.2], [.8, .6]], requires_grad=True)
    if kind == "matmul":
        fn = lambda a, b: a.T @ b
    elif kind == "linear":
        fn = lambda a, b: d.functional.linear(a.T, b.T)
    elif kind == "layer_norm":
        fn = lambda a, b: d.functional.layer_norm(a.T, 2, weight=b[0])
    elif kind == "mse":
        fn = lambda a, b: d.functional.mse_loss(a.T, b.T)
    else:
        fn = lambda a, b: d.functional.cross_entropy(a.T * b, [0, 1])
    assert gradcheck(fn, (x, weight))


def test_grad_mode_restored_and_nonsmooth_point_rejected():
    x = d.tensor(.2, requires_grad=True)
    with d.no_grad():
        assert gradcheck(lambda a: a * a, x)
        assert not d.is_grad_enabled()
    assert not gradcheck(lambda a: a.relu(), d.tensor(0., requires_grad=True), raise_exception=False)


def test_nonfinite_output_and_shape_changes_are_errors():
    x = d.tensor(.2, requires_grad=True)
    with pytest.raises(ValueError, match="finite"):
        gradcheck(lambda a: a * float("nan"), x)
    with pytest.raises(ValueError, match="shape changed"):
        gradcheck(lambda a: a if a.numpy().item() <= .2 else a.reshape(1), x)


@pytest.mark.parametrize("shape", [(), (1,), (1, 3)])
def test_wrong_gradient_shape_is_rejected(monkeypatch, shape):
    x = d.tensor([.2, .4, .6], requires_grad=True)
    monkeypatch.setattr(d.Tensor, "grad", property(lambda self: d.tensor(np.ones(shape, dtype=np.float32), dtype="float32")))
    with pytest.raises(GradcheckError, match="gradient shape"):
        gradcheck(lambda a: a.sum(), x)
    assert not gradcheck(lambda a: a.sum(), x, raise_exception=False)


@pytest.mark.parametrize('raise_exception', [True, False])
def test_gradcheck_rejects_boolean_outputs(raise_exception):
    x = d.tensor([.2, .3], requires_grad=True)
    with pytest.raises(TypeError, match='float32 output'):
        gradcheck(lambda a: a > 0., x, raise_exception=raise_exception)


def test_gradcheck_rejects_boolean_output_only_under_perturbation():
    x = d.tensor(.2, requires_grad=True)
    calls = []
    def changes_dtype(value):
        calls.append(d.is_grad_enabled())
        return value * value if d.is_grad_enabled() else value > 0.
    with pytest.raises(TypeError, match='float32 output'):
        gradcheck(changes_dtype, x)
    assert calls == [True, False]
    assert x.grad is None


def test_gradcheck_rejects_empty_boolean_output():
    x = d.tensor(np.empty((0, 2)), requires_grad=True, dtype="float32")
    with pytest.raises(TypeError, match='float32 output'):
        gradcheck(lambda value: value > 0., x)
