"""Numerical first-order gradient validation for small float32 functions."""
import numbers
import numpy as np
from .tensor import Tensor
from .grad_mode import enable_grad, no_grad


class GradcheckError(RuntimeError):
    """Analytical and numerical Jacobians disagree."""


def gradcheck(func, inputs, *, eps=1e-3, atol=1e-3, rtol=1e-2,
              raise_exception=True):
    """Compare every output/input Jacobian entry using central differences.

    ``inputs`` is a Tensor or tuple/list of Tensors, with at least one requiring
    gradients. ``func`` must be deterministic, smooth at these inputs and return
    one finite float32 Tensor. Inputs and existing gradients are left untouched; checks
    evaluate independent copies of logical values. Layouts are contiguous at
    function entry, so construct views inside ``func`` to validate view paths.

    Float32 needs larger tolerances than a float64 checker. Unresolvable finite
    difference steps are rejected. Memory scales with the full Jacobian and
    execution requires two function calls per differentiable input element,
    plus one backward per output element. Use small tensors. This checks first
    derivatives only and does not prove differentiability at nonsmooth points.
    """
    for name, value in (("eps", eps), ("atol", atol), ("rtol", rtol)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real number")
        if not np.isfinite(value) or value < 0 or (name == "eps" and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'eps' else 'nonnegative'}")
    if type(raise_exception) is not bool:
        raise TypeError("raise_exception must be a bool")
    if isinstance(inputs, Tensor):
        inputs = (inputs,)
    if not isinstance(inputs, (tuple, list)) or not all(isinstance(x, Tensor) for x in inputs):
        raise TypeError("inputs must be a Tensor or tuple/list of Tensors")
    if any(x.dtype != "float32" for x in inputs):
        raise TypeError("gradcheck currently requires float32 inputs")
    if not inputs or not any(x.requires_grad for x in inputs):
        raise ValueError("at least one input must require gradients")
    arrays = [x.numpy() for x in inputs]
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("gradcheck inputs must be finite")
    flags = [x.requires_grad for x in inputs]

    def evaluate(values):
        args = [Tensor(x, requires_grad=flag) for x, flag in zip(values, flags)]
        output = func(*args)
        if not isinstance(output, Tensor):
            raise TypeError("gradcheck function must return a Tensor")
        if output.dtype != "float32":
            raise TypeError("gradcheck currently requires a float32 output")
        data = output.numpy()
        if not np.isfinite(data).all():
            raise ValueError("gradcheck output must be finite")
        return args, output, data

    with enable_grad():
        args, output, reference = evaluate(arrays)
    analytical = [np.zeros((reference.size, x.size), dtype=np.float64) if flag else None
                  for x, flag in zip(arrays, flags)]
    if output.requires_grad:
        for out_index in range(reference.size):
            for arg in args:
                if arg.requires_grad:
                    arg.zero_grad()
            seed = np.zeros_like(reference)
            seed.flat[out_index] = 1
            output.backward(Tensor(seed), retain_graph=True)
            for jacobian, arg in zip(analytical, args):
                if jacobian is not None:
                    gradient = arg.grad
                    if gradient is not None:
                        if gradient.shape != arg.shape:
                            if not raise_exception:
                                return False
                            raise GradcheckError(f"gradient shape mismatch: expected {arg.shape}, got {gradient.shape}")
                        jacobian[out_index] = gradient.numpy().ravel()

    for input_index, (array, jacobian) in enumerate(zip(arrays, analytical)):
        if jacobian is None:
            continue
        for flat_index, value in enumerate(array.flat):
            with np.errstate(over="ignore", invalid="ignore"):
                plus = np.float32(float(value) + eps)
                minus = np.float32(float(value) - eps)
            if (not np.isfinite(plus) or not np.isfinite(minus)
                    or plus == value or minus == value):
                raise ValueError("finite difference step cannot resolve this float32 input")
            samples = []
            for changed in (plus, minus):
                perturbed = [x.copy() for x in arrays]
                perturbed[input_index].flat[flat_index] = changed
                with no_grad():
                    _, _, sample = evaluate(perturbed)
                if sample.shape != reference.shape:
                    raise ValueError("gradcheck output shape changed under perturbation")
                samples.append(sample.astype(np.float64).ravel())
            numerical = (samples[0] - samples[1]) / (float(plus) - float(minus))
            actual = jacobian[:, flat_index]
            if not np.allclose(actual, numerical, atol=atol, rtol=rtol):
                if not raise_exception:
                    return False
                raise GradcheckError(f"Jacobian mismatch for input {input_index}, element {flat_index}: "
                                     f"analytical={actual}, numerical={numerical}")
    return True


from .function import Function
