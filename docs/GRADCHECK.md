# Numerical gradient checks

`densecore.autograd.gradcheck` compares every first-order input/output Jacobian entry using central finite differences. It supports one float32 Tensor result and one float32 Tensor or tuple/list of float32 Tensor inputs. At least one input must require gradients. Frozen inputs are held fixed. Original tensors and existing gradients are not modified.

```python
import densecore as d

x = d.tensor([[.2, .7], [.5, -.3]], requires_grad=True)
assert d.autograd.gradcheck(lambda a: (a.T[::2] * a.T[::2]).sum(), x)
```

The function must be deterministic and smooth near the input. Defaults are eps=1e-3, atol=1e-3, rtol=1e-2 for float32; this is not a float64 precision checker. Tune tolerances for the function and avoid nondifferentiable points. Inputs/outputs must be finite, and finite-difference steps that cannot change the float32 input are rejected. A changing output shape is an error. Boolean outputs are rejected on every evaluation, including empty results and finite-difference samples. A locally constant comparison is not a floating-point derivative check.

Analytical Jacobians use one backward seed per output element, including disconnected inputs. Numerical differences use fresh logical-value copies with gradients disabled. Function entry copies are contiguous; construct views inside the function to validate their gradient paths. The checker allocates the full Jacobian and runs two forwards per differentiable input element, so use small tensors. It does not check higher-order derivatives, stochastic functions, mutation or preservation of alias relationships between arguments.

Mismatches raise GradcheckError. Pass raise_exception=False to return False for a Jacobian mismatch; invalid inputs or outputs still raise errors. This utility supplements PyTorch parity and regression tests rather than proving arbitrary numerical correctness.

Tests include chained slicing/permutation, broadcasting, matmul, Linear, LayerNorm, MSE, cross entropy, scalar/empty tensors, frozen and disconnected inputs, wrong detach gradients, nonsmooth points and original-gradient preservation. Full suite after this unit: 441 Python tests pass with two existing NumPy empty-mean reference warnings. Rust was unchanged from the preceding 59-test checkpoint.

Review regression tests also reject wrong gradient shapes before NumPy can broadcast them into a Jacobian row. Both mismatch-raising and boolean-return modes cover this case. Invalid numerical evaluations still raise errors, including when a finite-difference step leaves the function domain.
