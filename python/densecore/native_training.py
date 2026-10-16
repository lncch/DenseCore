"""Strict opt-in fused dense training; ordinary autograd execution is unchanged."""
import threading
import numpy as np
from . import nn, optim, functional
from .tensor import Tensor
from .grad_mode import is_grad_enabled
from .compilation import (_capture, _signature_matches, _GraphExecutor,
                          _RELU, _WRAP, _NORMALIZE)

_LOCK = threading.RLock()
_NATIVE_SLOT = Tensor.__dict__["_native"]
_PARAMETER_SETATTR = nn.Parameter.__setattr__


class _TrainingRunner:
    def __init__(self):
        self.signature = None
        self.parameters = ()
        self.executor = None
        self.slots = ()
        self.publication = None

    def prepare(self, model, loss):
        if (self.executor is not None and getattr(self, "loss", None) == loss
                and _signature_matches(model, self.signature, strict_training=True)):
            return
        captured = _capture(model)
        if captured is None:
            raise ValueError("compiled training requires supported built-in modules")
        signature, parameters, operations, epsilons, features = captured
        if any(op[0] not in {"linear", "relu", "layer_norm"} for op in operations):
            raise ValueError("compiled training supports Linear/LayerNorm/ReLU/Sequential only")
        if not _signature_matches(model, signature, strict_training=True):
            raise ValueError("compiled training requires canonical built-in parameter registration")
        count = len(parameters) + 1
        def shift(slot):
            return slot + 1 if slot is not None and slot >= count else slot
        training = [(op, shift(x), shift(w), shift(b)) for op,x,w,b in operations]
        training.append((loss, count + len(operations), count, None))
        executor = _GraphExecutor(count + 1, training, epsilons + [None], features + [None])
        self.signature, self.parameters, self.executor = signature, tuple(parameters), executor
        self.loss = loss
        self.slots = tuple(range(1, count))
        self.publication = (self.parameters, _NATIVE_SLOT)


def compiled_train_step(model, x, target, optimizer, *, loss="mse"):
    """Run fused forward/loss/backward/update and return a detached scalar loss.

    Requires compile(model), built-in Linear/LayerNorm/ReLU/Sequential, exact trainable
    Parameters and SGD/Adam covering them in capture order. Inputs and targets
    must be detached float32 CPU Tensors. Shared parameters use one input slot.
    Existing eager leaf gradients are preserved, not accumulated or cleared.
    Do not call backward or optimizer.step again for this batch. Fused callers
    serialize through native handle publication. Concurrent ordinary mutation of
    model/optimizer state is unsupported. Fatal allocation failures and process
    interruption cannot provide recovery guarantees.
    """
    with _LOCK:
        if not isinstance(model, nn.Module) or "_compiled_runner" not in model.__dict__:
            raise ValueError("compiled training requires compile(model)")
        if type(optimizer) not in (optim.SGD, optim.Adam):
            raise TypeError("compiled training requires built-in SGD or Adam")
        if type(x) is not Tensor:
            raise TypeError("compiled training input must be an exact Tensor")
        if loss not in ("mse", "cross_entropy"):
            raise ValueError("compiled training loss must be mse or cross_entropy")
        if loss == "cross_entropy":
            if len(x.shape) != 2 or x.shape[0] == 0:
                raise ValueError("compiled cross_entropy requires a nonempty rank2 input")
            if type(target) is Tensor:
                raise TypeError("cross_entropy expects integer label metadata, not a Tensor")
            labels = np.asarray(target)
            if labels.ndim != 1 or labels.dtype.kind not in "iu" or len(labels) != x.shape[0]:
                raise ValueError("cross_entropy labels require a rank1 integer array matching batch")
            if np.any(labels < 0) or np.any(labels >= 2**24):
                raise ValueError("class labels must be nonnegative and exactly representable as float32")
            target = Tensor(np.asarray(labels, dtype=np.float32))
        if type(target) is not Tensor:
            raise TypeError("compiled training target must be an exact Tensor")
        if not is_grad_enabled() or x.requires_grad or target.requires_grad:
            raise ValueError("compiled training requires gradients enabled and detached inputs/targets")
        if (Tensor.relu is not _RELU or Tensor._wrap is not _WRAP
                or functional.layer_norm is not _NORMALIZE
                or Tensor.__dict__["_native"] is not _NATIVE_SLOT
                or nn.Parameter.__setattr__ is not _PARAMETER_SETATTR):
            raise ValueError("compiled training rejects patched tensor dispatch")
        runner = model.__dict__.get("_compiled_training_runner")
        if type(runner) is not _TrainingRunner:
            runner = _TrainingRunner()
            object.__setattr__(model, "_compiled_training_runner", runner)
        runner.prepare(model, loss)
        parameters = runner.parameters
        if any(p._hook_slot is not None and p._hook_slot.callbacks for p in parameters):
            raise ValueError("compiled training does not support tensor hooks")
        if optimizer._grouped():
            raise ValueError("compiled training supports single-group optimizers only")
        if (not parameters or len(optimizer._params) != len(parameters)
                or any(a is not b for a,b in zip(optimizer._params, parameters))
):
            raise ValueError("compiled training requires matching ordered trainable model parameters")
        inputs = [x._native] + [p._native for p in parameters] + [target._native]
        result = object.__new__(Tensor)
        result._contexts = ()
        result._hook_slot = None
        loss, _ = runner.executor.train_step(inputs, runner.slots, optimizer._native,
                                            runner.publication)
        _NATIVE_SLOT.__set__(result, loss)
        return result
