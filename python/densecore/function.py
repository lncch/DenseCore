"""Single-output first-order custom autograd with callbacks outside native locks."""
import threading
from .tensor import Tensor
from .grad_mode import no_grad, is_grad_enabled
from ._native import _BackwardSession, _custom_result

_callback_state = threading.local()


class _Context:
    def __init__(self, needs_input_grad):
        self.needs_input_grad = needs_input_grad
        self._phase = "forward"
        self._saved = ()

    def save_for_backward(self, *tensors):
        if self._phase != "forward":
            raise RuntimeError("save_for_backward is only available during forward")
        if any(not isinstance(t, Tensor) or t.dtype != "float32" for t in tensors):
            raise TypeError("save_for_backward currently accepts float32 Tensors only")
        self._saved = tuple(t._native._saved_snapshot() for t in tensors)

    @property
    def saved_tensors(self):
        if self._phase not in ("forward", "backward"):
            raise RuntimeError("saved_tensors is only available during forward/backward")
        for native in self._saved:
            native._check_current()
        return tuple(Tensor._wrap(native.detach()) for native in self._saved)


class _Record:
    def __init__(self, identifier, ctx, backward):
        self.identifier, self.ctx, self.backward = identifier, ctx, backward
        self.lock = threading.Lock()


def _records(owners, slots=False):
    from .hooks import _HookSlot
    stack, visited, result = [owners], set(), {}
    while stack:
        item = stack.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        if isinstance(item, _Record):
            if not slots:
                result[item.identifier] = item
        elif isinstance(item, _HookSlot):
            if slots:
                result[item.identity.id] = item
        elif isinstance(item, tuple):
            stack.extend(item)
        else:
            raise RuntimeError("invalid custom autograd ownership graph")
    return result


class Function:
    """Subclass static forward/backward; call apply with float32 Tensor inputs.

    One Tensor output and first-order gradients are supported. Forward/backward
    execute under no_grad. Backward returns one detached gradient or None per
    input. Saved tensors are available during forward/backward only.
    """
    @classmethod
    def apply(cls, *inputs):
        if not inputs or any(not isinstance(t, Tensor) or t.dtype != "float32" for t in inputs):
            raise TypeError("Function.apply requires nonempty float32 Tensor inputs")
        native_inputs = tuple(t._native for t in inputs)
        flags = tuple(t.requires_grad for t in inputs)
        enabled = is_grad_enabled()
        ctx = _Context(flags)
        try:
            with no_grad():
                output = cls.forward(ctx, *inputs)
            if not isinstance(output, Tensor) or output.dtype != "float32":
                raise TypeError("custom forward must return one float32 Tensor")
            if not enabled or not any(flags):
                return output.detach()
            native, identifier = _custom_result(output._native, native_inputs, ctx._saved)
            record = _Record(identifier, ctx, cls.backward)
            return Tensor._wrap(native, contexts=(record, Tensor._parent_contexts(inputs)))
        finally:
            ctx._phase, ctx._saved = "idle", ()

    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError("define a static forward method")

    @staticmethod
    def backward(ctx, gradient):
        raise NotImplementedError("define a static backward method")


def _backward(output, gradient, retain_graph):
    from .hooks import snapshot, invoke
    hooks = snapshot(output._contexts)
    if not hooks and not output._native.has_custom:
        return output._native.backward(None if gradient is None else gradient._native,
                                       retain_graph=retain_graph)
    if getattr(_callback_state, "active", False):
        raise RuntimeError("reentrant custom backward is unsupported")
    records = _records(output._contexts)
    session = _BackwardSession(output._native, None if gradient is None else gradient._native,
                               retain_graph, list(hooks))
    try:
        while True:
            pending = session.advance()
            if pending is None:
                return
            kind, identifier, upstream, saved = pending
            if kind == "hook":
                slot, callbacks = hooks[identifier]
                invoke(session, slot, callbacks, Tensor._wrap(upstream))
                continue
            record = records.get(identifier)
            if record is None:
                raise RuntimeError("custom autograd context ownership was not propagated")
            if not record.lock.acquire(blocking=False):
                raise RuntimeError("custom backward context is already running")
            try:
                record.ctx._phase, record.ctx._saved = "backward", tuple(saved)
                _callback_state.active = True
                with no_grad():
                    gradients = record.backward(record.ctx, Tensor._wrap(upstream))
                if isinstance(gradients, Tensor) or gradients is None:
                    gradients = (gradients,)
                if not isinstance(gradients, (tuple, list)):
                    raise TypeError("custom backward must return gradients or None in a tuple/list")
                if any(g is not None and (not isinstance(g, Tensor) or g.dtype != "float32") for g in gradients):
                    raise TypeError("custom backward gradients must be float32 Tensors or None")
                session.resume([None if g is None else g._native for g in gradients])
            finally:
                _callback_state.active = False
                record.ctx._phase, record.ctx._saved = "idle", ()
                record.lock.release()
    finally:
        session.abort()
