"""GC-visible tensor hook ownership; native identities contain only Weak<Node>."""
import threading
import weakref
from .grad_mode import no_grad

_active_slots = weakref.WeakSet()


class _HookSlot:
    __slots__ = ("identity", "callbacks", "next_id", "lock", "__weakref__")
    def __init__(self, native):
        self.identity = native._identity()
        self.callbacks = None
        self.next_id = 0
        self.lock = None

    def sync(self, native):
        identity = native._identity()
        if identity.id != self.identity.id:
            self.identity.set_hooks(False)
            self.identity = identity
            identity.set_hooks(bool(self.callbacks))


class _Handle:
    def __init__(self, slot, identifier):
        self._slot = weakref.ref(slot)
        self._identifier = identifier

    def remove(self):
        slot = self._slot()
        if slot is not None:
            slot.callbacks.pop(self._identifier, None)
            if not slot.callbacks:
                slot.identity.set_hooks(False)
                _active_slots.discard(slot)


def register(tensor, callback):
    if not callable(callback):
        raise TypeError("hook must be callable")
    if tensor.dtype != 'float32' or not tensor.requires_grad:
        raise RuntimeError("hooks require a float32 Tensor requiring gradients")
    tensor._native._check_current()
    tensor._sync_hook_slot()
    slot = tensor._hook_slot
    if slot.callbacks is None:
        slot.callbacks = {}
        slot.lock = threading.Lock()
    identifier = slot.next_id
    slot.next_id += 1
    slot.callbacks[identifier] = callback
    slot.identity.set_hooks(True)
    _active_slots.add(slot)
    return _Handle(slot, identifier)


def snapshot(owners):
    from .function import _records
    result = {}
    for slot in _records(owners, slots=True).values():
        if slot.callbacks and slot.identity.alive:
            result[slot.identity.id] = (slot, tuple(slot.callbacks.values()))
    return result


def invoke(session, slot, callbacks, gradient):
    from .tensor import Tensor
    from .function import _callback_state
    if not slot.lock.acquire(blocking=False):
        raise RuntimeError("tensor hook context is already running")
    previous = getattr(_callback_state, 'hook_active', False)
    try:
        _callback_state.hook_active = True
        for callback in callbacks:
            with no_grad():
                replacement = callback(gradient)
            if replacement is not None:
                if not isinstance(replacement, Tensor) or replacement.dtype != 'float32':
                    raise TypeError("hook must return a float32 Tensor or None")
                gradient = replacement
            session.check_hook(gradient._native)
        session.resume_hook(gradient._native)
    finally:
        _callback_state.hook_active = previous
        slot.lock.release()
