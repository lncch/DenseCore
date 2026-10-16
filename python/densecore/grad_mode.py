"""Thread-local gradient recording for synchronous execution."""
from contextlib import ContextDecorator
import inspect
import threading
from ._native import _is_grad_enabled, _set_grad_enabled

class _Mode(ContextDecorator):
    def __init__(self, enabled):
        self._enabled=enabled
        self._local=threading.local()

    def __enter__(self):
        if not hasattr(self._local,"stack"):
            self._local.stack=[]
        self._local.stack.append(_set_grad_enabled(self._enabled))
        return self

    def __exit__(self, *exc):
        _set_grad_enabled(self._local.stack.pop())
        return False

    def __call__(self, func):
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func) or inspect.isgeneratorfunction(func):
            raise TypeError("gradient mode decorators require synchronous functions")
        return super().__call__(func)

class no_grad(_Mode):
    def __init__(self):super().__init__(False)

class enable_grad(_Mode):
    def __init__(self):super().__init__(True)

def is_grad_enabled():
    return _is_grad_enabled()
