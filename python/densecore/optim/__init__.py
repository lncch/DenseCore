"""Native SGD/Adam with ordered, versioned optimizer state."""
import math
import numbers
import operator
from ..tensor import Tensor
from .._native import _Optimizer, _Tensor

def _real(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value

class _Base:
    def _initialize(self, params, config):
        self.defaults = dict(config)
        self._params = list(params)
        if self._params and all(isinstance(g, dict) for g in self._params):
            groups = [dict(g) for g in self._params]
            if any("params" not in g for g in groups):
                raise ValueError("each parameter group needs a 'params' entry")
            if len(groups) > 1:
                return self._initialize_groups(groups, config)
            group = groups[0]
            self._params = list(group.pop("params"))
            config = {**config, **group}
        if any(not isinstance(p, Tensor) for p in self._params):
            raise TypeError("optimizer parameters must be Tensors or parameter-group dicts")
        self._config = self._validate_config(config)
        self._native = self._make_native(self._config)

    def _initialize_groups(self, groups, config):
        """Several groups: one single-group optimizer per nonempty group, stepped in order.
        shortcut: empty groups are dropped, so param_groups lists only nonempty ones."""
        self._groups = []
        for group in groups:
            members = list(group.pop("params"))
            if members:
                child = object.__new__(type(self))
                child._initialize([{"params": members, **group}], config)
                self._groups.append(child)
        if not self._groups:
            raise ValueError("optimizer requires at least one parameter")
        self._params = [p for child in self._groups for p in child._params]
        if len({id(p) for p in self._params}) != len(self._params):
            raise ValueError("a parameter appears in more than one group")

    def _grouped(self):
        return self.__dict__.get("_groups")

    def _validate_config(self, config):
        expected = {"lr", "momentum", "weight_decay"} if self._kind == "sgd" else {"lr", "betas", "eps", "weight_decay"}
        if not isinstance(config, dict) or set(config) != expected:
            raise ValueError("optimizer hyperparameter keys mismatch")
        result = {k: _real(k, v) for k, v in config.items() if k != "betas"}
        if "betas" in config:
            values = config["betas"]
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                raise ValueError("betas must contain two real numbers")
            result["betas"] = tuple(_real("beta", v) for v in values)
        return result

    def _make_native(self, config):
        args = dict(config)
        if self._kind != "sgd":
            args["beta1"], args["beta2"] = args.pop("betas")
        return _Optimizer(self._kind, [p._native for p in self._params], **args)

    def zero_grad(self, set_to_none=True):
        if set_to_none is not True:
            raise NotImplementedError("zero_grad clears gradients to None; zero-filled gradients are unsupported")
        for p in self._params:
            p.zero_grad()

    @property
    def param_groups(self):
        """PyTorch-style group dicts, the same objects on every access (lr schedulers keep
        them). Setting a hyperparameter key applies it from the next step; other keys are
        stored, as in PyTorch."""
        groups = self.__dict__.get("_param_group_list")
        if groups is None:
            children = self._groups if self._grouped() else [self]
            groups = self.__dict__["_param_group_list"] = [_ParamGroup(child) for child in children]
        return groups

    def _set_hyper(self, key, value):
        config = self._validate_config({**self._config, key: value})
        def flat(cfg):
            return {**{k: v for k, v in cfg.items() if k != "betas"},
                    **(dict(zip(("beta1", "beta2"), cfg["betas"])) if "betas" in cfg else {})}
        old, new = flat(self._config), flat(config)
        try:
            for name in new:
                if new[name] != old[name]:
                    self._native.set_hyper(name, new[name])
        except ValueError:
            for name in old:  # keep the native config equal to self._config
                self._native.set_hyper(name, old[name])
            raise
        self._config = config
        return config[key]

    def _set_lr(self, lr):
        self._set_hyper("lr", lr)

    def step(self):
        if self._grouped():
            for child in self._groups:
                child.step()
            return
        updated = self._native.step([p._native for p in self._params])
        for p, native in zip(self._params, updated):
            p._native = native
            if p._hook_slot is not None and p._hook_slot.callbacks:
                p._sync_hook_slot()

    def state_dict(self):
        if self._grouped():
            return {"version": 2, "kind": self._kind, "groups": [c.state_dict() for c in self._groups]}
        records = []
        for p, (step, first, second) in zip(self._params, self._native.snapshot()):
            def copied(values):
                return Tensor._wrap(_Tensor(values, list(p.shape)))
            if self._kind == "sgd":
                records.append({"step": step, "momentum_buffer": copied(first) if step and self._config["momentum"] else None})
            else:
                records.append({"step": step, "first_moment": copied(first) if step else None,
                                "second_moment": copied(second) if step else None})
        return {"version": 1, "kind": self._kind, "hyperparameters": dict(self._config), "state": records}

    def _prepare_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"version", "kind", "hyperparameters", "state"}:
            raise ValueError("invalid optimizer state dictionary")
        if type(state["version"]) is not int or state["version"] != 1 or state["kind"] != self._kind:
            raise ValueError("optimizer state version or kind mismatch")
        config = self._validate_config(state["hyperparameters"])
        records = state["state"]
        if not isinstance(records, list) or len(records) != len(self._params):
            raise ValueError("optimizer state count mismatch")
        parsed = []
        for p, record in zip(self._params, records):
            expected = {"step", "momentum_buffer"} if self._kind == "sgd" else {"step", "first_moment", "second_moment"}
            if not isinstance(record, dict) or set(record) != expected:
                raise ValueError("optimizer state record keys mismatch")
            if isinstance(record["step"], bool):
                raise TypeError("optimizer step must be an integer")
            step = operator.index(record["step"])
            if not 0 <= step <= 2**64 - 1:
                raise ValueError("optimizer step outside uint64 range")
            def buffer(name, required):
                value = record[name]
                if not required:
                    if value is not None:
                        raise ValueError("unexpected optimizer buffer")
                    return []
                if not isinstance(value, Tensor) or value.requires_grad or value.shape != p.shape:
                    raise ValueError("optimizer buffers must be detached Tensors matching parameter shape")
                if value.dtype != "float32":
                    raise TypeError("optimizer buffers must be float32 tensors")
                return value.numpy().ravel().tolist()
            if self._kind == "sgd":
                first = buffer("momentum_buffer", step > 0 and config["momentum"] > 0)
                second = []
            else:
                first = buffer("first_moment", step > 0)
                second = buffer("second_moment", step > 0)
            parsed.append((step, first, second))
        replacement = self._make_native(config)
        replacement.load(parsed)
        return replacement, config

    def load_state_dict(self, state):
        if self._grouped():
            if (not isinstance(state, dict) or state.get("version") != 2 or state.get("kind") != self._kind
                    or not isinstance(state.get("groups"), list) or len(state["groups"]) != len(self._groups)):
                raise ValueError("optimizer state does not match these parameter groups")
            # Validate every group before changing any, so a bad state changes nothing.
            prepared = [c._prepare_state_dict(g) for c, g in zip(self._groups, state["groups"])]
            for child, (native, config) in zip(self._groups, prepared):
                child._native, child._config = native, config
            self._refresh_param_groups()
            return
        self._native, self._config = self._prepare_state_dict(state)
        self._refresh_param_groups()

    def _refresh_param_groups(self):
        for group in self.__dict__.get("_param_group_list") or ():
            dict.update(group, group._optimizer._config)

class _ParamGroup(dict):
    def __init__(self, optimizer):
        super().__init__(optimizer._config, params=list(optimizer._params))
        self._optimizer = optimizer

    def __setitem__(self, key, value):
        if key == "params":
            raise NotImplementedError("parameters cannot change after construction")
        if key in self._optimizer._config:
            value = self._optimizer._set_hyper(key, value)
        super().__setitem__(key, value)

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

class SGD(_Base):
    _kind = "sgd"
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        self._initialize(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

class Adam(_Base):
    _kind = "adam"
    def __init__(self, params, lr=1e-3, betas=(.9,.999), eps=1e-8, weight_decay=0.0):
        self._initialize(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

class AdamW(_Base):
    """Adam with decoupled weight decay: parameters shrink by lr*weight_decay each step."""
    _kind = "adamw"
    def __init__(self, params, lr=1e-3, betas=(.9,.999), eps=1e-8, weight_decay=1e-2):
        self._initialize(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

from . import lr_scheduler  # noqa: E402  (needs the classes above)
