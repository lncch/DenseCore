"""Guarded compilation of built-in dense modules; unsupported modules run eagerly."""
from . import nn, functional
from .tensor import Tensor
from ._native import _GraphExecutor

_FORWARDS = {cls: cls.forward for cls in (nn.Linear, nn.LayerNorm, nn.ReLU, nn.Sequential)}
_CALL = nn.Module.__call__
_RELU = Tensor.relu
_WRAP = Tensor._wrap
_NORMALIZE = functional.layer_norm


def _capture(model):
    layers, signature = [], []

    def visit(module):
        cls = type(module)
        if (cls not in _FORWARDS or 'forward' in module.__dict__
                or cls.forward is not _FORWARDS[cls] or cls.__call__ is not _CALL):
            return False
        signature.append((id(module), cls, module.training))
        if cls is nn.Sequential:
            signature.append(tuple((name, id(child)) for name, child in module._modules.items()))
            return all(visit(child) for child in module._modules.values())
        if cls in (nn.Linear,nn.LayerNorm):
            if module.weight is not None and not isinstance(module.weight, nn.Parameter):
                return False
            if module.bias is not None and not isinstance(module.bias, nn.Parameter):
                return False
            signature.append((id(module.weight), None if module.weight is None else module.weight.shape,
                              id(module.bias), None if module.bias is None else module.bias.shape))
        if cls is nn.LayerNorm:
            from .functional import _normalization_eps,_normalization_shape
            signature.append((_normalization_shape(module.normalized_shape),_normalization_eps(module.eps)))
        layers.append(module)
        return True

    if not visit(model) or not layers:
        return None
    parameters, slots = [], {}

    def parameter_slot(parameter):
        identity = id(parameter)
        if identity not in slots:
            slots[identity] = len(parameters) + 1
            parameters.append(parameter)
        return slots[identity]

    # Determine all input slots before assigning intermediate node indices.
    for layer in layers:
        if type(layer) in (nn.Linear,nn.LayerNorm):
            if layer.weight is not None:parameter_slot(layer.weight)
            if layer.bias is not None:
                parameter_slot(layer.bias)
    operations, epsilons, normalized_features, value = [], [], [], 0
    input_count = len(parameters) + 1
    for layer in layers:
        if type(layer) is nn.Linear:
            operations.append(('linear', value, parameter_slot(layer.weight),
                               None if layer.bias is None else parameter_slot(layer.bias)))
        elif type(layer) is nn.LayerNorm:
            operations.append(('layer_norm',value,None if layer.weight is None else parameter_slot(layer.weight),None if layer.bias is None else parameter_slot(layer.bias)))
        else:
            operations.append(('relu', value, None, None))
        epsilons.append(layer.eps if type(layer) is nn.LayerNorm else None)
        normalized_features.append(functional._normalization_shape(layer.normalized_shape)[0] if type(layer) is nn.LayerNorm else None)
        value = input_count + len(operations) - 1
    return tuple(signature), parameters, operations,epsilons,normalized_features



def _signature_matches(model, signature, *, strict_training=False):
    """Validate the cached graph without rebuilding capture metadata."""
    if signature is None:
        return False

    position = 0
    stack = [model]

    while stack:
        module = stack.pop()
        cls = type(module)

        # Preserve the existing Python dispatch guards.
        if (cls not in _FORWARDS
                or 'forward' in module.__dict__
                or cls.forward is not _FORWARDS[cls]
                or cls.__call__ is not _CALL):
            return False

        if strict_training:
            if cls is nn.Linear:
                names = ("weight",) if module.bias is None else ("weight", "bias")
                if module._modules or tuple(module._parameters) != names:
                    return False
                if any(type(module._parameters[name]) is not nn.Parameter
                       or module._parameters[name] is not getattr(module, name)
                       for name in names):
                    return False
            elif cls is nn.LayerNorm:
                # Only canonical affine registration is safe for the native
                # training transaction; reject extra parameters and children.
                names = (() if module.weight is None else ("weight",)) + (
                    () if module.bias is None else ("bias",)
                )
                if (module._modules or tuple(module._parameters) != names
                        or any(type(module._parameters[name]) is not nn.Parameter
                               or module._parameters[name] is not getattr(module, name)
                               for name in names)
                        or type(module.elementwise_affine) is not bool
                        or (module.elementwise_affine != (module.weight is not None))
                        or (module.bias is not None and module.weight is None)):
                    return False
            elif cls in (nn.Sequential, nn.ReLU):
                if module._parameters or (cls is nn.ReLU and module._modules):
                    return False
            else:
                return False

        if position >= len(signature):
            return False

        expected = signature[position]
        position += 1

        # Compare fields directly instead of constructing a tuple.
        if (expected[0] != id(module)
                or expected[1] is not cls
                or expected[2] != module.training):
            return False

        if cls is nn.Sequential:
            if position >= len(signature):
                return False

            expected_children = signature[position]
            position += 1

            children = module._modules

            if len(children) != len(expected_children):
                return False

            for (name, child), (old_name, old_id) in zip(
                children.items(), expected_children
            ):
                if name != old_name or id(child) != old_id:
                    return False

            # Preserve the original depth-first visitation order.
            stack.extend(reversed(tuple(children.values())))
            continue

        if cls in (nn.Linear, nn.LayerNorm):
            weight = module.weight
            bias = module.bias

            if weight is not None and not isinstance(weight, nn.Parameter):
                return False

            if bias is not None and not isinstance(bias, nn.Parameter):
                return False

            if position >= len(signature):
                return False

            expected_params = signature[position]
            position += 1

            if (expected_params[0] != id(weight)
                    or expected_params[1] != (
                        None if weight is None else weight.shape
                    )
                    or expected_params[2] != id(bias)
                    or expected_params[3] != (
                        None if bias is None else bias.shape
                    )):
                return False

        if cls is nn.LayerNorm:
            if position >= len(signature):
                return False

            from .functional import _normalization_eps, _normalization_shape

            expected_norm = signature[position]
            position += 1

            if (expected_norm[0] != _normalization_shape(
                    module.normalized_shape
                )
                    or expected_norm[1] != _normalization_eps(module.eps)):
                return False

    return position == len(signature)



class _Runner:
    def __init__(self):
        self.signature = None
        self.executor = None
        self.parameters = None

    def __call__(self, model, args, kwargs):
        if len(args) != 1 or kwargs or type(args[0]) is not Tensor or len(args[0].shape) != 2 or Tensor.relu is not _RELU or Tensor._wrap is not _WRAP or functional.layer_norm is not _NORMALIZE:
            return model.forward(*args, **kwargs)
        if self.executor is not None and _signature_matches(model, self.signature):
            parameters = self.parameters
        else:
            captured = _capture(model)
            if captured is None:
                return model.forward(*args, **kwargs)
            signature, parameters, operations, epsilons, normalized_features = captured
            if self.executor is None or signature != self.signature:
                executor = _GraphExecutor(
                    len(parameters) + 1,
                    operations,
                    epsilons,
                    normalized_features,
                )
                self.executor, self.signature = executor, signature
                self.parameters = tuple(parameters)
        # Current native handles reflect optimizer steps and state loads without
        # retaining stale parameters in the cached execution plans.
        inputs = [args[0]._native] + [p._native for p in parameters]
        return Tensor._wrap(self.executor.execute(inputs), parents=(args[0], *parameters))


def compile(model):
    """Enable guarded CPU float32 Linear/LayerNorm/ReLU/Sequential compilation.

    Return the same module. Other modules run eagerly. Shapes and gradient masks
    specialize lazily, with at most four cached native plans. Every live training
    output owns its activations; no-grad inference reuses workspace buffers.
    """
    if not isinstance(model, nn.Module):
        raise TypeError('compile requires a Module')
    if '_compiled_runner' in model._parameters or '_compiled_runner' in model._modules:
        raise ValueError('compiled runner name conflicts with a registered member')
    object.__setattr__(model, '_compiled_runner', _Runner())
    return model
