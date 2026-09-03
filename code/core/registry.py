"""Model registry (timm-style).

Based on the registry from MultiMAE ``tmp/MultiMAE/utils/registry.py``
(originally by Ross Wightman / timm). Register a model factory with
``@register_model`` and later instantiate it by name via ``model_entrypoint``.
"""
import sys
from collections import defaultdict

__all__ = [
    'register_model', 'model_entrypoint', 'is_model', 'list_models',
    'is_model_in_modules', 'list_modules',
]

_module_to_models = defaultdict(set)   # module name -> {model names}
_model_to_module = {}                  # model name -> module name
_model_entrypoints = {}                # model name -> factory fn


def register_model(fn):
    """Class/function decorator that registers ``fn`` as a creatable model.

    The entrypoint name is the decorated function's name (e.g.
    ``project_vit_base_patch16_224``).
    """
    mod = sys.modules[fn.__module__]
    module_name = fn.__module__.split('.')[-1]
    model_name = fn.__name__

    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    return fn


def model_entrypoint(model_name):
    """Return the registered factory function for ``model_name``."""
    if model_name not in _model_entrypoints:
        raise RuntimeError(
            f"Unknown model '{model_name}'. Available models: {list_models()}")
    return _model_entrypoints[model_name]


def is_model(model_name):
    return model_name in _model_entrypoints


def list_modules():
    return sorted(_module_to_models)


def list_models(module=''):
    """List registered model names, optionally filtered by source module."""
    if module:
        return sorted(_module_to_models.get(module, set()))
    return sorted(_model_entrypoints)


def is_model_in_modules(model_name, module_names):
    """Check whether ``model_name`` was registered from one of ``module_names``."""
    mod = _model_to_module.get(model_name, '')
    return any(mod == name for name in module_names)
