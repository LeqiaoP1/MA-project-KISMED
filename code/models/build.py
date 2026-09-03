"""Model building helpers.

``create_model`` is the single entry point for instantiating a model by name
(as configured in YAML / CLI). Mirrors MultiMAE ``utils/model_builder.py``.
"""
from core.registry import model_entrypoint

__all__ = ['create_model', 'is_model', 'list_models']


def create_model(model_name: str, **kwargs):
    """Instantiate a registered model factory by ``model_name``.

    Example::

        model = create_model('project_vit_base_patch16_224', num_classes=10)
    """
    create_fn = model_entrypoint(model_name)
    return create_fn(**kwargs)


def is_model(model_name: str) -> bool:
    from core.registry import is_model as _is_model
    return _is_model(model_name)


def list_models(module: str = ''):
    from core.registry import list_models as _list_models
    return _list_models(module)
