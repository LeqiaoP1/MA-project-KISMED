"""Public model API."""
from .build import create_model, is_model, list_models
from .pretrained import (DEFAULT_SOURCE, PRETRAINED_SOURCES, VIT_VARIANTS,
                         available_specs, describe_sources, download_pretrained,
                         initial_dir, resolve_encoder_weights)

# Importing core.model / core.multimae runs their @register_model decorators;
# import here so that a bare `import models` registers every entrypoint
# (project_vit_* and project_multimae_*).
from core import model as _model  # noqa: F401,E402
from core import multimae as _multimae  # noqa: F401,E402

__all__ = [
    'create_model', 'is_model', 'list_models',
    'VIT_VARIANTS', 'PRETRAINED_SOURCES', 'DEFAULT_SOURCE', 'initial_dir',
    'available_specs', 'describe_sources', 'download_pretrained',
    'resolve_encoder_weights',
]
