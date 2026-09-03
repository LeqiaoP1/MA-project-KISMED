"""Public model API."""
from .build import create_model, is_model, list_models

# Importing core.model runs @register_model decorators; import here so that a
# bare `import models` still registers every entrypoint.
from core import model as _model  # noqa: F401,E402
