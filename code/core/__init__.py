"""Pure model code (no I/O / training logic) — mirrors MultiMAE's ``multimae/``
package. Importing this package also runs all ``@register_model`` decorators,
so the registered entrypoints become available to ``models.build.create_model``.
"""
from .registry import register_model, model_entrypoint, is_model, list_models
from .blocks import trunc_normal_, drop_path, DropPath, Mlp, Attention, Block
from .input_adapters import (PatchedInputAdapter, SemSegInputAdapter,
                            SignalInputAdapter)
from .output_adapters import SpatialOutputAdapter
from .criterion import MaskedMSELoss, MaskedL1Loss, MaskedCrossEntropyLoss
from .waveform_losses import (PearsonLoss, MultiResolutionSTFTLoss,
                              WaveformJointLoss)
from .model import (ProjectViT,
                    project_vit_small_patch16_224,
                    project_vit_base_patch16_224,
                    project_vit_large_patch16_224,
                    project_vit_huge_patch16_224)
