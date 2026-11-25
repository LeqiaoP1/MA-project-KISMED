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
# ADR-008 (Stage-2 objective): re-exported here for the same reason the Stage-3
# losses are -- one import surface for "the losses of this package". Both
# modules are leaf modules (torch only), so the import order above is unchanged.
from .resp_composite import (MaskedPearsonLoss, BandLimitedSpectralLoss,
                             CompositeRespLoss)
from .align_infonce import (AlignProjection, anatomy_weighted_pool,
                            symmetric_infonce)
from .model import (ProjectViT,
                    project_vit_small_patch16_224,
                    project_vit_base_patch16_224,
                    project_vit_large_patch16_224,
                    project_vit_huge_patch16_224)
