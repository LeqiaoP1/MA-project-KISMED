"""Download the Stage-1 "initial" ViT encoder weights.

Usage (from ``code/``)::

    python runners/run_download_weights.py --list
    python runners/run_download_weights.py base          # VideoMAE ViT-Base
    python runners/run_download_weights.py mae:base       # MAE ViT-Base
    python runners/run_download_weights.py base large     # VideoMAE B + L
    python runners/run_download_weights.py --all         # base + large

Specs accepted here are exactly the ones the training runners accept for
``--finetune`` / ``--pretrained_encoder``: a variant (``base``/``large``), an
explicit ``<source>:<variant>`` (``videomae:base``, ``mae:large``) or any
``timm:<model_id>``.

``base``/``large`` resolve to **VideoMAE** (MCG-NJU, Kinetics-400): its
``patch_embed.proj`` is already a ``Conv3d(3, D, (2,16,16))`` tubelet filter, so
the RGB tokenizer -- including the temporal kernel -- transfers verbatim, and
the objective (tube-masked video MAE) matches Stage 2. ``mae:base`` gives the
plain ImageNet MAE ViT-B (2-D filter, boxcar-inflated into the tubelet) as the
control. VideoMAE weights are CC-BY-NC 4.0.

Files are cached under ``<project_root>/models/initial/`` -- override with
``--weights_dir`` or ``$INITIAL_MODELS_DIR`` (scratch volume on the HPC). The
repo ``.gitignore`` already covers ``models/*``, so the weights are never
committed.

On the HPC, download **once on a login node** (which has internet) and let the
compute nodes reuse the cache::

    python runners/run_download_weights.py --all
    python runners/run_pretrain.py -c configs/pretrain/stage2_local_pretrained.yaml \
        --pretrained_encoder videomae:base
"""
import os
import sys

# allow `python runners/run_download_weights.py` from the code/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pretrained import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
