"""Download the Stage-1 "initial" ViT encoder weights.

Usage (from ``code/``)::

    python runners/run_download_weights.py --list
    python runners/run_download_weights.py base          # MAE ViT-Base
    python runners/run_download_weights.py small large   # DeiT-S + MAE ViT-L
    python runners/run_download_weights.py --all         # all four variants

Specs accepted here are exactly the ones the training runners accept for
``--finetune`` / ``--pretrained_encoder``: a variant (``small``/``base``/
``large``/``huge``), an explicit ``<source>:<variant>`` (``mae:large``,
``deit:small``) or any ``timm:<model_id>``.

Files are cached under ``<project_root>/models/initial/`` -- override with
``--weights_dir`` or ``$INITIAL_MODELS_DIR`` (scratch volume on the HPC). The
repo ``.gitignore`` already covers ``models/*``, so the weights are never
committed.

On the HPC, download **once on a login node** (which has internet) and let the
compute nodes reuse the cache::

    python runners/run_download_weights.py --all
    python runners/run_pretrain.py -c configs/pretrain/stage2_local_pretrained.yaml \
        --pretrained_encoder mae:base
"""
import os
import sys

# allow `python runners/run_download_weights.py` from the code/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pretrained import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
