"""Download the Stage-1 "initial" VideoMAE ViT-Base encoder weights.

Usage (from ``code/``)::

    python runners/run_download_weights.py --list
    python runners/run_download_weights.py videomae:k400   # Kinetics-400
    python runners/run_download_weights.py videomae:ssv2   # Something-Something-v2
    python runners/run_download_weights.py --all           # both corpora

Specs accepted here are exactly the ones ``run_pretrain.py`` accepts for
``--pretrained_encoder`` and the AU probe for ``--finetune``: a corpus spec
(``videomae:k400`` / ``videomae:ssv2`` / ``videomae`` + ``--dataset``), the
bare alias ``base`` (= ``videomae:k400``) or a local ``.pth`` path.

Stage 1 is **VideoMAE ViT-Base only**: both corpora are ``MCG-NJU`` Hub
checkpoints whose ``patch_embed.proj`` is already a
``Conv3d(3, D, (2,16,16))`` tubelet filter, so the RGB tokenizer -- including
the temporal kernel -- transfers verbatim, and the objective (tube-masked video
MAE) matches Stage 2. ``videomae:k400`` (Kinetics-400) is the default; the
``videomae:ssv2`` checkpoint enables the two-corpus A/B. VideoMAE weights are
CC-BY-NC 4.0.

Files are cached under ``<project_root>/models/initial/`` -- override with
``--weights_dir`` or ``$INITIAL_MODELS_DIR`` (scratch volume on the HPC). The
repo ``.gitignore`` already covers ``models/*``, so the weights are never
committed.

On the HPC, download **once on a login node** (which has internet) and let the
compute nodes reuse the cache::

    python runners/run_download_weights.py --all
    python runners/run_pretrain.py -c configs/pretrain/stage2_local_pretrained.yaml \
        --pretrained_encoder videomae:k400
"""
import os
import sys

# allow `python runners/run_download_weights.py` from the code/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pretrained import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
