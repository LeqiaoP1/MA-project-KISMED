"""The Stage-3 fold driver must forward the Stage-2 lineage to the job script.

Why this exists
---------------
``submit_stage3_kfold.sh`` takes ``STAGE2_NAME`` and preflights
``<STAGE2_ROOT>/<STAGE2_NAME>_fold<k>``. The actual training job, however,
resolves its encoder from ``STAGE2_DIR`` / ``STAGE2_RUN_NAME`` with
``stage2_hpc_tir_roi_resp`` as the default (``submit_waveform_tir_roi.sbatch``).
If the driver does not forward the name, a RENAMED lineage preflights green and
then loads the DEFAULT one -- a different geometry.

That is not hypothetical: on 2026-11-04 jobs 55906288-55906291 (the
``96respgrid`` sweep, folds 0-3) all died ~10 s in with

    ValueError: .../pretrain/stage2_hpc_tir_roi_resp_fold0/checkpoints/
    checkpoint-0149.pth: 1 of the Stage-2 tensors have the wrong shape for this
    Stage-3 geometry (e.g. ['positions.tir.pos_embed'])

because they were handed the 112 px encoder while the config asked for 96 px.
The geometry guard inside ``core/waveform_model.load_stage2_encoder`` caught it
loudly (no silently half-loaded encoder), but four queue slots were wasted.

A dry run (``bash scripts/hpc/submit_stage3_kfold.sh``) prints the exact ``env``
line per fold, so the forwarding is observable without a GPU -- it is pinned
here as text because the preflight needs real checkpoints and torch to run.

Run from ``code/``:  ``python -m pytest tests/test_stage3_driver_env.py -q``
"""
import os
import re

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(CODE_DIR, 'scripts', 'hpc', 'submit_stage3_kfold.sh')
JOB = os.path.join(CODE_DIR, 'scripts', 'hpc', 'submit_waveform_tir_roi.sbatch')


def _read(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return fh.read()


def test_job_script_still_defaults_to_the_unnamed_lineage():
    """The fallback the driver has to override must still be the documented one."""
    text = _read(JOB)
    assert 'STAGE2_RUN_NAME' in text
    assert 'stage2_hpc_tir_roi_resp' in text, (
        'the job script no longer defaults STAGE2_RUN_NAME to '
        'stage2_hpc_tir_roi_resp -- update the driver contract test if the '
        'default moved')


def test_driver_forwards_the_stage2_lineage_per_fold():
    """STAGE2_NAME must reach the job env, or renamed lineages load the wrong ckpt."""
    text = _read(DRIVER)
    assert 'STAGE2_NAME' in text, 'the driver no longer knows a Stage-2 basename'

    # the forwarding must live inside the per-fold submission block
    loop = text.split('for fold in "${READY[@]}"; do')[-1]
    assert re.search(r'STAGE2_RUN_NAME=\$STAGE2_NAME', loop), (
        'submit_stage3_kfold.sh does not forward STAGE2_RUN_NAME=$STAGE2_NAME '
        'to the job; a renamed Stage-2 lineage (e.g. *_96respgrid) would be '
        'preflighted here and then the DEFAULT 112 px encoder would be loaded '
        'by submit_waveform_tir_roi.sbatch')
    assert 'STAGE2_DIR=$STAGE2_DIR' in loop, (
        'the explicit STAGE2_DIR override must also be forwarded, otherwise an '
        'absolute encoder path is silently ignored')
