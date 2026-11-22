"""The acc_short chain wrapper: links, dependencies, and the progress guard.

Rationale under test: `acc`/`acc_long` have PriorityJobFactor=10 while
`acc_short` has 100, so a multi-hour run only starts as a chain of 30-minute
allocations. Each link must (a) depend on its predecessor with `afterany` so a
walltime kill hands off, and (b) save a checkpoint often enough that the link
actually advances.
"""
import os
import subprocess

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAIN = os.path.join(CODE_DIR, 'scripts', 'hpc', 'submit_pretrain_chain.sh')


def _dry_run(**extra):
    env = os.environ.copy()
    env.update({
        'EXP_ROOT': '/work/scratch/x/fourstream_v1',
        'FOLDS': '0',
        'RAW_ROOT': '/tmp/raw',
        'ROI_CACHE': '/tmp/roi',
        'RGB2D_CACHE': '/tmp/rgb2d',
        'RESP_CACHE': '/tmp/resp',
        'BP_ARTEFACT': '/tmp/bp',
        'DRY_RUN': '1',
        'LINKS': '3',
    })
    env.update(extra)
    return subprocess.run(['bash', CHAIN], cwd=CODE_DIR, env=env,
                          check=False, capture_output=True, text=True)


def test_chain_requires_folds_and_exp_root():
    env = os.environ.copy()
    env.pop('FOLDS', None)
    env.pop('EXP_ROOT', None)
    env['DRY_RUN'] = '1'
    proc = subprocess.run(['bash', CHAIN], cwd=CODE_DIR, env=env,
                          check=False, capture_output=True, text=True)
    assert proc.returncode != 0
    assert 'EXP_ROOT' in proc.stderr or 'FOLDS' in proc.stderr


def test_first_link_has_no_dependency_and_the_rest_chain():
    proc = _dry_run()
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('[chain] sbatch')]
    assert len(lines) == 3
    assert '--dependency' not in lines[0]
    assert '--dependency=afterany:DRYRUN1' in lines[1]
    assert '--dependency=afterany:DRYRUN2' in lines[2]


def test_links_request_acc_short_resources_and_never_stage3():
    proc = _dry_run()
    out = proc.stdout
    assert '--partition=acc_short' in out
    assert '--time=00:30:00' in out
    assert '--gres=gpu:a100:4' in out
    assert '--ntasks=4' in out
    assert 'run_waveform.py' not in out


def test_progress_guard_sets_save_ckpt_freq_unless_overridden():
    proc = _dry_run()
    assert '--save_ckpt_freq 5' in proc.stdout

    proc = _dry_run(SAVE_FREQ='2')
    assert '--save_ckpt_freq 2' in proc.stdout
