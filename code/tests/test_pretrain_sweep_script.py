"""The pretrain-only HPC wrapper must never launch Stage 3."""
import os
import subprocess

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(CODE_DIR, 'scripts', 'hpc', 'submit_pretrain_sweep.sh')
SBATCH = os.path.join(CODE_DIR, 'scripts', 'hpc', 'submit_pretrain_sweep.sbatch')


def test_pretrain_sweep_is_stage2_only():
    src = open(SCRIPT).read() + open(SBATCH).read()
    assert 'runners/run_partition.py' in src
    assert 'runners/run_pretrain.py' in src
    # Ignore comments: the header deliberately explains that Stage 3 is absent.
    code = '\n'.join(line.split('#', 1)[0] for line in src.splitlines())
    assert 'run_waveform.py' not in code


def test_dry_run_prints_stage2_commands_only(tmp_path):
    env = os.environ.copy()
    env.update({
        'EXP_ROOT': str(tmp_path / 'exp'),
        'FOLDS': '0',
        'RAW_ROOT': '/tmp/raw',
        'DATA_SET2': 'tir_roi_resp_rgb2d_bp',
        'ROI_CACHE': '/tmp/roi',
        'RGB2D_CACHE': '/tmp/rgb2d',
        'RESP_CACHE': '/tmp/resp',
        'BP_ARTEFACT': '/tmp/bp',
        'DRY_RUN': '1',
        'PY': 'python',
    })
    proc = subprocess.run(['bash', SCRIPT], cwd=CODE_DIR, env=env,
                          check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert 'runners/run_partition.py' in out
    assert 'runners/run_pretrain.py' in out
    assert 'run_waveform.py' not in out
    for flag in ('--roi_cache', '--rgb2d_cache', '--resp_cache',
                 '--bp_artefact'):
        assert flag in out
