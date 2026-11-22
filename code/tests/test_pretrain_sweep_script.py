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


def _dry_run_env(tmp_path, **extra):
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
    env.pop('SLURM_NTASKS', None)
    env.update(extra)
    return env


def test_single_task_launches_plain_python(tmp_path):
    proc = subprocess.run(['bash', SCRIPT], cwd=CODE_DIR, env=_dry_run_env(tmp_path),
                          check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    train_line = [ln for ln in proc.stdout.splitlines()
                  if 'runners/run_pretrain.py' in ln][0]
    assert '--ntasks' not in train_line


def test_multi_task_launches_srun_per_gpu(tmp_path):
    """utils/dist.py reads world_size from SLURM_NTASKS, so the ranks must be
    spawned: one plain python in a 4-task allocation would block in
    init_process_group() instead of failing."""
    proc = subprocess.run(
        ['bash', SCRIPT], cwd=CODE_DIR,
        env=_dry_run_env(tmp_path, SLURM_NTASKS='4', SRUN='true'),
        check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    train_line = [ln for ln in proc.stdout.splitlines()
                  if 'runners/run_pretrain.py' in ln][0]
    assert '--ntasks=4' in train_line
    partition_line = [ln for ln in proc.stdout.splitlines()
                      if 'runners/run_partition.py' in ln][0]
    assert '--ntasks' not in partition_line


def test_epochs_override_reaches_the_runner(tmp_path):
    proc = subprocess.run(
        ['bash', SCRIPT], cwd=CODE_DIR,
        env=_dry_run_env(tmp_path, EPOCHS='1'),
        check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    train_line = [ln for ln in proc.stdout.splitlines()
                  if 'runners/run_pretrain.py' in ln][0]
    assert '--epochs 1' in train_line
    # EPOCHS must never reach the partition step.
    partition_line = [ln for ln in proc.stdout.splitlines()
                      if 'runners/run_partition.py' in ln][0]
    assert '--epochs' not in partition_line
