"""ADR-001 pilots: Criterion 2 (determinism) and Criterion 4 (cost).

Both are measured on a SMALL model with synthetic tensors, because what the
criteria constrain is the RECIPE, not the corpus: two evaluations of one
checkpoint must agree bit-for-bit, and a validation pass must stay under 5% of
the run's wall clock / peak VRAM. Running this on the real corpus would cost
hours and answer the same question.

    python scripts/local/pilot_protocol.py --device cuda
    python scripts/local/pilot_protocol.py --device cpu --out /tmp/pilot.json

Output is one JSON document plus a printed table. The numbers are recorded in
`code/notes/plan/ADR-001-subject-kfold-plan.md` (P8).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch

from core import WaveformJointLoss
from core.multimae import _tiny_model
from engines.pretrain import evaluate_pretrain
from engines.waveform import evaluate_waveform_loss
from protocol import MIN_EVAL_FREQ, val_mask_seed

__all__ = ['main']


def _peak(device) -> float:
    if device.type != 'cuda':
        return 0.0
    return torch.cuda.max_memory_allocated() / 2 ** 20      # MiB


def _reset_peak(device) -> None:
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Stage 2: the stochastic-masking path (I6 is only non-trivial here)
# --------------------------------------------------------------------------- #
def stage2_batches(device, batch_size, n_batches):
    out = []
    for _ in range(n_batches):
        out.append({'tir': torch.rand(batch_size, 3, 20, 16, 16, device=device),
                    'resp': torch.randn(batch_size, 1, 80, device=device)})
    return out


def pilot_stage2(device, batch_size, n_batches, eval_freq, keep):
    model = _tiny_model().to(device)
    # ONE batch list, reused by both passes: a determinism check that draws new
    # random tensors per pass compares different data and always "fails".
    batches = stage2_batches(device, batch_size, n_batches)
    val_batches = batches[:max(1, round(n_batches * 0.10 / 0.70))]

    def val_pass():
        return evaluate_pretrain(model, val_batches, device,
                                 mask_base=1234, kept_override=keep)

    def train_pass():
        """One train epoch worth of steps: forward + backward + step."""
        model.train(True)
        optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)
        for batch in batches:
            out = model(batch)
            optimiser.zero_grad(set_to_none=True)
            out['loss'].backward()
            optimiser.step()
        model.train(False)

    _reset_peak(device)
    first = val_pass()
    second = val_pass()
    deterministic = (first['val_loss'] == second['val_loss']
                     and first['components'] == second['components'])
    _reset_peak(device)
    t0 = time.perf_counter()
    train_pass()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_s = time.perf_counter() - t0
    train_vram = _peak(device)

    _reset_peak(device)
    t0 = time.perf_counter()
    val_pass()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    val_s = time.perf_counter() - t0
    val_vram = _peak(device)

    return {
        'deterministic': bool(deterministic),
        'val_loss_first': first['val_loss'],
        'val_loss_second': second['val_loss'],
        'val_pass_s': val_s,
        'train_pass_s': epoch_s,
        'eval_freq': eval_freq,
        'wall_clock_fraction': (val_s / epoch_s) / eval_freq,
        'val_peak_mib': val_vram,
        'train_peak_mib': train_vram,
        'vram_increase_fraction': ((val_vram - train_vram) / train_vram
                                   if train_vram else 0.0),
        'mask_seed_batch0': val_mask_seed(1234, 0),
        'mask_seed_batch1': val_mask_seed(1234, 1),
    }


# --------------------------------------------------------------------------- #
# Stage 3: no stochastic forward, so the pilot only has to prove the cost
# --------------------------------------------------------------------------- #
class _TinyWaveform(torch.nn.Module):
    """Deterministic, shape-compatible stand-in for the waveform head."""

    def __init__(self, in_shape=(3, 20, 16, 16), out_len=80):
        super().__init__()
        self.flat = int(torch.tensor(in_shape).prod())
        self.out_len = out_len
        self.proj = torch.nn.Linear(self.flat, out_len)

    def forward(self, x):
        return self.proj(x.reshape(x.shape[0], -1))


def _waveform_batches(device, batch_size, n_batches):
    return [(torch.rand(batch_size, 3, 20, 16, 16, device=device),
             torch.randn(batch_size, 80, device=device))
            for _ in range(n_batches)]


def pilot_stage3(device, batch_size, n_batches, eval_freq):
    model = _TinyWaveform().to(device)
    criterion = WaveformJointLoss(fft_sizes=(32, 64))
    # One batch list for both passes (see pilot_stage2) and the corpus's own
    # 70/10 ratio, so the measured wall-clock fraction is the one Criterion 4
    # constrains: val pass / train epoch, divided by eval_freq.
    batches = _waveform_batches(device, batch_size, n_batches)
    val_batches = batches[:max(1, round(n_batches * 0.10 / 0.70))]

    def val_pass():
        return evaluate_waveform_loss(val_batches, model, criterion, device,
                                      with_spectral=False)

    def train_pass():
        model.train(True)
        optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)
        for samples, target in batches:
            loss = criterion(model(samples), target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        model.train(False)

    _reset_peak(device)
    first = val_pass()
    second = val_pass()
    deterministic = first['val_loss'] == second['val_loss']
    _reset_peak(device)
    t0 = time.perf_counter()
    train_pass()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_s = time.perf_counter() - t0
    train_vram = _peak(device)

    _reset_peak(device)
    t0 = time.perf_counter()
    val_pass()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    val_s = time.perf_counter() - t0
    val_vram = _peak(device)

    return {
        'deterministic': bool(deterministic),
        'val_loss_first': first['val_loss'],
        'val_loss_second': second['val_loss'],
        'val_pass_s': val_s,
        'train_pass_s': epoch_s,
        'eval_freq': eval_freq,
        'wall_clock_fraction': (val_s / epoch_s) / eval_freq,
        'val_peak_mib': val_vram,
        'train_peak_mib': train_vram,
        'vram_increase_fraction': ((val_vram - train_vram) / train_vram
                                   if train_vram else 0.0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser('ADR-001 Criterion 2/4 pilots')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--n_batches', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--val_kept_tir', default=0.25, type=float)
    parser.add_argument('--val_kept_resp', default=0.50, type=float)
    parser.add_argument('--out', default='', type=str)
    args = parser.parse_args(argv)

    device = torch.device(args.device if (args.device != 'cuda'
                                          or torch.cuda.is_available()) else 'cpu')
    if args.device == 'cuda' and device.type != 'cuda':
        print('[pilot] WARNING: --device cuda requested but CUDA is unavailable; '
              'running on CPU (VRAM columns are then meaningless).')
    keep = {'tir': args.val_kept_tir, 'resp': args.val_kept_resp}

    results = {'device': str(device),
               'torch': torch.__version__,
               'batch_size': args.batch_size,
               'n_batches': args.n_batches,
               'stage2': pilot_stage2(device, args.batch_size, args.n_batches,
                                      args.eval_freq, keep),
               'stage3': pilot_stage3(device, args.batch_size, args.n_batches,
                                      args.eval_freq),
               'criteria': {'determinism': 'bit-identical val records',
                            'wall_clock': '< 5% of the run',
                            'vram': '< 5% peak increase'}}

    print(f'\n[pilot] device={device}  eval_freq={args.eval_freq}  '
          f'min_eval_freq={MIN_EVAL_FREQ}')
    header = (f'{"stage":8s} {"determ.":8s} {"val_s":>8s} {"train_s":>8s} '
              f'{"wall%":>7s} {"vram+%":>8s}')
    print(header)
    print('-' * len(header))
    for name in ('stage2', 'stage3'):
        block = results[name]
        print(f'{name:8s} {str(block["deterministic"]):8s} '
              f'{block["val_pass_s"]:8.3f} {block["train_pass_s"]:8.3f} '
              f'{100 * block["wall_clock_fraction"]:6.2f}% '
              f'{100 * block["vram_increase_fraction"]:7.2f}%')

    ok_det = results['stage2']['deterministic'] and results['stage3']['deterministic']
    ok_wall = all(results[name]['wall_clock_fraction'] < 0.05
                  for name in ('stage2', 'stage3'))
    ok_vram = all(results[name]['vram_increase_fraction'] < 0.05
                  for name in ('stage2', 'stage3'))
    results['pass'] = {'criterion2_determinism': ok_det,
                       'criterion4_wall_clock': ok_wall,
                       'criterion4_vram': ok_vram}
    print(f'\n[pilot] criterion 2 (determinism): {"PASS" if ok_det else "FAIL"}')
    print(f'[pilot] criterion 4 (wall clock): {"PASS" if ok_wall else "FAIL"}')
    print(f'[pilot] criterion 4 (peak VRAM): {"PASS" if ok_vram else "FAIL"}')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
            handle.write('\n')
        print(f'[pilot] wrote {args.out}')
    return 0 if (ok_det and ok_wall and ok_vram) else 1


if __name__ == '__main__':
    raise SystemExit(main())
