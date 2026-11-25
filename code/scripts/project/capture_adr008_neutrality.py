"""Capture the ADR-008 objective neutrality fixture (Plan Phase 0, decision D1).

Why this exists
---------------
ADR-008 adds a composite objective whose **neutral defaults must be the old
run**: with ``(alpha, beta, gamma, lambda_align) = (1, 0, 0, 0)`` the model must
build no new modules and produce a *bit-identical* loss, per-stream loss and
parameter set. ``Criterion 1.1`` of the ADR makes that a gate, so the reference
has to be captured **before** ``core/`` changes -- that is all this script does.

What is captured
----------------
Two entries, both pinned by an explicit ``plan_from_counts`` allocation (no
random mask draw) and by ``torch.manual_seed(0)`` before construction:

``tiny``
    ``_tiny_model()`` (the ADR-002 3-stream geometry: 16 spatial tubes, 10
    tokens/tube, 20 signal tokens, ``n_budget = 80``), ``B = 2``, **Mode 1 and
    Mode 2**, with a backward pass. Records the loss values *and* the gradient
    norms of a fixed parameter subset, so a change in the reduction order shows
    up even when the forward value happens to agree. This is the entry the
    ``G1`` pytest consumes -- it runs in ~1 s.

``adr007_geometry``
    The shipped ADR-007 tokenizer geometry at ViT-Base (``input_size 96``,
    ``tubelet (2,16,16)``, ``num_frames 50``, ``sig_kernel 32``,
    ``seq_len 8000``, ``fs 1000``, ``n_budget 200``, ``physio_cap_ratio 0.40``,
    ``anatomy_aware``) with ``B = 1``, **Mode 1 and Mode 2**, forward only
    (``no_grad``). It exercises the exact ``[B, 250, 32]`` / ``L = 8000`` shapes
    the composite spectral term will see, so a shape or target-normalisation
    regression cannot hide behind the tiny model. Cost is ~20-30 s on CPU, which
    is why the pytest suite uses the ``tiny`` entry.

Both entries also record the parameter count and the sorted ``state_dict`` key
list: a neutral run that silently grew a new module (the failure R5 forbids)
changes the key list even if every loss value still matches.

Usage
-----
    cd code && python -m scripts.project.capture_adr008_neutrality
    # after Phase 1-3, verify nothing moved:
    cd code && python -m scripts.project.capture_adr008_neutrality --check
    # fast check (skips the ViT-Base entry):
    cd code && python -m scripts.project.capture_adr008_neutrality --check --skip-heavy

``--check`` exits non-zero and prints the first divergent field per entry.

Environment note
----------------
``torch.set_num_threads(1)`` is set at import time: CPU matmul reduction order
depends on the thread count, and a fixture is only meaningful if capture and
check execute the same kernel schedule. The ``G1`` test does the same.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from core.multimae import MultiModalMAE, _tiny_model  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(HERE, 'tests', 'fixtures', 'adr008_neutrality_fixture.json')

#: the exact ADR-007 arm geometry (``configs/pretrain/stage2_budget_tir_resp_2stream_anatomy.yaml``)
ADR007 = dict(
    streams=('tir', 'resp'), embed_dim=768, enc_depth=12, enc_num_heads=12,
    mlp_ratio=4.0, dec_depth=2, tubelet=(2, 16, 16), num_frames=50,
    input_size=96, sig_kernel=32, seq_len=8000, fps=25.0, fs=1000.0,
    temporal_stride=4, n_budget=200, video_cap_ratio=0.25,
    physio_cap_ratio=0.40, video_min_tubes=4, p_virtual_sensing=0.50,
    virtual_sensing_warmup_p=0.10, virtual_sensing_warmup_epochs=20,
    spatial_sampling='anatomy_aware', core_row_bounds=(2, 3),
    core_col_bounds=(2, 3), core_tubes_min=1, core_tubes_max=2,
    periph_min_tubes=1, physio_draw_min=1, signal_weight=0.5,
    pos_init='sincos3d', target_norm='none',
)

#: gradient probes for the tiny entry: (module path, parameter name). The
#: signal adapter is ``SignalEmbed.embed`` (a ``Conv1d``), the visual one is
#: ``TubeletEmbed.patch_embed`` (a ``Conv3d``); both ``heads`` are ``Linear``.
GRAD_PROBES = (
    ('enc_norm', 'weight'),
    ('adapters.resp.embed', 'weight'),
    ('adapters.tir.patch_embed', 'weight'),
    ('heads.resp', 'weight'),
    ('heads.tir', 'weight'),
)


def _hash_file(rel: str) -> str:
    with open(os.path.join(HERE, rel), 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def engine_fingerprint() -> dict:
    """SHA-256 of the files the captured numbers are tied to."""
    return {rel: _hash_file(rel) for rel in
            ('core/criterion.py', 'core/multimae.py', 'core/budget_masking.py')}


def _inputs(model, batch: int) -> dict:
    g = torch.Generator().manual_seed(7)
    x = {}
    for s in model.streams:
        if s in model.visual:
            shape = (batch, model.stream_channels.get(s, 3),
                     model.num_frames, model.input_size, model.input_size)
        else:
            shape = (batch, 1, model.seq_len)
        x[s] = torch.randn(*shape, generator=g)
    return x


def _plans(model, which: str):
    """The pinned Mode-1 / Mode-2 allocations for a model, from its own caps."""
    app = model.apportioner
    if which == 'tiny':
        return {'mode1': {'tir': 40, 'rgb2d': 40, 'resp': 0},
                'mode2': {'tir': 40, 'rgb2d': 30, 'resp': 10}}
    # ADR-007 geometry: Mode 1 = 200/0 (S_tir = 8), Mode 2 = 175/25 (S_tir = 7)
    return {'mode1': {'tir': 200, 'resp': 0},
            'mode2': {'tir': 175, 'resp': 25}}


def _record(model, which: str, plan_name: str, counts: dict,
            *, backward: bool) -> dict:
    app = model.apportioner
    plan = app.plan_from_counts(counts, kept_core=1 if model.anatomy_aware else None)
    x = _inputs(model, 2 if which == 'tiny' else 1)
    model.train(True)
    if backward:
        model.zero_grad(set_to_none=True)
        out = model(x, plan=plan)
        total = out['loss']
        total.backward()
        grads = {}
        for mod, name in GRAD_PROBES:
            obj = model
            for part in mod.split('.'):
                obj = getattr(obj, part)
            p = getattr(obj, name)
            grads[f'{mod}.{name}'] = (None if p.grad is None
                                      else float(p.grad.detach().norm(2)))
        # leave no state behind for the next mode
        model.zero_grad(set_to_none=True)
    else:
        with torch.no_grad():
            out = model(x, plan=plan)
            total = out['loss']
        grads = {}
    with torch.no_grad():
        return {
            'counts': {k: int(v) for k, v in counts.items()},
            'mode': int(out['mode']),
            'mode_resolved': int(out['mode_resolved']),
            'total': float(total),
            'losses': {k: float(v) for k, v in out['losses'].items()},
            'losses_mse': {k: float(v) for k, v in out['losses_mse'].items()},
            'grad_norms': grads,
        }


def _build(which: str):
    torch.manual_seed(0)
    if which == 'tiny':
        return _tiny_model()
    return MultiModalMAE(**ADR007)


def capture_one(which: str, *, backward: bool) -> dict:
    model = _build(which)
    entries = {}
    for plan_name, counts in _plans(model, which).items():
        # a FRESH model per mode: the pinned plans must not share optimizer /
        # RNG state, and the tiny model's dropout-free forward makes this exact
        m = _build(which)
        entries[plan_name] = _record(m, which, plan_name, counts,
                                     backward=backward)
    return {
        'geometry': {
            'streams': list(model.streams),
            'embed_dim': int(model.embed_dim),
            'enc_depth': len(model.enc_blocks),
            'n_budget': int(model.n_budget),
            'spatial_tubes': int(model.spatial_patches),
            'tokens_per_unit': int(model.grid_t),
            'signal_tokens': int(model.physio_tokens),
            'sig_kernel': int(model.sig_kernel),
            'seq_len': int(model.seq_len),
            'target_norm': str(model.target_norm),
            'spatial_sampling': str(model.spatial_sampling),
        },
        'n_parameters': int(sum(p.numel() for p in model.parameters())),
        'state_keys': sorted(model.state_dict().keys()),
        'modes': entries,
    }


def capture(*, skip_heavy: bool = False) -> dict:
    data = {
        'format': 'ADR-008 objective neutrality fixture v1',
        'purpose': ('bit-identical neutral objective (DoD 7 / gate G1, decision '
                    'D1): (alpha, beta, gamma, lambda_align) = (1, 0, 0, 0) must '
                    'reproduce these numbers, this parameter count and this key '
                    'list exactly'),
        'threads': int(torch.get_num_threads()),
        'torch': torch.__version__,
        'engine': engine_fingerprint(),
        'tiny': capture_one('tiny', backward=True),
    }
    if not skip_heavy:
        data['adr007_geometry'] = capture_one('adr007_geometry', backward=False)
    return data


def _diff(prefix: str, want, got, out: list) -> None:
    if isinstance(want, dict) and isinstance(got, dict):
        for key in want:
            if key not in got:
                out.append(f'{prefix}.{key}: missing in the re-run')
                continue
            _diff(f'{prefix}.{key}', want[key], got[key], out)
        for key in got:
            if key not in want:
                out.append(f'{prefix}.{key}: NEW in the re-run ({got[key]!r})')
        return
    if isinstance(want, list) and isinstance(got, list):
        if want != got:
            extra = [k for k in got if k not in want]
            missing = [k for k in want if k not in got]
            out.append(f'{prefix}: list differs (missing={missing[:6]}, '
                       f'new={extra[:6]})')
        return
    if want != got:
        out.append(f'{prefix}: want {want!r}, got {got!r}')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true',
                   help='compare against the committed fixture instead of writing it')
    p.add_argument('--skip-heavy', action='store_true',
                   help='do not build the ViT-Base ADR-007 geometry entry')
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)

    if not a.check:
        data = capture(skip_heavy=a.skip_heavy)
        with open(FIXTURE, 'w') as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write('\n')
        print(f'[fixture] wrote {FIXTURE}')
        for name in ('tiny', 'adr007_geometry'):
            if name not in data:
                continue
            rec = data[name]
            print(f'  {name}: {rec["n_parameters"]} params, '
                  f'{len(rec["state_keys"])} state keys')
            for mode, m in rec['modes'].items():
                print(f'    {mode}: total {m["total"]:.12f}  '
                      f'mse {m["losses_mse"]}')
        return 0

    with open(FIXTURE) as fh:
        ref = json.load(fh)
    got = capture(skip_heavy=a.skip_heavy or 'adr007_geometry' not in ref)
    problems: list = []
    for name in ('tiny', 'adr007_geometry'):
        if name not in ref:
            continue
        _diff(name, ref[name], got[name], problems)
    if problems:
        print(f'[fixture] FAILED: {len(problems)} divergence(s) -- the neutral '
              f'objective is NOT bit-identical:')
        for line in problems[:25]:
            print(f'  {line}')
        return 1
    print('[fixture] OK: neutral objective is bit-identical (values, grads, '
          'parameter count and state-dict keys).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
