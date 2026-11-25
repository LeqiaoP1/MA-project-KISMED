"""Capture the ADR-007 mask-sequence neutrality fixture (Plan Phase 0).

Why this exists
---------------
ADR-007 widens the constant-budget policy from "exactly TWO visual streams" to
"one or two". That widening touches ``ConstantBudgetApportioner._validate`` and
the draw paths that every existing run (ADR-002/005/006) shares. The neutrality
guarantee -- *a 2-visual run produces the bit-identical mask sequence it
produced before this change* -- can only be MEASURED against a reference taken
**before** the code changes, so the reference is captured here and committed as
``tests/fixtures/adr007_neutrality_fixture.json``.

The fixture stores, per epoch context, a SHA-256 over the full plan sequence plus
the generator's next raw draw. The next-draw value is the part that actually
detects a "consumed-then-ignored" random number: a change that draws a sample and
discards it would shift every later count, and would also move this value.

What is captured
----------------
The ADR-006 three-stream arm's own tuple (``stage2_budget_3regime_single_modality``
values, a superset of the ADR-005 arm): ``tir``+``rgb2d`` visual (36 tubes x 25
tokens) and ``resp`` signal (250 tokens), ``N = 350``, caps 0.25 / 0.50, floor 2
tubes, ``p_warm = 0.10`` below ``E_warm = 50`` then ``p_target = 0.50``, and the
single-carrier class at ``p_s = 0.20`` (warm-up 0.00) with ``S_v = 14``.

Usage
-----
    cd code && ../.env/bin/python -m scripts.project.capture_adr007_neutrality
    # then, after the Phase-1 change, verify:
    cd code && ../.env/bin/python -m scripts.project.capture_adr007_neutrality --check

``--check`` exits non-zero and prints the first divergent (epoch, step) if any
plan, plan label or trailing draw differs from the committed fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from core.budget_masking import (SIGNAL, VISUAL, ConstantBudgetApportioner,  # noqa: E402
                                 StreamSpec)
from core.mask_placement import counts_generator  # noqa: E402

#: the ADR-006 arm tuple -- see
#: ``configs/pretrain/stage2_budget_3regime_single_modality.yaml``.
N_BUDGET = 350
T = 25                      # tokens per visual tube
SPATIAL = 36                # 6x6 patch grid
PHYSIO = 250                # 1-D tokens per signal stream
VIDEO_CAP_RATIO = 0.25
PHYSIO_CAP_RATIO = 0.50
VIDEO_MIN_TUBES = 2
P_VIRTUAL = 0.50
P_WARM = 0.10
E_WARM = 50
P_SINGLE = 0.20
P_SINGLE_WARM = 0.00
SINGLE_CARRIER_TUBES = 14

MASK_BASE = 1234
STEPS = 50_000
EPOCH_CONTEXTS = (None, 0, 50, 149)

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'adr007_neutrality_fixture.json')


def build_apportioner() -> ConstantBudgetApportioner:
    """The ADR-006 three-stream apportioner, from the arm's own numbers."""
    streams = (
        StreamSpec('tir', VISUAL, SPATIAL, tokens_per_unit=T),
        StreamSpec('rgb2d', VISUAL, SPATIAL, tokens_per_unit=T),
        StreamSpec('resp', SIGNAL, PHYSIO, tokens_per_unit=1),
    )
    return ConstantBudgetApportioner(
        streams, n_budget=N_BUDGET,
        video_cap_ratio=VIDEO_CAP_RATIO,
        physio_cap_ratio=PHYSIO_CAP_RATIO,
        video_min_tubes=VIDEO_MIN_TUBES,
        p_virtual_sensing=P_VIRTUAL,
        virtual_sensing_warmup_p=P_WARM,
        virtual_sensing_warmup_epochs=E_WARM,
        single_carrier_tubes=SINGLE_CARRIER_TUBES,
        p_single_modality=P_SINGLE,
        single_modality_warmup_p=P_SINGLE_WARM,
    )


def plan_signature(plan) -> str:
    """One canonical, hash-stable line per plan.

    Counts and labels only -- placement is drawn per sample from a different
    generator and is covered by its own tests. ``regime`` and ``carrier`` are
    included so a change in the step-class logic cannot pass unnoticed.
    """
    tokens = ','.join(f'{k}={int(v)}' for k, v in sorted(plan.kept_tokens.items()))
    units = ','.join(f'{k}={int(v)}' for k, v in sorted(plan.kept_units.items()))
    return (f'{int(plan.mode)}/{int(plan.mode_by_counts)}/{plan.regime}/'
            f'{plan.carrier}|{tokens}|{units}')


def digest_of(epoch_context, steps: int = STEPS) -> dict:
    """``{digest, next_draw, n_steps}`` for one epoch context."""
    app = build_apportioner()
    gen_epoch = 0 if epoch_context is None else int(epoch_context)
    acc = hashlib.sha256()
    first = None
    for step in range(steps):
        g = counts_generator(MASK_BASE, gen_epoch, step)
        plan = app.sample_step(g, epoch=epoch_context)
        line = plan_signature(plan)
        if first is None:
            first = line
        acc.update(line.encode('ascii'))
        acc.update(b'\n')
        # The generator is re-derived per step (exactly the training path), so
        # the trailing-draw probe needs its own final generator below.
    g_last = counts_generator(MASK_BASE, gen_epoch, steps)
    trailing = build_apportioner().sample_step(g_last, epoch=epoch_context)
    # one extra RAW draw from a generator whose plan was consumed: if any step
    # consumed a different number of samples, this value moves.
    next_draw = float(torch.rand(1, dtype=torch.float64, generator=g_last).item())
    return {'digest': acc.hexdigest(), 'next_draw': next_draw,
            'n_steps': int(steps), 'first_plan': first,
            'trailing_plan': plan_signature(trailing)}


def engine_fingerprint() -> dict:
    """SHA-256 of the engine files the fixture is tied to."""
    out = {}
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for name in ('core/budget_masking.py', 'core/mask_placement.py'):
        with open(os.path.join(root, name), 'rb') as fh:
            out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def capture() -> dict:
    return {
        'format': 'ADR-007 neutrality fixture v1',
        'purpose': ('bit-identical 2-visual mask sequences before/after the '
                    'ADR-007 one-visual extension (plan DoD 8 / gate G8)'),
        'engine': engine_fingerprint(),
        'tuple': {
            'streams': ['tir:visual:36x25', 'rgb2d:visual:36x25',
                        'resp:signal:250x1'],
            'n_budget': N_BUDGET, 'tokens_per_unit': T,
            'video_cap_ratio': VIDEO_CAP_RATIO,
            'physio_cap_ratio': PHYSIO_CAP_RATIO,
            'video_min_tubes': VIDEO_MIN_TUBES,
            'p_virtual_sensing': P_VIRTUAL,
            'virtual_sensing_warmup_p': P_WARM,
            'virtual_sensing_warmup_epochs': E_WARM,
            'p_single_modality': P_SINGLE,
            'single_modality_warmup_p': P_SINGLE_WARM,
            'single_carrier_tubes': SINGLE_CARRIER_TUBES,
        },
        'mask_base': MASK_BASE,
        'steps_per_context': STEPS,
        'epoch_contexts': {str(e): digest_of(e) for e in EPOCH_CONTEXTS},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true',
                   help='compare against the committed fixture instead of '
                        'writing it')
    p.add_argument('--steps', type=int, default=STEPS)
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)

    if not a.check:
        data = capture()
        with open(FIXTURE, 'w') as fh:
            json.dump(data, fh, indent=2, sort_keys=False)
            fh.write('\n')
        print(f'[fixture] wrote {FIXTURE}')
        for ctx, rec in data['epoch_contexts'].items():
            print(f'  epoch {ctx:>4}: {rec["n_steps"]} steps  '
                  f'sha256 {rec["digest"][:16]}…  '
                  f'next_draw {rec["next_draw"]!r}')
        return 0

    with open(FIXTURE) as fh:
        ref = json.load(fh)
    bad = 0
    for ctx, want in ref['epoch_contexts'].items():
        epoch = None if ctx == 'None' else int(ctx)
        got = digest_of(epoch, steps=int(want['n_steps']))
        for key in ('digest', 'next_draw', 'first_plan', 'trailing_plan'):
            if got[key] != want[key]:
                bad += 1
                print(f'[fixture] MISMATCH epoch {ctx} {key}:\n'
                      f'    want {want[key]!r}\n    got  {got[key]!r}')
    if bad:
        print(f'[fixture] FAILED: {bad} divergence(s) -> the 2-visual mask '
              f'sequence is NOT neutral.')
        return 1
    print(f'[fixture] OK: {len(ref["epoch_contexts"])} epoch context(s), '
          f'{sum(r["n_steps"] for r in ref["epoch_contexts"].values())} plans '
          f'and every trailing draw are bit-identical.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
