"""Automated governance tests for code/notes/design/ and code/notes/plan/features/.

Enforces the repository ADR structure (AGENTS.md §1 & §3):
- G1: Every active design contract under ``code/notes/design/`` follows
  ``ADR-NNN-<slug>.md`` with unique ADR numbers across active + archive.
- G2: Every active implementation plan under ``code/notes/plan/features/``
  follows ``ADR-NNN-<slug>-plan.md`` and pairs 1-to-1 with an active ADR;
  every active ADR either has a matching plan or is explicitly allowlisted
  as a pre-existing baseline in ``BASELINES_WITHOUT_FEATURE_PLAN``.
- G3: ``code/notes/design/README.md`` and ``code/notes/plan/README.md`` list
  every active ADR and feature plan (zero drift).
- G4: Zero dangling references to retired non-ADR design or feature-plan
  filenames remain across active source files, configs, notes, notebooks,
  and thesis sources (``report/``).
"""
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_DIR = REPO_ROOT / 'code' / 'notes' / 'design'
DESIGN_ARCHIVE_DIR = DESIGN_DIR / 'archive'
FEATURES_DIR = REPO_ROOT / 'code' / 'notes' / 'plan' / 'features'
FEATURES_ARCHIVE_DIR = FEATURES_DIR / 'archive'
PLAN_README = REPO_ROOT / 'code' / 'notes' / 'plan' / 'README.md'
DESIGN_README = DESIGN_DIR / 'README.md'

ADR_FILE_RE = re.compile(r'^ADR-(\d{3})-[a-z0-9-]+\.md$')
PLAN_FILE_RE = re.compile(r'^ADR-(\d{3})-[a-z0-9-]+-plan\.md$')

# Pre-existing foundational baselines implemented prior to the feature-plan
# workflow; they map directly to code/notes/spec/ instead of plan/features/.
BASELINES_WITHOUT_FEATURE_PLAN = frozenset({'009', '014', '015'})

RETIRED_FILENAMES = (
    'Evaluation_RespWaveform_protocol.md',
    'TIR_ROI_RESP_TasksDistortion.md',
    'ROI_crop_and_rescaling.md',
    '2DTexture_ROI_cache_preprocessing.md',
    'Video_input_Preprocessing.md',
    'SolutionC_EarlyTokenMixing_stage2.md',
    'SolutionC_EarlyTokenMixing_stage3.md',
    'hpc_AMP.md',
    'task_tiers_regroup_plan.md',
    'ROI_crop_implementation_plan.md',
    '2DTexture_ROI_cache_implementation_plan.md',
    'video_input_preprocessing_implementation_plan.md',
    'hpc_AMP_implementation.md',
)

UNARCHIVED_MOVED_PATTERNS = (
    re.compile(r'design/(?!archive/)SpanMask_PhysioSignals\.md'),
    re.compile(r'design/(?!archive/)ADR-002-asymmetric-multimodal-masking\.md'),
    re.compile(r'design/(?!archive/)ADR-005-asymmetric-multimodal-masking-3streams\.md'),
    re.compile(r'features/(?!archive/)ADR-002-constant-budget-masking-plan\.md'),
    re.compile(r'features/(?!archive/)ADR-005-asymmetric-multimodal-masking-3streams-plan\.md'),
)


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError as exc:
        raise AssertionError(
            f'Non-UTF-8 text file encountered during ADR governance scan: '
            f'{path.relative_to(REPO_ROOT)} ({exc})'
        ) from exc


def _active_adrs():
    return sorted(
        p for p in DESIGN_DIR.glob('*.md')
        if p.name not in {'README.md', 'template.md'}
    )


def _active_feature_plans():
    return sorted(FEATURES_DIR.glob('*.md'))


def test_g1_design_directory_contains_only_valid_unique_adrs():
    adrs = _active_adrs()
    assert adrs, 'Expected active ADR files under code/notes/design/'

    seen_ids = {}
    for p in adrs:
        m = ADR_FILE_RE.match(p.name)
        assert m is not None, f'Non-ADR file in active design/: {p.name}'
        adr_id = m.group(1)
        assert adr_id not in seen_ids, f'Duplicate ADR-{adr_id}: {seen_ids[adr_id]} vs {p.name}'
        seen_ids[adr_id] = p.name

        head = '\n'.join(_read_utf8(p).splitlines()[:15])
        assert '**Status**:' in head, f'Missing **Status** header in {p.name}'

    if DESIGN_ARCHIVE_DIR.exists():
        for p in sorted(DESIGN_ARCHIVE_DIR.glob('ADR-*.md')):
            m = ADR_FILE_RE.match(p.name)
            assert m is not None, f'Invalid archived ADR filename: {p.name}'
            adr_id = m.group(1)
            assert adr_id not in seen_ids, f'ADR-{adr_id} exists in both active and archive/'
            seen_ids[adr_id] = f'archive/{p.name}'


def test_g2_feature_plans_follow_adr_naming_and_pair_with_active_adrs():
    active_adr_ids = {
        ADR_FILE_RE.match(p.name).group(1): p.name
        for p in _active_adrs()
    }
    plans = _active_feature_plans()
    assert plans, 'Expected active feature plans under code/notes/plan/features/'

    plan_ids = {}
    for p in plans:
        m = PLAN_FILE_RE.match(p.name)
        assert m is not None, f'Non-ADR filename in plan/features/: {p.name}'
        adr_id = m.group(1)
        assert adr_id not in plan_ids, f'Duplicate feature plans for ADR-{adr_id}: {plan_ids[adr_id]} vs {p.name}'
        assert adr_id in active_adr_ids, (
            f'Feature plan {p.name} has no matching active design/ADR-{adr_id}-*.md'
        )
        plan_ids[adr_id] = p.name

    # Reverse check: every active ADR either has a feature plan or is explicitly
    # allowlisted as a pre-existing baseline, and the allowlist has no stale IDs.
    unplanned = set(active_adr_ids) - set(plan_ids)
    assert unplanned == BASELINES_WITHOUT_FEATURE_PLAN, (
        f'Active ADRs without feature plans ({sorted(unplanned)}) do not match '
        f'BASELINES_WITHOUT_FEATURE_PLAN ({sorted(BASELINES_WITHOUT_FEATURE_PLAN)})'
    )


def test_g3_design_and_plan_registries_are_synchronized():
    design_readme = _read_utf8(DESIGN_README)
    plan_readme = _read_utf8(PLAN_README)

    for adr in _active_adrs():
        assert adr.name in design_readme, f'{adr.name} missing from code/notes/design/README.md'

    for plan in _active_feature_plans():
        assert plan.name in plan_readme, f'{plan.name} missing from code/notes/plan/README.md'
        assert plan.name in design_readme, f'{plan.name} missing from code/notes/design/README.md'

    if DESIGN_ARCHIVE_DIR.exists():
        for archived in sorted(DESIGN_ARCHIVE_DIR.glob('*.md')):
            assert f'archive/{archived.name}' in design_readme, (
                f'archive/{archived.name} missing from code/notes/design/README.md'
            )


def test_g4_zero_dangling_references_to_retired_or_unarchived_paths():
    skip_dirs = {'__pycache__', '.pytest_cache', '.git', 'archive'}
    exts = {'.md', '.py', '.yaml', '.yml', '.sh', '.sbatch', '.tex', '.bib'}
    this_file = Path(__file__).resolve()
    violations = []

    for base in (REPO_ROOT / 'code', REPO_ROOT / 'notebooks', REPO_ROOT / 'report'):
        if not base.exists():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.suffix not in exts:
                continue
            if path.resolve() == this_file:
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            if path == DESIGN_README:
                continue

            is_adr = path.parent == DESIGN_DIR and path.name.startswith('ADR-')
            for lineno, line in enumerate(_read_utf8(path).splitlines(), 1):
                if is_adr and line.startswith('- **Supersedes**:'):
                    continue
                for retired in RETIRED_FILENAMES:
                    if retired in line:
                        violations.append(f'{path.relative_to(REPO_ROOT)}:{lineno}: {retired}')
                for pat in UNARCHIVED_MOVED_PATTERNS:
                    if pat.search(line):
                        violations.append(f'{path.relative_to(REPO_ROOT)}:{lineno}: {pat.pattern}')

    assert not violations, 'Found dangling note references:\n' + '\n'.join(violations)
