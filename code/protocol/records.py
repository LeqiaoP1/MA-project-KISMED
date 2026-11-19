"""Artefact writers and readers (ADR-001 section 2.3).

One writer per artefact, one schema version each, atomic on disk. Runners never
format their own JSON: a selection record that cannot be read back is not a
record.
"""
import json
import os
from typing import Any, Dict, List, Mapping, Optional

from .constants import SCHEMA_EVAL, SCHEMA_SELECTION
from .plan import ProtocolError

__all__ = ['write_json', 'read_json', 'append_jsonl', 'read_jsonl',
           'make_eval_record', 'write_selection', 'read_selection']

_REQUIRED_EVAL = ('epoch', 'value_kind', 'criterion_value', 'counts', 'seed')


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` atomically (process-unique temp, then one replace)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path: str) -> Dict[str, Any]:
    """Read a JSON artefact; a missing file is a protocol error, not ``{}``."""
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        raise ProtocolError(f'expected an artefact at {path}, found nothing.') from None
    if not isinstance(payload, dict):
        raise ProtocolError(f'{path} must hold a JSON object; got '
                            f'{type(payload).__name__}.')
    return payload


def append_jsonl(path: str, payload: Mapping[str, Any]) -> None:
    """Append one line. Callers restrict this to rank 0."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    line = json.dumps(dict(payload), sort_keys=True) + '\n'
    with open(path, 'a') as handle:      # O_APPEND: one record, one write
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL artefact; a malformed line is an error, never skipped."""
    try:
        with open(path) as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
    except FileNotFoundError:
        raise ProtocolError(f'expected a record at {path}, found nothing.') from None
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ProtocolError(
                f'{path}:{index + 1} is not valid JSON ({exc}); the run record '
                f'was probably truncated.') from None
    return records


def _numeric(mapping: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Keep the scalar numbers of a component dict, drop everything else.

    ``evaluation.metrics.spectral_metrics`` returns non-numeric fields
    (``psd_band_bins=None``, label lists) alongside the numbers; a record is a
    flat numeric log, so those are dropped rather than coerced.
    """
    out: Dict[str, float] = {}
    for key, value in (mapping or {}).items():
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def make_eval_record(*, epoch: int, value_kind: str, criterion_value: float,
                     counts: Mapping[str, int], seed: Mapping[str, int],
                     components: Optional[Mapping[str, float]] = None,
                     pinned: Optional[Mapping[str, Any]] = None,
                     role_digest: str = '') -> Dict[str, Any]:
    """One validation-evaluation line (ADR-001 section 2.3).

    ``counts`` and ``seed`` are mandatory: a validation value without its sample
    count, its subject count and its seeds cannot be compared with the next
    epoch's.
    """
    if 'clips' not in counts or 'subjects' not in counts:
        raise ProtocolError(
            f'eval record counts must name clips and subjects; got {sorted(counts)}')
    if not seed:
        raise ProtocolError('eval record must record its seeds.')
    record = {
        'schema': SCHEMA_EVAL,
        'epoch': int(epoch),
        'value_kind': str(value_kind),
        'criterion_value': float(criterion_value),
        'components': _numeric(components),
        'counts': {str(k): int(v) for k, v in counts.items()},
        'seed': {str(k): int(v) for k, v in seed.items()},
        'pinned': dict(pinned or {}),
        'role_digest': str(role_digest),
    }
    missing = [key for key in _REQUIRED_EVAL if key not in record]
    if missing:                                   # defensive, keeps the schema honest
        raise ProtocolError(f'eval record is missing {missing}')
    return record


def write_selection(path: str, record: Mapping[str, Any]) -> None:
    """Write the selection record, refusing a record without its rule."""
    if record.get('schema') != SCHEMA_SELECTION:
        raise ProtocolError(
            f'selection record must carry schema {SCHEMA_SELECTION!r}; got '
            f'{record.get("schema")!r}')
    for key in ('rule', 'eval_freq', 'selected_epoch', 'criterion',
                'criterion_value'):
        if key not in record:
            raise ProtocolError(f'selection record is missing {key!r}.')
    write_json(path, record)


def read_selection(path: str) -> Dict[str, Any]:
    """Read a selection record and check its schema."""
    record = read_json(path)
    if record.get('schema') != SCHEMA_SELECTION:
        raise ProtocolError(
            f'{path} is not a selection record (schema '
            f'{record.get("schema")!r}).')
    return record
