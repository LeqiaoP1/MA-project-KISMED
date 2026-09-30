"""Named task GROUPS ("distortion levels") -> explicit task-label selection.

A BP4D+ session is ``<subject>_<task>`` with ``task`` one of ``T1`` .. ``T10``.
The tasks differ a lot in how much they disturb the recordings (head motion,
speech, pain), so a run often wants to be restricted to a SUBSET of them. This
module turns *named* subsets into the explicit task-label list that the existing
``discover_sessions(..., tasks=...)`` filter already understands.

**No mapping is hardcoded here** -- the group definitions live in the YAML config
and are passed in::

    task_groups:                # name -> list of task labels (free-form)
      low:      [T1, T2]
      moderate: [T3, T4, T5, T6, T10]
      high:     [T7, T8, T9]
    task_set: ''                # '' = all; 'low' or 'low,high' = union

The SELECTION is resolved to explicit task labels ONCE at dataset-build time::

    final_tasks = union(explicit --tasks CSV, every task of every selected group)

so nothing downstream (the datasets, the split logic, the loaders) has to know
about groups. An empty result means "all tasks" (the historical behaviour), and
an unknown group name raises immediately, because a silent fallback to "all
tasks" would silently train on a different corpus than the one requested.

Accepted forms
--------------
``task_groups`` (definitions)

* a mapping: ``{'low': ['T1', 'T2'], 'high': 'T7,T8,T9'}``
* a string: ``'low=T1|T2;high=T7|T8|T9'`` (``;`` separates groups, ``=`` separates
  a name from its tasks, and tasks are separated by ``,``, ``|`` or whitespace)
* ``None`` / ``''`` -> no groups defined

``task_set`` (selection)

* ``'low'`` / ``'low,high'`` / ``['low', 'high']`` (names are case-insensitive)
* ``None`` / ``''`` / ``'all'`` -> no group filter

``tasks`` (the pre-existing explicit list)

* ``'T1,T2'`` / ``['T1', 'T2']`` / ``None`` (labels are upper-cased)
"""
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ['TASK_LABEL_RE', 'normalise_task', 'normalise_group',
           'parse_task_groups', 'parse_task_set', 'TaskSelection',
           'levels_by_task', 'resolve_task_selection']

#: a BP4D+ task label, e.g. ``T1`` .. ``T10`` (case-insensitive on input).
TASK_LABEL_RE = re.compile(r'^[Tt]\d+$')

_GROUP_SEP = ';'
_ASSIGN_SEP = ('=', ':')
_ITEM_SEP = (',', '|')


class TaskGroupError(ValueError):
    """A malformed group definition or an unusable selection."""


# --------------------------------------------------------------------------- #
# normalisation helpers
# --------------------------------------------------------------------------- #
def _task_sort_key(label: str):
    """``'T10'`` sorts after ``'T9'`` (numeric suffix, not lexicographic)."""
    m = re.match(r'^[Tt](\d+)$', str(label))
    return (0, int(m.group(1)), '') if m else (1, 0, str(label))


def normalise_task(label) -> str:
    """``'t7'`` -> ``'T7'``; validates the ``T<digit>`` shape."""
    s = str(label).strip()
    if not TASK_LABEL_RE.match(s):
        raise TaskGroupError(
            f'invalid task label {label!r}: expected a BP4D+ task id like '
            f'"T1" or "T10" (letter T + digits).')
    return s.upper()


def normalise_group(name) -> str:
    """``'LOW'`` -> ``'low'`` (group names are case-insensitive)."""
    s = str(name).strip().lower()
    if not s:
        raise TaskGroupError('empty task-group name')
    return s


def _task_tuple(value) -> Tuple[str, ...]:
    """``'T1|T2'`` / ``['T1','T2']`` / ``'T1'`` -> ``('T1','T2')`` (sorted)."""
    if value is None:
        items: List[str] = []
    elif isinstance(value, str):
        items = [v for v in re.split(f'[{re.escape("".join(_ITEM_SEP))}\\s]+',
                                     value) if v]
    else:
        items = [str(v) for v in value]
    out: List[str] = []
    for it in items:
        lab = normalise_task(it)
        if lab not in out:
            out.append(lab)
    return tuple(sorted(out, key=_task_sort_key))


def _name_tuple(value) -> Tuple[str, ...]:
    """``'low,high'`` / ``['low','high']`` -> ``('low','high')`` (de-duped)."""
    if value is None:
        return ()
    if isinstance(value, str):
        items = [v for v in re.split(r'[,\s]+', value) if v]
    else:
        items = [str(v) for v in value]
    if any(x.strip().lower() == 'all' for x in items):
        return ()
    out: List[str] = []
    for it in items:
        name = normalise_group(it)
        if name not in out:
            out.append(name)
    return tuple(out)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_task_groups(spec) -> Dict[str, Tuple[str, ...]]:
    """Normalise a ``task_groups`` definition -> ``{name: (T.., ...)}``.

    Accepts a YAML mapping, the ``'low=T1|T2;high=T7'`` string form, or
    ``None``/``''``/``{}`` (-> no groups). Raises :class:`TaskGroupError` on an
    empty group or an invalid task label.
    """
    if spec is None:
        return {}
    if isinstance(spec, str):
        text = spec.strip()
        if not text or text.lower() == 'none':
            return {}
        raw: Dict[str, object] = {}
        for chunk in text.split(_GROUP_SEP):
            chunk = chunk.strip()
            if not chunk:
                continue
            sep = next((s for s in _ASSIGN_SEP if s in chunk), None)
            if sep is None:
                raise TaskGroupError(
                    f'task_groups {spec!r}: chunk {chunk!r} is not of the form '
                    f'"<name>=<T1|T2|...>" (separate groups with "{_GROUP_SEP}").')
            name, rest = chunk.split(sep, 1)
            raw[name.strip()] = rest
    elif isinstance(spec, Mapping):
        raw = dict(spec)
    else:
        raise TaskGroupError(
            f'task_groups must be a mapping or a "name=T1|T2;name2=..." string; '
            f'got {type(spec).__name__}.')

    out: Dict[str, Tuple[str, ...]] = {}
    for name, value in raw.items():
        key = normalise_group(name)
        tasks = _task_tuple(value)
        if not tasks:
            raise TaskGroupError(
                f'task_groups: group {key!r} is empty (list at least one task id).')
        if key in out:
            raise TaskGroupError(f'task_groups: duplicate group name {key!r}.')
        out[key] = tasks
    return out


def parse_task_set(spec) -> Tuple[str, ...]:
    """Normalise the ``task_set`` selection -> a tuple of lower-case names."""
    return _name_tuple(spec)


def _explicit_tasks(tasks) -> Tuple[str, ...]:
    return _task_tuple(tasks)


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
@dataclass
class TaskSelection:
    """:func:`resolve_task_selection` result (all fields already normalised)."""

    #: sorted explicit task labels, or ``None`` = no task filter (all tasks).
    tasks: Optional[Tuple[str, ...]]
    #: every DEFINED group: name -> its task labels.
    levels: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: reverse map: task -> the group names that contain it.
    levels_by_task: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: the group names actually requested via ``task_set`` (order preserved).
    requested: Tuple[str, ...] = ()
    #: explicit task labels that came from the ``tasks`` CSV (a subset of
    #: :attr:`tasks`).
    explicit: Tuple[str, ...] = ()
    #: which inputs contributed: 'groups' and/or 'tasks'.
    sources: Tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        """True when a task filter applies (some tasks were selected)."""
        return bool(self.tasks)

    @property
    def defined(self) -> Tuple[str, ...]:
        """The defined group names, sorted -- for error messages."""
        return tuple(sorted(self.levels))

    def levels_of(self, task: str) -> Tuple[str, ...]:
        """The group names containing ``task`` (empty when none does)."""
        return self.levels_by_task.get(normalise_task(task), ())

    def describe(self) -> str:
        """One-line summary for the run log / ``describe()``."""
        defined = ', '.join(f'{k}={",".join(v)}' for k, v in
                            sorted(self.levels.items())) or '(none)'
        if self.requested:
            req = ', '.join(self.requested)
        else:
            req = '(none)'
        if self.tasks is None:
            final = 'ALL tasks'
        else:
            final = ','.join(self.tasks)
        return (f'task_set "{req}" + explicit tasks '
                f'{"none" if not self.explicit else ",".join(self.explicit)} '
                f'-> {final}   [groups: {defined}]')


def levels_by_task(levels: Mapping[str, Sequence[str]]
                   ) -> Dict[str, Tuple[str, ...]]:
    """Invert ``{group: (tasks,)}`` -> ``{task: (groups,)}``."""
    out: Dict[str, List[str]] = {}
    for name in sorted(levels):
        for task in levels[name]:
            out.setdefault(task, []).append(name)
    return {k: tuple(v) for k, v in sorted(out.items(), key=lambda kv: _task_sort_key(kv[0]))}


def resolve_task_selection(tasks=None, task_set=None, task_groups=None
                           ) -> TaskSelection:
    """Resolve ``--tasks`` + ``--task_set`` + ``task_groups`` -> a selection.

    :param tasks: the pre-existing explicit task list (CSV string or sequence).
    :param task_set: the requested group name(s) (CSV string or sequence).
    :param task_groups: the group DEFINITIONS (mapping or string form).

    The final task list is the UNION of the explicit ``tasks`` and every task of
    every requested group. ``None``/empty everywhere means "all tasks" (the
    historical behaviour).

    :raises TaskGroupError: a requested group is not defined (the message lists
        the defined names), or a group definition is malformed.
    """
    levels = parse_task_groups(task_groups)
    requested = parse_task_set(task_set)
    explicit = _explicit_tasks(tasks)

    selected: List[str] = []
    if requested:
        if not levels:
            raise TaskGroupError(
                f'task_set={requested!r} was requested but no task_groups are '
                f'defined: add a "task_groups:" block to the config (a mapping '
                f'from a group name to its task labels, e.g. '
                f'"low: [T1, T2]"), or drop --task_set.')
        unknown = [n for n in requested if n not in levels]
        if unknown:
            raise TaskGroupError(
                f'unknown task group(s) {unknown!r}; defined groups are '
                f'{list(sorted(levels))}. Fix --task_set / the config\'s '
                f'task_set, or add the group to "task_groups:".')
        for name in requested:
            for task in levels[name]:
                if task not in selected:
                    selected.append(task)

    final = sorted(set(explicit) | set(selected), key=_task_sort_key)
    sources = tuple(s for s in ('groups', 'tasks')
                    if (s == 'groups' and requested) or (s == 'tasks' and explicit))
    return TaskSelection(
        tasks=tuple(final) or None,
        levels=levels,
        levels_by_task=levels_by_task(levels),
        requested=requested,
        explicit=explicit,
        sources=sources)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    fails: List[str] = []

    def check(label: str, got, want) -> None:
        if got != want:
            fails.append(f'{label}: got {got!r}, want {want!r}')

    def check_raises(label: str, fn) -> None:
        try:
            fn()
        except TaskGroupError:
            return
        except Exception as exc:                      # noqa: BLE001
            fails.append(f'{label}: raised {type(exc).__name__} ({exc}), '
                         f'expected TaskGroupError')
            return
        fails.append(f'{label}: did not raise')

    groups = {'low': ['T1', 't2'],
              'moderate': 'T3, T4, T5, T6, T10',
              'high': 'T7|T8|T9'}
    parsed = parse_task_groups(groups)
    check('parse mapping keys', sorted(parsed), ['high', 'low', 'moderate'])
    check('parse lower-case + sort', parsed['low'], ('T1', 'T2'))
    check('parse string value', parsed['high'], ('T7', 'T8', 'T9'))
    check('parse numeric sort', parsed['moderate'],
          ('T3', 'T4', 'T5', 'T6', 'T10'))

    check('parse string form', parse_task_groups('low=T1|T2;high=T7|T8|T9'),
          {'low': ('T1', 'T2'), 'high': ('T7', 'T8', 'T9')})
    check('parse empty', parse_task_groups(''), {})
    check('parse none', parse_task_groups(None), {})

    # empty group / bad label / bad string form
    check_raises('empty group', lambda: parse_task_groups({'low': []}))
    check_raises('bad label', lambda: parse_task_groups({'low': ['A1']}))
    check_raises('bad string form', lambda: parse_task_groups('low T1'))
    check_raises('duplicate group', lambda: parse_task_groups(
        {'low': ['T1'], 'LOW': ['T2']}))

    # --- union + levels_by_task ------------------------------------------
    sel = resolve_task_selection(tasks='T1,T9', task_set='low,high',
                                 task_groups=groups)
    check('union tasks', sel.tasks, ('T1', 'T2', 'T7', 'T8', 'T9'))
    check('requested order', sel.requested, ('low', 'high'))
    check('explicit', sel.explicit, ('T1', 'T9'))
    check('sources', sel.sources, ('groups', 'tasks'))
    check('levels_by_task T1', sel.levels_by_task.get('T1'), ('low',))
    check('levels_of T2', sel.levels_of('T2'), ('low',))
    check('active', sel.active, True)
    # T11 is a valid label that no group mentions -> no level
    check('levels_of unknown', sel.levels_of('T11'), ())
    check('levels_of T5', sel.levels_of('T5'), ('moderate',))

    # group-only, tasks-only, neither
    check('group-only', resolve_task_selection(None, 'high', groups).tasks,
          ('T7', 'T8', 'T9'))
    check('tasks-only', resolve_task_selection('t3', '', groups).tasks, ('T3',))
    empty = resolve_task_selection('', '', groups)
    check('no filter -> None', empty.tasks, None)
    check('no filter inactive', empty.active, False)
    check('no filter sources', empty.sources, ())
    check('all keyword', resolve_task_selection(None, 'all', groups).tasks, None)

    # multi-membership
    multi = resolve_task_selection(None, 'a,b', {'a': ['T1', 'T2'],
                                                 'b': ['T2', 'T3']})
    check('multi levels T2', multi.levels_by_task['T2'], ('a', 'b'))
    check('multi union', multi.tasks, ('T1', 'T2', 'T3'))

    # selection persists the definitions even when a name is not requested
    sel2 = resolve_task_selection('T5', '', groups)
    check('defined groups kept', sorted(sel2.levels), ['high', 'low', 'moderate'])
    check('tasks-only keeps levels_by_task', sel2.levels_of('T5'), ('moderate',))

    # --- failures ---------------------------------------------------------
    check_raises('unknown group', lambda: resolve_task_selection(
        None, 'bogus', groups))
    check_raises('no groups defined', lambda: resolve_task_selection(
        None, 'low', {}))
    check_raises('bad explicit task', lambda: resolve_task_selection(
        'nope', None, groups))

    try:
        resolve_task_selection(None, 'bogus', groups)
    except TaskGroupError as exc:
        if 'high' not in str(exc) or 'low' not in str(exc):
            fails.append(f'unknown-group message does not list the defined '
                         f'names: {exc}')

    print('=' * 72)
    print('task_groups self-test')
    print('=' * 72)
    print('example', resolve_task_selection('T1', 'low,high', groups).describe())
    if fails:
        for f in fails:
            print(f'[FAIL] {f}')
        print(f'FAIL: {len(fails)} check(s) failed')
        return 1
    print('PASS: all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(_self_test())
