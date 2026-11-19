"""Role-disjointness and the leakage abort (I1, I5, ADR-001 Criterion 3).

Two checks, deliberately at different times:

* :func:`assert_roles_disjoint` runs at build time, before a clip is read.
* :class:`LeakGuard` runs **inside the training step**, before
  ``optimizer.step()``, so a leaked sample aborts the run before it can touch a
  weight. Checking only at dataset construction would leave a window in which a
  bad sampler or a manual batch could still train on held-out subjects.
"""
from typing import Any, FrozenSet, Iterable, Mapping, Optional, Sequence

from .plan import FoldPlan, ProtocolError

__all__ = ['assert_roles_disjoint', 'LeakGuard']


def _roles_of(plan_or_roles: Any) -> Mapping[str, Sequence[str]]:
    """Accept a `FoldPlan` or a plain ``{role: subjects}`` mapping."""
    roles = getattr(plan_or_roles, 'roles', plan_or_roles)
    if not isinstance(roles, Mapping):
        raise ProtocolError(
            f'expected a FoldPlan or a role mapping; got '
            f'{type(plan_or_roles).__name__}')
    return roles


def assert_roles_disjoint(plan_or_roles: Any) -> None:
    """Raise if any subject id appears in more than one role (I1)."""
    roles = _roles_of(plan_or_roles)
    seen: dict = {}
    for role, members in roles.items():
        for subject in members:
            if subject in seen:
                raise ProtocolError(
                    f'{subject!r} is in both {seen[subject]!r} and {role!r}; '
                    f'roles must be disjoint at the SUBJECT level (I1).')
            seen[subject] = role


class LeakGuard:
    """Abort a training step that touches a held-out subject (Criterion 3).

    The guard is cheap by design: it checks every batch of the first epoch (a
    leak is then caught on step 1, not after a report) and one batch in every
    ``every`` afterwards. The caller passes the *subject ids of the batch it is
    about to step on*; nothing here needs the model, the loader or torch.
    """

    def __init__(self, plan: FoldPlan, role: str = 'train', every: int = 10,
                 first_epoch_all: bool = True):
        if role not in plan.roles:
            raise ProtocolError(f'unknown role {role!r}; got {sorted(plan.roles)}')
        if int(every) < 1:
            raise ProtocolError(f'every must be >= 1; got {every}')
        self.role = role
        self.every = int(every)
        self.first_epoch_all = bool(first_epoch_all)
        self._allowed: FrozenSet[str] = frozenset(plan.roles[role])
        self._held_out: FrozenSet[str] = frozenset(
            subject for name, members in plan.roles.items() if name != role
            for subject in members)
        self.partition_digest = plan.digest
        self.checked_batches = 0

    def should_check(self, epoch: int, batch_idx: int) -> bool:
        """The sampling schedule of the guard (see the class docstring)."""
        if self.first_epoch_all and int(epoch) == 0:
            return True
        return int(batch_idx) % self.every == 0

    def check(self, subjects: Iterable[str], *, epoch: int, batch_idx: int,
              step: Optional[int] = None) -> None:
        """Raise if any subject of this batch is outside ``self.role``."""
        if not self.should_check(epoch, batch_idx):
            return
        self.checked_batches += 1
        offending = sorted({str(s) for s in subjects} - self._allowed)
        if not offending:
            return
        held_out = [s for s in offending if s in self._held_out]
        where = f' (step {step})' if step is not None else ''
        raise ProtocolError(
            f'LEAK: epoch {epoch}, batch {batch_idx}{where}: '
            f'{len(offending)} subject(s) outside the {self.role!r} role reached '
            f'a gradient update, e.g. {offending[:5]}'
            + (f'; {len(held_out)} of them belong to a held-out role.'
               if held_out else '.')
            + f' partition digest {self.partition_digest[:12]}. '
            f'The run is aborted before the optimizer step.')
