"""Selection rule and the validation-pass determinism contract (I6, I7).

ADR-001 section 2.3 names the interface (``val_select``, ``eval_freq``,
``val_kept_<stream>``) and section 2.2 I7 requires the rule, the selected epoch
and the criterion value to be recorded. This module is the only place a
selection decision is made, so ``best.pth`` can only ever come from a validation
record.
"""
from typing import Any, Dict, Mapping, Optional

from .constants import (CRITERIA, MIN_EVAL_FREQ, ROLE_SHARE, SCHEMA_SELECTION,
                        SELECTION_RULES)
from .plan import ProtocolError

__all__ = ['val_mask_seed', 'SelectionTracker']


def val_mask_seed(base: int, batch_idx: int) -> int:
    """Mask seed of one validation batch (I6, Criterion 2).

    Constant ``base`` per evaluation plus the batch index: every epoch sees the
    SAME mask for the same batch (so the curve is comparable across epochs),
    while different batches do not share one mask pattern (so the loss is not
    measured under a single lucky corruption). Re-running an evaluation of the
    same checkpoint therefore reproduces it bit-for-bit.
    """
    return int(base) + int(batch_idx)


class SelectionTracker:
    """Apply the recorded selection rule and emit the selection record (I7).

    ``fixed_epoch`` (default) never selects: the delivered weights are the last
    epoch's, and the record says so. ``best_val`` may only read the validation
    criterion, whose direction is fixed per criterion (a loss is minimised, a
    correlation maximised) so the rule cannot be inverted by a caller.
    """

    def __init__(self, *, epochs: int, rule: str = 'fixed_epoch',
                 criterion: str = 'joint_loss', eval_freq: int = 5,
                 n_val_clips: int = 0, n_train_clips: int = 0,
                 dev_smoke: bool = False):
        rule = str(rule).strip().lower()
        criterion = str(criterion).strip().lower()
        if rule not in SELECTION_RULES:
            raise ProtocolError(
                f'val_select must be one of {SELECTION_RULES}; got {rule!r}.')
        if criterion not in CRITERIA:
            raise ProtocolError(
                f'val_criterion must be one of {tuple(CRITERIA)}; got '
                f'{criterion!r}.')
        if int(epochs) < 1:
            raise ProtocolError(f'epochs must be >= 1; got {epochs}')
        if int(eval_freq) < 0:
            raise ProtocolError(f'eval_freq must be >= 0 (0 = off); got {eval_freq}')
        if int(eval_freq) == 0 and rule == 'best_val':
            raise ProtocolError(
                'val_select=best_val needs a validation evaluation, but '
                'eval_freq=0 disables them; set eval_freq >= '
                f'{MIN_EVAL_FREQ} or choose val_select=fixed_epoch.')
        if 0 < int(eval_freq) < MIN_EVAL_FREQ and not dev_smoke:
            raise ProtocolError(
                f'eval_freq={eval_freq} evaluates the {ROLE_SHARE["val"]:.0%} '
                f'role every {eval_freq} epoch(s); against a '
                f'{ROLE_SHARE["train"]:.0%} train role that is about '
                f'{ROLE_SHARE["val"] / ROLE_SHARE["train"] / eval_freq:.1%} of '
                f'the run, above the 5% budget of Criterion 4. Use '
                f'eval_freq >= {MIN_EVAL_FREQ}.')

        self.epochs = int(epochs)
        self.rule = rule
        self.criterion = criterion
        self.direction = CRITERIA[criterion]
        self.eval_freq = int(eval_freq)
        self.n_val_clips = int(n_val_clips)
        self.n_train_clips = int(n_train_clips)
        self.dev_smoke = bool(dev_smoke)
        self._best_epoch: Optional[int] = None
        self._best_value: Optional[float] = None
        self._last_epoch: Optional[int] = None
        self._last_value: Optional[float] = None
        self.n_evals = 0

    @property
    def writes_best_checkpoint(self) -> bool:
        """Whether this rule delivers ``best.pth`` (only ``best_val`` does)."""
        return self.rule == 'best_val'

    def should_eval(self, epoch: int) -> bool:
        """Evaluate on this epoch? ``0`` means evaluation is off."""
        if self.eval_freq == 0:
            return False
        return epoch % self.eval_freq == 0 or epoch + 1 == self.epochs

    def observe(self, epoch: int, value: float) -> bool:
        """Feed one validation criterion value; True when it is the best so far."""
        if value is None:
            raise ProtocolError(
                f'epoch {epoch}: the validation criterion value is missing; a '
                f'selection record without it is not auditable (I7).')
        value = float(value)
        self.n_evals += 1
        self._last_epoch, self._last_value = int(epoch), value
        better = (self._best_value is None
                  or (value < self._best_value if self.direction == 'min'
                      else value > self._best_value))
        if better:
            self._best_epoch, self._best_value = int(epoch), value
        return better

    @property
    def best_epoch(self) -> Optional[int]:
        return self._best_epoch

    @property
    def best_value(self) -> Optional[float]:
        return self._best_value

    def record(self, pinned: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The ADR-001 selection record for the run that just finished."""
        if self.rule == 'fixed_epoch':
            selected_epoch = self.epochs - 1
            criterion_value = self._last_value
        else:
            if self._best_epoch is None:
                raise ProtocolError(
                    'val_select=best_val but no validation evaluation was '
                    'recorded; there is nothing to select on.')
            selected_epoch = self._best_epoch
            criterion_value = self._best_value
        return {
            'schema': SCHEMA_SELECTION,
            'rule': self.rule,
            'eval_freq': self.eval_freq,
            'selected_epoch': int(selected_epoch),
            'criterion': self.criterion,
            'direction': self.direction,
            'criterion_value': (None if criterion_value is None
                                else float(criterion_value)),
            'n_evals': int(self.n_evals),
            'n_val_clips': self.n_val_clips,
            'n_train_clips': self.n_train_clips,
            'pinned': dict(pinned or {}),
            'dev_smoke': self.dev_smoke,
        }
