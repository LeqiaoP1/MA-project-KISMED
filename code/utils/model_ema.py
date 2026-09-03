"""Model EMA (exponential moving average).

Minimal port of timm's ModelEma, sufficient for the thesis scaffold.
For the more elaborate ``ModelEmaV2`` see ``tmp/MultiMAE/utils/model_ema.py``.
"""
from copy import deepcopy

import torch

__all__ = ['ModelEma']


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=''):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(),
                                      model.state_dict().values()):
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
