"""AMP loss scaler that also returns the gradient norm (for logging / clipping).

Port of MultiMAE ``tmp/MultiMAE/utils/native_scaler.py`` (used by MAE and
VideoMAE). Requires ``--enable_amp`` / AMP on by default in the run scripts.
"""
import torch

__all__ = ['NativeScalerWithGradNormCount']


class NativeScalerWithGradNormCount:
    state_dict_key = 'amp_scaler'

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None,
                 create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            # ``clip_grad`` must be POSITIVE to clip. The guard mirrors the
            # non-AMP branch in engines/*: ``clip_grad_norm_(..., max_norm=0.0)
            # scales EVERY gradient to exactly zero (clip_coef = 0/(norm+eps)),
            # which silently disabled learning for every run whose config left
            # clip_grad at its 0.0 default ("no clipping"), while still
            # RETURNING a healthy-looking grad norm (measured before the
            # scaling). Stage-3 bp/resp and Stage-2 were both affected.
            if clip_grad is not None and clip_grad > 0:
                assert parameters is not None
                self._scaler.unscale_(optimizer)   # unscale the gradients
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device)
                         for p in parameters]), norm_type)
    return total_norm
