"""Optimizer factory.

Adapted from MultiMAE ``tmp/MultiMAE/utils/optim_factory.py`` and VideoMAE
``tmp/videomae/optim_factory.py``. Supports grouped (``lr_layer_decay``)
parameters if you pass an assigner, otherwise uses a single parameter group.
"""
import json
import torch

__all__ = ['create_optimizer', 'get_num_layer_for_multimae',
           'LayerDecayValueAssigner', 'build_layer_decay_assigner']


def create_optimizer(args, model, skip_list=None, get_num_layer=None,
                     get_layer_scale=None, filter_bias_and_bn=True):
    """Build an optimizer (default AdamW) from CLI/YAML ``args``.

    ``get_num_layer``/``get_layer_scale`` can implement layer-wise LR decay
    (see VideoMAE ``LayerDecayValueAssigner`` for the reference recipe).
    """
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = set(skip_list)
        elif hasattr(model, 'no_weight_decay'):
            skip = set(model.no_weight_decay())
        parameters = get_parameter_groups(
            model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    if opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_lower == 'sgd':
        opt_args['momentum'] = args.momentum if hasattr(args, 'momentum') else 0.9
        optimizer = torch.optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError(f'Optimizer "{args.opt}" is not implemented.')

    return optimizer


def get_parameter_groups(model, weight_decay, skip_list=(), get_num_layer=None,
                         get_layer_scale=None):
    """Group parameters by decay (and optionally by layer for LR decay)."""
    parameter_group_names = {}
    parameter_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list:
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name += f'_layer_{layer_id}'
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.
            parameter_group_names[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'lr_scale': scale,
            }
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'lr_scale': scale,
            }
        parameter_group_names[group_name]['params'].append(name)
        parameter_groups[group_name]['params'].append(param)

    print('Parameter groups:\n%s' % json.dumps(parameter_group_names, indent=2))
    return list(parameter_groups.values())


# --------------------------------------------------------------------------- #
# layer-wise LR decay (VideoMAE recipe, mapped to THIS repo's module names)
# --------------------------------------------------------------------------- #
def get_num_layer_for_multimae(var_name: str, num_max_layer: int) -> int:
    """Layer index of a parameter, for layer-wise LR decay.

    Mirrors VideoMAE's ``get_num_layer_for_vit`` (tmp/videomae/optim_factory.py)
    with this repo's names:

    * ``adapters.*`` (the tubelet tokenizer) and ``positions.*`` (positional
      embeddings / mask tokens) -> layer 0, the MOST decayed;
    * ``enc_blocks.<n>.*`` -> ``n + 1`` (a deeper block gets a larger LR);
    * everything else (``enc_norm``, ``waveform_head``) -> the top layer
      (scale 1.0).

    Rationale (spec §4.2): the lower layers carry the generic space-time
    features inherited from Stage 1/2 and must move slowly, while the last
    blocks and the head adapt to the new task.
    """
    if var_name.startswith(('adapters.', 'positions.')):
        return 0
    if var_name.startswith('enc_blocks.'):
        parts = var_name.split('.')
        if len(parts) > 1:
            try:
                return int(parts[1]) + 1
            except ValueError:
                pass
    return max(0, num_max_layer - 1)


class LayerDecayValueAssigner(object):
    """``layer_id -> lr multiplier`` lookup (VideoMAE ``LayerDecayValueAssigner``)."""

    def __init__(self, values):
        self.values = list(values)

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_multimae(var_name, len(self.values))


def build_layer_decay_assigner(model, layer_decay):
    """Build a :class:`LayerDecayValueAssigner` for an encoder+head model.

    ``layer_decay`` >= 1.0 (or ``None``) means OFF -> returns ``None``, so
    ``create_optimizer`` keeps its single-group behaviour. Values follow the
    VideoMAE recipe ``layer_decay ** (num_layers + 1 - i)`` over
    ``num_layers + 2`` groups: the top group is exactly 1.0 and the tokenizer is
    the most decayed. ``num_layers`` is read from ``model.enc_blocks`` (the
    pre-trained encoder depth); a model without one -> ``None``.
    """
    if layer_decay is None or float(layer_decay) >= 1.0:
        return None
    try:
        num_layers = len(model.enc_blocks)
    except AttributeError:
        return None
    if num_layers <= 0:
        return None
    decay = float(layer_decay)
    values = [decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
    print(f'[layer_decay] decay {decay:g} over {num_layers} encoder blocks '
          f'-> {len(values)} groups; lr scale {values[0]:.4g} (tokenizer) '
          f'.. {values[-1]:.4g} (top)')
    return LayerDecayValueAssigner(values)
