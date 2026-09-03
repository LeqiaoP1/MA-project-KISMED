"""Optimizer factory.

Adapted from MultiMAE ``tmp/MultiMAE/utils/optim_factory.py`` and VideoMAE
``tmp/videomae/optim_factory.py``. Supports grouped (``lr_layer_decay``)
parameters if you pass an assigner, otherwise uses a single parameter group.
"""
import json
import torch

__all__ = ['create_optimizer']


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
