"""Attention visualisation entry point (scaffold).

Usage (from ``code/``)::

    python runners/run_visualize.py --model project_vit_base_patch16_224 \
        --resume output/finetune/best.pth
"""
import argparse
import os
import sys

# allow running either `python runners/run_visualize.py` or `python -m runners.run_visualize`
# from the code/ root by ensuring code/ is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners._common import add_common_args, parse_args_with_config


def get_args():
    parser = argparse.ArgumentParser('Attention visualisation', add_help=False)
    add_common_args(parser)
    parser.add_argument('--model', default='project_vit_base_patch16_224',
                        type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    return parse_args_with_config(parser)


def main(args):
    from engines import run_attention_vis
    run_attention_vis(args)


if __name__ == '__main__':
    args = get_args()
    main(args)
