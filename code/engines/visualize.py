"""Attention visualisation (scaffold).

Port the full attention-map visualisation from VideoMAE
``tmp/videomae/run_videomae_vis.py`` when you have trained a checkpoint.
"""
import torch

from models import create_model


def run_attention_vis(args):
    """Load a checkpoint and run one forward pass to sanity-check shapes."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = create_model(args.model, num_classes=args.nb_classes)
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.to(device)
    model.eval()

    # Synthetic input: [B, C, T, H, W] or [B, C, H, W] depending on modality.
    dummy = torch.randn(args.batch_size, 3, args.input_size, args.input_size,
                        device=device)
    with torch.no_grad():
        out = model(dummy)
    print('Visualisation forward OK. Output shape:', tuple(out.shape))
    print('TODO: render attention maps (see tmp/videomae/run_videomae_vis.py).')
