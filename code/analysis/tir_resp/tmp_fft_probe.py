"""How much of the respiration band does each MR-STFT window size actually see?

Throwaway probe. Reads the saved Stage-3 targets and the model predictions and
reports, for each fft_size: the window duration, the bin spacing, which bins the
0.1-0.6 Hz respiration band falls into, and the measured share of STFT energy in
those bins vs above 2 Hz. Uses the SAME _magnitude() the loss uses.
"""
import numpy as np
import torch

FS = 100.0
L = 800                       # 8 s clip
RESP = (0.1, 0.6)             # eval_band
HF = 2.0
ROOTS = {'C': 'resp_tir_roi_local_matched', 'A': 'resp_tir_roi_crossmae'}


def magnitude(x, n_fft, hop_ratio=0.25):
    hop = max(1, int(n_fft * hop_ratio))
    window = torch.hann_window(n_fft)
    s = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=window, return_complex=True)
    return s.abs()


def main():
    t = np.load('../../../output/finetune/resp_tir_roi_local_matched/targets.npy')
    ps = {k: np.load(f'../../../output/finetune/{v}/preds.npy')
          for k, v in ROOTS.items()}
    t = torch.from_numpy(t.astype(np.float32))

    print(f'targets {tuple(t.shape)}  fs {FS:g} Hz  band {RESP} Hz')
    print()
    hdr = (f'{"n_fft":>6}{"win_s":>8}{"bins/s":>9}{"bin of 0.1-0.6Hz":>18}'
           f'{"resp-bin energy share":>24}{"pred energy >2Hz":>18}')
    print(hdr)
    print('-' * len(hdr))

    for n_fft in (64, 128, 256, 512, 1024):
        mag = magnitude(t, n_fft)
        df = FS / n_fft
        n_bins = mag.shape[1]
        lo, hi = int(np.floor(RESP[0] / df)), int(np.ceil(RESP[1] / df))
        lo = min(lo, n_bins - 1)
        hi = min(max(hi, lo + 1), n_bins)
        tot = mag.pow(2).sum(dim=(1, 2))
        share = (mag[:, lo:hi, :].pow(2).sum(dim=(1, 2)) / tot).mean().item()
        p = ps['C']
        pm = magnitude(torch.from_numpy(p.astype(np.float32)), n_fft)
        f = torch.fft.rfftfreq(n_fft, d=1.0 / FS)
        hfshare = (pm.pow(2)[:, f >= HF, :].sum(dim=(1, 2))
                   / pm.pow(2).sum(dim=(1, 2))).mean().item()
        print(f'{n_fft:>6}{n_fft / FS:>8.2f}{df:>9.3f}'
              f'{f"[{lo},{hi - 1}]":>18}{share * 100:>22.1f}%{hfshare * 100:>17.1f}%')

    print()
    print('bins covering 0.1-0.6 Hz: a band narrower than one bin means the')
    print('spectral-convergence term cannot separate the breath from the DC/trend')
    print('bin. Energy share is the MEASURED fraction of the target spectrum that')
    print('lands in those bins.')


if __name__ == '__main__':
    main()
