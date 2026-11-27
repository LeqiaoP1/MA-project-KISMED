"""Masked pre-training entry point.

Usage (from ``code/``)::

    python runners/run_pretrain.py -c configs/pretrain/example.yaml

NOTE: masked pre-training needs your (a) dataset in ``data/datasets.py`` and
(b) an encoder+decoder that returns ``(pred, target, mask)`` (port from
``tmp/videomae/modeling_pretrain.py`` / ``tmp/MultiMAE/multimae/multimae.py``).
Until then this script wires everything except those two thesis-specific parts.
"""
import argparse
import os
import sys
import time

# allow running either `python runners/run_pretrain.py` or `python -m runners.run_pretrain`
# from the code/ root by ensuring code/ is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engines import train_one_epoch_pretrain
from engines.pretrain import evaluate_pretrain
from runners._common import (add_common_args, add_protocol_args,
                             check_loader_not_empty, env_or, get_rank,
                             init_env, load_or_create_plan, make_data_loader,
                             parse_args_with_config)


def str2bool(value) -> bool:
    """argparse ``type`` that accepts a real bool, ``1/0`` or ``true/false``.

    Needed because the default is True: a bare ``store_true`` cannot express
    that, and ``--find_unused_parameters false`` has to be spellable on the
    command line. A YAML ``false`` reaches the variable as a genuine bool (the
    config loader applies it via ``set_defaults``, which skips ``type``), hence
    the isinstance short-circuit.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if text in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(
        f'expects a boolean (true/false/1/0), got {value!r}')


def get_args():
    parser = argparse.ArgumentParser('Project MAE pre-training', add_help=True)
    add_common_args(parser)
    add_protocol_args(parser)

    # model
    parser.add_argument('--model', default='project_multimae_base', type=str,
                        help='registered model name. project_multimae_{small,'
                             'base,large} run the multimodal masked '
                             'pre-training (the NAME sets the ViT geometry); '
                             'any project_vit_* runs the single-stream path. '
                             'small requires --pretrained_encoder none because '
                             'no small Stage-1 checkpoint is published.')
    # ViT geometry comes from the --model variant; 0 = "unset" => take it from
    # the name. Set a value only to OVERRIDE (e.g. an ablation).
    parser.add_argument('--enc_embed_dim', default=0, type=int)
    parser.add_argument('--enc_depth', default=0, type=int)
    parser.add_argument('--enc_num_heads', default=0, type=int)
    # data / clip geometry (the multimodal dataset & model read these)
    parser.add_argument('--data_path', default=env_or('DATA_PATH'), type=str)
    parser.add_argument('--data_set', default='bp4d+', type=str,
                        help="Stage-2 data source: 'bp4d+' (default) = the "
                             "canonical paired layout; 'tir_roi' = the ADD-ON "
                             "thermal-ROI + respiration dataset built from the "
                             "RAW tree (see code/notes/plan/experiments/TirROI_Resp_plan.md); "
                             "'rgb_roi' = the ADD-ON RGB face-ROI dataset "
                             "(landmark crop from 2DFeatures + RAW physiology; "
                             "see code/data/rgb_roi_dataset.py); "
                             "'tir_roi_resp_rgb2d_bp' = the FOUR-STREAM Stage-2 "
                             "path (TIR-ROI + 2D texture + BP + respiration). "
                             "A (subject, "
                             "task) pair whose 2DFeatures frame count differs "
                             "from the jpeg count is EXCLUDED as 'unusable' "
                             "and its 1-D physiology is dropped too.")
    parser.add_argument('--raw_root', default=env_or('RAW_DATA_PATH', ''),
                        type=str,
                        help='raw BP4D root for data_set=tir_roi/rgb_roi; '
                             'default: $RAW_DATA_PATH, else '
                             '<repo>/data/raw/BP4D')
    parser.add_argument('--subjects', default='', type=str,
                        help='tir_roi/rgb_roi only: comma list of subjects '
                             '(empty=all). NARROWS the sessions read inside a '
                             'role; it never moves a subject between roles.')
    parser.add_argument('--tasks', default='', type=str,
                        help='tir_roi/rgb_roi only: comma list of tasks '
                             '(empty=all)')
    parser.add_argument('--task_set', default='', type=str,
                        help="tir_roi/rgb_roi only: named task group(s) from "
                             "the config's task_groups: block, e.g. 'low' or "
                             "'low,high' (empty=all). UNIONED with --tasks, "
                             'see code/data/task_groups.py')
    parser.add_argument('--task_groups', default='', type=str,
                        help='tir_roi/rgb_roi only: group definitions, '
                             "overriding the config, e.g. "
                             "'low=T1|T2;high=T7|T8|T9' (the config's "
                             'task_groups: mapping in string form)')
    # ---- ROI-crop options (shared by the tir_roi and rgb_roi paths) --------
    parser.add_argument('--roi_padding', default=0.2, type=float,
                        help='tir_roi/rgb_roi only: fraction of the landmark-box '
                             'extent added on EACH side before the crop '
                             '(0.2 = +40%% overall)')
    parser.add_argument('--crop_rule', default='square', type=str,
                        choices=['box', 'square'],
                        help="tir_roi/rgb_roi: how the padded landmark box "
                             "becomes a crop. 'square' (default for tir_roi) "
                             "squares it about its centre FIRST, so the rescale "
                             "has ONE scale factor and the anatomy is never "
                             "deformed; 'box' is the legacy per-axis (squashing) "
                             "geometry, kept for A/B runs. It changes WHICH "
                             "PIXELS reach the encoder while leaving every "
                             "tensor shape unchanged -> a Stage-3 run must "
                             "mirror it.")
    parser.add_argument('--max_sway_px', default=40.0, type=float,
                        help='tir_roi only: DROP a clip whose target-landmark '
                             'centroid travels further than this many SOURCE '
                             'pixels from its own within-clip median (0 = '
                             'gate disabled; study value 40). A static box over '
                             'a large head sweep is mostly background, so the '
                             'clip is excluded rather than absorbed into a '
                             'bigger crop. Changes WHICH CLIPS EXIST, not any '
                             'pixel -> mirror it in Stage 3. Task-biased: T4 '
                             'keeps 80 % at 40 px.')
    parser.add_argument('--min_signal_spread', default=0.0, type=float,
                        help='tir_roi only: DROP a clip whose respiration '
                             'window spread (max-min) is below this many VOLTS '
                             '(0.0 = off; for the study use 0.1). It is the '
                             'DEAD-SIGNAL guard, complementary to '
                             '--rail_touch_v: every corpus window below 0.1 V '
                             'sits with a median level of +9.1 V, i.e. a '
                             'channel pinned just inside the +/-10 V clamp with '
                             'a 0.05-0.09 V ripple (disconnected belt / loose '
                             'contact). Real breathing is >= 0.3 V p-p. Must '
                             'match the Stage-3 value.')
    parser.add_argument('--rail_touch_v', default=0.0, type=float,
                        help='tir_roi only: DROP a clip whose respiration '
                             'window contains ANY sample that touched the '
                             'recorder rail, i.e. abs(x) >= this many VOLTS '
                             '(0.0 = off). The recorder clamps at +/-10 V, so '
                             'the study value 9.90 rejects every window whose '
                             'label contains even ONE rail-valued reading -- a '
                             'dead / pinned sensor AND a clipped flat-topped '
                             'trough, so no unrepresentable target survives. '
                             'Must match the Stage-3 value.')
    parser.add_argument('--roi_landmarks', default='', type=str,
                        help='ROI paths (tir_roi AND rgb_roi): the landmarks '
                             'whose bounding box IS the crop. Empty = the path '
                             'default (tir_roi: the historical 12-point nose+mouth '
                             'set; rgb_roi: "face" = all 49). Otherwise a preset '
                             'name (nose_mouth | nostrils | nostril_mouth | '
                             'nose_tip) or a comma list of 1-indexed labels, e.g. '
                             '"10,21". Presets live in '
                             'data/tir_resp_dataset.ROI_LANDMARK_PRESETS (tir_roi) '
                             'and data/rgb_features.ROI_LANDMARKS_2D (rgb_roi). '
                             'NOTE: the 2-point "nostrils" box is a thin slab '
                             'whose extent tracks landmark jitter (measured '
                             '100x81 px on a still session but 91x15 px on a '
                             'moving one); prefer "nose_tip". This changes the '
                             'VISUAL INPUT, so a Stage-2 checkpoint trained with a '
                             'different value must be re-pretrained.')
    parser.add_argument('--roi_cache', default=env_or('ROI_CACHE', ''), type=str,
                        help='tir_roi* only: OUT_ROOT of an offline TIR-ROI cache '
                             '(data/roi_cache.py), i.e. the parent of the '
                             'CACHE_KEY directories. Empty (the default, or '
                             'ROI_CACHE) = decode .wmv as before. The cache is '
                             'matched by KEY against this run of roi_landmarks / '
                             'roi_padding / corpus, and the run ABORTS on a '
                             'mismatch rather than reading slightly wrong '
                             'pixels. It stores NATIVE union crops, so ONE cache '
                             'serves the 2 s and 1 s hops and both the 112 px '
                             'and 64 px lineages.')
    parser.add_argument('--resp_cache', default=env_or('RESP_CACHE', ''),
                        type=str,
                        help='tir_roi* only: OUT_ROOT of an offline respiration '
                             'cache (data/resp_cache.py), i.e. the parent of '
                             "the <CACHE_KEY>/ directories. Empty (default, or "
                             '$RESP_CACHE) = read Resp_Volts.txt per epoch as '
                             'before. Matched by KEY (corpus, fs_in/fs_out and '
                             'the validity knobs), and the run ABORTS on a '
                             'mismatch. With a cache, --resp_fs IS the '
                             "artefact's output rate and must equal --fs; "
                             'rail_touch_v / min_signal_spread must be 0 (the '
                             'stored per-sample mask is the only validity '
                             'rule).')
    parser.add_argument('--rgb2d_cache', default=env_or('RGB2D_CACHE', ''),
                        type=str,
                        help='tir_roi_resp_rgb2d / trimodal only: OUT_ROOT of an '
                             'offline 2D-texture face-ROI cache '
                             '(data/rgb2d_cache.py), i.e. the parent of the '
                             'CACHE_KEY directories. REQUIRED for that '
                             'data_set. Matched by KEY against this run of '
                             'rgb2d_landmarks / roi_padding / crop_rule / '
                             'clip_seconds / fps / corpus, and the run ABORTS on '
                             'a mismatch. Stores NATIVE hull crops, so ONE cache '
                             'serves any hop and any --input_size.')
    parser.add_argument('--rgb2d_cache_lru', default=8, type=int,
                        help='rgb2d only: number of OPEN shard handles per '
                             'worker (2 fds each). Resident memory is ~0 -- the '
                             'arrays are memory-mapped, not loaded.')
    parser.add_argument('--rgb2d_store', default=env_or('RGB2D_STORE', 'native'),
                        type=str,
                        help='rgb2d only: WHICH store to attach. native '
                             '(default) = source pixels, input_size stays free. '
                             'A pixel size (112 / px112) selects a cache the '
                             'builder RESIZED to N x N: 42x less I/O per clip '
                             '(the native store reads ~316 MB/clip and made '
                             'the tri-modal run I/O-bound), at the price of '
                             'input_size <= N. It is a cache-key field, so a '
                             'mismatch fails loudly instead of mixing stores.')
    parser.add_argument('--rgb2d_landmarks', default='face', type=str,
                        help="rgb2d only: 'face' (all 49 points, the default) or "
                             'a preset / CSV from data/rgb_features.ROI_LANDMARKS_2D. '
                             'MUST match the cache build (it is a cache-key field).')
    parser.add_argument('--rgb2d_min_valid_ratio', default=1.0, type=float,
                        help='rgb2d only: fraction of a window whose 49-point '
                             'track must be finite. 1.0 (the default) = zero '
                             'tolerance, matching the design contract.')
    parser.add_argument('--rgb2d_max_sway_px', default=110.0, type=float,
                        help='rgb2d only: max centroid excursion (source px) '
                             'inside one window; 0 = gate off. Same statistic '
                             'as --max_sway_px, in THIS camera\'s pixels: 110 px '
                             'here == the TIR 40 px (RGB reports 3.2x the '
                             'excursion for the same motion; see '
                             'code/notes/eval/dataset_inspection/rgb2d/tir_rgb2d_sway_ratio.py). '
                             'This is a LOAD-time selection rule and never a '
                             'cache-key field.')
    parser.add_argument('--rgb2d_area_min_frac', default=0.5, type=float,
                        help='rgb2d only: minimum in-frame share of a CLIP box; '
                             'below it the clip is dropped '
                             '(clips_dropped_2d_envelope). Black padding would '
                             'otherwise feed a mostly-empty clip to the model.')
    parser.add_argument('--bp_artefact', default=env_or('BP_ARTEFACT', ''),
                        type=str,
                        help='tir_roi_resp_rgb2d_bp only: OUT_ROOT of an '
                             'offline BP artefact (data/bp_artefact.py), i.e. '
                             'the parent of the CACHE_KEY directories. '
                             'REQUIRED for that data_set. Matched by the full '
                             'frozen key (fs_in==fs_out, gate thresholds, '
                             'min_segment_s, store); the run ABORTS on a '
                             'mismatch.')
    parser.add_argument('--bp_fs', default=None, type=float,
                        help='tir_roi_resp_rgb2d_bp only: BP artefact sample '
                             'rate. The artefact stores the source grid '
                             'unchanged, so this MUST equal --fs; default None '
                             '= use --fs.')
    parser.add_argument('--bp_cache_lru', default=8, type=int,
                        help='tir_roi_resp_rgb2d_bp only: number of BP shard '
                             'payloads kept per worker.')
    parser.add_argument('--decode_scale', default=1, type=int,
                        choices=[1, 2, 4, 8],
                        help='rgb_roi only: JPEG decode scale. 1 = native full '
                             'decode (most faithful, and the default for the '
                             'shared Stage-2/Stage-3 front-end); 2/4/8 = '
                             'libjpeg DCT downscale (faster; the ROI box is '
                             'rescaled into the decoded frame). Changing this '
                             'changes the visual front-end, so it must match '
                             'across Stage 2 and Stage 3.')
    parser.add_argument('--phys_fs', default=1000.0, type=float,
                        help='rgb_roi only: sample rate of the RAW 1-D files '
                             'under Physiology/<S>/<T>/ (BP_mmHg.txt, '
                             'Resp_Volts.txt). 1000 Hz '
                             'nominal; they are resampled onto the --fs grid.')
    parser.add_argument('--fs', default=100.0, type=float,
                        help='signal sample rate (Hz)')
    parser.add_argument('--fps', default=25.0, type=float,
                        help='RGB/TIR frame rate (Hz)')
    parser.add_argument('--clip_duration', default=4.0, type=float,
                        help='window length in seconds per MAE sample')
    parser.add_argument('--clip_stride', default=0.0, type=float,
                        help='window stride in seconds. < clip_duration yields '
                             'OVERLAPPING windows (more samples/session, e.g. '
                             'to reach 10-20k clips on the full HPC data). '
                             '0 => stride = clip_duration (non-overlapping).')
    parser.add_argument('--temporal_stride', default=1, type=int,
                        help='decimate frames INSIDE a window: keep 1 frame '
                             'every N (1 = every frame; 2/4/8 -> 12.5/6.25/'
                             '3.125 fps at 25 fps nominal). Distinct from '
                             '--clip_stride (window hop in seconds). Sets the '
                             'video geometry num_frames = clip_duration*fps/N.')
    parser.add_argument('--seq_len', default=0, type=int,
                        help='signal samples per window (0 => clip_duration*fs)')
    parser.add_argument('--input_size', default=64, type=int,
                        help='frame short-side resize/crop (square)')
    # multimodal MAE (Stage 2)
    parser.add_argument('--streams', default='rgb,tir,bp', type=str,
                        help='comma list of pretraining streams. Stage-2 '
                             'requires >=2: >=1 of rgb,tir (video) AND >=1 '
                             'of bp,resp (1-D physio, Stage-3 target), '
                             'e.g. rgb,bp or rgb,tir,bp,resp')
    parser.add_argument('--tubelet', default='2,16,16', type=str,
                        help='tubelet (t, ph, pw) for the video tokenizer')
    # ---- masking policy (ADR-002 constant-budget apportioner) ------------ #
    # These seven keys are the ONLY policy parameters. Each step draws a mode
    # and one kept-token count per stream; the counts of all streams sum to
    # n_budget. YAML keys mirror the flag names. See
    # code/notes/design/archive/ADR-002-asymmetric-multimodal-masking.md §2.
    parser.add_argument('--n_budget', default=350, type=int,
                        help='constant number of visible encoder tokens per '
                             'step (ADR-002 contract 1). Every step feeds '
                             'exactly this many tokens to the encoder.')
    parser.add_argument('--video_cap_ratio', default=0.25, type=float,
                        help='kept-ratio ceiling per VISUAL stream (0.25 => '
                             'at least 75 %% masked); the cap in tubes is '
                             'floor(ratio * spatial_patches).')
    parser.add_argument('--physio_cap_ratio', default=0.50, type=float,
                        help='kept-ratio ceiling per PHYSIOLOGICAL stream '
                             '(0.50 => at least half of every waveform is a '
                             'reconstruction target).')
    parser.add_argument('--video_min_tubes', default=2, type=int,
                        help='kept-tube floor per visual stream: the smallest '
                             'allocation that still carries spatial context.')
    parser.add_argument('--p_virtual_sensing', default=0.50, type=float,
                        help='probability of Mode 1 (pure virtual sensing: no '
                             'physiological token reaches the encoder). The '
                             'other steps run Mode 2 (waveform inpainting). '
                             'From ADR-005 §2.7 this is the TARGET '
                             'probability, applied from '
                             '--virtual_sensing_warmup_epochs onwards.')
    parser.add_argument('--virtual_sensing_warmup_p', default=0.10, type=float,
                        help='ADR-005 §2.7 p_warm: Mode-1 probability strictly '
                             'below --virtual_sensing_warmup_epochs. Must lie '
                             'in [0, --p_virtual_sensing].')
    parser.add_argument('--virtual_sensing_warmup_epochs', default=0, type=int,
                        help='ADR-005 §2.7 E_warm: number of leading epochs at '
                             'p_warm. 0 (default) DISABLES the curriculum, so '
                             'p(e) stays the constant --p_virtual_sensing and '
                             'a pre-ADR-005 configuration is unchanged. The '
                             'shipped arm sets 50 in its YAML.')
    # ---- ADR-006 single-carrier class (all three default to NEUTRAL) ----- #
    parser.add_argument('--single_carrier_tubes', default=0, type=int,
                        help='ADR-006 §2.4 S_v: kept tubes of a SOLE visual '
                             'carrier, which then holds the whole budget. '
                             '0 (default) DERIVES n_budget/tokens_per_unit; '
                             'any other value must equal that derivation or '
                             'the run is rejected at resolve time. An '
                             'assertion, not a knob.')
    parser.add_argument('--p_single_modality', default=0.0, type=float,
                        help='ADR-006 §2.7 p_s_target: probability, WITHIN a '
                             'sensor-absent step, that exactly ONE visual '
                             'stream carries the whole budget (the other '
                             'visuals keep 0). 0.0 (default) DISABLES the '
                             'class, so the mask sequence stays bit-identical '
                             'to ADR-005.')
    parser.add_argument('--single_modality_warmup_p', default=0.0, type=float,
                        help='ADR-006 §2.7 p_s_warm: single-carrier '
                             'probability strictly below '
                             '--virtual_sensing_warmup_epochs. Must lie in '
                             '[0, --p_single_modality] while the curriculum '
                             'is active.')
    parser.add_argument('--spatial_patches', default=0, type=int,
                        help='tokenizer-derived (default 0 = derive from '
                             'input_size/tubelet); pass it only to ASSERT the '
                             'identity, the run fails on a mismatch.')
    # ---- ADR-007 anatomy-aware stratified placement (all NEUTRAL) -------- #
    # Every default reproduces the pre-ADR-007 mask sequence, so a config that
    # omits all seven keys stays bit-identical. The arm's tuple lives in its
    # YAML; see code/notes/design/ADR-007-*.md §2.3/§2.5/§2.6.
    parser.add_argument('--spatial_sampling', default='uniform', type=str,
                        choices=['uniform', 'anatomy_aware'],
                        help="how the kept spatial TUBES are drawn: 'uniform' "
                             '(default, pre-ADR-007) or '
                             "'anatomy_aware' (ADR-007 §2.3: stratified "
                             'anatomical core + periphery, ONE visual stream '
                             'only, every sample keeps >= 1 core tube).')
    parser.add_argument('--core_row_bounds', default='2,3', type=str,
                        help='ADR-007 §2.3 inclusive (lo, hi) ROW span of '
                             'A_core on the patch grid. Ignored under '
                             '--spatial_sampling uniform.')
    parser.add_argument('--core_col_bounds', default='2,3', type=str,
                        help='ADR-007 §2.3 inclusive (lo, hi) COLUMN span of '
                             'A_core. Ignored under uniform sampling.')
    parser.add_argument('--core_tubes_min', default=1, type=int,
                        help='ADR-007 §2.3 S_core^min: kept core tubes per '
                             'step (>= 1, which is what makes the core hit '
                             'certain rather than probable).')
    parser.add_argument('--core_tubes_max', default=2, type=int,
                        help='ADR-007 §2.3 S_core^max: kept core tubes per '
                             'step; the rest of the budget is peripheral.')
    parser.add_argument('--periph_min_tubes', default=1, type=int,
                        help='ADR-007 §2.6(5) m_periph: floor on the '
                             'PERIPHERAL tubes, so a step always keeps spatial '
                             'context around the core.')
    parser.add_argument('--physio_draw_min', default=0, type=int,
                        help='ADR-007 §2.5: lower bound of the Mode-2 '
                             'physiological draw. 0 (default) keeps the '
                             'ADR-002..006 draw INCLUDING its 0 fall-through; '
                             '1 forbids that fall-through, so no Mode-2 step '
                             'can leak into Mode 1 (the ADR-007 arm).')
    # ---- ADR-008 objective: the twelve logical parameters of §2.8 --------- #
    # EVERY default below is NEUTRAL (no module is built, the forward keeps the
    # pre-ADR-008 statements), so a config that omits them is byte-identical to
    # the pure-MSE run. The arm's tuple lives in its YAML.
    parser.add_argument('--resp_loss_mse_weight', default=1.0, type=float,
                        help='ADR-008 alpha: weight of the masked time-domain '
                             'MSE inside the composite respiration objective '
                             '(default 1.0 = neutral).')
    parser.add_argument('--resp_loss_pearson_weight', default=0.0, type=float,
                        help='ADR-008 beta: weight of the masked negative '
                             'Pearson term (0.0 = disabled, no code path). '
                             'The collapse-resistant term: it repels a flat '
                             'zero-variance prediction.')
    parser.add_argument('--resp_loss_spectral_weight', default=0.0, type=float,
                        help='ADR-008 gamma: weight of the band-limited '
                             'spectral power term (0.0 = disabled, no FFT is '
                             'ever built).')
    parser.add_argument('--spectral_f_min', default=0.10, type=float,
                        help='ADR-008 lower respiratory passband edge (Hz); '
                             'must equal the evaluation passband '
                             '(0.10 Hz = 6 bpm).')
    parser.add_argument('--spectral_f_max', default=0.60, type=float,
                        help='ADR-008 upper respiratory passband edge (Hz) '
                             '(0.60 Hz = 36 bpm); must be <= fs/2.')
    parser.add_argument('--spectral_n_fft', default=0, type=int,
                        help='ADR-008 rFFT length for the spectral term. 0 '
                             '(default) = auto, 4 * 2**ceil(log2 L) = 32768 at '
                             'L = 8000 (df 0.0305 Hz, 16 in-band bins); any '
                             'explicit value must be >= L.')
    parser.add_argument('--align_weight', default=0.0, type=float,
                        help='ADR-008 lambda_align: weight of the symmetric '
                             'cross-modal InfoNCE term (0.0 = disabled, and NO '
                             'projection head is created). Needs exactly one '
                             'visual and one physiological stream.')
    parser.add_argument('--align_proj_dim', default=128, type=int,
                        help='ADR-008 d_proj: output width of the InfoNCE '
                             'projection MLPs (>= 16).')
    parser.add_argument('--align_temperature', default=0.07, type=float,
                        help='ADR-008 tau: InfoNCE cosine temperature (> 0).')
    parser.add_argument('--align_core_weight', default=1.0, type=float,
                        help='ADR-008 w_core: pooling weight on the anatomical '
                             'core tubes (A_core) for the thermal side of the '
                             'alignment. 1.0 (default) = unweighted mean '
                             'pooling; 2.0 emphasises the nostril-plume core. '
                             'Inert under spatial_sampling=uniform (A_core '
                             'is empty).')
    parser.add_argument('--align_gather_ddp', default=True, type=str2bool,
                        nargs='?', const=True,
                        help='ADR-008: gather the InfoNCE embeddings across '
                             'ranks (global negatives) during TRAINING. '
                             'Default true; the validation pass never '
                             'gathers, because its loader is unpadded.')
    parser.add_argument('--grad_balance_freq', default=0, type=int,
                        help='ADR-008 §2.7: step interval of the '
                             '||grad_tir|| / ||grad_resp|| diagnostic '
                             '(0 = disabled, zero extra autograd cost). This '
                             'is a logging cadence, NOT an objective '
                             'parameter, so it is recorded under `run` in '
                             'masking.json and never blocks a resume.')
    parser.add_argument('--physio_tokens', default=0, type=int,
                        help='tokenizer-derived (default 0 = derive from '
                             'seq_len/sig_kernel); pass it only to ASSERT the '
                             'identity, the run fails on a mismatch.')
    # ---- placement: WHICH tokens survive (not how many) ------------------ #
    parser.add_argument('--physio_placement', default='scattered', type=str,
                        choices=['scattered', 'span', 'masked_span'],
                        help="1-D placement of the kept tokens: 'scattered' "
                             "(uniform without replacement, default), "
                             "'span' (contiguous KEPT blocks) or "
                             "'masked_span' (contiguous MASKED gaps of "
                             'bounded length, ADR-017). The choice is '
                             'recorded with the run.')
    parser.add_argument('--physio_span_tokens', default=0, type=int,
                        help="span placement only: length of one contiguous "
                             'block in tokens (required when '
                             '--physio_placement span).')
    parser.add_argument('--physio_gap_min_tokens', default=0, type=int,
                        help='masked_span placement only (ADR-017): g_min, '
                             'the shortest masked run in tokens. Required '
                             '>= 1 with --physio_placement masked_span and 0 '
                             'otherwise; the resolved seconds and the '
                             'per-count gap numbers are printed at build.')
    parser.add_argument('--physio_gap_max_tokens', default=0, type=int,
                        help='masked_span placement only (ADR-017): g_max, '
                             'the longest masked run in tokens; '
                             '>= --physio_gap_min_tokens and '
                             '<= seq_len / sig_kernel.')
    parser.add_argument('--loss_weights', default='', type=str,
                        help='FULL per-stream override of the per-modality '
                             'masked-MSE weights: comma list, ONE value per '
                             '--streams modality in order. Empty (default) = '
                             'policy: lambda 1.0 for visual streams and '
                             'lambda --signal_weight for physio streams.')
    parser.add_argument('--signal_weight', default=0.5, type=float,
                        help='weight of each physio (1-D) stream in the '
                             'masked-MSE weighted sum (visual streams fixed '
                             'at 1.0). Tune in ~[0.5, 1.0] from the NORMALIZED '
                             'variance of the physio stream\'s masked target '
                             'tokens: high variance -> ~0.5 so the 1-D signal '
                             'does not dominate the gradient; low variance -> '
                             '~1.0 so it is not ignored.')
    parser.add_argument('--target_norm', default='token', type=str,
                        choices=['token', 'clip', 'none'],
                        help="normalisation of the 1-D reconstruction target. "
                             "'token' (default) = per-token z-score: every "
                             'sig_kernel window is rescaled by its OWN '
                             "mean/std. 'clip' = per-clip z-score: ONE "
                             'mean/std per (sample, stream), so the token '
                             'windows reassemble into a coherent waveform and '
                             'the Stage-2 target space matches Stage 3. '
                             "'none' = IDENTITY: use the 1-D target exactly as "
                             'the dataset returns it. NOTE this knob applies to '
                             'the NON-ROI paths only (e.g. bp4d+): for the '
                             'RAW-tree ROI lineages (tir_roi / rgb_roi) the '
                             'DATASET owns the normalisation via --physio_norm '
                             "and run_pretrain pins target_norm='none'.")
    parser.add_argument('--physio_norm', default='zscore', type=str,
                        choices=['clip', 'zscore'],
                        help="the ROI lineages' label normalisation, applied by "
                             'the DATASET (tir_roi = TIR-ROI+RESP and rgb_roi = '
                             'RGB+BP alike): a per-clip z-score, i.e. one '
                             'mean/std per clip from that clip\'s own samples. '
                             "'clip' and 'zscore' are synonyms; there is no "
                             'raw / session-relative alternative, because a '
                             'second target space makes the two stages '
                             'incomparable. The MODEL never normalises a 1-D '
                             'stream: run_pretrain pins target_norm="none" for '
                             'the ROI data_sets.')
    parser.add_argument('--dec_depth', default=2, type=int)
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    parser.add_argument('--sig_kernel', default=8, type=int,
                        help='signal token window (samples per token). MUST be '
                             'time-aligned with the video tubelet: one token '
                             'must cover the same seconds in every stream, '
                             'i.e. sig_kernel/fs == tubelet_t*temporal_stride/'
                             'fps (default 8/100 == 2*1/25 = 80 ms). The model '
                             'refuses to build on a misaligned geometry.')
    parser.add_argument('--pos_init', default='sincos3d', type=str,
                        choices=['sincos3d', 'random'],
                        help="positional-embedding init: 'sincos3d' (default) "
                             'initialises the video positions with a 3-D '
                             '(t, h, w) sincos grid and the physio positions '
                             'with the same 1-D temporal sincos (a space-time '
                             'prior at init, since the Stage-1 MAE pos_embed '
                             'cannot be reused); \'random\' = trunc_normal.')
    parser.add_argument('--pretrained_encoder', default='videomae', type=str,
                        help='Stage-1 checkpoint that initialises the shared '
                             'encoder. Corpus specs: videomae:k400 '
                             '(Kinetics-400), videomae:ssv2 '
                             '(Something-Something-v2), videomae (source only '
                             '-> --videomae_dataset), base (alias of '
                             'videomae:k400), or a local .pth path. The '
                             'explicit blank aliases none / random / scratch / '
                             'c0 select RANDOM initialisation (no checkpoint); '
                             'any other unknown spec is rejected. Downloads '
                             'into <project_root>/models/initial.')
    parser.add_argument('--videomae_dataset', default='k400', type=str,
                        choices=['k400', 'ssv2'],
                        help='VideoMAE pre-training corpus used when '
                             "--pretrained_encoder names the videomae source "
                             "without a dataset (e.g. 'videomae'): k400 "
                             '(Kinetics-400, default) or ssv2 '
                             '(Something-Something-v2). An explicit '
                             'videomae:k400 / videomae:ssv2 spec wins.')
    parser.add_argument('--patch_stream', default='', type=str,
                        help='ADR-005 I2: which VISUAL stream receives the '
                             'Stage-1 Conv3d tubelet. Empty (default) keeps '
                             'the adapter-ordering resolution (rgb when that '
                             'adapter exists, else the first visual adapter), '
                             'so no existing run changes. The three-stream arm '
                             'sets tir explicitly and records it.')
    # training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--find_unused_parameters', default=True,
                        type=str2bool, nargs='?', const=True,
                        help='DDP find_unused_parameters. Default True keeps '
                             'the historical behaviour, but DDP warns when the '
                             'forward pass turns out to use every parameter '
                             'and the flag costs an extra autograd traversal '
                             'per step -- pure CPU with no benefit, on a loop '
                             'that is already CPU-bound. Pass '
                             '"--find_unused_parameters false" only for a run '
                             'whose logs actually show that warning; a config '
                             'with genuinely unused parameters fails LOUDLY in '
                             'DDP ("Expected to have finished reduction") '
                             'rather than silently training wrongly.')
    # optimizer / lr  (official-MAE semantics)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', default=None, type=float,
                        help='peak (absolute) learning rate. Omit it to derive '
                             'the LR from --blr via lr = blr * batch_size * '
                             'world_size / 256 (large-batch MAE convention). '
                             'For small-batch LOCAL runs pass --lr explicitly: '
                             'the scaled LR would be ~1e-6 and the model would '
                             'not learn.')
    parser.add_argument('--blr', default=1.5e-4, type=float,
                        help='base lr, only used when --lr is not set: '
                             'lr = blr * batch_size * world_size / 256')
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--warmup_epochs', default=40, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--clip_grad', default=0.0, type=float)
    return parse_args_with_config(parser)


def pin_roi_target_norm(args) -> bool:
    """Pin the model's 1-D target normalisation to identity for ROI lineages.

    For every RAW-tree ROI lineage (``tir_roi`` = TIR-ROI+RESP, ``rgb_roi`` =
    RGB+BP, ``tir_roi_resp_rgb2d`` = TIR+2D+RESP, ``tir_roi_resp_rgb2d_bp`` =
    the four-stream arm) the DATASET returns a per-clip z-scored 1-D target.
    The model must therefore NOT normalise the 1-D stream again, so
    ``--target_norm`` is forced to ``'none'`` for these data_sets; any other
    value is reported as a no-op.

    Returns ``True`` when the pin was applied. No effect on non-ROI paths.
    """
    from data import (FOURSTREAM_DATA_SETS, RGB_ROI_DATA_SETS,
                      TIR_ROI_DATA_SETS, TIR_ROI_RGB2D_DATA_SETS)
    data_set = str(getattr(args, 'data_set', '') or '').strip().lower()
    if data_set not in (TIR_ROI_DATA_SETS + RGB_ROI_DATA_SETS
                        + TIR_ROI_RGB2D_DATA_SETS + FOURSTREAM_DATA_SETS):
        return False
    if str(getattr(args, 'target_norm', 'token')) != 'none':
        print(f'[data] data_set {data_set!r}: the dataset owns the 1-D target '
              f'normalisation (per-clip z-score) -> forcing '
              f"--target_norm 'none' (the model does not re-normalise).")
    args.target_norm = 'none'
    return True


def init_stage1_encoder(model, args):
    """Initialise the shared encoder from Stage-1 weights or leave it random.

    The explicit blank aliases (``''`` / ``none`` / ``random`` / ``scratch`` /
    ``c0``) are the random-init path: ``resolve_encoder_weights(...,
    require=False)`` returns an empty path and the model keeps its
    ``trunc_normal_`` initialisation. Any other unknown spec still raises, so
    a typo cannot silently train from scratch.

    :returns: ``{'path', 'loaded', 'random', 'patch_stream'}`` for the
        caller/log. ``patch_stream`` is the RESOLVED recipient (``''`` on the
        random-init path), which the masking manifest records (I2).
    """
    from core.multimae import load_pretrained_encoder
    from models.pretrained import resolve_encoder_weights

    ckpt = resolve_encoder_weights(
        args.pretrained_encoder,
        dataset=getattr(args, 'videomae_dataset', 'k400'),
        require=False)
    if not ckpt:
        print(f'[stage1] random init (spec {args.pretrained_encoder!r})')
        return {'path': '', 'loaded': 0, 'random': True}

    # ``patch_stream`` is only passed when declared: with the empty default the
    # loader resolves the recipient by adapter order exactly as before, and the
    # call shape stays identical for every existing caller (and its doubles).
    declared_patch_stream = str(getattr(args, 'patch_stream', '') or '')
    stats = (load_pretrained_encoder(model, ckpt,
                                     patch_stream=declared_patch_stream)
             if declared_patch_stream
             else load_pretrained_encoder(model, ckpt))
    # The RESOLVED recipient is recorded on the model, not in the return dict:
    # that dict's shape is pinned by test_pretrain_random_init.py, and the run
    # manifest is the only consumer (I2).
    try:
        model.patch_stream_resolved = str(stats.get('patch_stream', '') or '')
    except AttributeError:          # duck-typed double without __dict__
        pass
    # Provenance line: every fold must show the SAME pristine Stage-1 file.
    # Printing size+mtime (cheap) lets a fold sweep prove it started from the
    # shared read-only baseline and not from another fold's output.
    try:
        _st = os.stat(ckpt)
        _prov = (f'{_st.st_size / 1e6:.1f} MB, '
                 f'mtime {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_st.st_mtime))}')
    except OSError:
        _prov = 'stat unavailable'
    print(f'[stage1] init from {ckpt} '
          f'(spec {args.pretrained_encoder!r}, '
          f'{stats["loaded"]} encoder tensors, {_prov})')
    return {'path': ckpt, 'loaded': int(stats['loaded']), 'random': False}


def _objective_non_neutral(model) -> bool:
    """True when an ADR-008 objective term is active for this model.

    The SINGLE predicate shared by the manifest writer and the model: it is
    ``composite_active or align_weight > 0``, never ``beta/gamma`` alone, so
    ``alpha != 1.0`` (which rescales the MSE term even with both other terms
    off) is recorded too.
    """
    return bool(getattr(model, 'composite_active', False)
                or float(getattr(model, 'align_weight', 0.0) or 0.0) > 0.0)


def write_masking_manifest(args, model, *, plan, mask_base: int,
                           stage1: dict) -> dict:
    """Write the ADR-005 §2.8/§2.9 masking record into ``output_dir``.

    The apportioner is NOT an ``nn.Module``, so nothing of the policy is inside
    the checkpoint. This file is therefore the only record that makes a resumed
    run's mask sequence reproducible: it captures the resolved parameter tuple,
    the schedule, the placement policy, the conditioning mechanism and the
    kernel recipient (I2). Rank 0 writes it once, before the first step.

    A resumed run must not silently overwrite the record: if the file exists
    and its stable fields (geometry, policy, conditioning, kernel, mask_base)
    disagree, the run would replay a DIFFERENT mask sequence than the one the
    checkpoints were trained on, so this raises instead (plan Trap 4).

    :returns: the written record (also convenient for tests).
    """
    import json

    def _stable(rec: dict) -> dict:
        return {
            'streams': rec.get('streams'),
            'geometry': rec.get('geometry'),
            'policy': rec.get('policy'),
            'conditioning': rec.get('conditioning'),
            'kernel': rec.get('kernel'),
            # ADR-008 §2.7: the objective tuple is part of the reproducibility
            # record, so resuming with different objective weights must fail
            # loudly. It is compared only when EITHER side carries it (plan
            # decision D4): an unconditional key would make every in-flight
            # pre-ADR-008 run un-resumable, exactly the trap the ADR-007 plan
            # hit with the single-carrier keys.
            'objective': rec.get('objective'),
            'mask_base': (rec.get('run') or {}).get('mask_base'),
        }

    app = model.apportioner
    record = {
        'format': 'ADR-005 masking record v1',
        'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'argv': list(sys.argv[1:]),
        'streams': list(model.streams),
        'visual_streams': list(model.visual),
        'signal_streams': list(model.signal),
        'geometry': {
            'tubelet': [int(x) for x in model.tubelet],
            'input_size': int(model.input_size),
            'num_frames': int(model.num_frames),
            'temporal_stride': int(model.temporal_stride),
            'fps': float(model.fps),
            'fs': float(model.fs),
            'clip_duration': float(getattr(args, 'clip_duration', 0.0) or 0.0),
            'sig_kernel': int(model.sig_kernel),
            'seq_len': int(model.seq_len),
            'tokens_per_unit': int(app.tokens_per_unit),
            'signal_tokens_per_visual_token': int(
                model.signal_tokens_per_visual),
        },
        'policy': {
            'n_budget': int(model.n_budget),
            'video_cap_ratio': float(model.video_cap_ratio),
            'physio_cap_ratio': float(model.physio_cap_ratio),
            'video_min_tubes': int(model.video_min_tubes),
            'video_cap_units': dict(app.video_cap_units),
            'physio_cap_tokens': dict(app.physio_cap_tokens),
            'spatial_patches': int(model.spatial_patches),
            'physio_tokens': int(model.physio_tokens),
            'p_virtual_sensing': float(model.p_virtual_sensing),
            'virtual_sensing_warmup_p': float(model.virtual_sensing_warmup_p),
            'virtual_sensing_warmup_epochs': int(
                model.virtual_sensing_warmup_epochs),
            'physio_placement': str(model.physio_placement),
            'physio_span_tokens': int(model.physio_span_tokens),
        },
        # Contract 7, mechanism (a): the encoder latents are prefixed into the
        # decoder input, so a stream with 0 visible tokens still conditions.
        'conditioning': {'mechanism': 'prefix'},
        'kernel': {
            'recipient': str(getattr(model, 'patch_stream_resolved', '') or ''),
            'pretrained': not bool(stage1.get('random', True)),
            'checkpoint': str(stage1.get('path', '') or ''),
            'loaded_tensors': int(stage1.get('loaded', 0) or 0),
            # I2 technical debt, recorded rather than inferred: a random
            # tokenizer cannot support an R4 claim for these streams.
            'random_init_visuals': [
                s for s in model.visual
                if s != str(getattr(model, 'patch_stream_resolved', '')
                            or '')],
        },
        'run': {
            'mask_base': int(mask_base),
            'epochs': int(getattr(args, 'epochs', 0) or 0),
            'fold': int(getattr(plan, 'fold', -1)),
            'role_digest': str(getattr(plan, 'digest', '')),
        },
    }
    # ADR-008 §2.7: `grad_balance_freq` is a READ-ONLY logging cadence, not an
    # objective parameter. It is recorded under `run` (next to `epochs`), which
    # `_stable()` does not compare, so changing it can never block resuming a
    # preempted HPC run. Written only when armed, so a neutral record stays
    # byte-identical to the pre-ADR-008 file.
    _gb_freq = int(getattr(model, 'grad_balance_freq', 0) or 0)
    if _gb_freq != 0:
        record['run']['grad_balance_freq'] = _gb_freq
    # ADR-006 §2.4/§2.7: the single-carrier keys are written ONLY when they are
    # non-neutral. The resume guard below compares the NEW record's keys against
    # the stored file, so an unconditional write would make an in-flight
    # ADR-005 run fail to resume (`theirs.get(key)` is None for a key the old
    # record never had). With this guard an ADR-005 record is byte-identical to
    # the one the engine wrote before ADR-006.
    _p_s = float(getattr(args, 'p_single_modality', 0.0) or 0.0)
    _p_s_warm = float(getattr(args, 'single_modality_warmup_p', 0.0) or 0.0)
    _s_v_raw = int(getattr(args, 'single_carrier_tubes', 0) or 0)
    if _p_s > 0.0 or _p_s_warm > 0.0 or _s_v_raw != 0:
        record['policy'].update({
            'p_single_modality': float(app.p_single_modality),
            'single_modality_warmup_p': float(app.single_modality_warmup_p),
            'single_carrier_tubes': int(app.single_carrier_tubes),
        })
    # ADR-017: the masked-span gap bounds, written ONLY when non-neutral for the
    # same reason as the ADR-006/007/008 keys above — `_stable()` compares the
    # whole `policy` dict, so an unconditional pair of new keys would make every
    # in-flight pre-ADR-017 run refuse to resume. Everything stays JSON-native
    # (ints and a 2-float list) so the reload compares equal to the fresh record.
    _g_min = int(getattr(model, 'physio_gap_min_tokens', 0) or 0)
    _g_max = int(getattr(model, 'physio_gap_max_tokens', 0) or 0)
    if _g_min or _g_max:
        from core.mask_placement import admissible_gap_counts as _gap_counts

        _g_sec = [round(float(g) * float(model.sig_kernel) / float(model.fs), 6)
                  for g in (_g_min, _g_max)]
        # ADR-017 Criterion 2: N(M) per masked count, as JSON-native
        # ``{"<M>": [n, ...]}`` so the record reloads identically.
        _n_map = {}
        if str(model.physio_placement) == 'masked_span':
            for _k in app.mode2_counts(model.signal[0]):
                _m = int(model.physio_tokens) - int(_k)
                _n_map[str(_m)] = [int(x) for x in
                                   _gap_counts(_m, int(_k), _g_min, _g_max)]
        record['policy'].update({
            'physio_gap_min_tokens': _g_min,
            'physio_gap_max_tokens': _g_max,
            'physio_gap_seconds': _g_sec,
            'physio_gap_counts': _n_map,
        })
    # ADR-007 §2.9: the stratified policy is recorded for EVERY one-visual run
    # (including the ``uniform`` ablation, whose record must still say which
    # policy produced it), while a 2-visual ADR-005/006 manifest stays
    # byte-identical -- the resume guard compares key SETS, so an unconditional
    # write would make an in-flight run un-resumable.
    _spatial = str(getattr(model, 'spatial_sampling', 'uniform') or 'uniform')
    _draw_min = int(getattr(model, 'physio_draw_min', 0) or 0)
    if len(model.visual) == 1 or _spatial != 'uniform' or _draw_min != 0:
        _aware = bool(getattr(model, 'anatomy_aware', False))
        record['policy'].update({
            # 'ADR-007' only when the run IS the ADR's arm; anything else with
            # one visual stream is a comparator/ablation and says so.
            'adr': ('ADR-007' if (_aware and _draw_min == 1) else 'off-adr-007'),
            'spatial_sampling': _spatial,
            'physio_draw_min': _draw_min,
            # The bounds describe A_core, so under ``uniform`` (where no core
            # exists) they are nulled rather than serialized as the ignored
            # defaults; the tube-count knobs stay, because they document what
            # the run would have used.
            'core_row_bounds': ([int(x) for x in app.core_row_bounds]
                                if _aware else None),
            'core_col_bounds': ([int(x) for x in app.core_col_bounds]
                                if _aware else None),
            # Both forms: the 2-D coordinate SET that §2.9 names, and the flat
            # tube indices the placement consumes. Under ``uniform`` the
            # ignored defaults are NOT serialized, so no reader can mistake
            # (2, 3) for a realized A_core.
            'core_coords': ([[int(r), int(c)] for r, c in app.core_coords]
                            if _aware else None),
            'core_indices': ([int(i) for i in app.core_indices]
                             if _aware else []),
            'periph_tubes': (len(app.periph_indices) if _aware else None),
            'core_tubes_min': int(app.core_tubes_min),
            'core_tubes_max': int(app.core_tubes_max),
            'periph_min_tubes': int(app.periph_min_tubes),
            's_tir_min': (None if app.s_tir_min is None
                          else int(app.s_tir_min)),
            's_tir_max': (None if app.s_tir_max is None
                          else int(app.s_tir_max)),
        })
    # ADR-008 §2.7: the resolved objective tuple. Written ONLY for a run whose
    # objective is not the neutral pure-MSE one, so a legacy manifest stays
    # byte-identical (decision D4) while any objective run is fully
    # reproducible and its resume is guarded (see `_stable`).
    if _objective_non_neutral(model):
        _spec = getattr(getattr(model, 'resp_loss', None), 'spectral', None)
        record['objective'] = {
            'adr': 'ADR-008',
            'resp_loss_mse_weight': float(model.resp_loss_mse_weight),
            'resp_loss_pearson_weight': float(model.resp_loss_pearson_weight),
            'resp_loss_spectral_weight': float(model.resp_loss_spectral_weight),
            'spectral_f_min': float(model.spectral_f_min),
            'spectral_f_max': float(model.spectral_f_max),
            # the RESOLVED FFT length and the band bin set, so a reader never
            # has to re-derive them from (L, fs, f_min, f_max)
            'spectral_n_fft': (int(_spec.n_fft_resolved) if _spec is not None
                               else int(model.spectral_n_fft)),
            'spectral_band_bins': (_spec.band_bin_list if _spec is not None
                                   else []),
            'signal_weight': float(model.loss_weights[model.signal[0]]),
            'align_weight': float(model.align_weight),
            'align_proj_dim': int(model.align_proj_dim),
            'align_temperature': float(model.align_temperature),
            'align_core_weight': float(model.align_core_weight),
            'align_gather_ddp': bool(model.align_gather_ddp),
        }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, 'masking.json')
    if os.path.exists(path):
        try:
            with open(path, 'r') as fh:
                previous = json.load(fh)
        except (OSError, ValueError):
            previous = None
        if isinstance(previous, dict):
            mine, theirs = _stable(record), _stable(previous)
            changed = [k for k in mine
                       if not (k == 'objective' and mine[k] is None
                               and theirs.get(k) is None)
                       and mine[k] != theirs.get(k)]
            if changed:
                raise SystemExit(
                    f'{path} already records a different masking policy '
                    f'({", ".join(changed)} differ). Resuming this '
                    f'output_dir with different masking keys would replay a '
                    f'different mask sequence than the checkpoints were '
                    f'trained on. Restore the original keys or point '
                    f'--output_dir elsewhere (ADR-005 §2.8).')
            # Identical policy: keep the run's FIRST record untouched.
            return previous
    with open(path, 'w') as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    return record


def main(args):
    from utils import (WandbLogger, get_world_size, is_main_process,
                       resolve_amp)

    device = init_env(args)
    amp_config = resolve_amp(getattr(args, 'amp_dtype', 'none'), device)
    print(f'[amp] {amp_config.describe()}')

    # MAE linear-scaling rule: only derive the absolute LR from --blr when the
    # user did not pass --lr explicitly. Otherwise blr*batch/256 on a batch-1/2
    # local run silently yields ~1e-6 and the weights never move (flat loss).
    if args.lr is None:
        args.lr = args.blr * args.batch_size * get_world_size() / 256
        print(f'LR derived from blr (blr*batch*world/256): {args.lr:.3e}')
    else:
        print(f'LR set explicitly (--lr): {args.lr:.3e}')

    # ----- 1-D target normalisation is owned by the DATASET --------------- #
    pin_roi_target_norm(args)

    # ----- model ---------------------------------------------------------- #
    if args.model.startswith('project_multimae'):
        from core.multimae import build_pretraining_model
        model = build_pretraining_model(args)
        stage1 = init_stage1_encoder(model, args)
    else:
        raise SystemExit(
            f'--model {args.model!r}: Stage 1 only supports the multimodal '
            f'project_multimae_* entrypoints. The single-stream project_vit_* '
            f'path would build a RANDOM backbone, which is not a valid Stage-1 '
            f'model here.')
    model.to(device)
    if torch.cuda.is_available():
        # ADR-005 §3.3 (Resource): peak device memory is part of the record,
        # and the predecessor's figure does not transfer to this geometry.
        torch.cuda.reset_peak_memory_stats(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model = {args.model}, params = {n_params:,}')

    # ----- data: the ADR-001 roles of this fold --------------------------- #
    from data import build_pretraining_dataset
    plan = load_or_create_plan(args, stage='stage2')
    # Persisted into every checkpoint of this run: Stage 3 refuses a pair whose
    # digests differ, because two partitions can share a fold index.
    args.partition_digest = plan.digest
    dataset_train = build_pretraining_dataset(args, plan, role='train')
    # Record the respiration cache IDENTITY (directory name, machine-
    # independent) on the args so the checkpoint carries it: it is a corpus-
    # level contract like clip_stride -- it changes the LABEL SPACE, never a
    # tensor shape, so Stage 3 cannot detect a mismatch downstream.
    _stats = getattr(dataset_train, 'stats', {}) or {}
    if _stats.get('resp_cache_key'):
        args.resp_cache_key = _stats['resp_cache_key']
        print(f'[data] resp_cache_key = {args.resp_cache_key} '
              f'(recorded in the checkpoint)')
    elif str(getattr(args, 'resp_cache', '') or '').strip():
        print('[data] WARNING: --resp_cache was passed but the dataset '
              'reports no resp_cache_key (not a TIR-ROI dataset?)')
    # the task/group selection lives in the dataset -> make it visible in the
    # run log (a Stage-2 run has no entries.json of its own)
    _describe = getattr(dataset_train, 'describe', None)
    if callable(_describe):
        print('-' * 72)
        print(_describe())
        print('-' * 72)
    else:
        print(f'Dataset {type(dataset_train).__name__}: '
              f'{len(dataset_train)} sample(s)')
    data_loader_train = make_data_loader(args, dataset_train, shuffle=True)
    # fail fast + name the cause instead of crashing later in the logger (the
    # LR schedule below clamps steps_per_epoch to >= 1, so it would not notice)
    check_loader_not_empty(data_loader_train, 'train', args)

    # Validation role: the ONLY selection signal (I4/I7). Ordered, single-worker
    # and unpadded, so the same checkpoint evaluates to the same number.
    dataset_val = build_pretraining_dataset(args, plan, role='val')
    data_loader_val = make_data_loader(args, dataset_val, shuffle=False,
                                       drop_last=False, num_workers=0)
    check_loader_not_empty(
        data_loader_val, 'val', args,
        extra='Stage 2 requires a validation role (ADR-001 I4); check the fold\'s '
              'partition record and any --subjects filter.')

    # step-level warmup + cosine LR schedule (mirrors MultiMAE/VideoMAE)
    from utils import cosine_scheduler
    steps_per_epoch = max(
        1, (len(dataset_train) // (args.batch_size * get_world_size()))
        // max(1, args.update_freq))
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, steps_per_epoch,
        warmup_epochs=args.warmup_epochs)
    print(f'Step-level LR schedule: {len(lr_schedule_values)} steps, '
          f'peak {args.lr:.3e} -> min {args.min_lr:.3e}, '
          f'warmup {args.warmup_epochs} epoch(s)')

    model_without_ddp = model
    if args.distributed:
        # False only when the user asked for it (see --find_unused_parameters):
        # the flag costs an extra autograd traversal per step and this loop is
        # CPU-bound, so it is pure overhead whenever nothing is unused.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=bool(args.find_unused_parameters))
        model_without_ddp = model.module

    # ----- optimizer / scaler -------------------------------------------- #
    from utils import (NativeScalerWithGradNormCount, auto_resume_model,
                       create_optimizer)
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScalerWithGradNormCount()

    # ----- resume --------------------------------------------------------- #
    # auto_resume_model restores the model, optimizer and AMP scaler from
    # --resume, or from the newest pointer in <output_dir>/checkpoints
    # (latest_checkpoint.txt, written by save_model), and sets
    # args.start_epoch = checkpoint epoch + 1. Without this a job killed by the
    # Slurm wall clock restarted from epoch 0, which makes the 800-epoch
    # full-corpus schedule impossible on `acc` (24 h).
    # NOTE the LR schedule is NOT restored: it is rebuilt from
    # --epochs/steps_per_epoch above, so keep --epochs IDENTICAL across submits.
    auto_resume_model(args, model_without_ddp, optimizer, loss_scaler)
    start_epoch = int(getattr(args, 'start_epoch', 0) or 0)
    if start_epoch >= args.epochs:
        raise SystemExit(
            f'nothing to do: the checkpoint in {args.output_dir}/checkpoints '
            f'is already at epoch {start_epoch} and --epochs is {args.epochs}. '
            f'Raise --epochs to train longer, or point --output_dir elsewhere '
            f'/ remove latest_checkpoint.txt to start from scratch.')

    # ----- training loop -------------------------------------------------- #
    from utils import save_model
    # Optional W&B logging (rank 0 only, offline by default inside a Slurm job).
    # ``--log_wandb`` used to be parsed but never consumed, so the flag was a
    # silent no-op; utils/wandb_logger.py is its consumer. The run name is the
    # output-dir basename, i.e. one W&B run per FOLD of a sweep.
    budget_tag = 'n_budget{}'.format(int(getattr(args, 'n_budget', 350)))
    wandb_logger = WandbLogger.from_args(
        args, job_type='pretrain',
        tags=(str(getattr(args, 'data_set', '')),
              budget_tag,
              str(getattr(args, 'physio_placement', 'scattered')),
              f'fold{plan.fold}', f'digest{plan.digest[:8]}'),
        extra={'data/steps_per_epoch': steps_per_epoch,
               'data/n_samples': len(dataset_train),
               'train/world_size': get_world_size()})
    print(f'Start training for {args.epochs} epochs '
          f'(resuming at epoch {start_epoch})')

    # Selection is one object shared with Stage 3; it is also the only writer of
    # best.pth, so a delivered checkpoint can only come from the val role (I7).
    from protocol import (SelectionTracker, append_jsonl, make_eval_record,
                          write_selection)
    tracker = SelectionTracker(
        epochs=args.epochs, rule=getattr(args, 'val_select', 'fixed_epoch'),
        criterion=getattr(args, 'val_criterion', 'joint_loss'),
        eval_freq=int(getattr(args, 'eval_freq', 0) or 0),
        n_val_clips=len(dataset_val), n_train_clips=len(dataset_train),
        dev_smoke=bool(getattr(args, 'dev_smoke', False)))
    mask_base = int(getattr(args, 'val_seed', None)
                    or getattr(args, 'split_seed', 42))
    # ADR-005 §2.9 limits E_warm to [0, epochs]; the apportioner cannot see the
    # run length, so the upper half of the admissibility check lives here. An
    # E_warm beyond the run is not dangerous, only a curriculum that never
    # ends, and it must not pass silently.
    e_warm = int(getattr(args, 'virtual_sensing_warmup_epochs', 0) or 0)
    if e_warm > int(args.epochs):
        raise SystemExit(
            f'--virtual_sensing_warmup_epochs {e_warm} exceeds --epochs '
            f'{int(args.epochs)}: the curriculum would never reach its target '
            f'probability (ADR-005 §2.9 admits [0, epochs]).')
    if is_main_process():
        _masking = write_masking_manifest(
            args, model_without_ddp, plan=plan, mask_base=mask_base,
            stage1=stage1)
        print(f'[mask] wrote {os.path.join(args.output_dir, "masking.json")} '
              f'(p_warm={_masking["policy"]["virtual_sensing_warmup_p"]:g}, '
              f'p_target={_masking["policy"]["p_virtual_sensing"]:g}, '
              f'E_warm={_masking["policy"]["virtual_sensing_warmup_epochs"]}, '
              f'p_single={_masking["policy"].get("p_single_modality", 0.0):g}, '
              f'S_v={_masking["policy"].get("single_carrier_tubes", "derive")}, '
              f'patch_stream={_masking["kernel"]["recipient"] or "(random)"}, '
              f'spatial_sampling={_masking["policy"].get("spatial_sampling", "uniform")}, '
              f'A_core={_masking["policy"].get("core_coords")}, '
              f'physio_draw_min={_masking["policy"].get("physio_draw_min", 0)})')
        if _masking['policy'].get('physio_gap_min_tokens'):
            # ADR-017 Criterion 2: the banner names the placement, both bounds,
            # their seconds and every admissible gap count per masked count.
            print(f'[mask] placement={_masking["policy"]["physio_placement"]} '
                  f'gaps={_masking["policy"]["physio_gap_min_tokens"]}..'
                  f'{_masking["policy"]["physio_gap_max_tokens"]} tokens = '
                  f'{_masking["policy"]["physio_gap_seconds"]} s  '
                  f'N(M)={_masking["policy"]["physio_gap_counts"]}')
    # Mask RNG of the training loop (ADR-005 §2.3 contract 8). The COUNTS
    # generator is seeded from a rank-independent key, so every rank derives
    # the same plan without a collective; the PLACEMENT generator includes the
    # rank, so the samples differ across ranks. ``get_rank()`` is 0 outside DDP.
    rank = get_rank()
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    selection_path = os.path.join(args.output_dir, 'selection.json')
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        _t_epoch = time.time()
        stats = train_one_epoch_pretrain(
            model=model, data_loader=data_loader_train, optimizer=optimizer,
            device=device, epoch=epoch, loss_scaler=loss_scaler,
            max_norm=args.clip_grad, update_freq=args.update_freq,
            lr_schedule_values=lr_schedule_values,
            start_steps=epoch * steps_per_epoch,
            amp_config=amp_config,
            mask_base=mask_base, steps_per_epoch=steps_per_epoch,
            rank=rank, log_writer=wandb_logger)
        epoch_seconds = time.time() - _t_epoch
        print(f'Epoch {epoch} stats: {stats}')
        # ONE step axis for the whole run: the global optimizer step. The
        # per-step `train/loss` curve is written by the engine at `sched_idx`,
        # so epoch and validation points must land on that same axis -- an
        # epoch-indexed point after a per-step one is rejected as
        # non-monotonic by the backend.
        wb_step = (epoch + 1) * max(1, int(steps_per_epoch))
        # ``train/loss`` is the per-STEP series written by the engine; the
        # epoch mean is published separately as ``train_epoch/loss`` so the two
        # meanings never share one key.
        wandb_logger.log({**{f'train_epoch/{k}': v for k, v in stats.items()},
                          'train_epoch/seconds': epoch_seconds,
                          'epoch': epoch}, step=wb_step)

        if tracker.should_eval(epoch):
            _t_val = time.time()
            val = evaluate_pretrain(model, data_loader_val, device,
                                    mask_base=mask_base, epoch=epoch)
            improved = tracker.observe(epoch, val['val_loss'])
            record = make_eval_record(
                epoch=epoch, value_kind=tracker.criterion,
                criterion_value=val['val_loss'],
                counts={'clips': val['n_samples'],
                        'subjects': len(plan.roles['val'])},
                seed={'val_seed': plan.val_seed, 'mask_base': mask_base},
                components=val['components'], pinned=val['pinned'],
                role_digest=plan.digest)
            print(f'[val] epoch {epoch}: loss {val["val_loss"]:.6f} '
                  f'over {val["n_samples"]} sample(s) '
                  f'({time.time() - _t_val:.1f}s)'
                  + ('  <- best' if improved else ''))
            if is_main_process():
                append_jsonl(metrics_path, record)
            wandb_logger.log({f'val/{k}': v
                              for k, v in val['components'].items()},
                             step=wb_step)
            # ADR-008 §2.7: the objective summary block, a SIBLING of
            # `components` (which stays the MSE component set). Empty for a
            # run whose config enables no objective term.
            _objective = val.get('objective') or {}
            if _objective:
                wandb_logger.log({f'val/{k}': v
                                  for k, v in _objective.items()},
                                 step=wb_step)
                print('[val]   objective  '
                      + '  '.join(f'{k}={v:.4f}' for k, v
                                  in sorted(_objective.items())))
            # ADR-007 §2.9: the stratified observables live in their OWN dict,
            # never in ``components`` (that is the MSE component set and is
            # asserted as such by test_stage2_protocol). Logged on the same
            # step as the losses so a stratified curve is readable offline.
            _strat = val.get('stratification') or {}
            if _strat:
                wandb_logger.log({f'val/{k}': v for k, v in _strat.items()},
                                 step=wb_step)
                print('[val]   kept tubes  '
                      + '  '.join(f'{k.replace("tubes_", "")}={v:.2f}'
                                  for k, v in sorted(_strat.items())))
            wandb_logger.log({'val/loss': val['val_loss']}, step=wb_step)
            if improved and tracker.writes_best_checkpoint and is_main_process():
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save({'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'partition_digest': plan.digest},
                           os.path.join(args.output_dir, 'best.pth'))

        if is_main_process() and (epoch % args.save_ckpt_freq == 0
                                  or epoch + 1 == args.epochs):
            save_model(args, epoch, model, model_without_ddp, optimizer,
                       loss_scaler)
    if is_main_process():
        selection = tracker.record(pinned=None)
        selection['partition_digest'] = plan.digest
        write_selection(selection_path, selection)
        print(f'[selection] {selection["rule"]}: epoch '
              f'{selection["selected_epoch"]}, '
              f'{selection["criterion"]} '
              f'{selection["criterion_value"]} -> {selection_path}')
        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated(device) / 1e9
            reserv = torch.cuda.max_memory_reserved(device) / 1e9
            print(f'[resource] peak device memory: allocated {alloc:.2f} GB, '
                  f'reserved {reserv:.2f} GB (batch {args.batch_size}, '
                  f'precision {getattr(args, "amp_dtype", "none")})')
    wandb_logger.finish()


if __name__ == '__main__':
    args = get_args()
    main(args)
