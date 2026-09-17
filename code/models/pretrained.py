"""Downloadable "initial" (Stage-1) encoder weights for the project ViTs.

The entrypoints registered in :mod:`core.model` (e.g.
``project_vit_small_patch16_224``) build a **randomly initialised** network --
see :func:`models.build.create_model`. The *spatial priors* of plan Stage 1 come
from an external checkpoint that has to be fetched from the public internet.
This module knows which checkpoint belongs to which ViT **variant** and caches
the download under::

    <project_root>/models/initial/
    <project_root>/models/initial/videomae_base_patch16_224.pth
    <project_root>/models/initial/mae_pretrain_vit_base.pth

Only the ``base`` and ``large`` variants are served (see
``PRETRAINED_SOURCES``); the DeiT-Small and MAE ViT-Huge sources were removed
with the ``small``/``huge`` variants, because the project plan no longer uses
those geometries. Any other backbone is still reachable as
``timm:<model_id>`` or as a local path.

(``<project_root>`` is the repository root, i.e. the parent of ``code/``. Use
the ``INITIAL_MODELS_DIR`` environment variable or ``--weights_dir`` to place
them elsewhere, e.g. a scratch volume on the HPC.)

Spec grammar
------------
The runners accept a *variant spec* everywhere a checkpoint path is expected
(``--finetune``, ``--pretrained_encoder``)::

    ''            random init, no download                  (control C0)
    base          -> videomae:base    (Stage-1 default)
    large         -> videomae:large
    mae:base      explicit source: ``videomae`` | ``mae`` | ``timm``
    timm:vit_base_patch16_224.mae
                  any timm/HF checkpoint name when the built-in table has no
                  entry for what you want
    path/to.pth   a path that exists is returned untouched (never downloaded)

Examples (all from ``code/``)::

    python runners/run_download_weights.py --list
    python runners/run_download_weights.py base          # VideoMAE ViT-Base
    python runners/run_download_weights.py mae:base      # plain MAE ViT-Base
    python runners/run_download_weights.py --all         # base + large
    python runners/run_finetune.py --finetune large ...  # downloads, then trains

Sources
-------
``videomae``
          MCG-NJU **VideoMAE** (HuggingFace Hub ``MCG-NJU/videomae-*``, mirrored
          via ``HF_ENDPOINT``): ViT-B / ViT-L. **The Stage-1 source of
          choice**: ``patch_embed.proj`` is a ``Conv3d(3, D, (2,16,16))``
          tubelet filter -- the exact shape this repo's adapters use -- so the
          tokenizer transfers verbatim (no boxcar inflation and no
          motion-blind init), and the objective (tube-masked video MAE) is the
          same family as Stage 2. Hosted in the HF `transformers` layout, which
          the loader maps back to MAE keys. Licence: CC-BY-NC 4.0 (fine for
          academic work; state it if you redistribute).
``mae``   Facebook AI MAE (``dl.fbaipublicfiles.com``): ViT-B / ViT-L. The 2-D
          control for the VideoMAE source (same 148 tensors transfer, but the
          patch filter is boxcar-inflated instead of a real 3-D tubelet).
          ``timm`` also ships ``*.mae`` recipes, e.g.
          ``vit_base_patch16_224.mae``.
``timm``  HuggingFace-hosted ``timm`` weights. Honours the ``HF_ENDPOINT``
          environment variable (mirrors for restricted networks).

Download mechanics
------------------
Streamed with stdlib ``urllib`` (no extra dependency), written to a ``.part``
file with HTTP-Range **resume**, atomically renamed, and cached -- a second run
is a no-op. Under DDP only rank 0 downloads (the others wait on a barrier).
``--verify`` additionally ``torch.load``s the file and checks that it looks like
a ViT state dict; it is opt-in because the largest checkpoint is ~1.3 GB.

.. note::
   The loader functions (``core.multimae.load_pretrained_encoder``,
   ``core.au_probe.load_au_probe_weights``) map MAE/timm key layouts
   (``blocks.*`` -> ``enc_blocks.*``, ``patch_embed.proj.*`` -> the RGB tubelet
   adapter) and the HF VideoMAE layout
   (``core.multimae.canonicalise_vit_state_dict``). A 3-D ``Conv3d`` patch
   embed is copied verbatim, a 2-D one is boxcar-inflated over the tubelet.
   They raise on a geometry mismatch, so pick the variant that matches your
   ``enc_embed_dim``/``enc_depth``/``enc_num_heads``.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    'VIT_VARIANTS', 'PRETRAINED_SOURCES', 'DEFAULT_SOURCE', 'PROJECT_ROOT',
    'initial_dir', 'available_specs', 'describe_sources', 'plan_download',
    'download_pretrained', 'resolve_encoder_weights', 'main',
]

_CHUNK = 1 << 20          # 1 MiB read chunks
_MIN_BYTES = 1 << 20      # no real ViT checkpoint is smaller than 1 MiB

# ``code/models/pretrained.py`` -> [0]=code/models, [1]=code, [2]=repo root
_CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _CODE_DIR.parent

_FBAI = 'https://dl.fbaipublicfiles.com'

#: ViT geometry per variant (matches the ``@register_model`` entrypoints in
#: :mod:`core.model`). Used for user-facing hints and error messages.
#: NOTE: only the variants that still have a Stage-1 checkpoint source are
#: listed. The ``small`` (384/12/6) and ``huge`` (1280/32/16) geometries still
#: exist as model entrypoints, but their only sources (DeiT-S and MAE ViT-H)
#: were dropped with the plan; build them with an explicit path or a
#: ``timm:<model_id>`` spec if ever needed.
VIT_VARIANTS: Dict[str, Dict[str, int]] = {
    'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
}

#: ``source -> variant -> recipe``. ``candidates`` is a list of
#: ``(url, kind)`` pairs tried in order (``kind`` is ``'torch'`` or
#: ``'safetensors'``; the latter is converted to a ``.pth`` after download).
#: ``file`` is the final, cache-checked file name inside ``models/initial/``.
#: A candidate URL may contain ``{hf}``, which :func:`_lookup` substitutes with
#: the CURRENT ``$HF_ENDPOINT`` (mirrors for restricted networks).
PRETRAINED_SOURCES: Dict[str, Dict[str, dict]] = {
    'videomae': {
        # The Stage-1 source of choice: the patch embed IS a tubelet
        # Conv3d(3, D, (2,16,16)), i.e. the exact shape this repo's adapters
        # use, so the tokenizer transfers verbatim (no boxcar inflation) and
        # the model is not motion-blind at init. Objective (tube-masked video
        # MAE) also matches Stage 2. Weights: CC-BY-NC 4.0.
        'base': {
            'file': 'videomae_base_patch16_224.pth',
            'candidates': [
                ('{hf}/MCG-NJU/videomae-base/resolve/main/pytorch_model.bin',
                 'torch'),
                ('{hf}/MCG-NJU/videomae-base/resolve/main/model.safetensors',
                 'safetensors'),
            ],
            'note': 'VideoMAE ViT-B (768/12/12), tube-masked video MAE on '
                    'Kinetics-400; 3-D tubelet patch embed 2x16x16',
        },
        'large': {
            'file': 'videomae_large_patch16_224.pth',
            'candidates': [
                ('{hf}/MCG-NJU/videomae-large/resolve/main/pytorch_model.bin',
                 'torch'),
                ('{hf}/MCG-NJU/videomae-large/resolve/main/model.safetensors',
                 'safetensors'),
            ],
            'note': 'VideoMAE ViT-L (1024/24/16), tube-masked video MAE on '
                    'Kinetics-400; 3-D tubelet patch embed 2x16x16',
        },
    },
    'mae': {
        'base': {
            'file': 'mae_pretrain_vit_base.pth',
            'candidates': [
                (f'{_FBAI}/mae/pretrain/mae_pretrain_vit_base.pth', 'torch'),
            ],
            'note': 'MAE ViT-Base, self-supervised on ImageNet-1k',
        },
        'large': {
            'file': 'mae_pretrain_vit_large.pth',
            'candidates': [
                (f'{_FBAI}/mae/pretrain/mae_pretrain_vit_large.pth', 'torch'),
            ],
            'note': 'MAE ViT-Large, self-supervised on ImageNet-1k',
        },
    },
}

#: Variant -> source used when the spec does not name one explicitly.
#: ``base``/``large`` prefer VideoMAE: a real 3-D tubelet tokenizer transfers
#: verbatim (see ``PRETRAINED_SOURCES['videomae']``). ``mae:base`` /
#: ``mae:large`` are the 2-D ImageNet controls.
DEFAULT_SOURCE: Dict[str, str] = {
    'base': 'videomae',
    'large': 'videomae',
}

#: Convenience ``timm:`` suggestions shown when a variant is missing upstream.
_TIMM_SUGGESTIONS = {
    'base': 'vit_base_patch16_224.mae',
    'large': 'vit_large_patch16_224.mae',
}

_NONE_ALIASES = {'', 'none', 'random', 'scratch', 'c0'}


# --------------------------------------------------------------------------- #
# directory / process helpers
# --------------------------------------------------------------------------- #
def initial_dir(dest_dir: Optional[str] = None) -> Path:
    """Directory the checkpoints are cached in (created on demand).

    Resolution order: explicit ``dest_dir`` -> ``$INITIAL_MODELS_DIR`` ->
    ``<project_root>/models/initial``.
    """
    if dest_dir:
        return Path(dest_dir).expanduser()
    env = os.environ.get('INITIAL_MODELS_DIR')
    if env:
        return Path(env).expanduser()
    return PROJECT_ROOT / 'models' / 'initial'


def _dist_initialized() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _is_main() -> bool:
    try:
        from utils.dist import is_main_process
        return is_main_process()
    except Exception:
        return True


def _barrier():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def _log(msg: str, quiet: bool = False):
    if not quiet and _is_main():
        print(msg, flush=True)


def _hf_endpoint() -> str:
    return os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')


# --------------------------------------------------------------------------- #
# spec parsing / lookup (no network)
# --------------------------------------------------------------------------- #
def _split_spec(spec: str) -> Tuple[str, str, str]:
    """Classify ``spec`` -> ``(kind, source, name)`` with kind in
    ``{'none', 'path', 'remote'}``."""
    spec = (spec or '').strip()
    if spec.lower() in _NONE_ALIASES:
        return 'none', '', ''
    if os.path.exists(os.path.expanduser(spec)):
        return 'path', '', spec
    # Anything that *looks* like a path stays a path: fail loudly instead of
    # silently interpreting a typo'd checkpoint path as a variant name.
    if (os.sep in spec or '/' in spec or '\\' in spec
            or Path(spec).suffix.lower() in {'.pth', '.pt', '.bin', '.ckpt',
                                             '.safetensors'}):
        raise FileNotFoundError(
            f'--finetune/--pretrained_encoder path does not exist: {spec!r}. '
            f'Use a variant spec ({"/".join(sorted(DEFAULT_SOURCE))}) to '
            f'auto-download an initial encoder into {initial_dir()}.')
    if ':' in spec:
        source, name = spec.split(':', 1)
        return 'remote', source.strip().lower(), name.strip()
    variant = spec.lower()
    if variant not in VIT_VARIANTS:
        # A bare unknown name is a typo far more often than a timm model id,
        # so refuse it instead of firing a doomed Hub request.
        choices = ', '.join(sorted(VIT_VARIANTS))
        raise KeyError(
            f'Unknown variant {spec!r}. Choose one of {choices}, use a '
            f'<source>:<variant> spec (e.g. videomae:base, mae:large), or a '
            f"timm model id as 'timm:<model_id>'.")
    return 'remote', DEFAULT_SOURCE[variant], variant


def _lookup(source: str, name: str) -> Tuple[str, dict]:
    """Return ``(cache_key, recipe)`` for a (source, name) pair."""
    if source == 'timm':
        safe = name.replace('/', '__')
        return name, {
            'file': f'timm-{safe}.pth',
            'candidates': [
                (f'{_hf_endpoint()}/timm/{name}/resolve/main/pytorch_model.bin',
                 'torch'),
                (f'{_hf_endpoint()}/{name}/resolve/main/pytorch_model.bin',
                 'torch'),
                (f'{_hf_endpoint()}/timm/{name}/resolve/main/model.safetensors',
                 'safetensors'),
                (f'{_hf_endpoint()}/{name}/resolve/main/model.safetensors',
                 'safetensors'),
            ],
            'note': f'timm/HuggingFace weights ({name})',
        }
    table = PRETRAINED_SOURCES.get(source)
    if table is None:
        raise KeyError(
            f"Unknown weight source {source!r}. Known sources: "
            f"{', '.join(sorted(PRETRAINED_SOURCES))}, timm")
    if name not in table:
        alts = [f'{s}:{name}' for s, t in PRETRAINED_SOURCES.items()
                if name in t]
        if name in _TIMM_SUGGESTIONS:
            alts.append(f'timm:{_TIMM_SUGGESTIONS[name]}')
        raise KeyError(
            f"Source {source!r} has no entry for variant {name!r}. Available "
            f"in {source!r}: {', '.join(sorted(table))}. "
            + (f'Alternatives for {name!r}: {", ".join(alts)}.' if alts else ''))
    recipe = table[name]
    if any('{hf}' in url for url, _ in recipe['candidates']):
        # resolve $HF_ENDPOINT at CALL time (mirrors must work without reimport)
        recipe = dict(recipe, candidates=[(url.format(hf=_hf_endpoint()), kind)
                                          for url, kind in recipe['candidates']])
    return name, recipe


def plan_download(spec: str, dest_dir: Optional[str] = None) -> dict:
    """Resolve ``spec`` to a download plan **without** touching the network.

    :return: dict with ``kind``, ``source``, ``variant``, ``dest`` (Path),
        ``urls`` and ``note``. Raises for un-downloadable specs.
    """
    kind, source, name = _split_spec(spec)
    if kind == 'none':
        raise ValueError('empty/random spec has nothing to download')
    if kind == 'path':
        return {'kind': 'path', 'source': '', 'variant': '',
                'dest': Path(name), 'urls': [], 'note': 'local file'}
    variant, recipe = _lookup(source, name)
    return {
        'kind': 'remote',
        'source': source,
        'variant': variant,
        'dest': initial_dir(dest_dir) / recipe['file'],
        'candidates': list(recipe['candidates']),
        'urls': [u for u, _ in recipe['candidates']],
        'note': recipe.get('note', ''),
    }


def available_specs() -> List[str]:
    """Every spec the built-in table can serve directly."""
    specs = []
    for source, table in PRETRAINED_SOURCES.items():
        specs.extend(f'{source}:{v}' for v in sorted(table))
    return specs


def describe_sources() -> str:
    """Human-readable table of variants -> source -> file (for ``--list``)."""
    lines = [
        f'Initial (Stage-1) encoder weights -> {initial_dir()}',
        '',
        f"{'spec':<22} {'geometry (dim/depth/heads)':<27} file / example",
    ]
    for variant in ('base', 'large'):
        geo = VIT_VARIANTS[variant]
        geom = f"{geo['embed_dim']}/{geo['depth']}/{geo['num_heads']}"
        default = f'{DEFAULT_SOURCE[variant]}:{variant}'
        first = True
        for source, table in PRETRAINED_SOURCES.items():
            if variant not in table:
                continue
            spec = f'{source}:{variant}'
            mark = ' *' if spec == default else ''
            lines.append(f'{spec + mark:<22} {geom if first else "":<27} '
                         f'{table[variant]["file"]}')
            first = False
        sugg = _TIMM_SUGGESTIONS.get(variant)
        if sugg:
            lines.append(f'{"timm:<model_id>":<22} {"":<27} e.g. timm:{sugg}')
    lines += [
        '',
        ' * = default source for that variant',
        "Any HuggingFace/timm model id works as 'timm:<model_id>'; set "
        'HF_ENDPOINT for a mirror.',
        'Overloads anywhere a checkpoint path is accepted, e.g. '
        '--finetune base --pretrained_encoder mae:large',
        'stage2_local_pretrained.yaml pairs project_multimae_base with '
        "pretrained_encoder: videomae:base (3-D tubelet tokenizer transfers "
        'verbatim; a 5-D source kernel must equal `tubelet`).',
    ]
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _progress(name: str, done: int, total: int, quiet: bool):
    if quiet or not _is_main():
        return
    step = 100 * _CHUNK
    if done == total or done % step < _CHUNK:
        pct = f'{100.0 * done / total:3.0f}%' if total else '  ?%'
        size = f'{done / 1e6:.0f}' + (f'/{total / 1e6:.0f} MB' if total else ' MB')
        print(f'    ... {name} {size} {pct}', flush=True)


def _stream_download(url: str, tmp: Path, quiet: bool):
    """Stream ``url`` into ``tmp`` (HTTP-Range resume if ``tmp`` exists)."""
    pos = tmp.stat().st_size if tmp.exists() else 0
    headers = {'User-Agent': 'kismed-ma-project/1.0'}
    if pos:
        headers['Range'] = f'bytes={pos}-'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        if pos and getattr(resp, 'status', 200) != 206:
            pos = 0                      # server ignored Range -> start over
            _log('    (server ignored Range; restarting download)', quiet)
        total = int(resp.headers.get('Content-Length') or 0) + pos
        done = pos
        with open(tmp, 'ab' if pos else 'wb') as fh:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                _progress(tmp.name, done, total, quiet)
    return tmp


def _convert_safetensors(src: Path, dest: Path):
    try:
        from safetensors.torch import load_file
    except ImportError as e:                              # pragma: no cover
        raise RuntimeError(
            'safetensors is required to convert this checkpoint; install it '
            'with `pip install safetensors` (timm already depends on it)') from e
    import torch
    torch.save(load_file(str(src)), dest)


def _verify_checkpoint(path: Path):
    """``torch.load`` ``path`` and check it looks like a ViT state dict."""
    try:
        import torch
    except ImportError as e:                              # pragma: no cover
        raise RuntimeError(
            '--verify needs torch installed in the active interpreter') from e
    obj = torch.load(path, map_location='cpu', weights_only=True)
    state = obj
    if isinstance(obj, dict):
        for key in ('model', 'state_dict', 'module'):
            if isinstance(obj.get(key), dict):
                state = obj[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError(f'{path} does not look like a ViT checkpoint '
                           f'(not a state dict)')
    # Normalise the HF transformers VideoMAE layout first, so a perfectly good
    # `videomae.encoder.layer.*` file is not rejected for lacking `attn.qkv`.
    try:                                                  # pragma: no cover
        from core.multimae import canonicalise_vit_state_dict
        state = canonicalise_vit_state_dict(state)[0]
    except Exception:
        pass
    if not any(k.endswith('attn.qkv.weight') for k in state):
        raise RuntimeError(f'{path} does not look like a ViT checkpoint '
                           f'(no "*.attn.qkv.weight" key)')


def _write_sidecar(dest: Path, info: dict):
    """Best-effort provenance log (which URL produced this file)."""
    try:
        dest.with_name(dest.name + '.json').write_text(
            json.dumps(info, indent=2) + '\n')
    except OSError:
        pass


def download_pretrained(spec: str, dest_dir: Optional[str] = None,
                        force: bool = False, quiet: bool = False,
                        verify: bool = False) -> Path:
    """Download (once) the checkpoint for ``spec``; return its local path.

    Cached files are reused unless ``force``. Under DDP only rank 0 downloads;
    the other ranks wait on a barrier and then use the shared file.
    """
    plan = plan_download(spec, dest_dir)
    if plan['kind'] == 'path':
        return plan['dest']

    dest: Path = plan['dest']
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= _MIN_BYTES and not force:
        _log(f'[init-weights] cached: {dest}', quiet)
        return dest

    # non-main ranks: let rank 0 download, then pick up the result
    if _dist_initialized() and not _is_main():
        _barrier()
        if not dest.exists() or dest.stat().st_size < _MIN_BYTES:
            raise RuntimeError(f'rank-0 download failed: {dest} is missing '
                               f'or truncated')
        _log(f'[init-weights] using rank-0 download: {dest}', quiet)
        return dest

    geo = VIT_VARIANTS.get(plan['variant'])
    if geo:
        _log(f"[init-weights] {spec} => ViT-{plan['variant']} "
             f"(embed_dim {geo['embed_dim']}, depth {geo['depth']}, "
             f"heads {geo['num_heads']}) -- make sure the model/config "
             f'matches', quiet)

    errors: List[str] = []
    try:
        for url, kind in plan['candidates']:
            tmp = dest.with_name(f'{dest.name}.{kind}.part')
            try:
                _log(f'[init-weights] downloading {url}', quiet)
                _stream_download(url, tmp, quiet)
                if tmp.stat().st_size < _MIN_BYTES:
                    raise RuntimeError(f'suspiciously small file '
                                       f'({tmp.stat().st_size} bytes)')
                if kind == 'safetensors':
                    _convert_safetensors(tmp, dest)
                    tmp.unlink(missing_ok=True)
                else:
                    os.replace(tmp, dest)
                if verify:
                    _verify_checkpoint(dest)
                _write_sidecar(dest, {
                    'spec': spec, 'source': plan['source'],
                    'variant': plan['variant'], 'url': url,
                    'file': dest.name, 'bytes': dest.stat().st_size,
                })
                _log(f'[init-weights] saved {dest} '
                     f'({dest.stat().st_size / 1e6:.1f} MB)', quiet)
                return dest
            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                    RuntimeError) as e:
                errors.append(f'  {url}\n      -> {e}')
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
        raise RuntimeError(
            f"Could not download '{spec}'. Tried:\n" + '\n'.join(errors)
            + f'\nDownload it manually and pass the path, e.g.:\n'
              f'  wget -P {dest.parent} <url>\n'
              f'  --finetune {dest}')
    finally:
        _barrier()


def resolve_encoder_weights(spec: str, dest_dir: Optional[str] = None,
                            force: bool = False, quiet: bool = False,
                            verify: bool = False) -> str:
    """Map a checkpoint spec to a usable local path ('' when random init).

    This is the single entry point used by the runners: it accepts a variant
    spec (``base``, ``mae:large``, ``timm:vit_base_patch16_224.mae``), an
    existing path (returned unchanged, relative paths preserved) or '' / None
    (nothing to load).
    """
    kind, _source, name = _split_spec(spec)
    if kind == 'none':
        return ''
    if kind == 'path':
        return name
    return str(download_pretrained(spec, dest_dir=dest_dir, force=force,
                                   quiet=quiet, verify=verify))


# --------------------------------------------------------------------------- #
# CLI (also used by runners/run_download_weights.py)
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='download-weights',
        description='Download Stage-1 initial ViT encoder weights into '
                    f'{initial_dir()}.')
    parser.add_argument('specs', nargs='*',
                        help="variant specs, e.g. small base mae:large "
                             "timm:vit_large_patch16_224.mae")
    parser.add_argument('--all', action='store_true',
                        help='download the default checkpoint of every variant')
    parser.add_argument('--list', action='store_true',
                        help='show available variants/sources and exit')
    parser.add_argument('--weights_dir', default=None,
                        help='destination directory '
                             '(default: $INITIAL_MODELS_DIR or '
                             '<project_root>/models/initial)')
    parser.add_argument('--force', action='store_true',
                        help='re-download even if a cached file exists')
    parser.add_argument('--verify', action='store_true',
                        help='torch.load the file and check it is a ViT '
                             'state dict (slow: hundreds of MB)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    if args.list or not (args.specs or args.all):
        print(describe_sources())
        return 0

    specs = list(args.specs)
    if args.all:
        specs += [f'{DEFAULT_SOURCE[v]}:{v}' for v in DEFAULT_SOURCE]
    done: List[Tuple[str, str]] = []
    for spec in dict.fromkeys(specs):        # de-duplicate, keep order
        path = resolve_encoder_weights(spec, dest_dir=args.weights_dir,
                                       force=args.force, quiet=args.quiet,
                                       verify=args.verify)
        done.append((spec, path))

    if _is_main():
        print('\nReady:')
        for spec, path in done:
            print(f'  {spec:<24} {path}')
        if done:
            print(f'\nUse it with e.g.:  --finetune {done[0][1]}')
    return 0


if __name__ == '__main__':                                # pragma: no cover
    sys.exit(main())
