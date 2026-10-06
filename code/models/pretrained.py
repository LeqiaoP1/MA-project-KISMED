"""Downloadable "initial" (Stage-1) encoder weights for the project ViTs.

The entrypoints registered in :mod:`core.model` (e.g.
``project_vit_base_patch16_224``) build a **randomly initialised** network --
see :func:`models.build.create_model`. The *spatial priors* of plan Stage 1 come
from an external checkpoint that has to be fetched from the public internet.
This module knows which checkpoint belongs to which **VideoMAE pre-training
corpus** and caches the download under::

    <project_root>/models/initial/
    <project_root>/models/initial/videomae_base_patch16_224.pth       # K400
    <project_root>/models/initial/videomae_base_ssv2_patch16_224.pth  # SSV2

Stage 1 is **ViT-Base only** and is **always** initialised from a VideoMAE
checkpoint: the project compares VideoMAE pre-trained on Kinetics-400 (K400)
against VideoMAE pre-trained on Something-Something-v2 (SSV2) as the initial
weights. Both are ``MCG-NJU`` Hub checkpoints with the same ViT-B geometry
(768/12/12) and the same ``Conv3d(3, D, (2,16,16))`` tubelet patch embed this
repo's adapters use, so the tokenizer transfers verbatim. There is therefore
**no blank / random Stage-1 init** and **no ImageNet-MAE / timm / ViT-Large**
source in the built-in table.

(``<project_root>`` is the repository root, i.e. the parent of ``code/``. Use
the ``INITIAL_MODELS_DIR`` environment variable or ``--weights_dir`` to place
them elsewhere, e.g. a scratch volume on the HPC.)

Spec grammar
------------
Stage-1 init (``--pretrained_encoder`` on ``run_pretrain.py``) and the AU probe
(``--finetune base``) accept a *corpus spec*::

    videomae:k400   VideoMAE ViT-B SSL on Kinetics-400    (Stage-1 default)
    videomae:ssv2   VideoMAE ViT-B SSL on Something-Something-v2
    videomae        source only -> the corpus named by ``--videomae_dataset``
    base            alias of ``videomae:k400``
    path/to.pth     a path that exists is returned untouched (never downloaded)

A blank / ``none`` / ``random`` / ``scratch`` / ``c0`` spec is still recognised
by :func:`_split_spec` (it resolves to "nothing to load") because the Stage-3
runners treat an empty ``--finetune`` as "no checkpoint". Stage-1 callers pass
``require=True`` so that a blank spec is a **hard error** instead.

Examples (all from ``code/``)::

    python runners/run_download_weights.py --list
    python runners/run_download_weights.py videomae:k400    # Kinetics-400
    python runners/run_download_weights.py videomae:ssv2    # SSV2
    python runners/run_download_weights.py --all            # both corpora
    python runners/run_pretrain.py -c <cfg> --videomae_dataset ssv2

Sources
-------
``videomae``
          MCG-NJU **VideoMAE** (HuggingFace Hub ``MCG-NJU/videomae-*``, mirrored
          via ``HF_ENDPOINT``). Two ViT-Base checkpoints are served, one per
          pre-training corpus: ``MCG-NJU/videomae-base`` (Kinetics-400) and
          ``MCG-NJU/videomae-base-ssv2`` (Something-Something-v2). In both,
          ``patch_embed.proj`` is a ``Conv3d(3, D, (2,16,16))`` tubelet filter --
          the exact shape this repo's adapters use -- so the tokenizer
          transfers verbatim, and the objective (tube-masked video MAE) is the
          same family as Stage 2. Hosted in the HF `transformers` layout, which
          the loader maps back to MAE keys. Licence: CC-BY-NC 4.0 (fine for
          academic work; state it if you redistribute).

Download mechanics
------------------
Streamed with stdlib ``urllib`` (no extra dependency), written to a ``.part``
file with HTTP-Range **resume**, atomically renamed, and cached -- a second run
is a no-op. Under DDP only rank 0 downloads (the others wait on a barrier).
``--verify`` additionally ``torch.load``s the file and checks that it looks like
a ViT state dict; it is opt-in because the checkpoint is ~377 MB.

.. note::
   The loader functions (``core.multimae.load_pretrained_encoder``,
   ``core.au_probe.load_au_probe_weights``) map the MAE key layout
   (``blocks.*`` -> ``enc_blocks.*``, ``patch_embed.proj.*`` -> the RGB tubelet
   adapter) and the HF VideoMAE layout
   (``core.multimae.canonicalise_vit_state_dict``). Only a 3-D ``Conv3d`` patch
   embed is accepted (copied verbatim); the legacy 2-D boxcar inflation was
   removed with the non-VideoMAE sources. They raise on a geometry mismatch and
   on a checkpoint that contributes **zero** encoder tensors (e.g. a Stage-2
   MultiModalMAE checkpoint), so pick the corpus that matches your
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
    'VIT_VARIANTS', 'PRETRAINED_SOURCES', 'DEFAULT_DATASET', 'PROJECT_ROOT',
    'initial_dir', 'available_specs', 'describe_sources', 'plan_download',
    'download_pretrained', 'resolve_encoder_weights', 'main',
]

_CHUNK = 1 << 20          # 1 MiB read chunks
_MIN_BYTES = 1 << 20      # no real ViT checkpoint is smaller than 1 MiB

# ``code/models/pretrained.py`` -> [0]=code/models, [1]=code, [2]=repo root
_CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _CODE_DIR.parent

#: ViT geometry per variant. Stage 1 is **ViT-Base only** (the project does not
#: use any other backbone geometry), so ``base`` is the sole entry. The
#: non-Base ``project_multimae_*`` / ``project_vit_*`` entrypoints still exist
#: for the model registry, but no Stage-1 checkpoint source is served for them.
VIT_VARIANTS: Dict[str, Dict[str, int]] = {
    'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
}

#: ``source -> corpus -> recipe``. ``candidates`` is a list of ``(url, kind)``
#: pairs tried in order (``kind`` is ``'torch'`` or ``'safetensors'``; the
#: latter is converted to a ``.pth`` after download). ``file`` is the final,
#: cache-checked file name inside ``models/initial/``; ``variant`` ties the
#: corpus back to a :data:`VIT_VARIANTS` geometry. A candidate URL may contain
#: ``{hf}``, which :func:`_lookup` substitutes with the CURRENT
#: ``$HF_ENDPOINT`` (mirrors for restricted networks).
#:
#: Only VideoMAE ViT-Base is served, because Stage 1 compares the two VideoMAE
#: pre-training corpora (Kinetics-400 vs Something-Something-v2) as the initial
#: weights. The patch embed IS a tubelet ``Conv3d(3, D, (2,16,16))`` -- the
#: exact shape this repo's adapters use -- so the tokenizer transfers verbatim
#: and the model is not motion-blind at init. Weights: CC-BY-NC 4.0.
PRETRAINED_SOURCES: Dict[str, Dict[str, dict]] = {
    'videomae': {
        'k400': {
            'file': 'videomae_base_patch16_224.pth',
            'variant': 'base',
            'candidates': [
                ('{hf}/MCG-NJU/videomae-base/resolve/main/pytorch_model.bin',
                 'torch'),
                ('{hf}/MCG-NJU/videomae-base/resolve/main/model.safetensors',
                 'safetensors'),
            ],
            'note': 'VideoMAE ViT-B (768/12/12), tube-masked video MAE on '
                    'Kinetics-400; 3-D tubelet patch embed 2x16x16',
        },
        'ssv2': {
            'file': 'videomae_base_ssv2_patch16_224.pth',
            'variant': 'base',
            'candidates': [
                ('{hf}/MCG-NJU/videomae-base-ssv2/resolve/main/'
                 'pytorch_model.bin', 'torch'),
                ('{hf}/MCG-NJU/videomae-base-ssv2/resolve/main/'
                 'model.safetensors', 'safetensors'),
            ],
            'note': 'VideoMAE ViT-B (768/12/12), tube-masked video MAE on '
                    'Something-Something-v2; 3-D tubelet patch embed 2x16x16',
        },
    },
}

#: Corpus used when a spec names the ``videomae`` source without a dataset
#: (``videomae``) or uses the bare ``base`` alias. Also the default of the
#: ``--videomae_dataset`` CLI knob on ``run_pretrain.py``.
DEFAULT_DATASET = 'k400'

#: Shorthand alias -> corpus, so ``base`` / ``videomae:base`` keep working.
_DATASET_ALIASES = {'base': 'k400'}

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
    ``{'none', 'path', 'remote'}``.

    ``name`` is a *corpus* (``k400`` / ``ssv2``) for the ``videomae`` source; it
    may be the empty string when the spec names the source only (``videomae``),
    in which case :func:`_lookup` fills in the default corpus.
    """
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
            f'Use a corpus spec ({"/".join(available_specs())}) to '
            f'auto-download an initial encoder into {initial_dir()}.')
    if ':' in spec:
        source, name = spec.split(':', 1)
        return 'remote', source.strip().lower(), name.strip()
    bare = spec.lower()
    if bare in _DATASET_ALIASES:
        return 'remote', 'videomae', _DATASET_ALIASES[bare]
    if bare == 'videomae':
        return 'remote', 'videomae', ''
    if bare in PRETRAINED_SOURCES['videomae']:
        return 'remote', 'videomae', bare
    raise KeyError(
        f'Unknown Stage-1 weights spec {spec!r}. Stage 1 only accepts a '
        f'VideoMAE ViT-Base corpus: {", ".join(available_specs())} '
        f"(or 'videomae' + --videomae_dataset, or a local .pth path). "
        f'ImageNet-MAE, timm and ViT-Large sources were removed.')


def _lookup(source: str, name: str) -> Tuple[str, str, dict]:
    """Return ``(corpus, variant, recipe)`` for a ``(source, name)`` pair."""
    table = PRETRAINED_SOURCES.get(source)
    if table is None:
        raise KeyError(
            f'Unknown weight source {source!r}. The only Stage-1 source is '
            f"'videomae' (corpora: "
            f"{', '.join(sorted(PRETRAINED_SOURCES['videomae']))}).")
    if not name:
        name = DEFAULT_DATASET
    name = _DATASET_ALIASES.get(name, name)
    if name not in table:
        raise KeyError(
            f'Source {source!r} has no VideoMAE corpus {name!r}. Choose one of '
            f"{', '.join(sorted(table))} (e.g. 'videomae:{DEFAULT_DATASET}').")
    recipe = table[name]
    if any('{hf}' in url for url, _ in recipe['candidates']):
        # resolve $HF_ENDPOINT at CALL time (mirrors must work without reimport)
        recipe = dict(recipe, candidates=[(url.format(hf=_hf_endpoint()), kind)
                                          for url, kind in recipe['candidates']])
    return name, recipe['variant'], recipe


def plan_download(spec: str, dest_dir: Optional[str] = None,
                  dataset: str = DEFAULT_DATASET) -> dict:
    """Resolve ``spec`` to a download plan **without** touching the network.

    ``dataset`` supplies the corpus when ``spec`` names the ``videomae`` source
    without one (bare ``videomae``).

    :return: dict with ``kind``, ``source``, ``dataset``, ``variant``,
        ``dest`` (Path), ``urls`` and ``note``. Raises for un-downloadable specs.
    """
    kind, source, name = _split_spec(spec)
    if kind == 'none':
        raise ValueError('empty/random spec has nothing to download')
    if kind == 'path':
        return {'kind': 'path', 'source': '', 'dataset': '', 'variant': '',
                'dest': Path(name), 'urls': [], 'note': 'local file'}
    if not name:
        name = _DATASET_ALIASES.get(dataset, dataset)
    corpus, variant, recipe = _lookup(source, name)
    return {
        'kind': 'remote',
        'source': source,
        'dataset': corpus,
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
    """Human-readable table of the Stage-1 corpora (for ``--list``)."""
    lines = [
        f'Stage-1 initial encoder weights (VideoMAE ViT-Base) -> '
        f'{initial_dir()}',
        '',
        f"{'spec':<22} {'geometry (dim/depth/heads)':<27} file",
    ]
    geo = VIT_VARIANTS['base']
    geom = f"{geo['embed_dim']}/{geo['depth']}/{geo['num_heads']}"
    table = PRETRAINED_SOURCES['videomae']
    first = True
    for corpus in ('k400', 'ssv2'):
        spec = f'videomae:{corpus}'
        mark = ' *' if corpus == DEFAULT_DATASET else ''
        lines.append(f'{spec + mark:<22} {geom if first else "":<27} '
                     f'{table[corpus]["file"]}')
        first = False
    lines += [
        '',
        ' * = default corpus (Kinetics-400)',
        "Use 'videomae' (source only) with --videomae_dataset k400|ssv2,",
        "or the bare alias 'base' for videomae:k400.",
        'A local .pth path is also accepted anywhere a spec is.',
        'Stage 1 is VideoMAE-only: ImageNet-MAE, timm and ViT-Large were '
        'removed.',
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
                        verify: bool = False,
                        dataset: str = DEFAULT_DATASET) -> Path:
    """Download (once) the checkpoint for ``spec``; return its local path.

    Cached files are reused unless ``force``. Under DDP only rank 0 downloads;
    the other ranks wait on a barrier and then use the shared file.
    """
    plan = plan_download(spec, dest_dir, dataset=dataset)
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
                            verify: bool = False,
                            dataset: str = DEFAULT_DATASET,
                            require: bool = False) -> str:
    """Map a checkpoint spec to a usable local path.

    This is the single entry point used by the runners: it accepts a corpus
    spec (``videomae:k400`` / ``videomae:ssv2`` / ``videomae`` / ``base``), an
    existing path (returned unchanged, relative paths preserved) or '' / None.

    :param dataset: corpus used when ``spec`` names the ``videomae`` source
        without one (``videomae``); see ``--videomae_dataset``.
    :param require: when True (Stage-1 init), a blank / ``random`` / ``scratch``
        spec is a HARD ERROR instead of returning ``''``. This is what removes
        the blank Stage-1 backbone path; the Stage-3 runners keep
        ``require=False`` so an empty ``--finetune`` still means "no checkpoint".
    """
    kind, _source, name = _split_spec(spec)
    if kind == 'none':
        if require:
            raise ValueError(
                f'A VideoMAE initial encoder is required, but the spec is '
                f'{spec!r} (blank/random). Choose one of '
                f'{", ".join(available_specs())} (or a local .pth path); '
                f'random-init backbones were removed.')
        return ''
    if kind == 'path':
        return name
    return str(download_pretrained(spec, dest_dir=dest_dir, force=force,
                                   quiet=quiet, verify=verify,
                                   dataset=dataset))


# --------------------------------------------------------------------------- #
# CLI (also used by runners/run_download_weights.py)
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='download-weights',
        description='Download Stage-1 initial ViT encoder weights into '
                    f'{initial_dir()}.')
    parser.add_argument('specs', nargs='*',
                        help="corpus specs, e.g. videomae:k400 videomae:ssv2")
    parser.add_argument('--all', action='store_true',
                        help='download every VideoMAE ViT-Base corpus '
                             '(Kinetics-400 + SSV2)')
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        choices=['k400', 'ssv2'],
                        help="corpus for a bare 'videomae' spec "
                             f'(default: {DEFAULT_DATASET})')
    parser.add_argument('--list', action='store_true',
                        help='show the available corpora and exit')
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
        specs += [f'videomae:{c}' for c in ('k400', 'ssv2')]
    done: List[Tuple[str, str]] = []
    for spec in dict.fromkeys(specs):        # de-duplicate, keep order
        path = resolve_encoder_weights(spec, dest_dir=args.weights_dir,
                                       force=args.force, quiet=args.quiet,
                                       verify=args.verify,
                                       dataset=args.dataset)
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
