"""Prepare the bundled examples or full Fish-X LC-NE Test viewer manifest."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile
import zlib


DATASET_NAME = 'Fish-X LC-NE Test'
BUNDLE = Path(__file__).resolve().parents[1]/'examples/fish_x_lc_ne'
VOLUMES = ('raw', 'lc', 'mito', 'syn', 'ves', 'lcpre', 'lcsyn')
RESOLUTIONS = [[8, 8, 30], [16, 16, 30], [32, 32, 30],
               [64, 64, 60], [128, 128, 120], [256, 256, 240]]


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _matches(path, member):
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum = zlib.crc32(block, checksum)
    return checksum == member.CRC


def _extract(archive_path, destination, record):
    """Reuse matching cached members and atomically repair interrupted extraction."""
    if sha256(archive_path) != record['sha256']:
        raise ValueError(f'Bundled archive checksum mismatch: {archive_path}')
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) != record['files'] or len({m.filename for m in members}) != len(members):
            raise ValueError('Bundled archive membership mismatch')
        targets = []
        for member in members:
            relative = PurePosixPath(member.filename)
            target = destination.joinpath(*relative.parts)
            if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                    or '\\' in member.filename or ':' in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16) or member.is_dir()
                    or target.is_symlink() or not target.resolve().is_relative_to(destination)):
                raise ValueError('Invalid bundled archive member')
            targets.append(target)
        for member, target in zip(members, targets):
            if _matches(target, member):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.extract-', delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    with archive.open(member) as source:
                        shutil.copyfileobj(source, stream)
                except BaseException:
                    stream.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def _full_overview(data_root):
    if data_root is None:
        raise ValueError('--dataset full requires --data-root pointing to the extracted '
                         'Fish-X LC-NE Test volumes. See docs/LC_NE_TEST.md.')
    root = Path(data_root).expanduser().resolve()
    for name in VOLUMES:
        source = root/f'zfishlc_{name}'
        if not (source/'info').is_file():
            raise ValueError(f'Missing extracted volume: {source}. Download the full dataset '
                             'from Hugging Face and restore its archives first; see docs/LC_NE_TEST.md.')
        info = json.loads((source/'info').read_text(encoding='utf-8'))
        scales = info.get('scales', [])
        expected_type = 'image' if name in ('raw', 'mito', 'syn', 'ves') else 'segmentation'
        if (info.get('type') != expected_type or [s.get('resolution') for s in scales] != RESOLUTIONS
                or scales[0].get('voxel_offset') != [192, 288, 0]
                or scales[0].get('size') != [8640, 8160, 1583]):
            raise ValueError(f'{source} does not match the released Fish-X LC-NE Test geometry')
        for scale in scales:
            folder = source/scale['key']
            if not folder.is_dir() or not any(p.is_file() for p in folder.iterdir()):
                raise ValueError(f'No extracted chunks at {folder}. Complete archive restoration first.')
    return dict(kind='overview', name=f'{DATASET_NAME} overview',
                raw_path=str(root/'zfishlc_raw'), seg_path=str(root/'zfishlc_lc'),
                segments=list(range(30001, 30006)),
                image_layers={name: str(root/f'zfishlc_{name}') for name in ('mito', 'syn', 'ves')},
                label_layers={name: str(root/f'zfishlc_{name}') for name in ('lcpre', 'lcsyn')},
                visible_layers=[])


def prepare_dataset(mode, cache, data_root=None, *, bundle=BUNDLE):
    """Return a local manifest; full-volume downloading is a separate setup step."""
    if mode not in ('examples', 'full', 'nuclei'):
        raise ValueError('Dataset option must be examples, full, or nuclei')
    if mode == 'examples' and data_root is not None:
        raise ValueError('--data-root is only used with --dataset full or nuclei')
    overview = _full_overview(data_root) if mode == 'full' else None
    if mode == 'nuclei':
        if data_root is None:
            raise ValueError('--dataset nuclei requires --data-root containing zfishlc_raw at 32 x 32 x 30 nm')
        raw = Path(data_root).expanduser().resolve()/'zfishlc_raw'
        if not (raw/'info').is_file():
            raise ValueError('Missing extracted zfishlc_raw/info; download and restore the raw volume first')
        info = json.loads((raw/'info').read_text())
        scales = [s for s in info.get('scales', []) if s['resolution'] == [32, 32, 30]]
        if (info.get('type') != 'image' or len(scales) != 1
                or scales[0]['voxel_offset'] != [48, 72, 0] or scales[0]['size'] != [2160, 2040, 1583]):
            raise ValueError('Raw does not match the released 32 x 32 x 30 nm Fish-X LC-NE Test geometry')
        folder = raw/scales[0]['key']
        if not folder.is_dir() or not any(p.is_file() for p in folder.iterdir()):
            raise ValueError('No extracted 32 x 32 x 30 nm raw chunks')
        overview = dict(kind='overview', name=f'{DATASET_NAME} nuclei (32 x 32 x 30 nm)',
                        raw_path=str(raw), review_mode='nuclei', raw_resolution=[32, 32, 30],
                        segments=list(range(30001, 30006)), image_layers={}, label_layers={}, visible_layers=[])
    bundle, cache = Path(bundle).resolve(), Path(cache).expanduser().resolve()
    release = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    if sha256(bundle/'windows.json') != release['windows_sha256']:
        raise ValueError('Bundled window manifest checksum mismatch')
    manifest = json.loads((bundle/'windows.json').read_text(encoding='utf-8'))
    annotation_path = bundle/'test_windows.json'
    if sha256(annotation_path) != release['test_windows_sha256']:
        raise ValueError('Bundled test_windows annotation checksum mismatch')
    cache.mkdir(parents=True, exist_ok=True)
    for record in release['archives']:
        if ('examples' if mode == 'nuclei' else mode) not in record.get('modes', ['examples', 'full']):
            continue
        filename = record['file']
        if Path(filename).name != filename:
            raise ValueError('Archive names must be immediate files')
        _extract(bundle/filename, cache, record)
    if overview is None:
        overview = dict(kind='overview', name=f'{DATASET_NAME} overview',
                        seg_path=str(cache/'zfishlc_lc'), segments=list(range(30001, 30006)),
                        image_layers={}, label_layers={}, visible_layers=[])
    if mode == 'nuclei':
        overview['seg_path'] = str(cache/'zfishlc_lc')
    else:
        overview['test_windows_path'] = str(annotation_path)
    manifest['windows'].insert(0, overview)
    manifest.update(dataset=DATASET_NAME, dataset_option=mode, initial_index=0)
    # Each mode has a separate manifest but shares the same five cached boxes.
    path = cache/f'windows.{mode}.json'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=cache, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def review_folder(output, item, index, *, supplied_dataset):
    """Keep each neuron's saved objects stable when the overview is added."""
    if supplied_dataset:
        if item.get('review_mode') == 'nuclei':
            return Path(output)/'nuclei'
        return Path(output)/('overview' if item['kind'] == 'overview' else f'neuron_{item["seed"]}')
    return Path(output)/f'window_{index:03d}'
