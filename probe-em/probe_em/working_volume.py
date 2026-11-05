"""Bounded, lossless cropping of local unsharded precomputed datasets."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import time
import zipfile

import numpy as np
from scipy import ndimage


def largest_rectangle(valid):
    """Largest all-true XY rectangle, with exclusive upper bounds; O(XY)."""
    valid = np.asarray(valid, bool)
    if valid.ndim != 2 or not valid.any():
        raise ValueError('Coverage must be a nonempty 2D mask')
    heights = np.zeros(valid.shape[0], int)
    best_area, best = 0, None
    for y, row in enumerate(valid.T):
        heights = (heights + 1) * row
        stack = []
        for x, height in enumerate(np.r_[heights, 0]):
            start = x
            while stack and stack[-1][1] > height:
                left, previous = stack.pop()
                area = (x - left) * previous
                if area > best_area:
                    best_area, best = area, ((left, y - previous + 1), (x, y + 1))
                start = left
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return np.asarray(best, dtype=int)


def padding_mask(section, threshold=5):
    """Select bottom-right padding; retain unrelated dark tissue components."""
    dark = np.asarray(section) <= threshold
    if dark.ndim != 2 or not dark[-1, -1]:
        raise ValueError('Expected a dark bottom-right padding seed in each section')
    seed = np.zeros_like(dark)
    seed[-1, -1] = True
    return ndimage.binary_propagation(seed, mask=dark)


def scan_coverage(volume, threshold=5, guard=1):
    """Inspect every source Z plane at the caller's chosen XY resolution."""
    if guard < 0:
        raise ValueError('Coverage guard must be nonnegative')
    offset, size = np.asarray(volume.voxel_offset), np.asarray(volume.bounds.size())
    common = np.ones(tuple(size[:2]), bool)
    counts = []
    depth = int(volume.chunk_size[2])
    for z in range(int(offset[2]), int(offset[2] + size[2]), depth):
        stop = min(z + depth, int(offset[2] + size[2]))
        block = np.asarray(volume[:, :, z:stop])[..., 0]
        for index in range(block.shape[2]):
            missing = padding_mask(block[:, :, index], threshold)
            common &= ~missing
            counts.append(int(missing.sum()))
    if guard:
        common = ~ndimage.binary_dilation(~common, structure=np.ones((3, 3), bool),
                                          iterations=guard, border_value=0)
    bounds = largest_rectangle(common) + offset[:2]
    return common, dict(bounds_xy_voxels=bounds.tolist(),
                        bounds_xy_nm=(bounds * volume.resolution[:2]).tolist(),
                        resolution_xyz_nm=list(map(int, volume.resolution)),
                        sections=int(size[2]), threshold=threshold, guard_xy_voxels=guard,
                        padding_pixels_per_section=counts,
                        method='Largest common rectangle excluding bottom-right padding in every section')


def chunk_name(low, high):
    return '_'.join(f'{int(a)}-{int(b)}' for a, b in zip(low, high))


def chunk_bounds(scale):
    offset = np.asarray(scale.get('voxel_offset', [0, 0, 0]), int)
    size, chunk = np.asarray(scale['size'], int), np.asarray(scale['chunk_sizes'][0], int)
    # Z/Y/X order reuses neighboring input chunks in the bounded read cache.
    for z, y, x in itertools.product(*(range(0, int(size[d]), int(chunk[d])) for d in (2, 1, 0))):
        low = offset + [x, y, z]
        yield low, np.minimum(low + chunk, offset + size)


def crop_info(info, bounds_nm, jpeg_crop=None):
    bounds = np.asarray(bounds_nm, float)
    native = info['scales'][0]
    resolution = np.asarray(native['resolution'], float)
    offset = np.asarray(native.get('voxel_offset', [0, 0, 0]))
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0])
            or np.any(bounds[0] < offset * resolution)
            or np.any(bounds[1] > (offset + native['size']) * resolution)
            or not np.allclose(bounds / resolution, np.rint(bounds / resolution))):
        raise ValueError('Crop must be an integer native-grid box inside the source')
    if info.get('num_channels') != 1:
        raise ValueError('Working-volume crops require one channel')
    result = deepcopy(info)
    # These spatial resources need their own crop; never point at uncropped data.
    for key in ('mesh', 'skeletons', 'segment_properties'):
        result.pop(key, None)
    for source, target in zip(info['scales'], result['scales']):
        if 'sharding' in source or len(source['chunk_sizes']) != 1:
            raise ValueError('Cropping supports unsharded scales with one chunk size')
        scale_resolution = np.asarray(source['resolution'], float)
        low = np.floor(bounds[0] / scale_resolution).astype(int)
        high = np.ceil(bounds[1] / scale_resolution).astype(int)
        source_low = np.asarray(source.get('voxel_offset', [0, 0, 0]), int)
        source_high = source_low + source['size']
        if np.any(low < source_low) or np.any(high > source_high):
            raise ValueError('Every source scale must cover the crop')
        target['voxel_offset'], target['size'] = low.tolist(), (high - low).tolist()
        if source['encoding'] == 'jpeg':
            chunk = np.asarray(source['chunk_sizes'][0])
            whole_chunks = (np.all((low - source_low) % chunk == 0)
                            and np.all(((high - source_low) % chunk == 0) | (high == source_high)))
            aligned = (np.all((low[:2] - source_low[:2]) % 8 == 0)
                       and np.all((high[:2] - source_low[:2]) % 8 == 0)
                       and np.all(chunk[:2] % 8 == 0)
                       and low[2] == source_low[2] and high[2] == source_high[2])
            # Gzip/raw is lossless relative to decoded source pixels. Never silently
            # requantize JPEG when the crop cuts through an 8x8 DCT block.
            target['encoding'] = 'jpeg' if whole_chunks or (jpeg_crop and aligned) else 'raw'
        elif source['encoding'] not in ('raw', 'png', 'compressed_segmentation'):
            raise ValueError(f'Unsupported source encoding: {source["encoding"]}')
    return result


class ChunkReader:
    def __init__(self, source, info, scale, archive=None):
        self.source, self.info, self.scale = Path(source), info, scale
        self.archive = archive
        self.archive_lock = threading.Lock()
        self.offset = np.asarray(scale.get('voxel_offset', [0, 0, 0]), int)
        self.end = self.offset + scale['size']
        self.chunk = np.asarray(scale['chunk_sizes'][0], int)
        self.encoded = lru_cache(maxsize=96)(self._encoded)
        self.decoded = lru_cache(maxsize=48)(self._decoded)
        self.locks = [threading.Lock() for _ in range(64)]

    def stored(self, name):
        """Read an original chunk without decoding or changing outer compression."""
        if self.archive is not None:
            key = f'{self.source.name}/{self.scale["key"]}/{name}'
            compressed = key+'.gz' in self.archive.NameToInfo
            with self.archive_lock:
                try:
                    payload = self.archive.read(key+'.gz' if compressed else key)
                except KeyError as error:
                    raise FileNotFoundError(f'Missing archive chunk: {key}') from error
            if not payload:
                raise ValueError(f'Empty archive chunk: {key}')
            return payload, compressed
        path = self.source / self.scale['key'] / name
        compressed = path.with_name(path.name + '.gz')
        gzipped = compressed.is_file()
        payload = compressed.read_bytes() if gzipped else path.read_bytes()
        if not payload:
            raise ValueError(f'Empty source chunk: {path}')
        return payload, gzipped

    def _encoded(self, name):
        payload, compressed = self.stored(name)
        return gzip.decompress(payload) if compressed else payload

    def pieces(self, low, high):
        first = (low - self.offset) // self.chunk
        last = (high - 1 - self.offset) // self.chunk
        for index in itertools.product(*(range(int(a), int(b) + 1) for a, b in zip(first, last))):
            start = self.offset + np.asarray(index) * self.chunk
            end = np.minimum(start + self.chunk, self.end)
            yield start, end, self.encoded(chunk_name(start, end))

    def _decoded(self, start, end):
        from cloudvolume import chunks
        return chunks.decode(self.encoded(chunk_name(start, end)), self.scale['encoding'],
                             shape=tuple(b-a for a,b in zip(start,end))+(1,),
                             dtype=np.dtype(self.info['data_type']),
                             block_size=self.scale.get('compressed_segmentation_block_size'))

    def assemble(self, low, high, pieces):
        from cloudvolume import chunks
        output = np.empty(tuple(high - low) + (1,), dtype=self.info['data_type'], order='F')
        for start, end, payload in pieces:
            key = tuple(map(int,start)), tuple(map(int,end))
            with self.locks[hash(key) % len(self.locks)]:
                values = self.decoded(*key)
            a, b = np.maximum(start, low), np.minimum(end, high)
            output[tuple(slice(int(x), int(y)) for x, y in zip(a - low, b - low))] = values[
                tuple(slice(int(x), int(y)) for x, y in zip(a - start, b - start))]
        return output


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def run_jpeg_crop(command, data, environment):
    """Stream a bounded JPEG request without disk files or tiny pipe writes.

    Popen.communicate(input=...) limits writes to PIPE_BUF (512 bytes on macOS).
    A blocking writer keeps large transfers outside the Python thread lock while
    subprocess.run concurrently drains stdout and stderr.
    """
    read_fd, write_fd = os.pipe()
    errors = []

    def write():
        try:
            with os.fdopen(write_fd,'wb',buffering=0) as stream:
                remaining = memoryview(data)
                while remaining:
                    remaining = remaining[stream.write(remaining):]
        except BrokenPipeError:
            pass  # The helper's exit status and stderr describe rejected input.
        except BaseException as error:
            errors.append(error)

    writer = threading.Thread(target=write,name='jpeg-input')
    writer.start()
    try:
        completed = subprocess.run(command,stdin=read_fd,capture_output=True,env=environment)
    finally:
        os.close(read_fd)
        writer.join()
    if errors:
        raise errors[0]
    return completed


def certified_zero_segmentation(payload, shape, block_size, dtype):
    """Certify zero-valued constant blocks from the published cseg headers.

    False means the normal codec/verification path must be used. No inference is
    made from compressed byte length, a sample, or a label-count heuristic.
    """
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype('uint32'),np.dtype('uint64')) or len(payload)%4:
        return False
    words=np.frombuffer(payload,dtype='<u4')
    count=int(np.prod((np.asarray(shape[:3])+np.asarray(block_size)-1)//block_size))
    if not len(words) or words[0] != 1 or count < 1 or 1+2*count > len(words):
        return False
    headers=words[1:1+2*count:2]
    if np.any(headers >> 24):
        return False
    tables=(headers & 0xffffff).astype(np.int64)+1
    width=dtype.itemsize//4
    if np.any(tables < 1+2*count) or np.any(tables+width > len(words)):
        return False
    return all(not np.any(words[tables+index]) for index in range(width))


def crop_precomputed(source, destination, bounds_nm, *, info=None, jpeg_crop=None,
                     workers=8, resume=False, verify_resume=False, scratch_dir=None,
                     source_archive=None, archive_identity='sha256', progress=None):
    """Write all voxel chunks, then publish info; missing inputs fail explicitly.

    Matching whole chunks are copied byte for byte without codec work. Retiled
    nonuniform chunks are compared pixel by pixel. Empty segmentation chunks are
    certified from their compressed label palettes and a verified zero template.
    Resume requires the identical persisted plan; verify_resume also rereads pixels.
    """
    from cloudvolume import chunks
    from PIL import Image
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and working dataset must be separate directories')
    if workers < 1:
        raise ValueError('workers must be positive')
    scratch = Path(scratch_dir or tempfile.gettempdir()).resolve()
    if scratch == destination or destination in scratch.parents:
        raise ValueError('Temporary JPEG files must be outside the working dataset')
    scratch.mkdir(parents=True, exist_ok=True)
    archive_path = Path(source_archive).resolve() if source_archive else None
    if info is None and archive_path:
        with zipfile.ZipFile(archive_path) as archive:
            info = json.loads(archive.read(f'{source.name}/info'))
    info = info or json.loads((source / 'info').read_text())
    target = crop_info(info, bounds_nm, jpeg_crop)
    plan = dict(source=str(source), bounds_nm=np.asarray(bounds_nm).tolist(), source_info=info, info=target)
    if archive_path:
        if archive_identity not in ('sha256','central-directory'):
            raise ValueError('Archive identity must be sha256 or central-directory')
        checksum=hashlib.sha256()
        start=0
        if archive_identity == 'central-directory':
            with zipfile.ZipFile(archive_path) as index:
                start=index.start_dir
        with archive_path.open('rb') as stream:
            stream.seek(start)
            while block := stream.read(4*1024**2):
                checksum.update(block)
        identity=dict(path=str(archive_path),prefix=source.name)
        if archive_identity == 'sha256':
            identity['sha256']=checksum.hexdigest()
        else:
            metadata=archive_path.stat()
            identity.update(central_directory_sha256=checksum.hexdigest(),bytes=metadata.st_size,
                            mtime_ns=metadata.st_mtime_ns)
        plan['source_archive']=identity
    plan_path = destination / 'crop_plan.json'
    if destination.exists():
        previous = json.loads(plan_path.read_text()) if resume and plan_path.is_file() else None
        if previous is not None:
            # Older runs used raw/gzip for whole JPEG chunks without a helper.
            # Preserve that persisted codec instead of invalidating good output
            # merely because new runs can now copy the original JPEG bytes.
            legacy_target = deepcopy(target)
            for scale, saved in zip(legacy_target['scales'], previous.get('info', {}).get('scales', [])):
                if scale['encoding'] == 'jpeg' and saved.get('encoding') == 'raw':
                    scale['encoding'] = 'raw'
            legacy_plan = dict(plan, info=legacy_target)
            if previous == legacy_plan:
                target, plan = legacy_target, legacy_plan
        if previous != plan:
            raise FileExistsError('Destination exists; resume requires an identical crop plan')
    else:
        destination.mkdir(parents=True)
        write_json(plan_path, plan)
    started = time.monotonic()
    report = dict(status='running', source=str(source), bounds_nm=plan['bounds_nm'], scales=[],
                  pixel_policy='Exact decoded source pixels; exact integer labels; no requantization',
                  mesh_policy='Spatial resources must be cropped separately')
    write_json(destination / 'crop_status.json', report)
    archive = zipfile.ZipFile(archive_path) if archive_path else None
    try:
        for original, scale in zip(info['scales'], target['scales']):
            folder = destination / scale['key']
            folder.mkdir(parents=True, exist_ok=True)
            reader = ChunkReader(source, info, original, archive)
            encoding = scale['encoding']

            @lru_cache(maxsize=16)
            def zero_chunk(shape):
                zeros = np.zeros(shape+(1,),dtype=info['data_type'],order='F')
                payload = chunks.encode(zeros,encoding,block_size=scale['compressed_segmentation_block_size'])
                decoded = chunks.decode(payload,encoding,shape=zeros.shape,dtype=zeros.dtype,
                                        block_size=scale['compressed_segmentation_block_size'])
                if not np.array_equal(decoded,zeros):
                    raise ValueError('Zero segmentation template failed verification')
                return gzip.compress(payload,compresslevel=1,mtime=0)

            @lru_cache(maxsize=128)
            def background_palette(payload, shape):
                # Empty chunks commonly share identical compressed bytes. Cache
                # by content and shape, never just by filename or label count.
                return certified_zero_segmentation(payload,shape,
                    original['compressed_segmentation_block_size'],info['data_type'])

            def all_background(pieces):
                return all(background_palette(payload,tuple(end-start)+(1,))
                           for start,end,payload in pieces)

            def process(box):
                low, high = box
                base_path = folder / chunk_name(low, high)
                path = base_path
                if encoding in ('raw', 'compressed_segmentation'):
                    path = path.with_name(path.name + '.gz')
                # Copied inputs retain their original gzip wrapper, including
                # uncompressed raw/cseg or gzipped JPEG chunks. Earlier exports
                # may use the opposite wrapper under the same crop metadata.
                existing = False
                if resume:
                    for candidate in (path, base_path, base_path.with_name(base_path.name+'.gz')):
                        if candidate.is_file():
                            path, existing = candidate, True
                            break
                if existing and not verify_resume:
                    return path.stat().st_size, 'resumed'
                source_low = reader.offset + ((low-reader.offset)//reader.chunk)*reader.chunk
                source_high = np.minimum(source_low+reader.chunk, reader.end)
                if (not existing and encoding == original['encoding']
                        and np.array_equal(low, source_low) and np.array_equal(high, source_high)):
                    payload, compressed = reader.stored(chunk_name(low, high))
                    path = base_path.with_name(base_path.name+'.gz') if compressed else base_path
                    temporary = path.with_name(path.name+'.tmp')
                    temporary.write_bytes(payload)
                    if temporary.read_bytes() != payload:
                        raise ValueError(f'Copied chunk differs from source bytes: {path.name}')
                    temporary.replace(path)
                    return len(payload), 'copied'
                pieces = list(reader.pieces(low, high))
                if (encoding == original['encoding'] == 'compressed_segmentation'
                        and not existing and all_background(pieces)):
                    payload = zero_chunk(tuple(map(int,high-low)))
                    temporary = path.with_name(path.name+'.tmp')
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                    return len(payload),'uniform_verified'
                values = reader.assemble(low, high, pieces)
                if existing:
                    saved = path.read_bytes()
                    payload = gzip.decompress(saved) if path.suffix == '.gz' else saved
                    decoded = chunks.decode(payload, encoding, shape=values.shape, dtype=values.dtype,
                                            block_size=scale.get('compressed_segmentation_block_size'))
                    if not np.array_equal(decoded, values):
                        raise ValueError(f'Resumed chunk differs from source: {path.name}')
                    return len(saved), 'resumed_verified'
                if encoding == 'jpeg':
                    frames = [struct.pack('<I',len(pieces))]
                    for start,end,payload in pieces:
                        if start[2] != low[2] or end[2] != high[2]:
                            raise ValueError('JPEG coefficient assembly requires unchanged Z chunks')
                        frames.extend((struct.pack('<IiiI',int(end[1]-start[1]),int(start[0]-low[0]),
                                                   int(start[1]-low[1]),len(payload)),payload))
                    environment = dict(os.environ,TMPDIR=str(scratch),TMP=str(scratch),TEMP=str(scratch))
                    completed = run_jpeg_crop([str(jpeg_crop),'--stream',*map(str,high-low)],
                                              b''.join(frames),environment)
                    if completed.returncode:
                        raise ValueError(f'JPEG coefficient crop failed: {completed.stderr.decode(errors="replace").strip()}')
                    payload = completed.stdout
                elif encoding == 'png':
                    stream = io.BytesIO()
                    Image.fromarray(values.ravel(order='F').reshape(-1, int(high[0] - low[0]))).save(
                        stream, format='PNG', compress_level=3)
                    payload = stream.getvalue()
                else:
                    payload = chunks.encode(values, encoding, block_size=scale.get('compressed_segmentation_block_size'))
                decoded = chunks.decode(payload, encoding, shape=values.shape, dtype=values.dtype,
                                        block_size=scale.get('compressed_segmentation_block_size'))
                if not np.array_equal(decoded, values):
                    raise ValueError(f'Lossless verification failed: {path.name}')
                if path.suffix == '.gz':
                    payload = gzip.compress(payload, compresslevel=1, mtime=0)
                temporary = path.with_name(path.name + '.tmp')
                temporary.write_bytes(payload)
                temporary.replace(path)
                return len(payload), 'verified'

            stats = dict(key=scale['key'], encoding=encoding, chunks=0, bytes=0, verified=0, resumed=0,
                         resumed_verified=0, uniform_verified=0, copied=0)
            report['scales'].append(stats)
            iterator = iter(chunk_bounds(scale))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while batch := list(itertools.islice(iterator, 256)):
                    for byte_count, kind in pool.map(process, batch):
                        stats['chunks'] += 1
                        stats['bytes'] += byte_count
                        stats[kind] += 1
                    report['elapsed_seconds'] = round(time.monotonic() - started, 2)
                    write_json(destination / 'crop_status.json', report)
                    if progress:
                        progress(dict(dataset=destination.name, **stats, elapsed_seconds=report['elapsed_seconds']))
            reader.encoded.cache_clear()
            reader.decoded.cache_clear()
        write_json(destination / 'info', target)
        report['status'] = 'complete'
        write_json(destination / 'crop_status.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(destination / 'crop_status.json', report)
        raise
    finally:
        if archive is not None:
            archive.close()
