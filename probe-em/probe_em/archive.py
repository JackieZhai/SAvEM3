"""Portable ZIP64 parts for complete local precomputed working datasets.

Only the standard library is required. Each part is an ordinary independent ZIP,
with disjoint paths under the dataset name. JPEG and gzip payloads stay unchanged.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile


BLOCK = 4 * 1024**2


def digest(path):
    with Path(path).open('rb') as stream:
        return stream_digest(stream)


def stream_digest(stream):
    checksum = hashlib.sha256()
    while block := stream.read(BLOCK):
        checksum.update(block)
    return checksum.hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '\\' in name or ':' in name.split('/')[0]
            or '\x00' in name
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Unsafe archive path: {name!r}')
    return path


def inventory(source):
    """Include published voxel/mesh resources, excluding private processing paths."""
    source = Path(source)
    info = json.loads((source / 'info').read_text())
    status = json.loads((source / 'crop_status.json').read_text())
    if status.get('status') != 'complete':
        raise ValueError('Only a completed working crop can be archived')
    files = [source / 'info']
    resources = [scale['key'] for scale in info['scales']]
    resources += [info[key] for key in ('mesh', 'skeletons', 'segment_properties') if key in info]
    if len(resources) != len(set(resources)):
        raise ValueError('Duplicate resource directories in volume info')
    for resource in resources:
        safe_relative(resource)
        folder = source / resource
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f'Missing or linked volume resource: {resource}')
        for base, directories, names in os.walk(folder):
            if any((Path(base) / name).is_symlink() for name in directories):
                raise ValueError('Linked directories cannot be archived')
            for name in names:
                if name.startswith('.') or name.endswith(('.tmp', '.partial')) or name == 'crop_report.json':
                    continue
                files.append(Path(base) / name)
    # Portable provenance is deliberately separate from local crop plans/logs.
    if (source / 'provenance.json').is_file():
        files.append(source / 'provenance.json')
    result = []
    for path in sorted(files, key=lambda item: item.relative_to(source).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Only regular files can be archived: {path.name}')
        name = (PurePosixPath(source.name) / path.relative_to(source).as_posix()).as_posix()
        safe_relative(name)
        metadata = path.stat()
        result.append(dict(path=name, size=metadata.st_size, mtime_ns=metadata.st_mtime_ns))
    return info, result


def partition(entries, max_bytes):
    """Conservative stored-ZIP bound, including ZIP64 and data descriptors."""
    if max_bytes < 1024:
        raise ValueError('Archive limit must be at least 1024 bytes')
    parts, current, size = [], [], 128
    for entry in entries:
        cost = entry['size'] + 256 + 2 * len(entry['path'].encode('utf-8'))
        if cost + 128 > max_bytes:
            raise ValueError(f'One source file exceeds the archive limit: {entry["path"]}')
        if current and size + cost > max_bytes:
            parts.append(current)
            current, size = [], 128
        current.append(entry)
        size += cost
    if current:
        parts.append(current)
    return parts


class HashingWriter:
    """An unseekable ZIP sink hashes the final bytes, including data descriptors."""
    def __init__(self, stream):
        self.stream, self.hash, self.position = stream, hashlib.sha256(), 0

    def write(self, value):
        count = self.stream.write(value)
        if count != len(value):
            raise OSError('Incomplete archive write')
        self.hash.update(value)
        self.position += count
        return count

    def tell(self):
        return self.position

    def seek(self, *args):
        raise OSError('Archive is sequential')

    def flush(self):
        self.stream.flush()


def read_records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def verify_members(path, records):
    """Read every archived byte, validating ZIP CRC and the source SHA256."""
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [entry['path'] for entry in records]:
            raise ValueError('Archive membership differs from the source inventory')
        for entry in records:
            with archive.open(entry['path']) as stream:
                actual = stream_digest(stream)
            if actual != entry['sha256'] or archive.getinfo(entry['path']).file_size != entry['size']:
                raise ValueError(f'Archive member verification failed: {entry["path"]}')


def prefetched_entries(source, entries):
    """Hide network small-file latency with at most 128 MiB of pending payloads."""
    def read(entry):
        path = source.parent / entry['path']
        value = path.read_bytes()
        if len(value) != entry['size'] or path.stat().st_mtime_ns != entry['mtime_ns']:
            raise ValueError('Source changed while archiving')
        return value, hashlib.sha256(value).hexdigest()

    iterator, pending = iter(entries), deque()
    with ThreadPoolExecutor(max_workers=8) as pool:
        def submit():
            entry = next(iterator, None)
            if entry is not None:
                pending.append((entry, pool.submit(read, entry) if entry['size'] <= 8 * 1024**2 else None))
        for _ in range(16):
            submit()
        while pending:
            entry, future = pending.popleft()
            value = future.result() if future is not None else None
            yield entry, value
            submit()


def package_dataset(source, output, max_bytes=5 * 1024**3, resume=False, progress=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Archive output must be separate from the source dataset')
    safe_relative(source.name)
    info, entries = inventory(source)
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    plan = partition(entries, max_bytes)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    report = dict(format='precomputed-independent-zip64-v1', dataset=source.name, status='running',
                  max_archive_bytes=max_bytes, inventory_fingerprint=fingerprint,
                  source_files=len(entries), source_bytes=sum(entry['size'] for entry in entries),
                  info=info, parts=[])
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (not resume or previous.get('inventory_fingerprint') != fingerprint
                or previous.get('max_archive_bytes') != max_bytes):
            raise FileExistsError('Resume requires an unchanged dataset and archive size limit')
        report['parts'] = previous['parts']
    elif any(output.iterdir()):
        raise FileExistsError('Archive destination is not empty')
    save_json(manifest_path, report)
    for index, group in enumerate(plan):
        filename = (f'{source.name}.zip' if len(plan) == 1 else
                    f'{source.name}.part-{index+1:05d}-of-{len(plan):05d}.zip')
        path = output / filename
        if index < len(report['parts']):
            saved = report['parts'][index]
            if saved['file'] != filename or digest(path) != saved['sha256']:
                raise ValueError(f'Completed archive failed resume verification: {filename}')
            if digest(output / saved['inventory']) != saved['inventory_sha256']:
                raise ValueError('Completed member inventory failed resume verification')
            continue
        temporary = path.with_suffix('.zip.partial')
        records = []
        with temporary.open('wb') as stream:
            sink = HashingWriter(stream)
            with zipfile.ZipFile(sink, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for entry, prefetched in prefetched_entries(source, group):
                    original = source.parent / entry['path']
                    checksum = hashlib.sha256()
                    member = zipfile.ZipInfo(entry['path'], (1980, 1, 1, 0, 0, 0))
                    member.file_size = entry['size']
                    member.external_attr = (stat.S_IFREG | 0o644) << 16
                    count = 0
                    with archive.open(member, 'w', force_zip64=True) as writer:
                        if prefetched is not None:
                            value, source_hash = prefetched
                            writer.write(value)
                            count = len(value)
                        else:
                            with original.open('rb') as reader:
                                while block := reader.read(BLOCK):
                                    writer.write(block)
                                    checksum.update(block)
                                    count += len(block)
                            source_hash = checksum.hexdigest()
                    metadata = original.stat()
                    if count != entry['size'] or metadata.st_mtime_ns != entry['mtime_ns']:
                        raise ValueError('Source changed while archiving')
                    records.append(dict(path=entry['path'], size=count, sha256=source_hash))
            sink.flush()
        if temporary.stat().st_size > max_bytes:
            raise ValueError('Archive exceeds the requested size cap')
        verify_members(temporary, records)
        inventory_name = filename + '.files.jsonl.gz'
        inventory_path = output / inventory_name
        with inventory_path.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as zipped:
            for record in records:
                zipped.write((json.dumps(record, sort_keys=True) + '\n').encode())
        temporary.replace(path)
        part = dict(file=filename, bytes=sink.position, sha256=sink.hash.hexdigest(), files=len(records),
                    inventory=inventory_name, inventory_sha256=digest(inventory_path),
                    verification='Every member CRC and source SHA256 checked after writing')
        report['parts'].append(part)
        save_json(manifest_path, report)
        if progress:
            progress(dict(dataset=source.name, part=index+1, total_parts=len(plan), **part))
    report['status'] = 'complete'
    save_json(manifest_path, report)
    return report


def restore_dataset(package, destination, verify_only=False):
    """Verify every part before safe extraction; matching existing files can resume."""
    package, destination = Path(package).resolve(), Path(destination).resolve()
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('format') != 'precomputed-independent-zip64-v1':
        raise ValueError('Expected a completed precomputed archive manifest')
    dataset = manifest['dataset']
    if len(safe_relative(dataset).parts) != 1:
        raise ValueError('Expected a single dataset directory name')
    seen, verified = set(), []
    for part in manifest['parts']:
        for field in ('file', 'inventory'):
            if len(safe_relative(part[field]).parts) != 1:
                raise ValueError('Archive and inventory must be immediate files')
        archive_path = package / part['file']
        inventory_path = package / part['inventory']
        if digest(archive_path) != part['sha256'] or digest(inventory_path) != part['inventory_sha256']:
            raise ValueError(f'Package checksum mismatch: {part["file"]}')
        records = read_records(inventory_path)
        if len(records) != part['files']:
            raise ValueError('Member count mismatch')
        with zipfile.ZipFile(archive_path) as archive:
            if archive.namelist() != [record['path'] for record in records]:
                raise ValueError('Archive inventory mismatch')
            for record in records:
                relative = safe_relative(record['path'])
                member = archive.getinfo(record['path'])
                if (len(relative.parts) < 2 or relative.parts[0] != dataset or record['path'] in seen
                        or stat.S_ISLNK(member.external_attr >> 16) or member.is_dir()
                        or member.file_size != record['size']):
                    raise ValueError('Invalid or duplicated archive member')
                seen.add(record['path'])
        verified.append((archive_path, records))
    if len(seen) != manifest['source_files']:
        raise ValueError('Package file count mismatch')
    for archive_path, records in verified:
        if verify_only:
            verify_members(archive_path, records)
            continue
        with zipfile.ZipFile(archive_path) as archive:
            for record in records:
                target = destination / record['path']
                if os.name == 'nt' and ':' in record['path']:
                    raise ValueError('Legacy mesh filenames require a POSIX filesystem, for example WSL')
                if target.is_symlink() or destination not in target.resolve().parents:
                    raise ValueError('Extraction would escape the destination')
                if target.exists():
                    if target.is_file() and digest(target) == record['sha256']:
                        continue
                    raise FileExistsError(f'Existing file differs: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.extract-partial')
                if temporary.exists() or temporary.is_symlink():
                    raise FileExistsError(f'Extraction scratch already exists: {temporary}')
                checksum = hashlib.sha256()
                try:
                    with archive.open(record['path']) as reader, temporary.open('xb') as writer:
                        while block := reader.read(BLOCK):
                            writer.write(block)
                            checksum.update(block)
                    if checksum.hexdigest() != record['sha256']:
                        raise ValueError(f'Extracted checksum mismatch: {record["path"]}')
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pack = commands.add_parser('pack', help='Package a completed working dataset')
    pack.add_argument('--source', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--part-gib', type=float, default=5)
    pack.add_argument('--resume', action='store_true')
    restore = commands.add_parser('restore', help='Verify or extract all independent parts')
    restore.add_argument('--package', type=Path, required=True)
    restore.add_argument('--destination', type=Path, default=Path('.'))
    restore.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'pack':
        package_dataset(args.source, args.output, int(args.part_gib * 1024**3), args.resume,
                        progress=lambda item: print(json.dumps(item), flush=True))
    else:
        restore_dataset(args.package, args.destination, args.verify_only)


if __name__ == '__main__':
    main()
