"""Serve scoped local precomputed volumes with an on-demand disk chunk cache."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
from urllib.parse import unquote
import uuid


class PrecomputedFiles:
    """Immutable source snapshots; never reads a whole volume into memory."""

    def __init__(self, cache_directory):
        self.cache = Path(cache_directory).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.mounts = {}
        self.locks = [threading.Lock() for _ in range(64)]
        self.read_slots = threading.BoundedSemaphore(4)

    def register(self, name, source):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name) or name in self.mounts:
            raise ValueError('Volume mount names must be unique simple identifiers')
        source = Path(source).resolve()
        metadata = (source/'info').read_bytes()
        info = json.loads(metadata)
        if info.get('type') not in ('image', 'segmentation') or not info.get('scales'):
            raise ValueError('Expected a precomputed image or segmentation volume')
        identity = hashlib.sha256(str(source).encode()+b'\0'+metadata).hexdigest()
        destination = self.cache/identity
        destination.mkdir(exist_ok=True)
        (destination/'info').write_bytes(metadata)
        (self.cache/(identity+'.json')).write_text(json.dumps(
            {'source': str(source), 'info_sha256': hashlib.sha256(metadata).hexdigest(),
             'policy': 'Immutable source; copy requested chunks only'}, indent=2), encoding='utf-8')
        self.mounts[name] = (source, destination)
        return info

    @staticmethod
    def _inside(root, relative):
        candidate = (root/relative).resolve()
        if not candidate.is_relative_to(root):
            raise FileNotFoundError('File is outside the configured volume')
        return candidate

    def resolve(self, request):
        """Map a decoded volume request to one cached file and its encoding."""
        name, separator, relative = unquote(request).partition('/')
        parts = PurePosixPath(relative).parts
        if (not separator or name not in self.mounts or not parts or relative.startswith('/')
                or '..' in parts or '\\' in relative or '\0' in relative):
            raise FileNotFoundError('Unknown volume file')
        source, destination = self.mounts[name]
        lock = self.locks[int(hashlib.sha256(request.encode()).hexdigest()[:8], 16) % len(self.locks)]
        with lock:
            for suffix in ('', '.gz'):
                cached = self._inside(destination, relative+suffix)
                if cached.is_file():
                    return cached, cached.name.endswith('.gz')
            with self.read_slots:
                if not source.is_dir():
                    raise OSError('Volume source is unavailable for uncached chunks')
                for suffix in ('', '.gz'):
                    original = self._inside(source, relative+suffix)
                    if not original.is_file():
                        continue
                    cached = self._inside(destination, relative+suffix)
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cached.with_name(cached.name+'.tmp-'+uuid.uuid4().hex)
                    try:
                        shutil.copyfile(original, temporary)
                        temporary.replace(cached)
                    finally:
                        temporary.unlink(missing_ok=True)
                    return cached, cached.name.endswith('.gz')
        raise FileNotFoundError('Volume file not found')

    @staticmethod
    def _headers(handler):
        handler.send_header('Access-Control-Allow-Origin', '*')
        handler.send_header('Access-Control-Expose-Headers', 'Content-Range, Content-Encoding')
        handler.send_header('X-Content-Type-Options', 'nosniff')

    def serve(self, handler, request):
        try:
            path, compressed = self.resolve(request)
        except FileNotFoundError:
            handler.send_response(404)
            self._headers(handler)
            handler.send_header('Content-Length', '0')
            handler.end_headers()
            return
        except OSError:
            handler.send_response(503)
            self._headers(handler)
            handler.send_header('Content-Length', '0')
            handler.end_headers()
            return
        size = path.stat().st_size
        start, end, status = 0, size-1, 200
        byte_range = handler.headers.get('Range')
        # Gzip chunks are sent as complete HTTP-encoded entities. Uncompressed
        # shard/mesh files also support byte-range reads.
        if byte_range and not compressed:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', byte_range)
            if match and (match[1] or match[2]):
                start = int(match[1]) if match[1] else max(0, size-int(match[2]))
                end = min(size-1, int(match[2])) if match[1] and match[2] else size-1
            else:
                start = size
            if start >= size or end < start:
                handler.send_response(416)
                self._headers(handler)
                handler.send_header('Content-Range', f'bytes */{size}')
                handler.send_header('Content-Length', '0')
                handler.end_headers()
                return
            status = 206
        handler.send_response(status)
        self._headers(handler)
        handler.send_header('Content-Type', 'application/json' if path.name in ('info', 'info.gz') else 'application/octet-stream')
        handler.send_header('Content-Length', str(end-start+1))
        handler.send_header('Cache-Control', 'private, max-age=3600')
        if compressed:
            handler.send_header('Content-Encoding', 'gzip')
        else:
            handler.send_header('Accept-Ranges', 'bytes')
        if status == 206:
            handler.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        handler.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end-start+1
                while remaining:
                    block = stream.read(min(1024*1024, remaining))
                    if not block:
                        break
                    handler.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass
