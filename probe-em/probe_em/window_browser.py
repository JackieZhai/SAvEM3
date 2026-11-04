"""Lazy, persistent Neuroglancer windows behind a small local browser toolbar."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit


def read_windows(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('windows'):
        raise ValueError('Window manifest requires schema_version=1 and a nonempty windows list')
    windows = []
    for item in manifest['windows']:
        entry = dict(item)
        if not isinstance(entry.get('name'), str) or not entry['name'].strip():
            raise ValueError('Each window needs a nonempty name')

        def local(value):
            source = Path(value)
            return str((path.parent/source).resolve() if not source.is_absolute() else source.resolve())

        kind = entry.setdefault('kind', 'roi')
        if kind == 'overview':
            for key in ('raw_path', 'seg_path'):
                if not entry.get(key):
                    raise ValueError('An overview needs raw_path and seg_path precomputed directories')
                entry[key] = local(entry[key])
        elif kind == 'roi':
            if type(entry.get('seed')) is not int or not 0 < entry['seed'] < 2**64:
                raise ValueError('Each ROI needs a positive uint64 seed')
            if not entry.get('local_dir'):
                raise ValueError('Each ROI needs a local_dir containing raw and seg')
            entry['local_dir'] = local(entry['local_dir'])
        else:
            raise ValueError('Window kind must be roi or overview')
        if entry.get('align_field'):
            entry['align_field'] = local(entry['align_field'])
        for kind in ('image_layers', 'label_layers'):
            entry[kind] = {name: local(value) for name, value in entry.get(kind, {}).items()}
        windows.append(entry)
    initial = manifest.get('initial_index', 0)
    if not isinstance(initial, int) or not 0 <= initial < len(windows):
        raise ValueError('initial_index must identify a window in the manifest')
    return windows, initial


class WindowCollection:
    """Each window is built once and keeps its own controller and review state."""

    def __init__(self, windows, builder, initial=0):
        self.windows, self.builder, self.initial = windows, builder, initial
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='probe-em-window')
        self.lock = threading.RLock()
        self.futures = {}
        self.viewers = {}
        self.closed = False

    def _build(self, index):
        viewer = self.builder(index, self.windows[index])
        with self.lock:
            if self.closed:
                session = getattr(viewer, 'probe_em_interactive', None)
                if session is not None:
                    session.close()
                raise RuntimeError('Window browser closed while loading')
            self.viewers[index] = viewer
        return viewer

    def request(self, index, retry=False):
        with self.lock:
            if self.closed:
                raise RuntimeError('Window browser is closed')
            if not 0 <= index < len(self.windows):
                raise IndexError('Unknown test window')
            future = self.futures.get(index)
            if future is None or (retry and future.done() and future.exception() is not None):
                future = self.executor.submit(self._build, index)
                self.futures[index] = future
            if not future.done():
                return {'state': 'loading', 'index': index, 'name': self.windows[index]['name']}
            error = future.exception()
            if error is not None:
                return {'state': 'failed', 'index': index, 'error': f'{type(error).__name__}: {error}'}
            return {'state': 'ready', 'index': index, 'url': future.result().get_viewer_url()}

    def close(self):
        with self.lock:
            self.closed = True
            for viewer in self.viewers.values():
                session = getattr(viewer, 'probe_em_interactive', None)
                if session is not None:
                    session.close()
            self.viewers.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def status(self, index):
        with self.lock:
            if not 0 <= index < len(self.windows):
                raise IndexError('Unknown test window')
            viewer = self.viewers.get(index)
            feed = getattr(viewer, 'probe_em_toolbar_status', None)
            return {'index': index, 'messages': feed.snapshot() if feed is not None else []}


def serve_windows(collection, bind='127.0.0.1', port=0, volumes=None):
    token = secrets.token_urlsafe(18)
    prefix = '/'+token
    html = Path(__file__).with_name('window_browser.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, value, mime='application/json'):
            payload = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlsplit(self.path).path.rstrip('/')
            if path == prefix:
                self.respond(200, html, 'text/html; charset=utf-8')
            elif path == prefix+'/api/windows':
                self.respond(200, {'initial_index': collection.initial, 'windows': [
                    {'index': i, 'name': item['name'], 'kind': item.get('kind', 'roi')}
                    for i, item in enumerate(collection.windows)]})
            elif path.startswith(prefix+'/data/') and volumes is not None:
                volumes.serve(self, path[len(prefix+'/data/'):])
            elif path.startswith(prefix+'/api/windows/') and path.endswith('/status'):
                try:
                    self.respond(200, collection.status(int(path.split('/')[-2])))
                except (ValueError, IndexError):
                    self.respond(404, {'error': 'Unknown test window'})
            elif path.startswith(prefix+'/api/windows/'):
                try:
                    self.respond(200, collection.request(int(path.rsplit('/', 1)[-1])))
                except (ValueError, IndexError):
                    self.respond(404, {'error': 'Unknown test window'})
                except RuntimeError as error:
                    self.respond(503, {'error': str(error)})
            else:
                self.respond(404, {'error': 'Not found'})

        def do_OPTIONS(self):
            if urlsplit(self.path).path.startswith(prefix+'/data/') and volumes is not None:
                self.send_response(204)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'range')
                self.end_headers()
            else:
                self.respond(404, {'error': 'Not found'})

        def do_POST(self):
            path = urlsplit(self.path).path.rstrip('/')
            if path.startswith(prefix+'/api/windows/') and path.endswith('/retry'):
                try:
                    self.respond(200, collection.request(int(path.split('/')[-2]), retry=True))
                except (ValueError, IndexError):
                    self.respond(404, {'error': 'Unknown test window'})
                except RuntimeError as error:
                    self.respond(503, {'error': str(error)})
            else:
                self.respond(404, {'error': 'Not found'})

    server = ThreadingHTTPServer((bind, port), Handler)
    thread = threading.Thread(target=server.serve_forever, name='probe-em-browser', daemon=True)
    thread.start()
    return server, f'http://{bind}:{server.server_port}{prefix}/'
