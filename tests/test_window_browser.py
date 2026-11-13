"""Window loading is deduplicated and switching never replaces review state."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import urllib.request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'probe-em'))
from probe_em.window_browser import WindowCollection, read_windows, serve_windows


def test_manifest_paths_are_relative_to_manifest_and_seed_ids_are_exact(tmp_path):
    path = tmp_path/'windows.json'
    path.write_text(json.dumps({'schema_version': 1, 'windows': [dict(
        name='Neuron A', seed=2**63+9, local_dir='cache', label_layers={'gt': 'truth'})]}))
    windows, initial = read_windows(path)
    assert initial == 0 and windows[0]['seed'] == 2**63+9
    assert windows[0]['local_dir'] == str(tmp_path/'cache')
    assert windows[0]['label_layers']['gt'] == str(tmp_path/'truth')


def test_switching_preserves_state_and_deduplicates_loading_requests():
    started, release = threading.Event(), threading.Event()
    calls = []

    def build(index, item):
        calls.append(index)
        if index == 0:
            started.set(); assert release.wait(5)
        return SimpleNamespace(points=[], get_viewer_url=lambda: f'http://localhost/window/{index}')

    collection = WindowCollection([{'name': 'a'}, {'name': 'b'}], build)
    try:
        assert collection.request(0)['state'] == 'loading'
        assert started.wait(5)
        collection.request(0); collection.request(1); collection.request(0)
        release.set()
        first = collection.futures[0].result(timeout=5)
        collection.futures[1].result(timeout=5)
        first.points.append([1, 2, 3])
        assert collection.request(0)['state'] == 'ready'
        assert collection.futures[0].result().points == [[1, 2, 3]]
        assert calls == [0, 1]
        with pytest.raises(IndexError):
            collection.request(-1)
    finally:
        release.set(); collection.close()


def test_review_buttons_route_to_the_loaded_window_and_require_json():
    invoked = []
    session = SimpleNamespace(lock=threading.RLock(), future=None, merge_pending=None, close=lambda: None)
    viewer = SimpleNamespace(probe_em_interactive=session,
                             actions=SimpleNamespace(invoke=lambda name, payload: invoked.append(name)),
                             get_viewer_url=lambda: 'http://localhost/viewer')
    collection = WindowCollection([{'name': 'one'}], lambda *args: viewer)
    server, url = serve_windows(collection)
    try:
        collection.request(0); collection.futures[0].result(timeout=5)
        assert collection.status(0)['review'] == {'enabled': True, 'busy': False, 'confirmation_stage': None}
        for operation, expected in [('new-object', 'prompt-clear'), ('accept', 'prompt-accept'),
                                    ('confirm', 'prompt-merge-confirm'), ('cancel', 'prompt-merge-cancel')]:
            request = urllib.request.Request(url+'api/windows/0/review/'+operation, data=b'{}',
                                             headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(request) as response:
                assert json.load(response)['review']['enabled']
            assert invoked[-1] == expected
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url+'api/windows/0/review/confirm', data=b'{}', method='POST'))
        assert error.value.code == 415 and len(invoked) == 4
    finally:
        server.shutdown(); server.server_close(); collection.close()


def test_failure_requires_explicit_retry_and_http_routes_are_scoped():
    calls = []

    def build(index, item):
        calls.append(index)
        if len(calls) == 1:
            raise ValueError('Missing local chunk')
        return SimpleNamespace(get_viewer_url=lambda: 'http://localhost/recovered')

    collection = WindowCollection([{'name': 'a'}], build)
    server, url = serve_windows(collection)
    try:
        collection.request(0)
        with pytest.raises(ValueError):
            collection.futures[0].result(timeout=5)
        assert collection.request(0)['state'] == 'failed' and len(calls) == 1
        response = urllib.request.urlopen(url+'api/windows')
        assert json.load(response)['windows'][0]['name'] == 'a'
        request = urllib.request.Request(url+'api/windows/0/retry', method='POST', data=b'')
        urllib.request.urlopen(request).close()
        collection.futures[0].result(timeout=5)
        assert collection.request(0)['url'].endswith('/recovered')
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url+'api/windows/../anything')
    finally:
        server.shutdown(); server.server_close(); collection.close()


def test_closing_during_load_closes_the_eventual_session():
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    def build(index, item):
        started.set()
        assert release.wait(5)
        return SimpleNamespace(probe_em_interactive=SimpleNamespace(close=closed.set))

    collection = WindowCollection([{'name': 'a'}], build)
    try:
        collection.request(0)
        assert started.wait(5)
        collection.close()
        release.set()
        with pytest.raises(RuntimeError, match='closed while loading'):
            collection.futures[0].result(timeout=5)
        assert closed.is_set()
    finally:
        release.set()
        collection.close()
