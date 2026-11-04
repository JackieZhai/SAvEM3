"""Whole-volume overview coordinates, streaming and toolbar status isolation."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'probe-em'))
from probe_em.overview import build_overview, roi_boxes
from probe_em.precomputed_files import PrecomputedFiles
from probe_em.viewer_status import attach_toolbar_status, set_status
from probe_em.window_browser import WindowCollection, read_windows, serve_windows


def info(resolution=(4, 6, 30), offset=(10, 20, 40), size=(200, 300, 100), kind='image'):
    return {'type': kind, 'data_type': 'uint8' if kind == 'image' else 'uint32', 'num_channels': 1,
            'scales': [{'key': 'native', 'encoding': 'raw', 'chunk_sizes': [[8, 8, 8]],
                        'resolution': list(resolution), 'voxel_offset': list(offset), 'size': list(size)}]}


def test_overview_bounds_use_physical_coordinates_and_remain_unregistered(tmp_path):
    raw = tmp_path/'roi/raw';raw.mkdir(parents=True)
    (raw/'info').write_text(json.dumps(info((8, 12, 30), (8, 15, 50), (6, 8, 4))))
    item = dict(name='Entire lc', kind='overview', raw_path='raw', seg_path='labels')
    windows = [item, dict(name='Neuron A', seed=2**63+3, local_dir=tmp_path/'roi')]
    boxes = roi_boxes(windows, info())
    assert boxes[0]['point_a'] == [16, 30, 50]
    assert boxes[0]['point_b'] == [28, 46, 54]
    assert boxes[0]['bounds_nm'] == [[64, 180, 1500], [112, 276, 1620]]
    viewer, report = build_overview(item, windows, info(), info(kind='segmentation'),
                                    'precomputed://http://localhost/raw', 'precomputed://http://localhost/lc')
    assert report['alignment'] == {'enabled': False}
    assert viewer.state.show_slices is False
    annotation = viewer.state.layers['test_windows'].annotations[0]
    np.testing.assert_array_equal(annotation.point_a, [16, 30, 50])
    assert annotation.description == 'Neuron A'
    assert set(viewer.state.layers['lc'].segments) == {2**63+3}
    assert not hasattr(viewer, 'probe_em_interactive')
    assert 'keya' not in viewer.config_state.state.input_event_bindings.viewer
    (raw/'info').write_text(json.dumps(info((8, 12, 30), (200, 15, 50), (6, 8, 4))))
    with pytest.raises(ValueError, match='outside'):
        roi_boxes(windows, info())


def test_manifest_accepts_overview_without_seed_and_resolves_source_paths(tmp_path):
    manifest = tmp_path/'windows.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'windows': [
        dict(kind='overview', name='Entire lc', raw_path='data/raw', seg_path='data/lc')]}))
    windows, initial = read_windows(manifest)
    assert initial == 0 and windows[0]['kind'] == 'overview'
    assert windows[0]['raw_path'] == str(tmp_path/'data/raw')


def test_status_moves_to_toolbar_and_remains_live_per_viewer():
    import neuroglancer
    first, second = neuroglancer.Viewer(), neuroglancer.Viewer()
    set_status(first, 'alignment', 'Local alignment: ON')
    set_status(second, 'alignment', 'Local alignment: OFF')
    feed = attach_toolbar_status(first)
    assert not first.config_state.state.status_messages
    set_status(first, 'prediction', 'Propagation range: z=10..18')
    set_status(first, 'alignment', 'Local alignment: OFF')
    assert feed.snapshot() == [{'id': 'alignment', 'text': 'Local alignment: OFF'},
                               {'id': 'prediction', 'text': 'Propagation range: z=10..18'}]
    assert not first.config_state.state.status_messages
    assert second.config_state.state.status_messages['alignment'] == 'Local alignment: OFF'


def test_chunk_cache_deduplicates_reads_and_rejects_path_escape(tmp_path, monkeypatch):
    import shutil
    source = tmp_path/'source'; source.mkdir()
    (source/'info').write_text(json.dumps(info()))
    (source/'native').mkdir()
    payload = gzip.compress(b'one encoded volume chunk')
    (source/'native/chunk.gz').write_bytes(payload)
    (source/'native/untouched.gz').write_bytes(b'not requested')
    files = PrecomputedFiles(tmp_path/'cache');files.register('raw', source)
    assert not list((tmp_path/'cache').rglob('chunk.gz'))
    original = shutil.copyfile
    calls = []

    def copy(a, b):
        calls.append(a)
        return original(a, b)

    monkeypatch.setattr(shutil, 'copyfile', copy)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(files.resolve, ['raw/native/chunk']*8))
    assert len(calls) == 1 and len({str(path) for path, _ in results}) == 1
    assert results[0][0].read_bytes() == payload and results[0][1]
    assert not list((tmp_path/'cache').rglob('untouched.gz'))
    (source/'native/chunk.gz').unlink()
    assert files.resolve('raw/native/chunk')[0].read_bytes() == payload
    outside = tmp_path/'private';outside.write_text('outside volume')
    (source/'escape').symlink_to(outside)
    for request in ['raw/%2e%2e/private', 'raw/escape', 'raw/../private', 'raw/\\private', 'unknown/info']:
        with pytest.raises(FileNotFoundError):
            files.resolve(request)


def test_http_chunks_support_gzip_ranges_cors_and_status(tmp_path):
    source = tmp_path/'source';source.mkdir()
    (source/'info').write_text(json.dumps(info()))
    (source/'chunk.gz').write_bytes(gzip.compress(b'compressed voxels'))
    (source/'shard').write_bytes(b'0123456789')
    files = PrecomputedFiles(tmp_path/'cache');files.register('raw', source)
    from probe_em.viewer_status import ToolbarStatus
    viewer = SimpleNamespace(get_viewer_url=lambda: 'http://localhost/view',
                             probe_em_toolbar_status=ToolbarStatus({'task': 'Running inference'}))
    collection = WindowCollection([{'name': 'ROI'}], lambda *_: viewer)
    server, url = serve_windows(collection, volumes=files)
    try:
        collection.request(0);collection.futures[0].result(timeout=5)
        with urllib.request.urlopen(url+'data/raw/chunk') as response:
            assert response.headers['Content-Encoding'] == 'gzip'
            assert response.headers['Access-Control-Allow-Origin'] == '*'
            assert gzip.decompress(response.read()) == b'compressed voxels'
        request = urllib.request.Request(url+'data/raw/shard', headers={'Range': 'bytes=2-5'})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206 and response.read() == b'2345'
            assert response.headers['Content-Range'] == 'bytes 2-5/10'
        with urllib.request.urlopen(url+'api/windows/0/status') as response:
            assert json.load(response)['messages'] == [{'id': 'task', 'text': 'Running inference'}]
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url+'data/raw/%2e%2e/private')
        assert error.value.code == 404
        request = urllib.request.Request(url+'data/raw/shard', headers={'Range': 'bytes=20-'})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 416
        source.rename(tmp_path/'offline-source')
        with urllib.request.urlopen(url+'data/raw/chunk') as response:
            assert gzip.decompress(response.read()) == b'compressed voxels'
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url+'data/raw/not-yet-cached')
        assert error.value.code == 503
    finally:
        server.shutdown();server.server_close();collection.close()
