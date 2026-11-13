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
    item = dict(name='Fish-X LC-NE Test overview', kind='overview', raw_path='raw', seg_path='labels')
    windows = [item, dict(name='Neuron A', seed=2**63+3, local_dir=tmp_path/'roi')]
    boxes = roi_boxes(windows, info())
    assert boxes[0]['point_a'] == [16, 30, 50]
    assert boxes[0]['point_b'] == [28, 46, 54]
    assert boxes[0]['bounds_nm'] == [[64, 180, 1500], [112, 276, 1620]]
    viewer, report = build_overview(item, windows, info(), info(kind='segmentation'),
                                    'precomputed://http://localhost/raw', 'precomputed://http://localhost/lc')
    assert report['alignment'] == {'enabled': False}
    assert viewer.state.show_slices is False
    assert viewer.state.show_axis_lines is True
    assert viewer.state.to_json()['layout'] == '4panel-alt'
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
        dict(kind='overview', name='Fish-X LC-NE Test overview', raw_path='data/raw', seg_path='data/lc')]}))
    windows, initial = read_windows(manifest)
    assert initial == 0 and windows[0]['kind'] == 'overview'
    assert windows[0]['raw_path'] == str(tmp_path/'data/raw')


def test_manifest_accepts_label_only_overview_and_resolves_saved_boxes(tmp_path):
    manifest = tmp_path/'windows.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'windows': [dict(
        kind='overview', name='Fish-X LC-NE Test overview', seg_path='data/lc',
        test_windows_path='test_windows.json')]}))
    windows, initial = read_windows(manifest)
    assert initial == 0 and 'raw_path' not in windows[0]
    assert windows[0]['seg_path'] == str(tmp_path/'data/lc')
    assert windows[0]['test_windows_path'] == str(tmp_path/'test_windows.json')


def test_saved_test_windows_must_match_native_roi_geometry(tmp_path):
    item = dict(kind='overview', name='Fish-X LC-NE Test overview', seg_path='lc')
    raw = tmp_path/'roi/raw';raw.mkdir(parents=True)
    (raw/'info').write_text(json.dumps(info((8, 12, 30), (8, 15, 50), (6, 8, 4))))
    windows = [item, dict(name='Neuron A', seed=30001, local_dir=tmp_path/'roi')]
    viewer, _ = build_overview(item, windows, None, info(kind='segmentation'), None, 'lc')
    payload = viewer.state.layers['test_windows'].layer.to_json()
    payload['annotations'][0]['pointA'][0] += 1
    path = tmp_path/'boxes.json';path.write_text(json.dumps(payload))
    item['test_windows_path'] = str(path)
    with pytest.raises(ValueError, match='do not match'):
        build_overview(item, windows, None, info(kind='segmentation'), None, 'lc')


def test_portable_mesh_names_work_over_http_without_colons_on_disk(tmp_path):
    source = tmp_path/'source';(source/'mesh').mkdir(parents=True)
    metadata = dict(info(kind='segmentation'), mesh='mesh', probe_em_filename_encoding='percent')
    (source/'info').write_text(json.dumps(metadata))
    payload = gzip.compress(b'encoded mesh vertices and faces')
    (source/'mesh/30001%3A0').write_text('{"fragments": ["30001:0:fragment"]}')
    (source/'mesh/30001%3A0%3Afragment.gz').write_bytes(payload)
    files = PrecomputedFiles(tmp_path/'cache');files.register('lc', source)
    collection = WindowCollection([{'name': 'overview'}], lambda *_: None)
    server, url = serve_windows(collection, volumes=files)
    try:
        with urllib.request.urlopen(url+'data/lc/mesh/30001%3A0') as response:
            assert json.load(response)['fragments'] == ['30001:0:fragment']
        with urllib.request.urlopen(url+'data/lc/mesh/30001:0:fragment') as response:
            assert response.headers['Content-Encoding'] == 'gzip'
            assert gzip.decompress(response.read()) == b'encoded mesh vertices and faces'
        assert not any(':' in p.name for p in (tmp_path/'cache').rglob('*'))
        for request in ['lc/../outside', 'lc/%2e%2e/outside', 'lc/mesh/../../outside']:
            with pytest.raises(FileNotFoundError):
                files.resolve(request)
    finally:
        server.shutdown();server.server_close();collection.close()


def test_context_images_are_confined_to_overview_and_match_physical_extent(tmp_path):
    item = dict(name='Fish-X LC-NE Test overview', kind='overview', raw_path='raw', seg_path='labels', segments=[30001])
    sources = {name: (info(), f'precomputed://http://localhost/{name}') for name in ('mito', 'syn', 'ves')}
    viewer, report = build_overview(item, [item], info(), info(kind='segmentation'),
                                    'precomputed://http://localhost/raw', 'precomputed://http://localhost/lc', sources)
    assert report['image_layers'] == ['mito', 'syn', 'ves']
    for name in sources:
        assert not viewer.state.layers[name].visible
    with pytest.raises(ValueError, match='same physical volume'):
        build_overview(item, [item], info(), info(kind='segmentation'), 'raw', 'labels',
                       {'mito': (info(size=(2,3,4)), 'mito')})
    manifest = tmp_path/'windows.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'windows': [
        dict(name='Process', seed=30001, local_dir='roi', image_layers={'mito': 'mito'})]}))
    with pytest.raises(ValueError, match='belong to the overview'):
        read_windows(manifest)


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


def test_overview_label_context_preserves_type_and_checks_extent():
    item = dict(name='Fish-X LC-NE Test overview', raw_path='raw', seg_path='lc',
                kind='overview', visible_layers=['lcpre'])
    sources = {name: (info(kind='segmentation'), f'precomputed://http://localhost/{name}')
               for name in ('lcpre', 'lcsyn')}
    viewer, report = build_overview(item, [item], info(), info(kind='segmentation'),
                                    'raw', 'lc', label_sources=sources)
    assert report['label_layers'] == ['lcpre', 'lcsyn']
    assert viewer.state.layers['lcpre'].to_json()['type'] == 'segmentation'
    assert viewer.state.layers['lcpre'].visible
    assert not viewer.state.layers['lcsyn'].visible
    for invalid, message in [({'lcpre': (info(), 'source')}, 'segmentation volumes'),
                             ({'lc': (info(kind='segmentation'), 'source')}, 'distinct names'),
                             ({'lcpre': (info(size=(2, 3, 4), kind='segmentation'), 'source')},
                              'same physical volume')]:
        with pytest.raises(ValueError, match=message):
            build_overview(item, [item], info(), info(kind='segmentation'),
                           'raw', 'lc', label_sources=invalid)
    with pytest.raises(ValueError, match='distinct names'):
        build_overview(item, [item], info(), info(kind='segmentation'), 'raw', 'lc',
                       image_sources={'lcpre': (info(), 'image')}, label_sources=sources)


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
