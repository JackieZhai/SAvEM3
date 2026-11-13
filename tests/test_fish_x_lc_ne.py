"""Bundled data integrity and consistent five-box/full-dataset launch behavior."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probe-em'))
from probe_em.fish_x_lc_ne import (BUNDLE, DATASET_NAME, RESOLUTIONS, VOLUMES,
                                  _extract, prepare_dataset, review_folder, sha256)
from probe_em.window_browser import read_windows


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('fish-x')
    return root, prepare_dataset('examples', root/'dataset')


@pytest.fixture
def full_metadata(tmp_path):
    """Small metadata fixture; the full 201 GB release is not needed for unit tests."""
    root = tmp_path/'full'
    for name in VOLUMES:
        directory = root/f'zfishlc_{name}'
        scales = []
        for resolution in RESOLUTIONS:
            factor = [r//native for r, native in zip(resolution, RESOLUTIONS[0])]
            key = '_'.join(map(str, resolution))
            scale = dict(resolution=resolution, key=key,
                         voxel_offset=[v//f for v, f in zip([192, 288, 0], factor)],
                         size=[(v+f-1)//f for v, f in zip([8640, 8160, 1583], factor)])
            scales.append(scale)
            (directory/key).mkdir(parents=True)
            (directory/key/'fixture').write_bytes(b'fixture')
        (directory/'info').write_text(json.dumps(dict(
            type='image' if name in ('raw', 'mito', 'syn', 'ves') else 'segmentation', scales=scales)))
    return root


def test_bundled_boxes_have_native_geometry_labels_and_matching_local_fields(prepared):
    _, path = prepared
    windows, initial = read_windows(path)
    assert initial == 0 and len(windows) == 6
    assert windows[0]['kind'] == 'overview' and 'raw_path' not in windows[0]
    assert [entry['seed'] for entry in windows[1:]] == list(range(30001, 30006))
    assert json.loads(path.read_text())['dataset'] == DATASET_NAME
    for entry in windows[1:]:
        assert entry['kind'] == 'roi' and entry['image_layers'] == {}
        infos = [json.loads((Path(entry['local_dir'])/name/'info').read_text())
                 for name in ('raw', 'seg', 'target')]
        grids = [{key: info['scales'][0][key] for key in ('size', 'resolution', 'voxel_offset')}
                 for info in infos]
        assert grids[0] == grids[1] == grids[2]
        assert grids[0]['resolution'] == [8, 8, 30] and grids[0]['size'] == [512, 512, 63]
        field = Path(entry['align_field'])
        metadata = json.loads(field.with_suffix('.json').read_text())
        assert field.is_file() and metadata['voxel_offset_xyz'] == grids[0]['voxel_offset']
        assert metadata['algorithm'] == 'wide-context-ecc-preserve-global-trend-v1'
        assert metadata['reference_z'] == grids[0]['voxel_offset'][2]+31


def test_full_option_starts_at_overview_and_keeps_identical_boxes_and_review_paths(prepared, full_metadata):
    root, examples_path = prepared
    examples, _ = read_windows(examples_path)
    full, initial = read_windows(prepare_dataset('full', root/'dataset', full_metadata))
    assert initial == 0 and len(full) == 6
    assert full[1:] == examples[1:]
    assert full[0]['name'] == 'Fish-X LC-NE Test overview'
    assert set(full[0]['image_layers']) == {'mito', 'syn', 'ves'}
    assert set(full[0]['label_layers']) == {'lcpre', 'lcsyn'}
    assert full[0]['visible_layers'] == []
    from probe_em.overview import roi_boxes
    raw_info = json.loads((full_metadata/'zfishlc_raw/info').read_text())
    boxes = roi_boxes(full, raw_info)
    assert len(boxes) == 5
    for index, (entry, box) in enumerate(zip(examples[1:], boxes), 1):
        assert box['bounds_nm'] == entry['bounds_nm']
        first = review_folder(root, entry, index, supplied_dataset=True)
        second = review_folder(root, full[index], index, supplied_dataset=True)
        assert first == second == root/f'neuron_{entry["seed"]}'
    assert review_folder(root, examples[1], 1, supplied_dataset=False) == root/'window_001'


def test_extraction_reuses_good_files_and_repairs_damaged_or_missing_cache(prepared):
    root, path = prepared
    windows, _ = read_windows(path)
    raw = Path(windows[1]['local_dir'])/'raw'
    chunk = next((raw/'8_8_30').iterdir())
    original, modified = chunk.read_bytes(), chunk.stat().st_mtime_ns
    prepare_dataset('examples', root/'dataset')
    assert chunk.stat().st_mtime_ns == modified
    chunk.write_bytes(bytes([original[0] ^ 1])+original[1:])
    missing = Path(windows[2]['align_field'])
    expected = sha256(missing)
    missing.unlink()
    saved = root/'neuron_30001/interactive/review_objects.json'
    saved.parent.mkdir(parents=True)
    saved.write_text('saved objects')
    prepare_dataset('examples', root/'dataset')
    assert chunk.read_bytes() == original and sha256(missing) == expected
    assert saved.read_text() == 'saved objects'


def test_full_option_requires_extracted_volumes_before_creating_cache(tmp_path):
    cache = tmp_path/'cache'
    with pytest.raises(ValueError, match='requires --data-root'):
        prepare_dataset('full', cache)
    with pytest.raises(ValueError, match='Missing extracted volume'):
        prepare_dataset('full', cache, tmp_path/'not-extracted')
    with pytest.raises(ValueError, match='only used with --dataset full'):
        prepare_dataset('examples', cache, tmp_path)
    assert not cache.exists()


@pytest.mark.parametrize('invalid', ['wrong-grid', 'missing-volume', 'empty-scale'])
def test_full_option_rejects_incompatible_or_incomplete_setup(full_metadata, tmp_path, invalid):
    path = full_metadata/'zfishlc_lcsyn/info'
    if invalid == 'wrong-grid':
        info = json.loads(path.read_text())
        info['scales'][0]['voxel_offset'][0] += 1
        path.write_text(json.dumps(info))
    elif invalid == 'missing-volume':
        path.unlink()
    else:
        (path.parent/'256_256_240/fixture').unlink()
    with pytest.raises(ValueError, match='geometry|Missing extracted|No extracted chunks'):
        prepare_dataset('full', tmp_path/'cache', full_metadata)


def test_archive_integrity_and_containment_are_checked_before_writing(tmp_path):
    archive = tmp_path/'invalid.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('../outside', b'not a dataset member')
    destination = tmp_path/'cache'
    destination.mkdir()
    with pytest.raises(ValueError, match='checksum mismatch'):
        _extract(archive, destination, dict(sha256='0'*64, files=1))
    with pytest.raises(ValueError, match='Invalid bundled archive member'):
        _extract(archive, destination, dict(sha256=sha256(archive), files=1))
    assert not (tmp_path/'outside').exists() and list(destination.iterdir()) == []


def test_cli_prepares_default_and_full_options_without_model_or_server(prepared, full_metadata, capsys):
    spec = importlib.util.spec_from_file_location('fish_x_demo_cli', ROOT/'probe-em/scripts/demo_windows.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    root, _ = prepared
    script.main(['--prepare-only', '--output-dir', str(root)])
    output = capsys.readouterr().out
    assert 'Fish-X LC-NE Test: examples; 6 views' in output
    assert 'Window manifest:' in output
    script.main(['--dataset', 'full', '--data-root', str(full_metadata),
                 '--prepare-only', '--output-dir', str(root)])
    assert 'Fish-X LC-NE Test: full; 6 views' in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        script.main(['--dataset', 'full', '--windows', 'custom.json', '--prepare-only', '--output-dir', str(root)])
    assert error.value.code == 2


def test_bundled_lc_overview_loads_all_scales_meshes_and_saved_boxes(prepared, tmp_path):
    import gzip
    from probe_em.overview import build_overview
    from probe_em.precomputed_files import PrecomputedFiles
    _, path = prepared
    windows, _ = read_windows(path)
    item = windows[0]
    info = json.loads((Path(item['seg_path'])/'info').read_text())
    assert [scale['resolution'] for scale in info['scales']] == RESOLUTIONS
    assert info['scales'][0]['size'] == [8640, 8160, 1583]
    assert info['probe_em_filename_encoding'] == 'percent'
    viewer, report = build_overview(item, windows, None, info, None, 'precomputed://http://localhost/lc')
    assert {layer.name for layer in viewer.state.layers} == {'lc', 'test_windows'}
    assert report['raw_source'] is None and report['alignment'] == {'enabled': False}
    assert len(viewer.state.layers['test_windows'].annotations) == 5
    assert viewer.state.show_axis_lines and not viewer.state.show_slices
    assert viewer.state.to_json()['layout'] == '4panel-alt'
    assert not hasattr(viewer, 'probe_em_interactive')
    files = PrecomputedFiles(tmp_path/'served')
    files.register('lc', item['seg_path'])
    for neuron in range(30001, 30006):
        mesh, _ = files.resolve(f'lc/mesh/{neuron}:0')
        assert ':' not in mesh.name
        fragments = json.loads(mesh.read_text())['fragments']
        assert fragments
        for fragment in fragments:
            local, compressed = files.resolve(f'lc/mesh/{fragment}')
            assert ':' not in local.name and compressed
            assert len(gzip.decompress(local.read_bytes())) > 4
    from probe_em.viewer_status import attach_toolbar_status
    status = attach_toolbar_status(viewer).snapshot()
    assert 'bundled lc labels' in status[0]['text'] and 'full raw' not in status[0]['text']


def test_full_mode_does_not_extract_an_extra_bundled_lc_copy(full_metadata, tmp_path):
    path = prepare_dataset('full', tmp_path/'full-cache', full_metadata)
    windows, _ = read_windows(path)
    assert windows[0]['seg_path'] == str(full_metadata/'zfishlc_lc')
    assert not (path.parent/'zfishlc_lc').exists()


def test_bundled_starter_masks_and_annotations_match_all_five_raw_volumes(prepared, tmp_path):
    from probe_em.review_objects import ReviewObjects
    from probe_em.starter_session import seed_review
    spec = importlib.util.spec_from_file_location('bundled_review_demo', ROOT/'probe-em/scripts/demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    _, path = prepared
    windows, _ = read_windows(path)
    expected = json.loads((BUNDLE/'manifest.json').read_text())['starter_session']['windows']
    totals = [0, 0]
    for window, counts in zip(windows[1:], expected):
        raw, resolution, offset = demo.load_local(Path(window['local_dir'])/'raw')
        output = tmp_path/f'neuron_{window["seed"]}'/'interactive'
        saved = seed_review(window['starter_session'], output, raw, offset, resolution, window['local_dir'])
        restored = ReviewObjects(output, raw.shape, offset, resolution, window['local_dir'])
        assert saved['neuron_id'] == counts['neuron'] == window['seed']
        assert len(restored.objects) == counts['objects']
        annotations = [a for result in restored.objects.values() for a in result['annotations']]
        assert sum(a['label'] == 1 for a in annotations) == counts['positive']
        assert sum(a['label'] == 0 for a in annotations) == counts['negative']
        totals[0] += len(restored.objects)
        totals[1] += len(annotations)
    assert totals == [8, 55]
