"""Cache four disjoint online zebrafish ROIs, test Probe-EM, and serve four viewers.

Run ``prepare`` once online; ``trace`` and ``serve`` only read local volume data.
Seeds are selected by geometry before inference, not by the model's merge result.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import webbrowser

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

RAW_SOURCE = 'precomputed://https://ng.zebrafish.digital-brain.cn/srv/raw/'
SEG_SOURCE = 'precomputed://https://ng.zebrafish.digital-brain.cn/srv/merge_neuro_label/'
# Absolute XYZ mip-0 coordinates; exactly one server chunk per volume/region.
REGIONS = [
    {'name': 'region_1', 'offset': [32500, 19500, 13700], 'shape': [500, 500, 100]},
    {'name': 'region_2', 'offset': [34000, 19500, 14100], 'shape': [500, 500, 100]},
    {'name': 'region_3', 'offset': [32500, 21500, 14200], 'shape': [500, 500, 100]},
    {'name': 'region_4', 'offset': [34000, 21500, 13600], 'shape': [500, 500, 100]},
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def validate_regions(regions):
    names = set()
    for index, region in enumerate(regions):
        name = region['name']
        if name in names or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Region names must be unique, simple directory names')
        names.add(name)
        lo, shape = np.asarray(region['offset']), np.asarray(region['shape'])
        if (lo.shape != (3,) or shape.shape != (3,) or np.any(lo < 0)
                or np.any(shape <= 0) or np.any(lo != np.round(lo))
                or np.any(shape != np.round(shape)) or np.prod(shape, dtype=object) > 128_000_000):
            raise ValueError('Invalid XYZ region bounds or excessive ROI size')
        for previous in regions[:index]:
            other = np.asarray(previous['offset'])
            if np.all(lo < other + previous['shape']) and np.all(other < lo + shape):
                raise ValueError('Demo regions must not overlap')


def select_seed(labels, skeletons, resolution, offset):
    """Prefer uncut, modestly branched central fragments with interior endpoints."""
    ids, counts = np.unique(labels, return_counts=True)
    counts = dict(zip(map(int, ids), map(int, counts)))
    boundary = set(map(int, np.unique(np.concatenate([
        labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(),
        labels[:, -1].ravel(), labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))))
    shape, resolution, offset = map(np.asarray, (labels.shape, resolution, offset))
    candidates = []
    for key, skeleton in skeletons.items():
        label = int(key)
        if label == 0 or counts.get(label, 0) < 1000 or not skeleton['edges']:
            continue
        vertices = np.asarray(skeleton['vertices']) / resolution - offset
        degree = np.bincount(np.asarray(skeleton['edges']).ravel(), minlength=len(vertices))
        endpoints = vertices[degree == 1]
        if not 2 <= len(endpoints) <= 8 or np.ptp(vertices[:, 2]) < 5:
            continue
        clearance = np.minimum(endpoints, shape - 1 - endpoints) * resolution
        interior = int(np.sum(np.min(clearance, axis=1) >= 500))
        if not interior:
            continue
        distance = float(np.linalg.norm((vertices.mean(0) - (shape-1)/2) / shape))
        candidates.append((label in boundary, distance, label, len(endpoints), interior))
    if not candidates:
        raise ValueError('No suitable central seed with an interior endpoint; choose another ROI')
    cut, distance, seed, endpoint_count, interior_count = min(candidates)
    return seed, {'method': 'uncut-first-central-skeleton-v1', 'candidate_count': len(candidates),
                  'seed_voxels': counts[seed], 'seed_touches_roi_boundary': bool(cut),
                  'endpoint_count': endpoint_count, 'interior_endpoint_count': interior_count,
                  'normalized_center_distance': distance}


def prepare_region(root, region):
    from cloudvolume import CloudVolume
    from demo import load_local
    from export_savem3 import export_bundle
    from probe_em.demo_alignment import prepare_field

    directory = root / region['name']
    report_path = directory / 'region.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if (report['region'] != region or report['sources']['raw'] != RAW_SOURCE
                or report['sources']['seg'] != SEG_SOURCE):
            raise ValueError(f'Existing cache has different provenance: {directory}')
        for name in ('raw', 'seg'):
            data, resolution, offset = load_local(directory / 'cache' / name)
            if (array_hash(data) != report['sha256_decoded_xyz'][name]
                    or not np.array_equal(offset, region['offset'])
                    or not np.array_equal(resolution, report['resolution_xyz_nm'])):
                raise ValueError(f'Cached {name} checksum/coordinates changed: {directory}')
        print(f'[{region["name"]}] verified existing local cache', flush=True)
        return report

    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'cache').exists():
        raise FileExistsError(f'Incomplete cache at {directory}; choose a new --root (no overwrite)')
    lo, hi = np.array(region['offset']), np.array(region['offset']) + region['shape']

    def download(source):
        volume = CloudVolume(source, mip=0, fill_missing=False, parallel=False,
                             progress=False, cache=False)
        if np.any(lo < volume.bounds.minpt) or np.any(hi > volume.bounds.maxpt):
            raise ValueError(f'ROI outside source bounds: {source}')
        data = np.asarray(volume[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))])[..., 0]
        if data.shape != tuple(region['shape']):
            raise ValueError(f'Incomplete remote cutout: {data.shape}')
        return data, np.asarray(volume.resolution), volume.info

    print(f'[{region["name"]}] downloading XYZ {lo.tolist()} .. {hi.tolist()} from zebrafish', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        raw_result, seg_result = list(pool.map(download, (RAW_SOURCE, SEG_SOURCE)))
    raw, resolution, raw_info = raw_result
    labels, seg_resolution, seg_info = seg_result
    if not np.array_equal(resolution, seg_resolution) or raw.dtype != np.uint8:
        raise ValueError('Remote raw/seg resolution mismatch or unsupported raw dtype')
    if raw.std() < 1 or not np.any(labels):
        raise ValueError('ROI contains no usable EM texture or foreground segmentation')
    print(f'[{region["name"]}] downloaded; caching losslessly and skeletonizing ROI', flush=True)
    export_bundle(raw.transpose(2, 1, 0), labels.transpose(2, 1, 0),
                  directory / 'cache', resolution, lo, skeletonize=True)
    labels = labels.astype(np.uint64)
    hashes = {'raw': array_hash(raw), 'seg': array_hash(labels)}
    for name, expected in (('raw', raw), ('seg', labels)):
        actual, local_resolution, local_offset = load_local(directory / 'cache' / name)
        if (not np.array_equal(actual, expected) or not np.array_equal(local_offset, lo)
                or not np.array_equal(local_resolution, resolution)):
            raise ValueError(f'Local {name} round-trip did not preserve the downloaded data')
    skeletons = json.loads((directory / 'cache' / 'traced_skeletons.json').read_text())
    seed, selection = select_seed(labels, skeletons, resolution, lo)
    field, alignment = prepare_field(raw, resolution, lo, cache_dir=directory / 'alignment')
    report = {'schema_version': 1, 'region': region, 'seed': str(seed), 'selection': selection,
              'sources': {'raw': RAW_SOURCE, 'seg': SEG_SOURCE, 'mip': 0},
              'source_info': {'raw': raw_info, 'seg': seg_info},
              'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
              'resolution_xyz_nm': resolution.tolist(), 'sha256_decoded_xyz': hashes,
              'cache_roundtrip_exact': True, 'alignment': alignment,
              'skeleton_source': 'kimimaro on downloaded ROI labels; not whole-neuron skeletons',
              'foreground_fraction': float(np.count_nonzero(labels) / labels.size)}
    write_json(report_path, report)
    print(f'[{region["name"]}] ready; seed {seed}, rejected alignment pairs '
          f'{len(alignment["rejected_pair_slices"])}/{raw.shape[2]-1}', flush=True)
    return report


def trace_region(root, region, checkpoint, model_config, node_limit, resume):
    import torch
    from run_probe_em import DEFAULT_CONFIG, run_one_seed, validate_config
    torch.set_num_threads(2)
    directory = root / region['name']
    report = json.loads((directory / 'region.json').read_text())
    seed = int(report['seed'])
    config = dict(DEFAULT_CONFIG, raw_path=(directory / 'cache' / 'raw').as_uri(),
                  seg_path=(directory / 'cache' / 'seg').as_uri(),
                  checkpoint_sam=str(Path(checkpoint).resolve()), model_cfg_sam=model_config,
                  seed_ids=[seed], target_mip=0, device='cpu', max_workers=1, slice_workers=2,
                  debug_limit=node_limit, output_root=str(directory / 'trace_results'),
                  resume=resume, align_z=True, align_z_field=report['alignment']['field_path'])
    validate_config(config)
    config_path = directory / 'trace_config.json'
    if config_path.is_file():
        previous = json.loads(config_path.read_text())
        mutable = {'resume', 'debug_limit', 'max_workers', 'seed_ids', 'seed_list_file'}
        if ({key: value for key, value in previous.items() if key not in mutable}
                != {key: value for key, value in config.items() if key not in mutable}):
            raise FileExistsError('Different trace configuration already cached; choose a new --root')
    write_json(config_path, config)
    result = run_one_seed(seed, config)
    if result[1] == 'failed' or result[1].startswith('error:'):
        raise RuntimeError(f'Probe-EM failed: {result}')
    print(f'[{region["name"]}] trace: {result}', flush=True)
    return result


def alignment_quality(raw, offset, field):
    """Adjacent-section intensity NCC, a sanity check, NOT registration accuracy.

    Both scores use the same valid interior pixels (subsampled XY by 2). Failed
    pairs with a reused transform contribute unchanged scores, not fake gains.
    """
    from probe_em.demo_alignment import translation
    from probe_em.z_align import warp_slice

    def ncc(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        a, b = a-a.mean(), b-b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denominator) if denominator > 0 else None

    scores = []
    shift = translation(offset[:2])
    ones = np.ones(raw.shape[:2], np.uint8)
    for index in range(1, raw.shape[2]):
        z = int(offset[2]) + index
        relative = np.linalg.inv(shift) @ np.linalg.inv(field.transform(z-1)) @ field.transform(z) @ shift
        valid = warp_slice(ones, relative, seg=True)[::2, ::2] > 0
        previous = raw[:, :, index-1][::2, ::2][valid]
        original = raw[:, :, index][::2, ::2][valid]
        warped = warp_slice(raw[:, :, index], relative, border_mode='constant')[::2, ::2][valid]
        scores.append({'z': z, 'before': ncc(previous, original), 'after': ncc(previous, warped)})
    usable = [score for score in scores if score['before'] is not None and score['after'] is not None]
    if not usable:
        raise ValueError('No valid image pairs for the alignment sanity check')
    return {'metric': 'adjacent-section intensity NCC on shared valid pixels; not ground-truth accuracy',
            'mean_before': float(np.mean([score['before'] for score in usable])),
            'mean_after': float(np.mean([score['after'] for score in usable])),
            'pairs': scores}


def audit_region(root, region):
    from demo import load_local, read_trace_ids
    from probe_em.demo_alignment import prepare_field, register_arrays
    directory = root / region['name']
    if not (directory / 'region.json').is_file():
        raise FileNotFoundError(f'Prepare this region before auditing: {directory}')
    metadata = prepare_region(root, region)  # Verified local-only reuse, including SHA-256.
    raw, resolution, offset = load_local(directory / 'cache' / 'raw')
    seg, _, _ = load_local(directory / 'cache' / 'seg')
    field, alignment = prepare_field(raw, resolution, offset, metadata['alignment']['field_path'])
    aligned_raw, aligned_seg, aligned_offset = register_arrays(raw, seg, offset, field)
    ids, result_folder = read_trace_ids(directory / 'trace_results', int(metadata['seed']))
    status = json.loads((result_folder / 'status.json').read_text())
    config = json.loads((directory / 'trace_config.json').read_text())
    original_ids, aligned_ids = set(map(int, np.unique(seg))), set(map(int, np.unique(aligned_seg)))
    if (set(ids)-original_ids or set(ids)-aligned_ids or aligned_ids-original_ids-{0}
            or status['status'] not in ('complete', 'limited') or status.get('errors')):
        raise ValueError(f'Trace or label-preservation audit failed: {directory}')
    quality = alignment_quality(raw, offset, field)
    report = {'cache_roundtrip_exact': metadata['cache_roundtrip_exact'],
              'checksum_verified': True, 'raw_source': RAW_SOURCE, 'seed': metadata['seed'],
              'trace_status': status, 'trace_ids_present_before_and_after': True,
              'registered_shape_xyz': list(aligned_raw.shape),
              'registered_offset_xyz': aligned_offset.tolist(), 'alignment': alignment,
              'alignment_intensity_sanity_check': quality,
              'pec_result_images': len(list(result_folder.glob('temp_vis_*/*_result.jpg'))),
              'asp_result_images': len(list(result_folder.glob('temp_vis3d_*/*_collision.jpg'))),
              'node_limit_per_run': config['debug_limit'], 'checkpoint': config['checkpoint_sam'],
              'model_config': config['model_cfg_sam'], 'trace_alignment_enabled': config['align_z'],
              'scope': 'ROI-limited workflow sanity check, not a full-neuron or ground-truth accuracy evaluation'}
    write_json(directory / 'validation.json', report)
    print(f'[{region["name"]}] audit passed; {len(ids)} IDs; NCC '
          f'{quality["mean_before"]:.3f} -> {quality["mean_after"]:.3f}', flush=True)
    return report


def display_alignment(directory, metadata, legacy=False):
    report_path = directory / 'alignment_optimized' / 'report.json'
    if not legacy and report_path.is_file():
        optimized = json.loads(report_path.read_text())
        if (not optimized.get('validated')
                or optimized.get('raw_sha256') != metadata['sha256_decoded_xyz']['raw']):
            raise ValueError('Optimized registration is unvalidated or belongs to different raw data')
        return optimized['alignment'], True
    return metadata['alignment'], False


def serve(root, regions, align_z=True, open_browser=False, port=0, legacy_alignment=False):
    import neuroglancer
    from demo import build_viewer, read_trace_ids
    neuroglancer.set_server_bind_address(bind_address='127.0.0.1', bind_port=port)
    viewers, summaries = [], []
    for region in regions:
        directory = root / region['name']
        metadata = json.loads((directory / 'region.json').read_text())
        selected_alignment, optimized = display_alignment(directory, metadata, legacy_alignment)
        seed = int(metadata['seed'])
        ids, result_folder = read_trace_ids(directory / 'trace_results', seed)
        trace_status = json.loads((result_folder / 'status.json').read_text())
        config = json.loads((directory / 'trace_config.json').read_text())
        if trace_status['status'] not in ('complete', 'limited'):
            raise ValueError(f'Refusing to display a failed/incomplete test as valid: {result_folder}')
        viewer, report = build_viewer(directory / 'cache', ids, seed, directory / 'reviews',
                                     align_z=align_z,
                                     align_field=selected_alignment['field_path'] if align_z else None)
        report.update(region=region, trace_status=trace_status, sources=metadata['sources'],
                      trace_alignment_enabled=config['align_z'], node_limit_per_run=config['debug_limit'],
                      trace_alignment_field=config['align_z_field'],
                      display_registration_version='optimized' if optimized else 'original',
                      trace_decisions_recomputed=False,
                      scope='ROI-limited short-run demo, not a complete neuron')
        url = viewer.get_viewer_url()
        report['viewer_url'] = url
        with viewer.config_state.txn() as state:
            state.status_messages['region'] = (f'{region["name"]}: online zebrafish ROI cached locally; '
                                               f'trace status={trace_status["status"]}; ROI-limited test. '
                                               f'Display registration: {"v3" if optimized else "original"}; '
                                               f'{len(selected_alignment["rejected_pair_slices"])}/'
                                               f'{region["shape"][2]-1} adjacent pairs rejected. '
                                               f'Trace IDs held fixed; unmeasured sections: '
                                               f'{selected_alignment.get("quality_summary", {}).get("interpolated_or_extrapolated_slices", "see original QC")}.')
        viewer_directory = directory / ('viewer_optimized' if optimized else 'viewer')
        report['viewer_directory'] = str(viewer_directory)
        write_json(viewer_directory / 'report.json', report)
        write_json(viewer_directory / 'viewer_state.json', viewer.state.to_json())
        (viewer_directory / 'viewer_url.txt').write_text(url + '\n', encoding='utf-8')
        summaries.append(report)
        viewers.append(viewer)  # Keep all four LocalVolume owners alive.
        print(f'[{region["name"]}] Neuroglancer: {url}', flush=True)
    if any(report['display_registration_version'] == 'optimized' for report in summaries):
        if (root / 'viewers.json').is_file() and not (root / 'viewers_original.json').exists():
            write_json(root / 'viewers_original.json', json.loads((root / 'viewers.json').read_text()))
        write_json(root / 'viewers_optimized.json', summaries)
    write_json(root / 'viewers.json', summaries)
    if open_browser:
        for report in summaries:
            webbrowser.open_new_tab(report['viewer_url'])
    print('Four independent viewers ready. Keep this process alive. A toggles local registration.', flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        neuroglancer.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'trace', 'audit', 'serve'])
    parser.add_argument('--root', default='demo_session_regions')
    parser.add_argument('--region', choices=[region['name'] for region in REGIONS])
    parser.add_argument('--checkpoint')
    parser.add_argument('--model-config', default='sam2_hiera_t.yaml')
    parser.add_argument('--node-limit', type=int, default=1, help='Short-run node budget, not full-neuron tracing')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--align-z', action=argparse.BooleanOptionalAction, default=True,
                        help='Viewer registration (default on); trace stage always uses the cached field')
    parser.add_argument('--open', action='store_true', dest='open_browser')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--legacy-alignment', action='store_true',
                        help='serve: use the original cached field instead of validated optimized registration')
    args = parser.parse_args()
    validate_regions(REGIONS)
    regions = [region for region in REGIONS if not args.region or region['name'] == args.region]
    root = Path(args.root).resolve()
    if args.stage == 'prepare':
        for region in regions:
            prepare_region(root, region)
    elif args.stage == 'trace':
        if not args.checkpoint or args.node_limit < 1:
            parser.error('trace requires --checkpoint and a positive --node-limit')
        for region in regions:
            trace_region(root, region, args.checkpoint, args.model_config, args.node_limit, args.resume)
    elif args.stage == 'audit':
        for region in regions:
            audit_region(root, region)
    else:
        serve(root, regions, args.align_z, args.open_browser, args.port, args.legacy_alignment)


if __name__ == '__main__':
    main()
