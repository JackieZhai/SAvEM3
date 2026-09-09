"""Local Neuroglancer demo: raw, input instances, Probe-EM trace, editable review.

Use --local-dir volumes_local with the existing zebrafish cache, or a volume
bundle from export_savem3.py. Local alignment is ON by default; --no-align-z
skips it. No raw/seg data is fetched from a remote server.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_local(path, max_voxels=128_000_000):
    from cloudvolume import CloudVolume
    path = Path(path).resolve()
    if not (path / 'info').is_file():
        raise FileNotFoundError(f'Precomputed info missing: {path / "info"}')
    volume = CloudVolume(path.as_uri(), mip=0, parallel=False, fill_missing=False, progress=False)
    if int(np.prod(volume.bounds.size(), dtype=np.int64)) > max_voxels:
        raise ValueError('Demo ROI exceeds --max-voxels; export a smaller ROI')
    data = np.asarray(volume[volume.bounds])[..., 0]
    return data, np.asarray(volume.resolution), np.asarray(volume.voxel_offset)


def read_trace_ids(results_dir, seed, suffix='sam'):
    folder = Path(results_dir) / f'{seed}_results_{suffix}'
    filename = folder / f'trace_{seed}_ids.txt'
    if not filename.is_file():
        raise FileNotFoundError(f'Trace result missing: {filename}')
    ids = sorted({int(value) for value in filename.read_text().split()})
    if not ids or any(value <= 0 or value > np.iinfo(np.uint64).max for value in ids):
        raise ValueError('Trace IDs must be positive uint64 integers')
    return ids, folder


def build_viewer(local_dir, traced_ids, seed, correction_dir, max_voxels=128_000_000,
                 *, align_z=True, align_field=None, align_cache=None,
                 align_motion='translation', align_reference_z=None,
                 align_max_shift_nm=800., align_coarse_resolution_nm=32., align_min_ncc=.25):
    import neuroglancer
    from probe_em.demo_alignment import prepare_field, register_arrays, map_points, skeleton_lines
    raw, raw_res, raw_offset = load_local(Path(local_dir) / 'raw', max_voxels)
    seg, seg_res, seg_offset = load_local(Path(local_dir) / 'seg', max_voxels)
    if raw.shape != seg.shape or not np.array_equal(raw_res, seg_res) or not np.array_equal(raw_offset, seg_offset):
        raise ValueError('Raw and segmentation must have identical shape, resolution and voxel offset')
    if seg.dtype.kind not in 'ui' or np.any(seg < 0):
        raise ValueError('Segmentation must contain nonnegative integer labels')
    available = {int(value) for value in np.unique(seg)}
    missing = set(traced_ids) - available
    if seed not in available:
        raise ValueError(f'Seed {seed} is outside the local ROI')
    dimensions = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=seg_res)
    field = None
    field_info = {}
    alignment = {'enabled': bool(align_z)}
    if align_z:
        field, field_info = prepare_field(raw, seg_res, seg_offset, align_field, align_cache,
                                         align_motion, align_reference_z,
                                         max_shift_nm=align_max_shift_nm,
                                         coarse_resolution_nm=align_coarse_resolution_nm,
                                         min_ncc=align_min_ncc)
        aligned_raw, aligned_seg, aligned_offset = register_arrays(raw, seg, seg_offset, field, max_voxels)
        aligned_ids = {int(value) for value in np.unique(aligned_seg)}
        if seed not in aligned_ids:
            raise ValueError('Registration lost the seed label; inspect the field or use --no-align-z')
        alignment.update(field_info, registered_shape_xyz=list(aligned_seg.shape),
                         registered_voxel_offset_xyz=aligned_offset.tolist(),
                         lost_trace_ids=[str(value) for value in sorted((set(traced_ids) & available)-aligned_ids)])
        print('[demo alignment] ' + json.dumps(alignment), flush=True)
    else:
        aligned_raw = aligned_seg = aligned_offset = None

    def sources(raw_array, seg_array, offset):
        return (neuroglancer.LocalVolume(raw_array, dimensions=dimensions, voxel_offset=offset),
                neuroglancer.LocalVolume(seg_array, dimensions=dimensions, voxel_offset=offset))

    original_sources = sources(raw, seg, seg_offset)
    registered_sources = sources(aligned_raw, aligned_seg, aligned_offset) if field is not None else None
    present = sorted(set(traced_ids) & available)
    skeleton_layers = {}
    cache = Path(local_dir) / 'traced_skeletons.json'
    if cache.is_file():
        skeletons = json.loads(cache.read_text())
        for registered in ([False, True] if field is not None else [False]):
            annotations = []
            for label in present:
                skeleton = skeletons.get(str(label))
                if not skeleton:
                    continue
                for index, a, b in skeleton_lines(skeleton, seg_res, field if registered else None):
                    annotations.append(neuroglancer.LineAnnotation(
                        id=f'{label}-{index}', point_a=a, point_b=b))
            skeleton_layers[registered] = neuroglancer.LocalAnnotationLayer(
                dimensions=dimensions, annotations=annotations)

    mode = {'aligned': bool(align_z)}
    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.dimensions = dimensions
        raw_source, seg_source = registered_sources if mode['aligned'] else original_sources
        state.layers['raw'] = neuroglancer.ImageLayer(source=raw_source)
        state.layers['input_segments'] = neuroglancer.SegmentationLayer(source=seg_source)
        state.layers['input_segments'].visible = False
        state.layers['probe_trace'] = neuroglancer.SegmentationLayer(source=seg_source, segments=present)
        state.layers['manual_correction'] = neuroglancer.SegmentationLayer(source=seg_source, segments=present)
        state.layers['manual_correction'].visible = False
        if skeleton_layers:
            state.layers['skeletons'] = skeleton_layers[mode['aligned']]
        state.layout = '4panel'
        state.show_slices = False  # Hide slice planes in 3D; keep the three 2D views.
        points = np.argwhere(seg == seed)
        position = (points[len(points) // 2] + seg_offset).astype(float)
        state.position = map_points(field, [position])[0] if mode['aligned'] else position
        state.selected_layer.layer = 'manual_correction'

    def status(message):
        with viewer.config_state.txn() as state:
            state.status_messages['probe-em'] = message

    def mode_status():
        suffix = f"ON (reference z={field.ref_z})" if mode['aligned'] else 'OFF'
        status(f'Seed {seed}: {len(present)} segments. Local alignment: {suffix}. '
               'a: compare alignment; q: restore review; g: save review. '
               f'{len(missing)} traced IDs outside original ROI.')

    def toggle_alignment(_):
        if field is None:
            status('Alignment was disabled at startup. Restart without --no-align-z to enable comparison.')
            return
        new_mode = not mode['aligned']
        with viewer.txn() as state:
            # Move the navigation center with its data, then swap ALL geometry together.
            state.position = map_points(field, [state.position], inverse=not new_mode)[0]
            raw_source, seg_source = registered_sources if new_mode else original_sources
            state.layers['raw'].layer.source = raw_source
            for name in ('input_segments', 'probe_trace', 'manual_correction'):
                state.layers[name].layer.source = seg_source
            if skeleton_layers:
                visible = state.layers['skeletons'].visible
                state.layers['skeletons'] = skeleton_layers[new_mode]
                state.layers['skeletons'].visible = visible
        mode['aligned'] = new_mode
        mode_status()

    def restore(_):
        with viewer.txn() as state:
            state.layers['manual_correction'].segments = present
        status(f'Restored {len(present)} traced segments for review')

    def save(_):
        selected = sorted(int(value) for value in viewer.state.layers['manual_correction'].segments)
        if not selected or seed not in selected or any(value not in available for value in selected):
            status('Select a valid segment set containing the seed before saving')
            return
        directory = Path(correction_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path = directory / f'{seed}_{stamp}.json'
        path.write_text(json.dumps({'seed': str(seed), 'segments': [str(value) for value in selected],
                        'source': str(Path(local_dir).resolve()), 'resolution_xyz_nm': seg_res.tolist(),
                        'voxel_offset_xyz': seg_offset.tolist(), 'created_at': stamp,
                        'display_space': 'registered' if mode['aligned'] else 'original',
                        'display_voxel_offset_xyz': (aligned_offset if mode['aligned'] else seg_offset).tolist(),
                        'alignment': dict(field_info, enabled=mode['aligned'])}, indent=2), encoding='utf-8')
        status(f'Saved review: {path}')

    viewer.actions.add('toggle-alignment', toggle_alignment)
    viewer.actions.add('restore-trace', restore)
    viewer.actions.add('save-review', save)
    with viewer.config_state.txn() as state:
        state.input_event_bindings.viewer['keya'] = 'toggle-alignment'
        state.input_event_bindings.viewer['keyq'] = 'restore-trace'
        state.input_event_bindings.viewer['keyg'] = 'save-review'
    mode_status()
    return viewer, {'seed': str(seed), 'segments': [str(value) for value in present],
                    'missing_ids': [str(value) for value in sorted(missing)],
                    'shape_xyz': list(seg.shape), 'resolution_xyz_nm': seg_res.tolist(),
                    'voxel_offset_xyz': seg_offset.tolist(), 'alignment': alignment,
                    'show_slices': False,
                    'layers': [layer.name for layer in viewer.state.layers]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--suffix', default='sam')
    parser.add_argument('--align-z', action=argparse.BooleanOptionalAction, default=True,
                        help='Local slice registration (default: on); --no-align-z disables it')
    parser.add_argument('--align-field', help='Optional existing ZAffineField .npz; otherwise estimate locally')
    parser.add_argument('--align-motion', choices=['translation', 'euclidean', 'affine'], default='translation')
    parser.add_argument('--align-reference-z', type=int, help='Absolute reference z; default is the middle section')
    parser.add_argument('--align-max-shift-nm', type=float, default=800.,
                        help='Automatic estimation: physical pair displacement limit in nm (default 800)')
    parser.add_argument('--align-coarse-resolution-nm', type=float, default=32.,
                        help='Automatic estimation: coarsest XY pixel size in nm (default 32)')
    parser.add_argument('--align-min-ncc', type=float, default=.25,
                        help='Automatic estimation: minimum coarse-image correlation (default .25)')
    parser.add_argument('--correction-dir', default='demo_reviews')
    parser.add_argument('--output-dir', default='demo_session')
    parser.add_argument('--max-voxels', type=int, default=128_000_000)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='Validate volumes, actions and state, then exit')
    args = parser.parse_args()
    import neuroglancer
    ids, result_folder = read_trace_ids(args.results_dir, args.seed, args.suffix)
    if not args.check:
        neuroglancer.set_server_bind_address(bind_address=args.bind, bind_port=args.port)
    if not args.align_z and (args.align_field or args.align_reference_z is not None):
        parser.error('--align-field/--align-reference-z require alignment to be enabled')
    if args.align_field and args.align_reference_z is not None:
        parser.error('A supplied field already defines its reference z; omit --align-reference-z')
    viewer, report = build_viewer(args.local_dir, ids, args.seed, args.correction_dir, args.max_voxels,
                                 align_z=args.align_z, align_field=args.align_field,
                                 align_cache=Path(args.output_dir) / 'alignment',
                                 align_motion=args.align_motion, align_reference_z=args.align_reference_z,
                                 align_max_shift_nm=args.align_max_shift_nm,
                                 align_coarse_resolution_nm=args.align_coarse_resolution_nm,
                                 align_min_ncc=args.align_min_ncc)
    report['trace_results'] = str(result_folder.resolve())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # LocalVolume sources require the serving Python process; this is a diagnostic
    # snapshot, not a standalone/public viewer URL.
    (output / 'viewer_state.json').write_text(json.dumps(viewer.state.to_json(), indent=2), encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if args.check:
        print('Demo validation passed (viewer state only; no browser rendering tested).', flush=True)
        return
    url = viewer.get_viewer_url()
    (output / 'viewer_url.txt').write_text(url + '\n', encoding='utf-8')
    print(f'Neuroglancer demo: {url}', flush=True)
    print('Keep this process running. a: toggle alignment; enable manual_correction to edit; q restores, g saves.', flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        neuroglancer.stop()


if __name__ == '__main__':
    main()
