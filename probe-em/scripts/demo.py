"""Local Neuroglancer demo: raw, input instances, Probe-EM trace, editable review.

Use --local-dir volumes_local with the existing zebrafish cache, or a volume
bundle from export_savem3.py. Local alignment is ON by default; --no-align-z
skips it. No raw/seg data is fetched from a remote server.
Meshes default to near-isotropic sampling; slices retain native resolution.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import threading
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
                 align_max_shift_nm=800., align_coarse_resolution_nm=32., align_min_ncc=.25,
                 mesh_isotropic=True, mesh_target_nm=None, mesh_cache=None,
                 comparison_ids=None, interactive=None, align_backend='ecc',
                 sofima_python=None, sofima_options=None, image_layers=None, label_layers=None):
    import neuroglancer
    from probe_em.demo_alignment import prepare_field, register_arrays, map_points, skeleton_lines
    from probe_em.meshes import MeshVolume
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
                                         min_ncc=align_min_ncc, backend=align_backend,
                                         sofima_python=sofima_python, sofima_options=sofima_options)
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
                MeshVolume(seg_array, dimensions=dimensions, voxel_offset=offset,
                           mesh_isotropic=mesh_isotropic, mesh_target_nm=mesh_target_nm,
                           mesh_cache=mesh_cache))

    original_sources = sources(raw, seg, seg_offset)
    registered_sources = sources(aligned_raw, aligned_seg, aligned_offset) if field is not None else None
    # Auxiliary sources must share the native physical grid. Their display warp
    # always follows raw, even when the saved prediction used another estimator.
    from probe_em.alignment_field import warp_field_slice
    extra_sources, extra_ids, extra_paths = {False: {}, True: {}}, {}, {}
    reserved = {'raw', 'input_segments', 'probe_trace', 'manual_correction', 'skeletons',
                'positive_points', 'negative_points', 'sam_preview', 'sam_accepted', *(comparison_ids or {})}
    for is_label, paths in ((False, image_layers), (True, label_layers)):
        for name, path in (paths or {}).items():
            if not name or name in reserved or name in extra_paths:
                raise ValueError(f'Reserved or duplicate auxiliary layer name: {name}')
            data, resolution, offset = load_local(path, max_voxels)
            if data.shape != raw.shape or not np.array_equal(resolution, raw_res) or not np.array_equal(offset, raw_offset):
                raise ValueError(f'Auxiliary layer {name} must share the raw shape, resolution and voxel offset')
            if is_label and (data.dtype.kind not in 'ui' or np.any(data < 0)):
                raise ValueError(f'Auxiliary labels must be nonnegative integers: {name}')
            extra_paths[name] = {'path': str(Path(path).resolve()), 'type': 'segmentation' if is_label else 'image'}
            if is_label:
                extra_ids[name] = [int(i) for i in np.unique(data) if i]
            for aligned in ([False, True] if field is not None else [False]):
                shown, origin = data, offset
                if aligned:
                    shown, origin = np.empty(aligned_raw.shape, data.dtype), aligned_offset
                    for index in range(data.shape[2]):
                        shown[:, :, index] = warp_field_slice(field, data[:, :, index], int(offset[2])+index,
                                                              offset[:2], origin[:2], shown.shape[:2], seg=is_label)
                if is_label:
                    extra_sources[aligned][name] = MeshVolume(
                        shown, dimensions=dimensions, voxel_offset=origin, mesh_isotropic=mesh_isotropic,
                        mesh_target_nm=mesh_target_nm, mesh_cache=mesh_cache)
                else:
                    extra_sources[aligned][name] = neuroglancer.LocalVolume(shown, dimensions=dimensions, voxel_offset=origin)
    present = sorted(set(traced_ids) & available)
    meshes = {'policy': 'near_isotropic' if mesh_isotropic else 'native',
              'original': original_sources[1].precompute_meshes(present)}
    if registered_sources is not None:
        meshes['registered'] = registered_sources[1].precompute_meshes(sorted(set(present) & aligned_ids))
    print('[demo meshes] ' + json.dumps({
        space: {key: report[key] for key in ('downsample_xyz', 'mesh_resolution_xyz_nm', 'cache_directory')}
        for space, report in meshes.items() if isinstance(report, dict)}), flush=True)
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
    interactive_session = None
    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.dimensions = dimensions
        raw_source, seg_source = registered_sources if mode['aligned'] else original_sources
        state.layers['raw'] = neuroglancer.ImageLayer(source=raw_source)
        state.layers['input_segments'] = neuroglancer.SegmentationLayer(source=seg_source)
        state.layers['input_segments'].visible = False
        state.layers['probe_trace'] = neuroglancer.SegmentationLayer(
            source=seg_source, segments=present, hover_highlight=False, pick=False)
        state.layers['manual_correction'] = neuroglancer.SegmentationLayer(
            source=seg_source, segments=present, hover_highlight=False, pick=True)
        state.layers['manual_correction'].visible = False
        for name, ids in (comparison_ids or {}).items():
            if name in ('raw', 'input_segments', 'probe_trace', 'manual_correction', 'skeletons',
                        'positive_points', 'negative_points', 'sam_preview', 'sam_accepted'):
                raise ValueError(f'Reserved comparison layer name: {name}')
            state.layers[name] = neuroglancer.SegmentationLayer(
                source=seg_source, segments=sorted(set(map(int, ids)) & available),
                hover_highlight=False, pick=False)
            state.layers[name].visible = False
        for name, source in extra_sources[mode['aligned']].items():
            if name in extra_ids:
                state.layers[name] = neuroglancer.SegmentationLayer(
                    source=source, segments=extra_ids[name], hover_highlight=False, pick=False)
            else:
                state.layers[name] = neuroglancer.ImageLayer(source=source)
            state.layers[name].visible = False
        if skeleton_layers:
            state.layers['skeletons'] = skeleton_layers[mode['aligned']]
        state.layout = '4panel'
        state.show_slices = False  # Hide slice planes in 3D; keep the three 2D views.
        points = np.argwhere(seg == seed)
        position = (points[len(points) // 2] + seg_offset).astype(float)
        state.position = map_points(field, [position])[0] if mode['aligned'] else position
        state.selected_layer.layer = 'manual_correction'

    def status(message):
        from probe_em.viewer_status import set_status
        set_status(viewer, 'probe-em', message)

    def mode_status():
        suffix = f"ON (reference z={field.ref_z})" if mode['aligned'] else 'OFF'
        if mode['aligned']:
            backend = field_info.get('backend', 'ecc').upper()
            if backend == 'SOFIMA':
                count = sum(bool(np.any(section)) for section in field.displacement)
                backend += f'; elastic refinement on {count}/{len(field.displacement)} sections'
            suffix += f' [{backend}]'
        mesh_resolution = meshes['original']['mesh_resolution_xyz_nm']
        mesh_scale = 'x'.join(f'{value:g}' for value in mesh_resolution)
        status(f'Seed {seed}: {len(present)} segments. Local alignment: {suffix}. '
               f'Mesh grid: {mesh_scale} nm (tiny missing IDs: native fallback). '
               'a: compare alignment; r: restore result; q: restore review; g: save review. '
               f'{len(missing)} traced IDs outside original ROI.')

    def toggle_alignment(_):
        with interactive_session.lock if interactive_session is not None else nullcontext():
            toggle_alignment_locked()

    def toggle_alignment_locked():
        if field is None:
            status('Alignment was disabled at startup. Restart without --no-align-z to enable comparison.')
            return
        new_mode = not mode['aligned']
        with viewer.txn() as state:
            # Move the navigation center with its data, then swap ALL geometry together.
            state.position = map_points(field, [state.position], inverse=not new_mode)[0]
            raw_source, seg_source = registered_sources if new_mode else original_sources
            state.layers['raw'].layer.source = raw_source
            for name in ('input_segments', 'probe_trace', 'manual_correction', *(comparison_ids or {})):
                state.layers[name].layer.source = seg_source
            for name, source in extra_sources[new_mode].items():
                state.layers[name].layer.source = source
            if skeleton_layers:
                visible = state.layers['skeletons'].visible
                state.layers['skeletons'] = skeleton_layers[new_mode]
                state.layers['skeletons'].visible = visible
            if interactive_session is not None:
                interactive_session.transform_layers(state, new_mode)
        mode['aligned'] = new_mode
        mode_status()

    def restore(_):
        with viewer.txn() as state:
            state.layers['manual_correction'].segments = present
        status(f'Restored {len(present)} traced segments for review')

    def restore_result(_):
        # Spatial picking is off for reference layers; sidebar edits are still possible.
        # Restore references explicitly without replacing the user's review selection.
        with viewer.txn() as state:
            state.layers['probe_trace'].segments = present
            for name, ids in (comparison_ids or {}).items():
                state.layers[name].segments = sorted(set(map(int, ids)) & available)
            for name, ids in extra_ids.items():
                state.layers[name].segments = ids
        mode_status()

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
                        'mesh_policy': meshes['policy'],
                        'mesh_fingerprint': meshes['registered' if mode['aligned'] else 'original']['fingerprint'],
                        'alignment': dict(field_info, enabled=mode['aligned'])}, indent=2), encoding='utf-8')
        status(f'Saved review: {path}')

    viewer.actions.add('toggle-alignment', toggle_alignment)
    viewer.actions.add('restore-trace', restore)
    viewer.actions.add('restore-result', restore_result)
    viewer.actions.add('save-review', save)
    with viewer.config_state.txn() as state:
        state.input_event_bindings.viewer['keya'] = 'toggle-alignment'
        state.input_event_bindings.viewer['keyq'] = 'restore-trace'
        state.input_event_bindings.viewer['keyr'] = 'restore-result'
        state.input_event_bindings.viewer['keyg'] = 'save-review'
    mode_status()
    if interactive is not None:
        from probe_em.interactive import InteractiveSession
        options = dict(interactive)
        output_dir = options.pop('output_dir')
        propagation = {key: options.pop(key) for key in ('propagation_radius', 'propagation_max_slices') if key in options}
        interactive_session = InteractiveSession(
            viewer, raw, seg_offset, seg_res, field,
            aligned_raw.shape if aligned_raw is not None else None, aligned_offset,
            mode, output_dir, options, mesh_isotropic=mesh_isotropic,
            mesh_target_nm=mesh_target_nm, source=local_dir, **propagation)
        viewer.probe_em_interactive = interactive_session
    return viewer, {'seed': str(seed), 'segments': [str(value) for value in present],
                    'missing_ids': [str(value) for value in sorted(missing)],
                    'shape_xyz': list(seg.shape), 'resolution_xyz_nm': seg_res.tolist(),
                    'voxel_offset_xyz': seg_offset.tolist(), 'alignment': alignment,
                    'show_slices': False,
                    'trace_hover_highlight': False,
                    'trace_spatial_picking': False,
                    'meshes': meshes,
                    'auxiliary_layers': extra_paths,
                    'interactive': {'enabled': interactive is not None,
                                    'backend': interactive.get('backend') if interactive else None,
                                    'scope': 'per-section image prediction and bounded SAM2 video propagation',
                                    'inference_space': 'image: original native; video: display space at task start',
                                    'propagation_range_inclusive': interactive_session.propagation_range if interactive else None},
                    'layers': [layer.name for layer in viewer.state.layers]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', required=True)
    parser.add_argument('--results-dir', help='Completed trace directory; optional in interactive mode')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--suffix', default='sam')
    parser.add_argument('--align-z', action=argparse.BooleanOptionalAction, default=True,
                        help='Local slice registration (default: on); --no-align-z disables it')
    parser.add_argument('--align-field', help='Optional affine or elastic field .npz; otherwise estimate locally')
    parser.add_argument('--align-motion', choices=['translation', 'euclidean', 'affine'], default='translation')
    parser.add_argument('--align-backend', choices=['ecc', 'sofima'], default='ecc',
                        help='Estimator when --align-field is absent; SOFIMA adds bounded elastic refinement')
    parser.add_argument('--sofima-python', help='Python executable in an isolated SOFIMA environment')
    parser.add_argument('--sofima-config', type=Path, help='Optional SOFIMA parameter JSON (physical units)')
    parser.add_argument('--align-reference-z', type=int, help='Absolute reference z; default is the middle section')
    parser.add_argument('--align-max-shift-nm', type=float, default=800.,
                        help='Automatic estimation: physical pair displacement limit in nm (default 800)')
    parser.add_argument('--align-coarse-resolution-nm', type=float, default=32.,
                        help='Automatic estimation: coarsest XY pixel size in nm (default 32)')
    parser.add_argument('--align-min-ncc', type=float, default=.25,
                        help='Automatic estimation: minimum coarse-image correlation (default .25)')
    parser.add_argument('--mesh-isotropic', action=argparse.BooleanOptionalAction, default=True,
                        help='Near-isotropic mesh sampling (default on); slices stay at native resolution')
    parser.add_argument('--mesh-target-nm', type=float,
                        help='Optional target mesh scale in nm; default is the coarsest native axis')
    parser.add_argument('--correction-dir', default='demo_reviews')
    parser.add_argument('--output-dir', default='demo_session')
    parser.add_argument('--max-voxels', type=int, default=128_000_000)
    parser.add_argument('--image-layer', action='append', default=[], metavar='NAME=PATH',
                        help='Optional local precomputed context image on the same grid; repeatable')
    parser.add_argument('--label-layer', action='append', default=[], metavar='NAME=PATH',
                        help='Optional local precomputed GT or prediction on the same grid; repeatable')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable point prompts, image prediction and bounded SAM2 propagation')
    parser.add_argument('--prompt-backend', choices=['sam', 'hq_sam', 'sam2', 'saem2'], default='sam2')
    parser.add_argument('--prompt-checkpoint', help='Local checkpoint for the selected backend (HQ weights for SAEM2)')
    parser.add_argument('--prompt-decoder', help='Additional trained SAEM2 decoder checkpoint')
    parser.add_argument('--prompt-config', default='sam2_hiera_t.yaml', help='Installed SAM 2 model config')
    parser.add_argument('--prompt-model-type', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b')
    parser.add_argument('--prompt-device', default='auto', help='auto, cpu, mps, cuda or cuda:N')
    parser.add_argument('--sam2-runtime', choices=['torch', 'onnx'], default='torch',
                        help='Image encoder runtime; decoder and temporal memory remain in PyTorch')
    parser.add_argument('--sam2-onnx-dir', type=Path, help='Validated local export from export_sam2_onnx.py')
    parser.add_argument('--onnx-provider', choices=['cpu', 'coreml', 'cuda', 'directml'], default='cpu')
    parser.add_argument('--onnx-threads', type=int, default=0)
    parser.add_argument('--feature-cache-mb', type=float, default=512,
                        help='Bounded SAM2 image-feature cache on CPU; 0 disables it (default 512 MiB)')
    parser.add_argument('--propagate-radius', type=int, default=4,
                        help='Initial propagation range extends this many sections around the starting view (default 4)')
    parser.add_argument('--propagate-max-slices', type=int, default=32,
                        help='Maximum sections per explicit V propagation task (default 32)')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='Validate volumes, actions and state, then exit')
    args = parser.parse_args()
    def named_paths(values):
        result = {}
        for value in values:
            name, separator, path = value.partition('=')
            if not separator or not name.strip() or not path.strip() or name in result:
                parser.error('Auxiliary layers require unique NAME=PATH entries')
            result[name] = path
        return result
    image_layers, label_layers = named_paths(args.image_layer), named_paths(args.label_layer)
    if args.sofima_config and args.align_backend != 'sofima':
        parser.error('--sofima-config requires --align-backend sofima')
    sofima_options = json.loads(args.sofima_config.read_text()) if args.sofima_config else None
    if not args.results_dir and not args.interactive:
        parser.error('--results-dir is required unless --interactive is enabled')
    if args.interactive and not args.prompt_checkpoint:
        parser.error('--interactive requires --prompt-checkpoint')
    if args.interactive and args.prompt_backend == 'saem2' and not args.prompt_decoder:
        parser.error('--prompt-backend saem2 requires --prompt-decoder')
    interactive = dict(backend=args.prompt_backend, checkpoint=args.prompt_checkpoint,
                       decoder_checkpoint=args.prompt_decoder, model_config=args.prompt_config,
                       model_type=args.prompt_model_type, device=args.prompt_device,
                       sam2_runtime=args.sam2_runtime, onnx_dir=args.sam2_onnx_dir,
                       onnx_provider=args.onnx_provider, onnx_threads=args.onnx_threads,
                       feature_cache_mb=args.feature_cache_mb,
                       propagation_radius=args.propagate_radius, propagation_max_slices=args.propagate_max_slices,
                       output_dir=Path(args.output_dir) / 'interactive') if args.interactive else None
    import neuroglancer
    ids, result_folder = read_trace_ids(args.results_dir, args.seed, args.suffix) if args.results_dir else ([], None)
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
                                 align_min_ncc=args.align_min_ncc,
                                 mesh_isotropic=args.mesh_isotropic, mesh_target_nm=args.mesh_target_nm,
                                 mesh_cache=Path(args.output_dir) / 'meshes', interactive=interactive,
                                 align_backend=args.align_backend, sofima_python=args.sofima_python,
                                 sofima_options=sofima_options, image_layers=image_layers, label_layers=label_layers)
    report['trace_results'] = str(result_folder.resolve()) if result_folder else None
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # LocalVolume sources require the serving Python process; this is a diagnostic
    # snapshot, not a standalone/public viewer URL.
    (output / 'viewer_state.json').write_text(json.dumps(viewer.state.to_json(), indent=2), encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if args.check:
        if args.interactive:
            viewer.probe_em_interactive.close()
        print('Demo validation passed (viewer state only; no browser rendering tested).', flush=True)
        return
    url = viewer.get_viewer_url()
    (output / 'viewer_url.txt').write_text(url + '\n', encoding='utf-8')
    print(f'Neuroglancer demo: {url}', flush=True)
    print('Keep this process running. a: toggle alignment; r: restore result; '
          'enable manual_correction to edit; q restores review, g saves.', flush=True)
    if args.interactive:
        from probe_em.interactive import HELP
        print('Interactive image/video prediction: ' + HELP, flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        neuroglancer.stop()
    finally:
        if args.interactive:
            viewer.probe_em_interactive.close()


if __name__ == '__main__':
    main()
