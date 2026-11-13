"""Open Fish-X LC-NE Test examples, its full overview, or a custom window manifest."""
import argparse
import json
from pathlib import Path
import sys
import threading
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_em.fish_x_lc_ne import DATASET_NAME, prepare_dataset, review_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--dataset', choices=['examples', 'full'],
                        help='Fish-X LC-NE Test: bundled lc overview and five boxes (default), or full dataset')
    source.add_argument('--windows', type=Path, help='Custom local window manifest JSON')
    parser.add_argument('--data-root', type=Path, help='Extracted full dataset directory containing zfishlc_*')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare and validate data without starting the viewer')
    parser.add_argument('--no-starter-session', action='store_true',
                        help='Do not seed saved example prompts and traces into a fresh output directory')
    parser.add_argument('--checkpoint', help='SAM2 checkpoint; required unless --prepare-only')
    parser.add_argument('--model-config', default='sam2_hiera_t.yaml')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--sam2-runtime', choices=['torch', 'onnx'], default='torch')
    parser.add_argument('--sam2-onnx-dir', type=Path)
    parser.add_argument('--onnx-provider', choices=['cpu', 'coreml', 'cuda', 'directml'], default='cpu')
    parser.add_argument('--onnx-threads', type=int, default=0)
    parser.add_argument('--feature-cache-mb', type=float, default=512)
    parser.add_argument('--propagate-max-slices', type=int, default=64)
    parser.add_argument('--propagate-radius', type=int, default=4)
    parser.add_argument('--no-align-z', action='store_true')
    parser.add_argument('--max-voxels', type=int, default=128_000_000)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overview-cache', type=Path, help='Disk cache for requested whole-volume chunks')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args(argv)
    if not args.prepare_only and not args.checkpoint:
        parser.error('--checkpoint is required unless --prepare-only is used')
    if args.windows and args.data_root:
        parser.error('--data-root belongs to --dataset full, not --windows')
    output = args.output_dir.expanduser().resolve()
    supplied_dataset = args.windows is None
    try:
        manifest = (prepare_dataset(args.dataset or 'examples', output/'dataset', args.data_root)
                    if supplied_dataset else args.windows.expanduser().resolve())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    from probe_em.window_browser import WindowCollection, read_windows, serve_windows
    windows, initial = read_windows(manifest)
    if supplied_dataset:
        print(f'{DATASET_NAME}: {args.dataset or "examples"}; {len(windows)} views', flush=True)
    if args.prepare_only:
        print(f'Window manifest: {manifest}', flush=True)
        return

    import neuroglancer
    import numpy as np
    from demo import build_viewer
    from probe_em.demo_alignment import map_points
    from probe_em.prompt_models import PointModel
    from probe_em.precomputed_files import PrecomputedFiles
    from probe_em.overview import build_overview
    from probe_em.viewer_status import attach_toolbar_status
    neuroglancer.set_server_bind_address(bind_address=args.bind, bind_port=0)
    output.mkdir(parents=True, exist_ok=True)
    volumes = PrecomputedFiles(args.overview_cache or output/'overview_cache')
    overview_info = {}
    for index, item in enumerate(windows):
        if item['kind'] == 'overview':
            overview_info[index] = {
                kind: volumes.register(f'overview_{index}_{kind}', item[kind+'_path'])
                for kind in ('raw', 'seg') if item.get(kind+'_path')}
            overview_info[index]['images'] = {
                name: (volumes.register(f'overview_{index}_image_{number}', path),
                       f'overview_{index}_image_{number}')
                for number, (name, path) in enumerate(item['image_layers'].items())}
            overview_info[index]['labels'] = {
                name: (volumes.register(f'overview_{index}_label_{number}', path),
                       f'overview_{index}_label_{number}')
                for number, (name, path) in enumerate(item['label_layers'].items())}
    options = dict(backend='sam2', checkpoint=args.checkpoint, model_config=args.model_config, device=args.device,
                   sam2_runtime=args.sam2_runtime, onnx_dir=args.sam2_onnx_dir, onnx_provider=args.onnx_provider,
                   onnx_threads=args.onnx_threads, feature_cache_mb=args.feature_cache_mb)
    # Weights/features are shared, but each controller owns its prompts and fresh
    # propagation state. PointModel serializes calls across window workers.
    model = PointModel(**options)

    def build(index, item):
        folder = review_folder(output, item, index, supplied_dataset=supplied_dataset)
        if item['kind'] == 'overview':
            sources = {kind: f'precomputed://{url}data/overview_{index}_{kind}'
                       for kind in ('raw', 'seg') if item.get(kind+'_path')}
            viewer, report = build_overview(item, windows, overview_info[index].get('raw'), overview_info[index]['seg'],
                                           sources.get('raw'), sources['seg'], image_sources={
                                               name: (info, f'precomputed://{url}data/{mount}')
                                               for name, (info, mount) in overview_info[index]['images'].items()},
                                           label_sources={
                                               name: (info, f'precomputed://{url}data/{mount}')
                                               for name, (info, mount) in overview_info[index]['labels'].items()})
            publish(viewer, report, folder, item, index)
            return viewer
        viewer, report = build_viewer(item['local_dir'], item.get('traced_ids', []), item['seed'],
            folder/'reviews', args.max_voxels, align_z=not args.no_align_z,
            align_field=item.get('align_field') if not args.no_align_z else None,
            align_cache=folder/'alignment', mesh_cache=folder/'meshes',
            image_layers=item['image_layers'], label_layers=item['label_layers'],
            interactive=dict(options, output_dir=folder/'interactive', propagation_radius=args.propagate_radius,
                             propagation_max_slices=args.propagate_max_slices,
                             starter_session=None if args.no_starter_session else item.get('starter_session')))
        session = viewer.probe_em_interactive
        try:
            session.model = model
            with viewer.txn() as state:
                position = np.asarray(item.get('position_xyz', state.position), float)
                if 'position_xyz' in item:
                    if position.shape != (3,) or not np.isfinite(position).all():
                        raise ValueError('position_xyz must be three finite native voxel coordinates')
                    if np.any(position-session.offset < 0) or np.any(position-session.offset >= session.raw.shape):
                        raise ValueError('Initial position is outside this window')
                    position[2] = np.floor(position[2])
                    if session.mode['aligned']:
                        position = map_points(session.field, [position])[0]
                    position[2] += .5
                    state.position = position
                z = int(np.floor(position[2]))
                session.propagation_range = [max(int(session.offset[2]), z-args.propagate_radius),
                                              min(int(session.offset[2])+session.raw.shape[2]-1, z+args.propagate_radius)]
                state.cross_section_scale = max(session.raw.shape[:2])/470
                state.projection_scale = max(session.raw.shape[:2])*1.6
                state.projection_orientation = [.2, .3, 0, .9327379053088815]
                state.show_axis_lines = True
                state.show_default_annotations = False
                if 'visible_layers' in item:
                    names = {layer.name for layer in state.layers}
                    if set(item['visible_layers'])-names:
                        raise ValueError('visible_layers contains an unknown layer')
                    for name in (*item['image_layers'], *item['label_layers']):
                        state.layers[name].visible = name in item['visible_layers']
                for name, color in item.get('colors', {}).items():
                    if name not in item['label_layers']:
                        raise ValueError('Color entries must name auxiliary label layers')
                    state.layers[name].segment_colors = {'1': color}
                    state.layers[name].selected_alpha = .22 if name == 'ground_truth' else .35
                    state.layers[name].object_alpha = .35 if name == 'ground_truth' else .8
            from probe_em.starter_session import restore_starter_view
            if not restore_starter_view(session, viewer.probe_em_starter_state):
                session.status('D/F: previous/next slice. Add P/N points; I predicts; V propagates. Each test window retains its points.')
            report['interactive']['propagation_range_inclusive'] = session.propagation_range
            publish(viewer, report, folder, item, index)
        except BaseException:
            session.close()
            raise
        return viewer

    def publish(viewer, report, folder, item, index):
        attach_toolbar_status(viewer)
        report['window'] = item['name']
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (folder/'viewer_url.txt').write_text(url+f'#window={index}\n', encoding='utf-8')
        (folder/'neuroglancer_url.txt').write_text(viewer.get_viewer_url()+'\n', encoding='utf-8')

    collection = WindowCollection(windows, build, initial,
                                  dataset_name=DATASET_NAME if supplied_dataset else None)
    server, url = serve_windows(collection, args.bind, args.port, volumes=volumes)
    (output/'viewer_url.txt').write_text(url+'\n')
    print(f'SAEM²-SAEvM³ window browser: {url}', flush=True)
    collection.request(initial)
    if args.open:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        collection.close()
        server.shutdown()
        server.server_close()
        neuroglancer.stop()


if __name__ == '__main__':
    main()
