"""Paired SAEM2 / HQ-SAM / SAM2 image trials on the five existing cached ROIs.

Only one interior point per section comes from the existing seed label. Label
agreement is NOT accuracy. Independent slice masks are NOT a tracked neuron.
No online labels or previous trace/viewer artifacts are modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import webbrowser

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'probe-em'))
from demo import load_local
from run_probe_em import atomic_json

CASES = ('original', 'region_1', 'region_2', 'region_3', 'region_4')


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def interior_point(mask):
    """Use the largest connected seed component and a deterministic interior point."""
    parts, count = ndi.label(mask, np.ones((3, 3), bool))
    if count == 0:
        return None
    sizes = np.bincount(parts.ravel())
    sizes[0] = 0
    component = parts == sizes.argmax()
    distance = ndi.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
    row, col = np.unravel_index(distance.argmax(), distance.shape)
    return np.array([col, row], dtype=np.float32)


def agreement(mask, reference):
    mask, reference = np.asarray(mask), np.asarray(reference)
    if mask.shape != reference.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected same-grid binary masks, not logits/probabilities')
    mask, reference = mask.astype(bool), reference.astype(bool)
    intersection = int(np.count_nonzero(mask & reference))
    area, ref_area = int(mask.sum()), int(reference.sum())
    return {'pixels': area, 'input_seed_pixels': ref_area,
            'input_seed_iou_not_accuracy': intersection/max(1, int(np.count_nonzero(mask | reference))),
            'input_seed_coverage_not_accuracy': intersection/max(1, ref_area),
            'area_ratio_to_input_seed': area/max(1, ref_area)}


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a new trial directory; existing results are retained')
    try:
        _run(args)
    except (Exception, KeyboardInterrupt) as error:
        path = args.output/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
            atomic_json(path, manifest)
        raise


def _run(args):
    import torch
    from saem2.predictor import SAEM2Predictor
    from probe_em.runtime import PredictorCache
    from probe_em.demo_alignment import register_arrays
    from probe_em.z_align import ZAffineField
    if args.slices < 1 or args.slices % 2 != 1:
        raise ValueError('--slices must be positive and odd')
    torch.set_num_threads(2)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new trial directory; existing results are retained')
    output.mkdir(parents=True, exist_ok=False)
    weights = output/'weights'
    weights.mkdir()
    manifest = {'scope': 'Independent point-prompted 2D comparison, not 3D tracking or ground-truth evaluation',
                'checkpoint_provenance': args.provenance, 'model_type': args.model_type,
                'device': 'cpu', 'dtype': 'float32', 'torch_threads': 2,
                'sam2_model_config': args.sam2_config,
                'registration_enabled': args.align_z, 'slices_requested': args.slices,
                'prompt_policy': 'One interior point per section from the existing seed label; no mask/box from labels',
                'model_output': 'SAEM2 single SAM + HQ logits > 0; separate full-box membrane probability',
                'status': 'running', 'weights': {}}
    for key in ('decoder_checkpoint', 'hq_checkpoint', 'sam2_checkpoint'):
        path = getattr(args, key).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        target = weights/(key+path.suffix)
        shutil.copy2(path, target)
        manifest['weights'][key] = {'source': str(path), 'cache': str(target), 'sha256': fingerprint(target)}
    atomic_json(output/'manifest.json', manifest)
    predictor = SAEM2Predictor(manifest['weights']['hq_checkpoint']['cache'],
                              manifest['weights']['decoder_checkpoint']['cache'], args.model_type, 'cpu')
    sam2 = PredictorCache(manifest['weights']['sam2_checkpoint']['cache'], args.sam2_config, 'cpu').image()
    reports = []
    for name in args.cases:
        started = time.monotonic()
        case = json.loads((args.cases_root/name/'case.json').read_text())
        raw, resolution, offset = load_local(Path(case['cache'])/'raw')
        labels, seg_resolution, seg_offset = load_local(Path(case['cache'])/'seg')
        if (raw.shape != labels.shape or not np.array_equal(resolution, seg_resolution)
                or not np.array_equal(offset, seg_offset) or raw.dtype != np.uint8):
            raise ValueError('Expected matching native raw/label grids and uint8 EM')
        if args.align_z:
            field = ZAffineField.load(case['alignment_field'])
            raw, labels, offset = register_arrays(raw, labels, offset, field)
        seed = int(case['seed'])
        area_z = np.count_nonzero(labels == seed, axis=(0, 1))
        if not area_z.any():
            raise ValueError('Seed missing from trial ROI')
        center = int(area_z.argmax())
        start, end = max(0, center-args.slices//2), min(raw.shape[2], center+args.slices//2+1)
        offset = offset.copy()
        offset[2] += start
        raw, labels = raw[:, :, start:end], labels[:, :, start:end]
        masks = {key: np.zeros(raw.shape, np.uint8) for key in ('saem2', 'hq_sam', 'sam2')}
        membrane = np.zeros(raw.shape, np.float32)
        rows = []
        directory = output/name
        directory.mkdir()
        with torch.inference_mode():
            for z in range(raw.shape[2]):
                image = np.repeat(raw[:, :, z].T[..., None], 3, axis=2)
                reference = labels[:, :, z].T == seed
                point = interior_point(reference)
                predictor.set_image(image)
                # The membrane head is conditioned by a full-image box, not GT labels.
                membrane[:, :, z] = predictor.predict(box=[0, 0, image.shape[1], image.shape[0]])['membrane'].T
                record = {'z': int(offset[2]+z), 'point_xy': point.tolist() if point is not None else None}
                if point is None:
                    record['status'] = 'no_seed_prompt; masks not inferred'
                else:
                    prompt = {'point_coords': point[None], 'point_labels': np.ones(1, np.int32)}
                    saem2_result = predictor.predict(**prompt)['mask']
                    hq_result = predictor.backbone.predict(**prompt, multimask_output=False)[0][0]
                    sam2.set_image(image)
                    sam2_result = sam2.predict(**prompt, multimask_output=False)[0][0]
                    for key, mask in zip(masks, (saem2_result, hq_result, sam2_result)):
                        masks[key][:, :, z] = mask.T
                        record[key] = agreement(mask, reference)
                    record['status'] = 'paired_prediction'
                rows.append(record)
                atomic_json(directory/'slice_progress.json', rows)
                print(name, record['z'], record['status'], flush=True)
        np.savez_compressed(directory/'arrays.npz', raw=raw, input_seed=(labels == seed).astype(np.uint8),
                            membrane=membrane, **masks)
        report = {'case': name, 'seed': case['seed'], 'source_cache': case['cache'],
                  'raw_source': case['raw_source'], 'resolution_xyz_nm': resolution.tolist(),
                  'voxel_offset_xyz': offset.tolist(), 'shape_xyz': list(raw.shape),
                  'alignment_field': case['alignment_field'] if args.align_z else None,
                  'alignment_field_sha256': fingerprint(case['alignment_field']) if args.align_z else None,
                  'slices': rows, 'elapsed_seconds': time.monotonic()-started,
                  'status': 'complete_image_trial; not a tracked neuron'}
        atomic_json(directory/'report.json', report)
        reports.append(report)
        print(name, 'saved:', directory, 'seconds:', round(report['elapsed_seconds'], 1), flush=True)
    manifest['status'] = 'complete_image_trials'
    atomic_json(output/'manifest.json', manifest)
    atomic_json(output/'reports.json', reports)


def serve(args):
    import neuroglancer
    from probe_em.meshes import MeshVolume
    manifest = json.loads((args.output/'manifest.json').read_text())
    if manifest['status'] != 'complete_image_trials':
        raise ValueError('Cannot serve unfinished trials')
    neuroglancer.set_server_bind_address(bind_address='127.0.0.1', bind_port=args.port)
    viewers, reports = [], []
    for name in args.cases:
        directory = args.output.resolve()/name
        report = json.loads((directory/'report.json').read_text())
        with np.load(directory/'arrays.npz') as archive:
            data = {key: archive[key] for key in archive.files}
        if (directory/'membrane_seeded.npy').exists():
            data['membrane_seeded'] = np.load(directory/'membrane_seeded.npy')
        if (directory/'saem2_multicut/selected_xyz.npy').exists():
            post = json.loads((directory/'saem2_multicut/status.json').read_text())
            if post['status'] != 'complete_bounded_segmentation':
                raise ValueError('Cannot display incomplete multicut output')
            data['saem2_multicut'] = np.load(directory/'saem2_multicut/selected_xyz.npy')
            report['postprocessing'] = post
        dimensions = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=report['resolution_xyz_nm'])
        offset = np.array(report['voxel_offset_xyz'])
        viewer = neuroglancer.Viewer()
        meshes = {}
        with viewer.txn() as state:
            state.dimensions = dimensions
            state.layers['raw'] = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(
                data['raw'], dimensions=dimensions, voxel_offset=offset))
            state.layers['saem2_membrane'] = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(
                data['membrane'], dimensions=dimensions, voxel_offset=offset),
                shader='void main() { float v = toNormalized(getDataValue()); emitRGBA(vec4(v, 0.0, 0.0, v * 0.8)); }')
            state.layers['saem2_membrane'].visible = False
            mask_keys = ['input_seed', 'sam2', 'hq_sam', 'saem2']
            if 'membrane_seeded' in data:
                mask_keys.append('membrane_seeded')
            if 'saem2_multicut' in data:
                mask_keys.append('saem2_multicut')
            for key in mask_keys:
                source = MeshVolume(data[key], dimensions=dimensions, voxel_offset=offset,
                                    mesh_cache=directory/'mesh_cache')
                ids = [1] if data[key].any() else []
                meshes[key] = source.precompute_meshes(ids)
                layer_name = key+'_slice_masks' if key != 'saem2_multicut' else 'saem2_membrane_3d'
                state.layers[layer_name] = neuroglancer.SegmentationLayer(
                    source=source, segments=ids, hover_highlight=False, pick=False, selected_alpha=.35)
                state.layers[layer_name].visible = key == ('saem2_multicut' if 'saem2_multicut' in data else 'saem2')
            state.layout = '4panel'
            state.show_slices = False
            center = len(report['slices'])//2
            point = report['slices'][center]['point_xy']
            state.position = offset + [*(point or [data['raw'].shape[0]/2, data['raw'].shape[1]/2]), center]
        with viewer.config_state.txn() as config:
            config.status_messages['saem2'] = (
                f'{name}: SAEM2 ViT-B historical weights (training provenance unconfirmed). '
                f'Alignment: {"ON" if manifest["registration_enabled"] else "OFF"}. '
                'Small slab trial, NOT a complete neuron. 3D = membrane/watershed/local multicut. '
                'Other mask layers use independent slice prompts; threshold components can leak.')
        url = viewer.get_viewer_url()
        report.update(viewer_url=url, show_slices=False, meshes=meshes)
        atomic_json(directory/'viewer_state.json', viewer.state.to_json())
        atomic_json(directory/'viewer_report.json', report)
        reports.append(report)
        viewers.append(viewer)
        atomic_json(args.output/'viewers.json', reports)
        print(name, url, flush=True)
        if args.open_browser:
            webbrowser.open_new_tab(url)
    threading.Event().wait()


def membrane_seed(args):
    from cloudvolume import CloudVolume
    from saem2.membrane import seeded_component
    from probe_em.demo_alignment import translation
    from probe_em.z_align import ZAffineField, warp_slice
    for name in args.cases:
        directory = args.output/name
        if (directory/'membrane_seeded.npy').exists():
            raise FileExistsError('Existing membrane-seeded result is retained; use a new experiment directory')
        report = json.loads((directory/'report.json').read_text())
        source = CloudVolume((Path(report['source_cache'])/'raw').as_uri(), mip=0, fill_missing=False)
        source_valid = np.ones(tuple(source.bounds.size())[:2], np.uint8)
        source_offset = np.asarray(source.bounds.minpt)
        output_offset = np.array(report['voxel_offset_xyz'])
        field = ZAffineField.load(report['alignment_field']) if report['alignment_field'] else None
        with np.load(directory/'arrays.npz') as data:
            output = np.zeros(data['raw'].shape, np.uint8)
            rows = []
            for i, row in enumerate(report['slices']):
                matrix = translation(-output_offset[:2]) @ (field.transform(row['z']) if field else np.eye(3)) @ translation(source_offset[:2])
                valid = warp_slice(source_valid, matrix, seg=True, output_shape=output.shape[:2]).T.astype(bool)
                mask, info = seeded_component(data['membrane'][:, :, i].T, row['point_xy'], valid)
                output[:, :, i] = mask.T
                info.update(z=row['z'])
                if row['point_xy'] is not None:
                    info.update(agreement(mask, data['input_seed'][:, :, i].T))
                rows.append(info)
        np.save(directory/'membrane_seeded.npy', output)
        atomic_json(directory/'membrane_seed_report.json', {
            'algorithm': '4-connected probability<0.5 component at seed; enclosed-hole fill; valid-ROI clipping',
            'scope': 'Independent 2D baseline, not 3D tracking or a paper accuracy result', 'slices': rows})
        print(name, 'membrane components saved', flush=True)


def prepare_postprocess(args):
    from cloudvolume import CloudVolume
    from probe_em.demo_alignment import translation
    from probe_em.z_align import ZAffineField, warp_slice
    for name in args.cases:
        directory = args.output/name
        target = directory/'postprocess_inputs.npz'
        if target.exists():
            raise FileExistsError(target)
        report = json.loads((directory/'report.json').read_text())
        source = CloudVolume((Path(report['source_cache'])/'raw').as_uri(), mip=0, fill_missing=False)
        ones = np.ones(tuple(source.bounds.size())[:2], np.uint8)
        source_offset = np.asarray(source.bounds.minpt)
        output_offset = np.array(report['voxel_offset_xyz'])
        field = ZAffineField.load(report['alignment_field']) if report['alignment_field'] else None
        with np.load(directory/'arrays.npz') as data:
            valid = np.zeros_like(data['raw'], dtype=bool)
            for i, row in enumerate(report['slices']):
                matrix = translation(-output_offset[:2]) @ (field.transform(row['z']) if field else np.eye(3)) @ translation(source_offset[:2])
                valid[:, :, i] = warp_slice(ones, matrix, seg=True, output_shape=valid.shape[:2]).astype(bool)
            boundary = data['membrane'].copy()
            boundary[~valid] = 1
            # Only the central seed point initializes the 3D selection. Other
            # slice prompts/labels do not enter watershed or multicut.
            center = len(report['slices'])//2
            point = report['slices'][center]['point_xy']
            if point is None:
                raise ValueError('The central section has no seed prompt')
            np.savez_compressed(target, boundary=boundary.transpose(2, 1, 0),
                                valid=valid.transpose(2, 1, 0),
                                seed_zyx=np.array([center, point[1], point[0]], dtype=np.int64))
        print(name, 'ZYX postprocessing inputs saved', flush=True)


def summarize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = []
    for name in args.cases:
        directory = args.output/name
        report = json.loads((directory/'report.json').read_text())
        rows = [row for row in report['slices'] if row['status'] == 'paired_prediction']
        item = {'case': name, 'paired_slices': len(rows),
                'unprompted_slices': len(report['slices'])-len(rows), 'ground_truth_available': False}
        for key in ('saem2', 'hq_sam', 'sam2'):
            item[key] = {metric: float(np.median([row[key][metric] for row in rows]))
                         for metric in ('input_seed_iou_not_accuracy', 'area_ratio_to_input_seed')}
        extra_path = directory/'membrane_seed_report.json'
        if extra_path.exists():
            extra = json.loads(extra_path.read_text())['slices']
            selected = [row for row in extra if 'input_seed_iou_not_accuracy' in row]
            item['membrane_seeded'] = {metric: float(np.median([row[metric] for row in selected]))
                                      for metric in ('input_seed_iou_not_accuracy', 'area_ratio_to_input_seed')}
            item['membrane_roi_edge_warnings'] = sum(row.get('touches_valid_roi_boundary', False) for row in extra)
        with np.load(directory/'arrays.npz') as archive:
            data = {key: archive[key] for key in archive.files}
        if (directory/'membrane_seeded.npy').exists():
            data['membrane_seeded'] = np.load(directory/'membrane_seeded.npy')
        if (directory/'saem2_multicut/selected_xyz.npy').exists():
            data['saem2_multicut'] = np.load(directory/'saem2_multicut/selected_xyz.npy')
            post = json.loads((directory/'saem2_multicut/status.json').read_text())
            item['saem2_multicut'] = {**agreement(data['saem2_multicut'], data['input_seed']),
                                      'selected_sections': post['selected_sections'],
                                      'reaches_slab_end': post['reaches_first_or_last_section']}
        if data:
            z = max(range(len(report['slices'])), key=lambda i: int(data['input_seed'][:, :, i].sum()))
            point = np.array(report['slices'][z]['point_xy'])
            raw = data['raw'][:, :, z].T
            keys = ('raw', 'input_seed', 'hq_sam', 'sam2', 'saem2', 'membrane')
            if 'membrane_seeded' in data:
                keys += ('membrane_seeded',)
            if 'saem2_multicut' in data:
                keys += ('saem2_multicut',)
            fig, axes = plt.subplots(1, len(keys), figsize=(3*len(keys), 3.5))
            for axis, key in zip(axes, keys):
                axis.imshow(raw, cmap='gray', vmin=0, vmax=255)
                if key == 'membrane':
                    axis.imshow(data[key][:, :, z].T, cmap='magma', vmin=0, vmax=1, alpha=.7)
                elif key != 'raw':
                    mask = data[key][:, :, z].T.astype(bool)
                    overlay = np.zeros((*mask.shape, 4))
                    overlay[mask] = [0, 1, 0, .35]
                    axis.imshow(overlay)
                axis.scatter(*point, color='red', marker='+', s=40)
                axis.set_xlim(max(0, point[0]-90), min(raw.shape[1], point[0]+90))
                axis.set_ylim(min(raw.shape[0], point[1]+90), max(0, point[1]-90))
                axis.set_title(key if key != 'input_seed' else 'input seed (NOT GT)')
                axis.axis('off')
            fig.suptitle(f'{name}, z={report["slices"][z]["z"]}: same point; historical SAEM2 ViT-B; not tracing/accuracy')
            fig.tight_layout()
            fig.savefig(directory/'comparison.png', dpi=150)
            plt.close(fig)
        summary.append(item)
        print(json.dumps(item), flush=True)
    atomic_json(args.output/'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'serve', 'summarize', 'membrane-seed', 'prepare-postprocess'])
    parser.add_argument('--output', type=Path, default=ROOT/'probe-em/demo_session_saem2')
    parser.add_argument('--cases-root', type=Path, default=ROOT/'probe-em/demo_session_trace_quality')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--decoder-checkpoint', type=Path)
    parser.add_argument('--hq-checkpoint', type=Path)
    parser.add_argument('--sam2-checkpoint', type=Path)
    parser.add_argument('--sam2-config', default='sam2_hiera_t.yaml')
    parser.add_argument('--model-type', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b')
    parser.add_argument('--provenance', default='Historical checkpoint; training data/run not verified')
    parser.add_argument('--slices', type=int, default=9)
    parser.add_argument('--align-z', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    if args.command == 'run':
        if any(getattr(args, key) is None for key in ('decoder_checkpoint', 'hq_checkpoint', 'sam2_checkpoint')):
            parser.error('run requires decoder, HQ-SAM and SAM2 checkpoints')
        run(args)
    elif args.command == 'summarize':
        summarize(args)
    elif args.command == 'membrane-seed':
        membrane_seed(args)
    elif args.command == 'prepare-postprocess':
        prepare_postprocess(args)
    else:
        serve(args)


if __name__ == '__main__':
    main()
