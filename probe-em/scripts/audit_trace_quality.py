"""Rerun the same five cached ROIs and expose baseline/guarded/review layers.

No source volume or old experiment is modified. Counts below are diagnostics,
not precision/recall: these ROIs do not have proofread ground truth.
"""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import sys
import threading
import webbrowser
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import load_local, build_viewer
from run_probe_em import DEFAULT_CONFIG, atomic_json, run_one_seed


def read_ids(folder, seed):
    return [int(value) for value in (folder / f'trace_{seed}_ids.txt').read_text().split()]


def baseline_audit(cache, folder, seed):
    labels, resolution, offset = load_local(cache / 'seg')
    ids = read_ids(folder, seed)
    segments = []
    for uid in ids:
        mask = labels == uid
        components, count = ndi.label(mask, np.ones((3, 3, 3), bool))
        sizes = np.sort(np.bincount(components.ravel())[1:])[::-1]
        points = np.column_stack(np.where(mask))
        segments.append({'id': str(uid), 'voxels': int(mask.sum()), 'components_26': int(count),
                         'largest_components': sizes[:10].tolist(),
                         'touches_roi': bool(np.any((points == 0) | (points == np.array(labels.shape)-1)))})
    hss = set()
    contact_total = contact_misses = 0
    for path in folder.glob('temp_vis_*/neighbors.csv'):
        for row in csv.DictReader(path.open()):
            uid = int(row['neighbor_id'])
            hss.add(uid)
            point = np.array([int(float(row['contact_'+axis])) for axis in 'xyz'])-offset
            if np.all(point >= 0) and np.all(point < labels.shape):
                contact_total += 1
                contact_misses += int(int(labels[tuple(point)]) != uid)
    asp = set()
    for path in folder.glob('temp_vis3d_*/3d_collision_report.csv'):
        asp.update(int(row['neighbor_id']) for row in csv.DictReader(path.open()) if float(row['score']) > .6)
    status_path = folder / 'status.json'
    return {'seed': str(seed), 'ids': [str(uid) for uid in ids], 'segments': segments,
            'status': json.loads(status_path.read_text()) if status_path.exists() else {'status': 'legacy_no_status'},
            'asp_accepted': [str(uid) for uid in sorted(asp)],
            'asp_outside_hss': [str(uid) for uid in sorted(asp-hss)],
            'shape_xyz': list(labels.shape), 'resolution_xyz_nm': resolution.tolist(),
            'offset_xyz': offset.tolist(), 'ground_truth_available': False,
            'seg_sha256_xyz_uint64': hashlib.sha256(np.ascontiguousarray(labels, dtype=np.uint64)).hexdigest(),
            'contact_queries_in_roi': contact_total, 'contacts_not_on_nominated_label': contact_misses}


def prepare(args):
    from export_savem3 import export_bundle
    from probe_em.demo_alignment import prepare_field
    root = args.output.resolve()
    for name in args.cases:
        directory = root / name
        if (directory / 'case.json').exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        if name == 'original':
            cache, seed = args.single_cache.resolve(), 1486287284
            old_folder = args.single_results.resolve() / f'{seed}_results_sam'
            raw, res, offset = load_local(cache / 'raw')
            labels, _, _ = load_local(cache / 'seg')
            # The historical single cache lacks precomputed ROI skeletons.
            export_bundle(raw.transpose(2, 1, 0), labels.transpose(2, 1, 0),
                          directory / 'cache', res, offset, skeletonize=True)
            audit = baseline_audit(cache, old_folder, seed)
            cache = directory / 'cache'
            _, alignment = prepare_field(raw, res, offset, cache_dir=directory / 'alignment')
            field = alignment['field_path']
        else:
            region = args.regions.resolve() / name
            metadata = json.loads((region / 'region.json').read_text())
            seed, cache = int(metadata['seed']), region / 'cache'
            old_folder = region / 'trace_results' / f'{seed}_results_sam'
            field = json.loads((region / 'alignment_optimized/report.json').read_text())['alignment']['field_path']
            audit = baseline_audit(cache, old_folder, seed)
        config = dict(DEFAULT_CONFIG, raw_path=(cache/'raw').as_uri(), seg_path=(cache/'seg').as_uri(),
                      checkpoint_sam=str(args.checkpoint.resolve()), model_cfg_sam=args.model_config,
                      target_mip=0, debug_limit=args.node_limit, seed_ids=[seed], device='cpu',
                      slice_workers=2, max_workers=1, align_z=True, align_z_field=field,
                      verification_policy='guarded', keep_inputs=True,
                      output_root=str(directory/'trace_results'), resume=False)
        atomic_json(directory/'trace_config.json', config)
        atomic_json(directory/'baseline.json', audit)
        atomic_json(directory/'case.json', {'name': name, 'seed': str(seed), 'cache': str(cache),
                                           'old_results': str(old_folder), 'alignment_field': field,
                                           'raw_source': 'precomputed://https://ng.zebrafish.digital-brain.cn/srv/raw/',
                                           'model_scope': 'generic SAM2; not EM-finetuned NeuroSAM2'})
        print(f'Prepared {name}; same cached ROI, seed={seed}', flush=True)


def summarize(directory):
    case = json.loads((directory/'case.json').read_text())
    seed = int(case['seed'])
    evidence_root = directory/'trace_results'/f'{seed}_results_sam'
    folder = Path(case.get('active_results', str(evidence_root)))
    status = json.loads((folder/'status.json').read_text())
    accepted = set(read_ids(folder, seed))
    baseline = json.loads((directory/'baseline.json').read_text())
    state = json.loads((folder/'trace_state.json').read_text())
    completed = set(state['visited'])
    proposals, reasons, preparations = set(), {}, {}
    topology_report = directory/'candidate_topology.json'
    if topology_report.exists():
        proposals.update(int(row['segment']) for row in json.loads(topology_report.read_text()) if row['rejected'])
    for path in [*evidence_root.glob('topology_*.json'), evidence_root/'resumed_topology.json']:
        if not path.is_file():
            continue
        if path.name.startswith('topology_') and path.stem.split('_')[-1] not in completed:
            continue
        proposals.update(int(row['segment']) for row in json.loads(path.read_text()) if row['rejected'])
    for path in evidence_root.glob('temp_vis3d_*/*_evidence.json'):
        if path.parent.name.split('_')[-1] not in completed:
            continue
        for item in json.loads(path.read_text()):
            if item['paper_accepted'] and not item['is_connected']:
                proposals.add(int(item['neighbor_id']))
            for reason in item['reasons']:
                reasons[reason] = reasons.get(reason, 0)+1
    for path in evidence_root.glob('temp_vis_*/preparation.json'):
        if path.parent.name.split('_')[-1] not in completed:
            continue
        for item in json.loads(path.read_text()):
            key = item.get('reason', item['status'])
            preparations[key] = preparations.get(key, 0)+1
    comparison = {'case': case['name'], 'status': status, 'baseline_ids': baseline['ids'],
                  'guarded_ids': [str(uid) for uid in sorted(accepted)],
                  'review_ids': [str(uid) for uid in sorted(proposals-accepted)],
                  'removed_from_baseline': [str(uid) for uid in sorted(set(map(int, baseline['ids']))-accepted)],
                  'added_to_baseline': [str(uid) for uid in sorted(accepted-set(map(int, baseline['ids'])))],
                  'asp_review_reasons': reasons, 'preparation_counts': preparations,
                  'completed_nodes': len(completed), 'pending_ids': state['pending'],
                  'result_directory': str(folder), 'evidence_directory': str(evidence_root),
                  'accuracy_claim': 'No ground truth: these are decision changes, not measured accuracy gains.'}
    atomic_json(directory/'comparison.json', comparison)
    return comparison


def snapshot_checkpoint(directory, topology=False):
    """Publish only the last atomic node checkpoint after the worker has stopped.

    Partial-node inputs/evidence remain in the raw run for diagnosis/resume;
    they do not influence the snapshot's IDs, proposals or completion counts.
    """
    import networkx as nx
    from run_probe_em import NeuronTracer
    case = json.loads((directory/'case.json').read_text())
    seed = int(case['seed'])
    source = directory/'trace_results'/f'{seed}_results_sam'
    state = json.loads((source/'trace_state.json').read_text())
    previous = json.loads((source/'status.json').read_text())
    if state['errors'] or not state['visited']:
        raise ValueError('A snapshot needs an error-free completed-node checkpoint')
    rejected = []
    if topology:
        from probe_em.topology import candidate_topology
        config = json.loads((directory/'trace_config.json').read_text())
        audit = [candidate_topology(int(uid), 0, config['seg_path']) for uid in state['accepted'] if int(uid) != seed]
        atomic_json(directory/'candidate_topology.json', audit)
        rejected = [row['segment'] for row in audit if row['rejected']]
        graph = nx.DiGraph()
        graph.add_nodes_from(state['accepted'])
        graph.add_edges_from(state['edges'])
        graph.remove_nodes_from(rejected)
        reachable = nx.descendants(graph, str(seed)) | {str(seed)}
        state['accepted'] = sorted(reachable, key=int)
        state['pending'] = [uid for uid in state['pending'] if uid in reachable]
        state['edges'] = [[a, b] for a, b in graph.edges if a in reachable and b in reachable]
        state['topology_rejected'] = rejected
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = directory/'checkpoints'/stamp/f'{seed}_results_sam'
    folder.mkdir(parents=True, exist_ok=False)
    tracer = NeuronTracer('unused', 'unused', device='cpu')
    tracer.all_merged_ids = set(map(int, state['accepted']))
    tracer.graph = nx.DiGraph()
    tracer.graph.add_nodes_from(tracer.all_merged_ids)
    tracer.graph.add_edges_from((int(a), int(b)) for a, b in state['edges'])
    tracer.save_results(seed, folder)
    atomic_json(folder/'trace_state.json', state)
    status = dict(previous, status='limited' if state['pending'] else 'complete',
                  stopped_at_checkpoint=bool(state['pending']),
                  accepted_count=len(state['accepted']), pending_count=len(state['pending']),
                  completed_nodes=len(state['visited']), errors=[],
                  scope='Completed-node prefix; uncheckpointed inference excluded',
                  topology_rejected=rejected,
                  source_results=str(source))
    atomic_json(folder/'status.json', status)
    if not topology:
        atomic_json(source/'status.json', status)
    case['active_results'] = str(folder)
    atomic_json(directory/'case.json', case)
    print(json.dumps(summarize(directory)), flush=True)


def validate_geometry(directory):
    """Compare tracing and display on three maximum-displacement sections."""
    from cloudvolume import CloudVolume
    from probe_em.z_align import AlignedVolume, ZAffineField, warp_slice
    from probe_em.demo_alignment import registered_bounds, translation
    case = json.loads((directory/'case.json').read_text())
    volume = CloudVolume((Path(case['cache'])/'seg').as_uri(), mip=0, fill_missing=False)
    field = ZAffineField.load(case['alignment_field'])
    aligned = AlignedVolume(volume, field)
    shape, offset = registered_bounds(volume.bounds.size(), volume.bounds.minpt, field, 128_000_000)
    sections = sorted(range(int(volume.bounds.minpt[2]), int(volume.bounds.maxpt[2])),
                      key=lambda z: np.linalg.norm(field.transform(z)[:2, 2]), reverse=True)[:3]
    results = []
    for z in sections:
        lo, hi = np.asarray(volume.bounds.minpt), np.asarray(volume.bounds.maxpt)
        source = np.asarray(volume[lo[0]:hi[0], lo[1]:hi[1], z:z+1])[:, :, 0, 0]
        matrix = translation(-offset[:2]) @ field.transform(z) @ translation(lo[:2])
        expected = warp_slice(source, matrix, seg=True, output_shape=shape[:2])
        actual = aligned[offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1], z:z+1][:, :, 0, 0]
        equal = np.array_equal(expected, actual)
        if not equal:
            raise ValueError(f'Trace/display mismatch: {case["name"]}, z={z}')
        results.append({'z': z, 'exact_native_label_match': bool(equal)})
    report = {'case': case['name'], 'sections': results, 'ground_truth_metric': False}
    atomic_json(directory/'geometry_validation.json', report)
    return report


def serve(args):
    import neuroglancer
    neuroglancer.set_server_bind_address(bind_address='127.0.0.1', bind_port=args.port)
    viewers, reports = [], []
    def ready_cases():
        pending = set(args.cases)
        while pending:
            for name in args.cases:
                if name not in pending:
                    continue
                if args.wait_for_qc:
                    directory = args.output.resolve()/name
                    case = json.loads((directory/'case.json').read_text())
                    if not case.get('active_results') or not (directory/'candidate_topology.json').is_file():
                        continue
                yield name
                pending.remove(name)
            if pending:
                threading.Event().wait(5)
    for name in ready_cases():
        directory = args.output.resolve()/name
        case = json.loads((directory/'case.json').read_text())
        comparison = summarize(directory)
        if comparison['status']['status'] not in ('limited', 'complete'):
            raise ValueError(f'Cannot display failed trace as successful: {name}')
        viewer, report = build_viewer(Path(case['cache']), list(map(int, comparison['guarded_ids'])),
                                     int(case['seed']), directory/'reviews',
                                     align_field=case['alignment_field'], mesh_cache=directory/'mesh_cache',
                                     comparison_ids={'baseline_trace': comparison['baseline_ids'],
                                                     'uncertain_proposals': comparison['review_ids']})
        with viewer.config_state.txn() as state:
            state.status_messages['quality'] = (f'{name}: guarded tracing; {comparison["status"]["status"]}. '
                'Baseline and uncertain proposal layers are hidden; toggle to compare. '
                'Generic SAM2, no ground truth; this is a proofreading demo, not a corrected neuron.')
        report.update(comparison=comparison, viewer_url=viewer.get_viewer_url(), region={'name': name})
        report['viewer_directory'] = str(directory/'viewer')
        atomic_json(directory/'viewer/report.json', report)
        atomic_json(directory/'viewer/viewer_state.json', viewer.state.to_json())
        (directory/'viewer/viewer_url.txt').write_text(report['viewer_url']+'\n', encoding='utf-8')
        viewers.append(viewer)
        reports.append(report)
        atomic_json(args.output/'viewers.json', sorted(reports, key=lambda item: args.cases.index(item['region']['name'])))
        print(f'[{name}] {report["viewer_url"]}', flush=True)
        if args.open_browser:
            webbrowser.open_new_tab(report['viewer_url'])
    atomic_json(args.output/'viewers.json', sorted(reports, key=lambda item: args.cases.index(item['region']['name'])))
    threading.Event().wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'summarize', 'serve', 'baseline', 'geometry', 'snapshot', 'topology'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regions', type=Path, default=Path('demo_session_regions'))
    parser.add_argument('--single-cache', type=Path, default=Path('volumes_local'))
    parser.add_argument('--single-results', type=Path, default=Path('trace_results'))
    parser.add_argument('--cases', nargs='+', choices=('original', 'region_1', 'region_2', 'region_3', 'region_4'),
                        default=['original', 'region_1', 'region_2', 'region_3', 'region_4'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--model-config', default='sam2_hiera_t.yaml')
    parser.add_argument('--node-limit', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--open-browser', action='store_true')
    parser.add_argument('--wait-for-qc', action='store_true', help='Serve each case after its topology-checked snapshot is ready')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.checkpoint is None or not args.checkpoint.is_file():
            parser.error('prepare needs an existing --checkpoint')
        prepare(args)
    elif args.command == 'run':
        import torch
        torch.set_num_threads(2)
        for name in args.cases:
            directory = args.output.resolve()/name
            config = json.loads((directory/'trace_config.json').read_text())
            config.update(resume=args.resume, debug_limit=args.node_limit)
            result = run_one_seed(config['seed_ids'][0], config)
            print(name, result, flush=True)
            if result[1] == 'failed' or result[1].startswith('error:'):
                raise RuntimeError(f'Failed: {name}; inspect status.json')
            case = json.loads((directory/'case.json').read_text())
            case.pop('active_results', None)
            atomic_json(directory/'case.json', case)
            print(json.dumps(summarize(directory)), flush=True)
    elif args.command == 'snapshot':
        for name in args.cases:
            snapshot_checkpoint(args.output.resolve()/name)
    elif args.command == 'topology':
        for name in args.cases:
            snapshot_checkpoint(args.output.resolve()/name, topology=True)
    elif args.command == 'geometry':
        atomic_json(args.output/'geometry_validation.json', [validate_geometry(args.output/name) for name in args.cases])
    elif args.command == 'baseline':
        for name in args.cases:
            directory = args.output/name
            case = json.loads((directory/'case.json').read_text())
            atomic_json(directory/'baseline.json', baseline_audit(
                Path(case['cache']), Path(case['old_results']), int(case['seed'])))
    elif args.command == 'summarize':
        atomic_json(args.output/'comparison.json', [summarize(args.output/name) for name in args.cases])
    else:
        serve(args)


if __name__ == '__main__':
    main()
