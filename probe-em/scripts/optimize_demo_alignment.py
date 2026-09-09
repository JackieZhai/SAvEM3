"""Validate optimized display registration, leaving original trace decisions intact."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import load_local, read_trace_ids
from demo_regions import alignment_quality, array_hash, write_json
from probe_em.demo_alignment import prepare_field, register_arrays
from probe_em.z_align import ZAffineField, warp_slice
from probe_em.registration import ncc


def common_support_scores(raw, offset, baseline, optimized):
    """Use identical pixels for original, old and new transforms in each pair."""
    shift = np.eye(3)
    shift[:2, 2] = offset[:2]
    ones = np.ones(raw.shape[:2], np.uint8)
    pairs = []
    for index in range(1, raw.shape[2]):
        z = int(offset[2]) + index
        matrices = [np.linalg.inv(shift) @ np.linalg.inv(field.transform(z-1)) @ field.transform(z) @ shift
                    for field in (baseline, optimized)]
        valid = np.logical_and.reduce([warp_slice(ones, matrix, seg=True) > 0 for matrix in matrices])[::2, ::2]
        reference = raw[:, :, index-1][::2, ::2][valid]
        versions = [raw[:, :, index]] + [warp_slice(raw[:, :, index], matrix, border_mode='constant')
                                         for matrix in matrices]
        scores = [ncc(reference, version[::2, ::2][valid]) for version in versions]
        pairs.append({'z': z, 'original': scores[0], 'v2': scores[1], 'v3': scores[2],
                      'shared_valid_fraction': float(valid.mean())})
    return {'means': {name: float(np.mean([pair[name] for pair in pairs]))
                      for name in ('original', 'v2', 'v3', 'shared_valid_fraction')}, 'pairs': pairs}


def comparison_figure(raw, offset, resolution, baseline, optimized, output):
    # Analytical resampling of real EM data, not a generated/illustrative image.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(raw.shape[0]*.15, raw.shape[0]*.85) + offset[0]
    y = float(offset[1] + raw.shape[1]/2)
    coordinates = np.stack([x, np.full(len(x), y), np.ones(len(x))])
    sections = []
    for field in (None, baseline, optimized):
        rows = []
        for index in range(raw.shape[2]):
            z = int(offset[2])+index
            mapped = coordinates if field is None else np.linalg.inv(field.transform(z)) @ coordinates
            rows.append(cv2.remap(raw[:, :, index].T,
                                 (mapped[0]-offset[0]).astype(np.float32)[None],
                                 (mapped[1]-offset[1]).astype(np.float32)[None],
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0])
        sections.append(np.stack(rows))
    figure, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    for axis, image, title in zip(axes, sections, ('Original', 'Previous v2', 'Optimized v3')):
        axis.imshow(image, cmap='gray', vmin=0, vmax=255, aspect=resolution[2]/resolution[0])
        axis.set_title(title)
        axis.set_xlabel('X (same reference coordinates)')
        axis.set_ylabel('Z sections')
    figure.suptitle(f'{output.parents[1].name}: XZ, reference y={y:g}; same contrast and scale')
    figure.savefig(output, dpi=150)
    plt.close(figure)


def optimize(directory):
    raw, resolution, offset = load_local(directory / 'cache' / 'raw')
    seg, _, _ = load_local(directory / 'cache' / 'seg')
    metadata = json.loads((directory / 'region.json').read_text())
    output = directory / 'alignment_optimized'
    field, info = prepare_field(raw, resolution, offset, cache_dir=output)
    old_field = ZAffineField.load(metadata['alignment']['field_path'])
    baseline = alignment_quality(raw, offset, old_field)
    improved = alignment_quality(raw, offset, field)
    comparison = common_support_scores(raw, offset, old_field, field)
    _, aligned_seg, aligned_offset = register_arrays(raw, seg, offset, field)
    ids, trace_folder = read_trace_ids(directory / 'trace_results', int(metadata['seed']))
    present = set(map(int, np.unique(aligned_seg)))
    if set(ids)-present:
        raise ValueError('Optimized display registration lost trace labels')
    if comparison['means']['v3'] < comparison['means']['v2']-.005:
        raise ValueError('Optimized registration regressed the adjacent-image sanity check')
    report = {'validated': True, 'region': metadata['region'], 'seed': metadata['seed'],
              'raw_sha256': array_hash(raw), 'original_field': metadata['alignment']['field_path'],
              'alignment': info, 'baseline': baseline, 'optimized': improved,
              'common_support_comparison': comparison,
              'registered_shape_xyz': list(aligned_seg.shape), 'registered_offset_xyz': aligned_offset.tolist(),
              'trace_ids_preserved': [str(value) for value in ids],
              'trace_results_unchanged': str(trace_folder),
              'scope': 'Display registration only; original one-node PEC/ASP decisions held fixed for comparison'}
    write_json(output / 'report.json', report)
    print(json.dumps({'region': directory.name, 'before': baseline['mean_before'],
                      'v2': baseline['mean_after'], 'v3': improved['mean_after'],
                      'adjacent_rejected': len(info['rejected_pair_slices']),
                      'quality': info['quality_summary']}), flush=True)
    return report


if __name__ == '__main__':
    cv2.setNumThreads(1)
    root = Path(sys.argv[1] if len(sys.argv)>1 else 'demo_session_regions').resolve()
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(optimize, sorted(root.glob('region_*'))))
    write_json(root / 'optimized_alignment.json', reports)
    # Matplotlib is not thread-safe; produce comparison figures serially.
    for report in reports:
        directory = root / report['region']['name']
        raw, resolution, offset = load_local(directory / 'cache' / 'raw')
        comparison_figure(raw, offset, resolution, ZAffineField.load(report['original_field']),
                          ZAffineField.load(report['alignment']['field_path']),
                          directory / 'alignment_optimized' / 'comparison_xz.png')
