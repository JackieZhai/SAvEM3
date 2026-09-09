"""Boundary → slice watershed → 2D multicut → WaterZ / local 3D multicut.

The historical 'LMC' output name is retained, but no lifted edges are added.
See README.md for probability polarity and coordinate conventions.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.volume import (load_volume, save_volume, prediction_maps, compact_labels,
                          unique_slice_labels, edge_affinity_means, validate_labels)

load_h5_main = load_volume


def watershed_oversegment(affs, threshold=0.25, sigma_seeds=2.0):
    from elf.segmentation.watershed import distance_transform_watershed
    boundary, _ = prediction_maps(affs)
    sections = [distance_transform_watershed(section, threshold=threshold,
                sigma_seeds=sigma_seeds)[0] for section in boundary]
    return unique_slice_labels(np.stack(sections).astype(np.uint64))


def project_foreground(rag, node_labels, fragments):
    from elf.segmentation.features import project_node_labels_to_pixels
    result = project_node_labels_to_pixels(rag, np.asarray(node_labels, dtype=np.uint64) + 1)
    result[fragments == 0] = 0
    return compact_labels(result)


def protect_background(costs, uv):
    costs = np.array(costs, dtype=np.float64, copy=True)
    costs[np.any(uv == 0, axis=1)] = -max(float(np.abs(costs).sum()) + 1, 1.0)
    return costs


def premerge_2d_multicut(fragments, bmap, beta=0.5):
    import elf.segmentation.features as features
    import elf.segmentation.multicut as multicut
    validate_labels(fragments)
    if bmap.shape != fragments.shape:
        raise ValueError("Boundary and fragment shapes differ")
    out = np.zeros_like(fragments, dtype=np.uint64)
    for z in range(len(fragments)):
        frag = compact_labels(fragments[z])
        if not np.any(frag):
            continue
        rag = features.compute_rag(frag)
        if rag.numberOfEdges == 0:
            out[z] = frag
            continue
        statistics = features.compute_boundary_mean_and_length(rag, bmap[z].astype(np.float32))
        costs = multicut.transform_probabilities_to_costs(statistics[:, 0],
                                                         edge_sizes=statistics[:, -1], beta=beta)
        costs = protect_background(costs, rag.uvIds())
        nodes = multicut.multicut_kernighan_lin(rag, costs)
        out[z] = project_foreground(rag, nodes, frag)
    return unique_slice_labels(out)


def waterz_agglomerate(affs, fragments, thresholds=(0.3, 0.4, 0.5)):
    import waterz
    validate_labels(fragments)
    _, affinity = prediction_maps(affs, "affinity")
    if fragments.shape != affinity.shape[1:]:
        raise ValueError("Affinity and fragment shapes differ")
    fragments = compact_labels(fragments)
    thresholds = list(thresholds)
    if thresholds != sorted(thresholds) or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError("WaterZ thresholds must be sorted and in [0, 1]")
    scoring = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    results = {}
    for threshold, result in zip(thresholds, waterz.agglomerate(
            np.ascontiguousarray(affinity, dtype=np.float32), thresholds,
            fragments=np.ascontiguousarray(fragments, dtype=np.uint64),
            scoring_function=scoring, discretize_queue=256)):
        # WaterZ may reuse its output buffer between yields.
        result = result.astype(np.uint64, copy=True)
        result[fragments == 0] = 0
        results[threshold] = result
    return results


def lmc_agglomerate(affs, fragments=None, edge_feats=None, weights=None, beta=0.5):
    import elf.segmentation.features as features
    import elf.segmentation.multicut as multicut
    boundary, affinity = prediction_maps(affs)
    if fragments is None:
        fragments = watershed_oversegment(boundary)
    validate_labels(fragments)
    if not np.any(fragments):
        return fragments.astype(np.uint64)
    original_ids = np.unique(fragments)
    if original_ids[0] != 0:
        original_ids = np.concatenate((np.zeros(1, dtype=original_ids.dtype), original_ids))
    fragments = compact_labels(fragments)
    rag = features.compute_rag(fragments)
    uv = rag.uvIds()
    if not len(uv):
        return compact_labels(fragments)
    mean_affinity, _ = edge_affinity_means(fragments, affinity, uv)
    # ELF expects boundary / cut probability, not affinity.
    costs = multicut.transform_probabilities_to_costs(1.0 - mean_affinity, beta=beta)
    weights = weights or {"cos": 1.0, "iou": 1.0, "ioa": 0.0, "iob": 0.0}
    if edge_feats:
        for index, (u, v) in enumerate(uv):
            u, v = original_ids[int(u)], original_ids[int(v)]
            evidence = edge_feats.get(f"{u},{v}", edge_feats.get(f"{v},{u}", {}))
            # Positive costs penalize a cut: similarity must be ADDED.
            costs[index] += sum(weight * evidence.get(name, 0.0) for name, weight in weights.items())
    costs = protect_background(costs, uv)
    nodes = multicut.multicut_kernighan_lin(rag, costs)
    return project_foreground(rag, nodes, fragments)


def evaluate(seg, gt):
    from skimage.metrics import adapted_rand_error, variation_of_information
    validate_labels(seg)
    validate_labels(gt)
    if seg.shape != gt.shape or not np.any(gt):
        raise ValueError("Evaluation requires matching shapes and nonempty foreground GT")
    arand = adapted_rand_error(gt, seg, ignore_labels=(0,))[0]
    split, merge = variation_of_information(gt, seg, ignore_labels=(0,))
    return float(arand), float(split + merge), float(split), float(merge)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--affs', required=True, help='ZYX / 1ZYX boundary, or 3ZYX affinity H5/TIFF/NPY')
    parser.add_argument('--input-kind', choices=['auto', 'boundary', 'affinity'], default='auto')
    parser.add_argument('--gt')
    parser.add_argument('--out-dir', default='./postprocess')
    parser.add_argument('--ws-threshold', type=float, default=0.25)
    parser.add_argument('--sigma-seeds', type=float, default=2.0)
    parser.add_argument('--thresholds', default='0.3,0.4,0.5')
    parser.add_argument('--skip-2d-mc', action='store_true')
    parser.add_argument('--method', choices=['both', 'waterz', 'multicut'], default='both')
    args = parser.parse_args()
    boundary, affinity = prediction_maps(load_volume(args.affs), args.input_kind)
    fragments = watershed_oversegment(boundary, args.ws_threshold, args.sigma_seeds)
    output = Path(args.out_dir)
    save_volume(output / 'fragments_ws.h5', fragments)
    if not args.skip_2d_mc:
        fragments = premerge_2d_multicut(fragments, boundary)
        save_volume(output / 'seg_mc2d.hdf', fragments)
    save_volume(output / 'fragments.h5', fragments)
    save_volume(output / 'affinities.h5', affinity, kind='affinity', offsets='-z,-y,-x')
    results = {}
    if args.method in ('both', 'waterz'):
        for threshold, seg in waterz_agglomerate(affinity, fragments,
                                                [float(v) for v in args.thresholds.split(',')]).items():
            results[f'seg_waterz_t{threshold}'] = seg
    if args.method in ('both', 'multicut'):
        results['seg_lmc'] = lmc_agglomerate(affinity, fragments)
    report = {'input': str(Path(args.affs).resolve()), 'input_kind': args.input_kind,
              'shape_zyx': list(boundary.shape), 'multicut_solver': 'local RAG Kernighan-Lin',
              'premerge_2d': not args.skip_2d_mc, 'metrics': {}}
    gt = load_volume(args.gt) if args.gt else None
    for name, seg in results.items():
        save_volume(output / f'{name}.hdf', seg)
        if gt is not None:
            values = evaluate(seg, gt)
            report['metrics'][name] = dict(zip(['arand', 'voi', 'voi_split', 'voi_merge'], values))
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
