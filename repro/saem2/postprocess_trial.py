"""SAEM2 membrane -> slice watershed -> local 3D multicut on cached trial slabs.

Uses the existing ELF postprocessing environment. No SAM2 or training dependencies
are imported. No old segmentation labels enter the graph or its costs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from repro.savem3.distill_postprocess import watershed_oversegment, lmc_agglomerate


def process(directory, threshold=.25, beta=.5):
    output = directory/'saem2_multicut'
    output.mkdir(exist_ok=False)
    status_path = output/'status.json'
    report = {'status': 'running', 'method': 'slice watershed + local 3D multicut (no lifted edges)',
              'watershed_threshold': threshold, 'sigma_seeds_pixels': 2., 'multicut_beta': beta,
              'scope': 'Small contiguous slab, not full-ROI tracing or a paper accuracy result',
              'uses_old_labels_for_graph': False}
    status_path.write_text(json.dumps(report, indent=2))
    started = time.monotonic()
    try:
        with np.load(directory/'postprocess_inputs.npz') as data:
            boundary, valid, seed = data['boundary'], data['valid'], data['seed_zyx']
        if (boundary.ndim != 3 or valid.shape != boundary.shape or seed.shape != (3,)
                or not np.isfinite(boundary).all() or boundary.min() < 0 or boundary.max() > 1
                or np.any(seed < 0) or np.any(seed >= boundary.shape) or not valid[tuple(seed)]):
            raise ValueError('Invalid ZYX boundary, valid ROI or seed')
        fragments = watershed_oversegment(boundary, threshold=threshold, sigma_seeds=2.)
        fragments[~valid] = 0
        segmentation = lmc_agglomerate(boundary, fragments=fragments, beta=beta)
        selected_id = int(segmentation[tuple(seed)])
        if selected_id == 0:
            raise ValueError('Seed ended in background')
        selected = segmentation == selected_id
        np.save(output/'segmentation_zyx.npy', segmentation)
        np.save(output/'selected_xyz.npy', selected.transpose(2, 1, 0).astype(np.uint8))
        report.update(status='complete_bounded_segmentation', elapsed_seconds=time.monotonic()-started,
                      input_boundary_sha256=hashlib.sha256(np.ascontiguousarray(boundary)).hexdigest(),
                      shape_zyx=list(boundary.shape), seed_zyx=seed.tolist(),
                      fragment_count=len(np.unique(fragments[fragments != 0])),
                      instance_count=len(np.unique(segmentation[segmentation != 0])),
                      selected_local_id=str(selected_id), selected_voxels=int(selected.sum()),
                      selected_sections=int(np.any(selected, axis=(1, 2)).sum()),
                      reaches_first_or_last_section=bool(selected[0].any() or selected[-1].any()))
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(directory.name, json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['original', 'region_1', 'region_2', 'region_3', 'region_4'])
    parser.add_argument('--watershed-threshold', type=float, default=.25)
    parser.add_argument('--beta', type=float, default=.5)
    args = parser.parse_args()
    if not 0 < args.watershed_threshold < 1 or not 0 < args.beta < 1:
        parser.error('Threshold and beta must be in (0,1)')
    for name in args.cases:
        if name not in ('original', 'region_1', 'region_2', 'region_3', 'region_4'):
            parser.error('Unknown fixed ROI case')
        process(args.input/name, args.watershed_threshold, args.beta)


if __name__ == '__main__':
    main()
