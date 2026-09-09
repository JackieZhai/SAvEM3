"""Controlled registration ablations on the four cached online zebrafish ROIs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))
from demo import load_local
from demo_regions import alignment_quality, write_json
from probe_em.z_align import ZAffineField, _estimate_pair_affine


def run_region(directory):
    raw, resolution, offset = load_local(directory / 'cache' / 'raw')
    metadata = json.loads((directory / 'region.json').read_text())
    baseline = ZAffineField.load(metadata['alignment']['field_path'])
    ref_index = baseline.ref_z - int(offset[2])
    output = directory / 'alignment_investigation'
    output.mkdir(exist_ok=True)
    result = {'region': directory.name, 'baseline': alignment_quality(raw, offset, baseline),
              'baseline_rejected': len(metadata['alignment']['rejected_pair_slices']),
              'resolution_xyz_nm': resolution.tolist(), 'experiments': {}}
    for name, factors, max_nm in [('native_320nm', [1], 320),
                                  ('coarse32nm_320nm', [4], 320),
                                  ('pyramid_320nm', [4, 2, 1], 320)]:
        start = time.monotonic()
        scales = []
        for factor in factors:
            shape = (raw.shape[0]//factor, raw.shape[1]//factor)
            images = [cv2.resize(raw[:, :, index].T, shape, interpolation=cv2.INTER_AREA).T
                      for index in range(raw.shape[2])]
            scale = np.array(raw.shape[:2]) / shape
            scales.append((images, scale))
        steps = np.zeros((raw.shape[2], 2))
        measurements = []
        for index in range(1, raw.shape[2]):
            estimate = None
            for images, scale in scales:
                change_scale = np.diag([*scale, 1.])
                init = None if estimate is None else np.linalg.inv(change_scale) @ estimate @ change_scale
                pair = _estimate_pair_affine(images[index-1], images[index], init=init,
                                            max_trans=1e6, motion='translation')
                if pair is not None:
                    proposed = change_scale @ pair @ np.linalg.inv(change_scale)
                    if np.linalg.norm(proposed[:2, 2] * resolution[:2]) <= max_nm:
                        estimate = proposed
            if estimate is not None:
                steps[index] = estimate[:2, 2]
            measurements.append({'z': int(offset[2])+index, 'accepted': estimate is not None,
                                 'translation_xy_vox': steps[index].tolist()})
        positions = np.cumsum(steps, axis=0)
        positions -= positions[ref_index]
        mats = {int(offset[2])+index: np.array([[1., 0., dx], [0., 1., dy], [0., 0., 1.]])
                for index, (dx, dy) in enumerate(positions)}
        field = ZAffineField(int(offset[2]), int(offset[2])+raw.shape[2]-1, mats,
                             baseline.ref_z, baseline.window_bbox)
        field.save(output / (name + '.npz'))
        quality = alignment_quality(raw, offset, field)
        summary = {'runtime_seconds': time.monotonic()-start, 'quality': quality,
                   'rejected_pairs': sum(not m['accepted'] for m in measurements),
                   'accepted_above_old_10px_limit': int(sum(np.linalg.norm(steps[i]) > 10 for i in range(1, len(steps)))),
                   'stats': field.stats(), 'measurements': measurements}
        result['experiments'][name] = summary
        write_json(output / 'ablation.json', result)
        print(json.dumps({'region': directory.name, 'method': name, 'rejected': summary['rejected_pairs'],
                          'accepted_above_10px': summary['accepted_above_old_10px_limit'],
                          'mean_ncc': quality['mean_after'], 'max_shift': field.stats()['max_abs_xy_shift_vox']}), flush=True)
    return result


if __name__ == '__main__':
    cv2.setNumThreads(1)
    root = Path(sys.argv[1] if len(sys.argv)>1 else 'demo_session_regions').resolve()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_region, sorted(root.glob('region_*'))))
    write_json(root / 'alignment_ablation.json', results)
