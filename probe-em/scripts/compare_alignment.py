"""Compare ECC and SOFIMA on exactly the same locally cached ROI and support."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import load_local
from probe_em.alignment_field import warp_field_slice
from probe_em.demo_alignment import prepare_field, registered_bounds
from probe_em.registration import ncc


def compare(raw, seg, resolution, offset, fields):
    if 'unregistered' not in fields or fields['unregistered'] is not None or raw.shape[2] < 2:
        raise ValueError('Comparison needs an unregistered baseline and at least two sections')
    bounds = [registered_bounds(raw.shape, offset, field, 128_000_000) for field in fields.values() if field is not None]
    lower = np.min([np.asarray(offset)]+[np.asarray(lo) for shape, lo in bounds], axis=0)
    upper = np.max([np.asarray(offset)+raw.shape]+[np.asarray(lo)+shape for shape, lo in bounds], axis=0)
    shape = tuple((upper-lower)[:2])
    phase = (np.asarray(offset[:2])-lower[:2]) % 2
    sampling = (slice(int(phase[0]), None, 2), slice(int(phase[1]), None, 2))
    names = list(fields)
    previous, previous_valid = {}, {}
    scores, labels_seen = [], set()
    ones = np.ones(raw.shape[:2], np.uint8)
    for index in range(raw.shape[2]):
        z = int(offset[2])+index
        images, valid = {}, {}
        for name, field in fields.items():
            images[name] = warp_field_slice(field, raw[:, :, index], z, offset[:2], lower[:2], shape)[sampling]
            valid[name] = warp_field_slice(field, ones, z, offset[:2], lower[:2], shape, seg=True)[sampling] > 0
            if name == 'sofima':
                labels_seen.update(map(int, np.unique(warp_field_slice(
                    field, seg[:, :, index], z, offset[:2], lower[:2], shape, seg=True))))
        if index:
            shared = np.logical_and.reduce([valid[name] & previous_valid[name] for name in names])
            original = valid['unregistered'] & previous_valid['unregistered']
            if shared.sum() < 32:
                raise ValueError('Insufficient common support for a fair registration comparison')
            row = {'z': z, 'shared_pixels': int(shared.sum()), 'support_fraction': float(shared.sum()/original.sum())}
            row.update({name: ncc(previous[name][shared], images[name][shared]) for name in names})
            scores.append(row)
        previous, previous_valid = images, valid
    source_ids = set(map(int, np.unique(seg)))
    elastic = fields.get('sofima')
    error = None
    if elastic is not None:
        xyz = np.array([[x, y, z] for z in range(int(offset[2]), int(offset[2])+raw.shape[2])
                        for x in np.linspace(offset[0], offset[0]+raw.shape[0]-1, 5)
                        for y in np.linspace(offset[1], offset[1]+raw.shape[1]-1, 5)])
        error = np.linalg.norm(elastic.inverse_pts(elastic.forward_pts(xyz))-xyz, axis=1)
        if labels_seen-source_ids-{0}:
            raise ValueError('Warp introduced a label ID not present in the input')
    return {'metric': 'Adjacent-section NCC on identical valid pixels across all selected methods; not anatomical accuracy',
            'sampling_xy': 2, 'mean_ncc': {name: float(np.mean([row[name] for row in scores])) for name in names},
            'mean_support_fraction': float(np.mean([row['support_fraction'] for row in scores])),
            'pairs': scores, 'new_label_ids': [] if elastic is not None else None,
            'source_label_count': len(source_ids-{0}),
            'lost_label_ids': [str(i) for i in sorted(source_ids-labels_seen-{0})] if elastic is not None else None,
            'roundtrip_max_error_vox': float(error.max()) if error is not None else None,
            'elastic_geometry': elastic.stats() if elastic is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--sofima-python', required=True)
    parser.add_argument('--sofima-config', type=Path)
    parser.add_argument('--z-start', type=int, help='Optional absolute inclusive z')
    parser.add_argument('--z-stop', type=int, help='Optional absolute exclusive z')
    args = parser.parse_args()
    started = time.monotonic()
    raw, resolution, offset = load_local(args.local_dir/'raw')
    seg, seg_res, seg_offset = load_local(args.local_dir/'seg')
    if raw.shape != seg.shape or not np.array_equal(resolution, seg_res) or not np.array_equal(offset, seg_offset):
        raise ValueError('Raw and labels must share a grid')
    lo = int(offset[2]) if args.z_start is None else args.z_start
    hi = int(offset[2])+raw.shape[2] if args.z_stop is None else args.z_stop
    if not int(offset[2]) <= lo < hi <= int(offset[2])+raw.shape[2] or hi-lo < 2:
        raise ValueError('Choose at least two sections inside the cached ROI')
    raw = raw[:, :, lo-int(offset[2]):hi-int(offset[2])]
    seg = seg[:, :, lo-int(offset[2]):hi-int(offset[2])]
    offset = offset.copy()
    offset[2] = lo
    options = json.loads(args.sofima_config.read_text()) if args.sofima_config else None
    ecc, ecc_info = prepare_field(raw, resolution, offset, cache_dir=args.output_dir/'ecc')
    elastic, elastic_info = prepare_field(raw, resolution, offset, cache_dir=args.output_dir/'sofima',
                                         backend='sofima', sofima_python=args.sofima_python, sofima_options=options)
    report = compare(raw, seg, resolution, offset, {'unregistered': None, 'ecc': ecc, 'sofima': elastic})
    report.update(source=str(args.local_dir.resolve()), shape_xyz=list(raw.shape),
                  offset_xyz=offset.tolist(), resolution_xyz_nm=resolution.tolist(),
                  ecc=ecc_info, sofima=elastic_info, elapsed_seconds=time.monotonic()-started)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('mean_ncc', 'mean_support_fraction', 'roundtrip_max_error_vox',
                                                 'lost_label_ids', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
