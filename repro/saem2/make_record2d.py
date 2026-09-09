"""M2 preparation: export data-bank CloudVolume layers as SAEM² section TIFFs.

Dataloader contract (saem2/utils/dataloader_isbi_2d_v4.py):
    prepared_segments_mul_mem/{ds}/%05d.tif    Membrane maps (img > 0 means membrane)
    prepared_segments_mul_2d/{ds}/%05d.tif    Instance labels (mem2label in json2d_create.py)
    prepared_embedding/{ds}/embed/%05d.tif    SAM embeddings (savem3/precompute_teacher.py)
    record2d_train.json                        {dataset: {layer: [label_id...]}}

This script exports membrane TIFFs from the mem layer in location.py. If absent,
derive membranes from seg using fastmorph.erode followed by == 0, as in seg2mem.
Then run json2d_create.py followed by savem3/precompute_teacher.py --write-tif.

Dataset-name mapping (location.py -> historical SALEM2 records):
    snemi→SNEMI, ac3→AC3, cremi_a→cremiA, cremi_b→cremiB, cremi_c→cremiC,
    fib25→FIB25, hemibrain_*→HB-<region>, axonem-h_<x>-<y>-<z>→AxonEM-H/seg_<x>-<y>-<z>,
    axonem-m_*→AxonEM-M/seg_*, j0126_*→J0126/<block>, segem_*→SegEM/<block>

Usage:
    python make_record2d.py --datasets snemi,ac3 --out-root $SAVEM3_DATA_ROOT
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_engine'))
from common import DATA_ROOT, ranges_of, cv_path, open_cv, encoder_plane  # noqa: E402

from skimage import io  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402


def salem2_name(dataset):
    """Map a location.py dataset name to a historical SALEM2 record name."""
    if dataset == 'snemi':
        return 'SNEMI'
    if dataset == 'ac3':
        return 'AC3'
    if dataset.startswith('cremi'):
        return {'cremi_a': 'cremiA', 'cremi_b': 'cremiB', 'cremi_c': 'cremiC'}.get(dataset, dataset)
    if dataset == 'fib25':
        return 'FIB25'
    if dataset.startswith('hemibrain'):
        return 'HB-' + dataset.split('_', 1)[1]          # HB-fb-inner ...
    if dataset.startswith('axonem-h'):
        return 'AxonEM-H/seg_' + dataset.split('_', 1)[1]  # seg_950-0-0
    if dataset.startswith('axonem-m'):
        return 'AxonEM-M/seg_' + dataset.split('_', 1)[1]
    if dataset.startswith('j0126'):
        return 'J0126/' + dataset.split('_', 1)[1]
    if dataset.startswith('segem'):
        return 'SegEM/' + dataset.split('_', 1)[1]
    return dataset


def export_mem_tifs(dataset, out_root, xy_nm=4.0):
    """Export membrane TIFFs; derive missing membranes from seg using seg2mem conventions."""
    ds_name = salem2_name(dataset)
    out_dir = os.path.join(out_root, 'prepared_segments_mul_mem', ds_name)
    os.makedirs(out_dir, exist_ok=True)

    mem_vol = open_cv(dataset, 'mem')
    seg_vol = open_cv(dataset, 'seg')
    if mem_vol is None and seg_vol is None:
        print(f'[{dataset}] No mem/seg layer; skipping')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]

    need_erode = mem_vol is None
    for z in tqdm(range(zs, ze), desc=ds_name):
        if need_erode:
            import fastmorph
            seg = seg_vol[xs:xe, ys:ye, z][..., 0].copy()
            seg = fastmorph.erode(seg, parallel=4)
            mem = ((seg == 0).astype(np.uint8)) * 255
        else:
            mem = (mem_vol[xs:xe, ys:ye, z][..., 0] > 0).astype(np.uint8) * 255
        resolution = (seg_vol if need_erode else mem_vol).resolution
        mem = encoder_plane(mem, resolution, xy_nm, labels=True).astype(np.uint8)
        io.imsave(os.path.join(out_dir, '%05d.tif' % (z - zs)), mem)
    print(f'[{dataset}] -> {out_dir} ({ze - zs} sections)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='snemi,ac3')
    ap.add_argument('--out-root', type=str, default=None,
                    help='Default: $SAVEM3_DATA_ROOT (parent of prepared_segments_mul_mem)')
    ap.add_argument('--xy-nm', type=float, default=4.0, help='Must match precompute_teacher.py --xy-nm')
    args = ap.parse_args()
    out_root = args.out_root or DATA_ROOT
    for name in [d for d in args.datasets.split(',') if d]:
        print(f'== {name} ==')
        export_mem_tifs(name, out_root, args.xy_nm)


if __name__ == '__main__':
    main()
