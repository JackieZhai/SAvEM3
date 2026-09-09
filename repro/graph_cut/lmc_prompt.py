"""Prompt-augmented local RAG multicut (historical LMC output name).

ELF expects cut probabilities. Positive costs penalize a cut; high HQ feature
cosine and SAM mask IoU therefore ADD attractive evidence. No additional lifted
edges are created. Weights are implementation choices requiring validation.
See README.md for the relationship to SAvEM3 and SAM-Graph.
"""
import argparse
import json
import os

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.volume import prediction_maps
from repro.savem3.distill_postprocess import lmc_agglomerate


def load_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['main'][:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fragments', required=True)
    ap.add_argument('--affs', required=True, help='Three-channel affinity or single-channel boundary-probability h5')
    ap.add_argument('--edge-feats', required=True, help='edge_feats.json produced by prompt_graph_cut.py')
    ap.add_argument('--w-cos', type=float, default=1.0)
    ap.add_argument('--w-iou', type=float, default=1.0)
    ap.add_argument('--w-ioa', type=float, default=0.0, help='SAM-Graph IoA edge-feature weight')
    ap.add_argument('--w-iob', type=float, default=0.0, help='SAM-Graph IoB edge-feature weight')
    ap.add_argument('--gt', default=None)
    ap.add_argument('--out-dir', default='./seg')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import elf.segmentation.features as feats
    import elf.segmentation.multicut as mc

    frag = load_h5(args.fragments).astype('uint64')
    affs = load_h5(args.affs).astype(np.float32)
    _, affs3 = prediction_maps(affs)
    with open(args.edge_feats) as f:
        ef = json.load(f)
    seg = lmc_agglomerate(affs3, frag, ef,
                         {'cos': args.w_cos, 'iou': args.w_iou,
                          'ioa': args.w_ioa, 'iob': args.w_iob})

    import h5py
    with h5py.File(os.path.join(args.out_dir, 'seg_lmc_prompt.hdf'), 'w') as f:
        f.create_dataset('main', data=seg, compression='gzip')

    if args.gt:
        gt = load_h5(args.gt).astype('uint64')
        from skimage.metrics import adapted_rand_error, variation_of_information
        arand = adapted_rand_error(gt, seg, ignore_labels=(0,))[0]
        vs, vm = variation_of_information(gt, seg, ignore_labels=(0,))
        print(f'LMC+prompt: VoI={vs + vm:.4f} (split {vs:.4f} / merge {vm:.4f}), ARand={arand:.4f}')
    print(f'Done -> {args.out_dir}/seg_lmc_prompt.hdf')


if __name__ == '__main__':
    main()
