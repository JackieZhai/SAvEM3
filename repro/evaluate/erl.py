"""M6: sparse large-volume evaluation with Expected Run Length (ERL).

This entry point calls the official AxonEM-challenge evaluation:
    https://github.com/PytorchConnectomics/AxonEM-challenge
    erl_wrapper/eval_erl.py::compute_erl / expected_run_length
    erl_wrapper/test_axonEM.py::test_AxonEM

Official input contract:
    - Prediction: h5 (first dataset by default) or tif, in (z, y, x) order
    - GT: trusted AxonEM `gt_*_skel_stats.p` files (sequential pickle objects
      [gt_graph, gt_res]; gt_graph is a networkx_lite graph with node attributes
      skeleton_id/x/y/z; gt_res is the physical resolution in [z,y,x] order)
    - Optional `gt_*_mask.h5`: non-neuron mask for false-merge accounting

Usage:
    python erl.py --seg path/to/pred.h5 --gt-stats path/to/gt_human_32nm_skel_stats.p
                  [--gt-mask path/to/gt_human_32nm_mask.h5]
                  [--num-chunk 1] [--merge-threshold 50]
                  [--erl-intervals 0-20000-40000-150000]

The old z-path approximation remains in erl_approx.py for sanity checks only;
do not use it for paper ERL results.
"""
import argparse
import os
import sys

import numpy as np

AXONEM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "AxonEM-challenge", "erl_wrapper")
if not os.path.isfile(os.path.join(AXONEM_ROOT, 'eval_erl.py')):
    AXONEM_ROOT = os.environ.get('SAVEM3_AXONEM_ROOT', os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
        'SAvEM3', 'repro', 'evaluate', 'AxonEM-challenge', 'erl_wrapper'))
sys.path.insert(0, AXONEM_ROOT)


def parse_resolution(text):
    text = text.lower().replace(",", "x")
    res = [float(x) for x in text.split("x")]
    if len(res) != 3:
        raise ValueError(f"Cannot parse resolution {text!r}; expected three ZYX values, e.g. 30x32x32")
    return np.asarray(res)


def node_positions_voxel(gt_graph, gt_res):
    """Return (N, 3) voxel coordinates in (z, y, x) order from the official GT graph.

    - networkx_lite: nodes._nodes columns are skeleton_id/x/y/z
    - Standard networkx: node attributes z/y/x
    """
    # Lite graph (default AxonEM-challenge pickle format).
    nodes_obj = getattr(gt_graph, "nodes", None)
    is_lite = (nodes_obj is not None and hasattr(nodes_obj, "_nodes")
               and isinstance(getattr(nodes_obj, "_nodes"), np.ndarray))
    if is_lite:
        # Match erl_wrapper/test_axonEM.py:
        # _nodes is ordered [skeleton_id, x, y, z]; select z/y/x and divide by resolution.
        xyz = nodes_obj._nodes[:, -1:0:-1]  # z,y,x
        if xyz.shape[1] != 3:
            raise ValueError("Expected four networkx_lite node-attribute columns")
        pos = (xyz // gt_res).astype(np.int64)
        return pos

    # Standard networkx graph.
    pos = []
    for _, data in gt_graph.nodes(data=True):
        if not all(k in data for k in ("z", "y", "x")):
            raise ValueError("networkx node is missing z/y/x attributes")
        pos.append((data["z"], data["y"], data["x"]))
    pos = (np.asarray(pos, dtype=np.float64) // gt_res).astype(np.int64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("Cannot extract node coordinates from gt_graph")
    return pos


def main():
    ap = argparse.ArgumentParser(description="Official AxonEM-challenge ERL evaluation")
    ap.add_argument("--seg", required=True, help="Predicted instance segmentation h5/tif")
    ap.add_argument("--gt-stats", required=True,
                    help="Trusted AxonEM GT pickle with sequential [gt_graph, gt_res] objects")
    ap.add_argument("--gt-mask", default=None, help="Optional GT non-neuron mask h5")
    ap.add_argument("--num-chunk", type=int, default=1,
                    help="Number of prediction h5 read chunks (increase to reduce memory use)")
    ap.add_argument("--merge-threshold", type=int, default=50,
                    help="Minimum segment/skeleton voxel overlap counted as a false merge")
    ap.add_argument("--erl-intervals", type=str, default=None,
                    help="E.g. 0-20000-40000-150000; omit for overall ERL")
    args = ap.parse_args()
    if not os.path.isfile(os.path.join(AXONEM_ROOT, 'eval_erl.py')):
        raise FileNotFoundError('Install AxonEM-challenge or set SAVEM3_AXONEM_ROOT to its erl_wrapper directory')
    from data_io import read_pkl
    from eval_erl import compute_segment_lut, compute_erl

    print("Loading GT graph/resolution ...")
    objs = read_pkl(args.gt_stats)
    if len(objs) < 2:
        raise ValueError(
            f"{args.gt_stats} must contain at least two pickle objects [gt_graph, gt_res]")
    gt_graph, gt_res = objs[0], np.asarray(objs[1])
    if gt_res.shape != (3,):
        raise ValueError(f"gt_res must have three resolution values; got shape {gt_res.shape}")

    node_position = node_positions_voxel(gt_graph, gt_res)
    print(f"Nodes: {len(node_position)}, resolution: {tuple(gt_res)}")

    print("Computing skeleton-node -> segment LUT ...")
    node_segment_lut, mask_segment_id = compute_segment_lut(
        args.seg, node_position, args.gt_mask, args.num_chunk)

    intervals = None
    if args.erl_intervals:
        intervals = [int(x) for x in args.erl_intervals.split("-") if x != ""]
        if len(intervals) < 2:
            intervals = None

    print("Computing ERL ...")
    scores = compute_erl(gt_graph, node_segment_lut, mask_segment_id,
                         args.merge_threshold, intervals)
    if isinstance(scores, np.ndarray):
        print("ERL by interval:")
        for i, row in enumerate(scores):
            print(f"  interval {i}: ERL={row[0]:.1f}, length^2/length={row[1]:.1f}")
    else:
        print(f"ERL = {scores[0]:.1f}  (sum l^2/sum l = {scores[1]:.1f})")


if __name__ == "__main__":
    main()
