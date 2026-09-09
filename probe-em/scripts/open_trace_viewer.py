"""Open a Neuroglancer viewer for a completed Probe-EM trace.

Layers (all in mip0 voxel coordinates, voxel size 8x8x30 nm, offset 0):
  1. `raw`            — raw EM volume (remote precomputed).
  2. `original_seg`   — the original segmentation volume (remote precomputed).
  3. `traced_seg`     — a local segmentation containing ONLY the traced
                        segments, remapped to consecutive labels 1..N
                        (mapping printed to the console).
  4. `trace_path_lines`   — annotation layer: skeleton lines of every traced
                        segment (the traced neuron path), amber.
  5. `trace_path_points`  — annotation layer: one point per traced segment at
                        its skeleton centroid; hover shows the segment ID.
  6. `merge_points`       — annotation layer: the geometric contact points of
                        verified merges (target -> neighbor), red.

The viewer uses the "4panel" layout (xy / xz / yz / 3d) so the traced volume
and the trace path can be inspected against raw and the original segmentation.

By default the remote precomputed URLs are used. After running
`scripts/backup_local_volumes.py`, pass `--use-local` to serve the local
file:// backups instead (same voxel_offset/resolution, identical coordinates).

Reference: https://github.com/google/neuroglancer
"""

import argparse
import csv
import json
import os
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import neuroglancer
from cloudvolume import CloudVolume

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_URL = "precomputed://https://ng.zebrafish.digital-brain.cn/srv/raw/"
SEG_URL = "precomputed://https://ng.zebrafish.digital-brain.cn/srv/merge_neuro_label/"
MIP0_RES = np.array([8.0, 8.0, 30.0])  # nm, mip0 of both volumes
# Viewer coordinate space: with scales equal to the mip0 voxel size (nm),
# position / annotation coordinates are plain mip0 voxel indices, matching
# the remote precomputed layers (voxel_size 8x8x30 nm, offset 0).
DIMENSIONS = neuroglancer.CoordinateSpace(
    names=["x", "y", "z"], units=["nm", "nm", "nm"], scales=[8.0, 8.0, 30.0]
)


def read_trace_ids(result_folder, seed):
    path = os.path.join(result_folder, f"trace_{seed}_ids.txt")
    with open(path, "r", encoding="utf-8") as f:
        return sorted({int(tok) for tok in f.read().split() if tok.lstrip("-").isdigit()})


def read_merge_tree(result_folder, seed):
    path = os.path.join(result_folder, f"trace_{seed}_tree.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_merge_contacts(result_folder, traced_ids):
    """Scan per-node neighbors CSVs; keep rows whose neighbor was merged in."""
    traced = set(traced_ids)
    contacts = []
    for name in ("neighbors.csv", "neighbors_z.csv"):
        for rel in _find_files(result_folder, name):
            try:
                with open(rel, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        try:
                            tid = int(row["target_id"])
                            nid = int(row["neighbor_id"])
                        except (KeyError, ValueError):
                            continue
                        if nid in traced and nid != tid:
                            contacts.append(
                                {
                                    "target_id": tid,
                                    "neighbor_id": nid,
                                    "point": [
                                        float(row.get("contact_x", row.get("x", 0))),
                                        float(row.get("contact_y", row.get("y", 0))),
                                        float(row.get("contact_z", row.get("z", 0))),
                                    ],
                                }
                            )
            except Exception as e:
                print(f"[warn] failed to parse {rel}: {e}")
    return contacts


def _find_files(base, filename):
    hits = []
    for root, _dirs, files in os.walk(base):
        if filename in files:
            hits.append(os.path.join(root, filename))
    return hits


def fetch_skeletons(seg_vol, ids):
    skels = {}
    for tid in ids:
        try:
            skel = seg_vol.skeleton.get(tid)
            skels[tid] = skel if skel is not None and len(skel.vertices) else None
        except Exception as e:
            print(f"[warn] skeleton unavailable for {tid}: {e}")
            skels[tid] = None
        if skels[tid] is not None:
            print(f"  seg {tid}: {len(skels[tid].vertices)} skeleton vertices, "
                  f"{len(skels[tid].edges)} edges")
        else:
            print(f"  seg {tid}: no skeleton (fragment)")
    return skels


def skeleton_vox(skel):
    """Skeleton vertices come back in absolute nm (mip0 space). -> mip0 voxels."""
    return skel.vertices / MIP0_RES


class CachedSkeleton:
    """Minimal skeleton container loaded from the local backup JSON."""

    def __init__(self, vertices, edges):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.edges = np.asarray(edges, dtype=np.int64)


def load_skeleton_cache(path, traced_ids):
    """Load traced_skeletons.json; ids without an entry are fragments (None)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    skels = {}
    for tid in traced_ids:
        entry = data.get(str(tid))
        skels[tid] = (
            CachedSkeleton(entry["vertices"], entry["edges"]) if entry else None
        )
    return skels


def load_local_layer(path):
    """Load a local file:// precomputed volume as a Neuroglancer LocalVolume.

    The backup keeps the original voxel_offset, so coordinates stay aligned
    with the remote volumes.
    """
    vol = CloudVolume(f"file://{path}", mip=0, parallel=False,
                      fill_missing=True, progress=False)
    data = np.asarray(vol[vol.bounds]).squeeze()
    print(f"[info] loaded local layer {path}: {data.shape} {data.dtype} "
          f"offset={vol.voxel_offset}")
    return neuroglancer.LocalVolume(
        data,
        dimensions=DIMENSIONS,
        voxel_offset=[int(v) for v in vol.voxel_offset],
    )


def load_local_array(path):
    """Load a local file:// precomputed volume as (data, offset)."""
    vol = CloudVolume(f"file://{path}", mip=0, parallel=False, progress=False)
    data = np.asarray(vol[vol.bounds]).squeeze()
    return data, np.array(vol.voxel_offset)


def build_traced_volume(seg_vol, skels, contacts, traced_ids, padding):
    """Fetch mip0 seg over the traced bounding box; remap traced ids -> 1..N."""
    pts = []
    for tid, skel in skels.items():
        if skel is not None:
            pts.append(skeleton_vox(skel))
    for c in contacts:
        pts.append(np.array([c["point"]]))
    if not pts:
        raise ValueError("No geometry available to define the traced bounding box.")
    all_pts = np.vstack(pts)
    lo = np.floor(all_pts.min(axis=0)).astype(int) - padding
    hi = np.ceil(all_pts.max(axis=0)).astype(int) + padding + 1
    bounds = np.array(seg_vol.bounds.maxpt)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, bounds)
    print(f"[info] traced bbox (mip0 vox): {lo} .. {hi}  "
          f"size={hi - lo}")

    cutout = np.asarray(seg_vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).squeeze()
    remapped = np.zeros(cutout.shape, dtype=np.uint32)
    label_map = {}
    for label, tid in enumerate(traced_ids, start=1):
        label_map[tid] = label
        mask = cutout == tid
        remapped[mask] = label
        print(f"  label {label} <- seg {tid}: {int(mask.sum())} mip0 voxels in box")
    contacts = find_nearest_contacts(remapped, lo)
    return remapped, label_map, lo, contacts


def find_nearest_contacts(remapped, offset):
    """Closest voxel pair between every pair of traced labels in the box.

    Works for merges verified by 3D propagation too: those fragments do not
    touch the seed in the segmentation (there is a z-gap between them), so the
    closest pair of voxels is the natural merge-point annotation.
    """
    from scipy.spatial import cKDTree

    labels = sorted(np.unique(remapped).tolist())
    labels = [int(x) for x in labels if x != 0]
    entries = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            a = np.argwhere(remapped == la).astype(np.float64)
            b = np.argwhere(remapped == lb).astype(np.float64)
            tree = cKDTree(a)
            d, idx = tree.query(b, k=1)
            j = int(np.argmin(d))
            pt = offset + b[j]
            entries.append(
                {
                    "label_a": la,
                    "label_b": lb,
                    "point": [float(x) for x in pt],
                    "dist_vox": float(d[j]),
                }
            )
    return entries


def make_annotation_layers(skels, merge_entries, label_map, dimensions, frag_pts=None,
                           coord_map=None):
    """Build annotation layers; ``coord_map`` maps (N,3) voxel points into the
    layer coordinate space (used by the aligned-space viewer)."""
    rev_map = {label: tid for tid, label in label_map.items()}
    lines = neuroglancer.LocalAnnotationLayer(
        dimensions=dimensions,
        shader="""
void main() {
  setColor(vec4(1.0, 0.82, 0.2, 1.0));
}
""",
    )
    points = neuroglancer.LocalAnnotationLayer(
        dimensions=dimensions,
        shader="""
void main() {
  setColor(vec4(1.0, 1.0, 0.35, 1.0));
  setPointMarkerSize(9.0);
}
""",
    )
    merges = neuroglancer.LocalAnnotationLayer(
        dimensions=dimensions,
        shader="""
void main() {
  setColor(vec4(1.0, 0.25, 0.25, 1.0));
  setPointMarkerSize(11.0);
}
""",
    )

    for tid, skel in skels.items():
        if skel is None:
            continue
        vox = skeleton_vox(skel)
        if coord_map is not None:
            vox = coord_map(vox)
        for i, (a, b) in enumerate(skel.edges):
            lines.annotations.append(
                neuroglancer.LineAnnotation(
                    id=f"seg_{tid}_edge_{i}",
                    point_a=[float(v) for v in vox[a]],
                    point_b=[float(v) for v in vox[b]],
                )
            )
        centroid = vox.mean(axis=0)
        points.annotations.append(
            neuroglancer.PointAnnotation(
                id=f"seg_{tid}",
                point=[float(v) for v in centroid],
                description=(
                    f"segment {tid} (label {label_map[tid]}): "
                    f"{len(skel.vertices)} vertices, {len(skel.edges)} edges"
                ),
            )
        )

    for tid, info in (frag_pts or {}).items():
        points.annotations.append(
            neuroglancer.PointAnnotation(
                id=f"seg_{tid}",
                point=[float(v) for v in info["point"]],
                description=(
                    f"fragment {tid} (label {label_map[tid]}): "
                    f"no skeleton, {info['n_vox']} voxels in traced box"
                ),
            )
        )

    seen = set()
    for e in merge_entries:
        key = tuple(sorted([e["label_a"], e["label_b"]]))
        if key in seen:
            continue
        seen.add(key)
        id_a = rev_map.get(e["label_a"], e["label_a"])
        id_b = rev_map.get(e["label_b"], e["label_b"])
        dist = e.get("dist_vox")
        dist_s = "" if dist is None else f" (gap {dist:.1f} vox)"
        merges.annotations.append(
            neuroglancer.PointAnnotation(
                id=f"merge_{id_a}_{id_b}",
                point=[float(v) for v in e["point"]],
                description=(
                    f"merge {id_a} (label {e['label_a']}) <-> "
                    f"{id_b} (label {e['label_b']}){dist_s}"
                ),
            )
        )

    return {
        "trace_path_lines": lines,
        "trace_path_points": points,
        "merge_points": merges,
    }


def aligned_bbox(field, skels, contacts, padding):
    """Aligned-space bounding box of the traced geometry (forward-mapped)."""
    pts = []
    for tid, skel in skels.items():
        if skel is not None:
            pts.append(field.forward_pts(skeleton_vox(skel)))
    for c in contacts:
        pts.append(field.forward_pts(np.array([c["point"]])))
    if not pts:
        raise SystemExit("No traced geometry available.")
    allp = np.vstack(pts)
    lo = np.floor(allp.min(axis=0)).astype(int) - padding
    hi = np.ceil(allp.max(axis=0)).astype(int) + padding + 1
    return np.maximum(lo, 0), hi


def build_aligned_data(field, raw_vol0, seg_vol0, skels, contacts, traced_ids, padding):
    """Fetch the traced region in the ALIGNED space and remap traced labels.

    Returns (aligned_raw, aligned_seg, remapped_traced, label_map, offset,
    vol_contacts) with everything in aligned mip0 voxel coordinates.
    """
    from probe_em.z_align import AlignedVolume

    lo, hi = aligned_bbox(field, skels, contacts, padding)
    hi = np.minimum(hi, np.array(seg_vol0.bounds.maxpt))
    print(f"[info] aligned-space region (mip0 vox): {lo} .. {hi}  size={hi - lo}")

    av_raw = AlignedVolume(raw_vol0, field)
    av_seg = AlignedVolume(seg_vol0, field)
    print("[info] fetching aligned raw cutout ...")
    raw_cut = np.asarray(av_raw[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).squeeze()
    print(f"[info] aligned raw: {raw_cut.shape} {raw_cut.dtype}")
    print("[info] fetching aligned seg cutout ...")
    seg_cut = np.asarray(av_seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).squeeze()
    print(f"[info] aligned seg: {seg_cut.shape} {seg_cut.dtype}")

    remapped = np.zeros(seg_cut.shape, dtype=np.uint32)
    label_map = {}
    for label, tid in enumerate(traced_ids, start=1):
        label_map[tid] = label
        mask = seg_cut == tid
        remapped[mask] = label
        print(f"  aligned label {label} <- seg {tid}: {int(mask.sum())} voxels")
    vol_contacts = find_nearest_contacts(remapped, lo)
    return raw_cut, seg_cut, remapped, label_map, lo, vol_contacts


def save_aligned_volumes(out_dir, raw_data, seg_data, traced_data, offset,
                         label_map, raw_url, seg_url, field_path):
    """Persist the aligned volumes as file:// precomputed (offline reuse)."""
    from backup_local_volumes import write_volume

    os.makedirs(out_dir, exist_ok=True)
    write_volume(os.path.join(out_dir, "raw"), raw_data, "image", "uint8", "jpeg", offset)
    write_volume(os.path.join(out_dir, "seg"), seg_data, "segmentation", "uint32",
                 "compressed_segmentation", offset)
    write_volume(os.path.join(out_dir, "traced"), traced_data, "segmentation", "uint32",
                 "compressed_segmentation", offset)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            f"# Aligned-space volumes (z-affine aligned)\n\n"
            f"field: {field_path}\nraw: {raw_url}\nseg: {seg_url}\n"
            f"region (aligned mip0 vox): {list(offset)} .. "
            f"{list(np.array(offset) + np.array(traced_data.shape))}\n"
            f"traced label map: {label_map}\n"
            f"view: python scripts/open_trace_viewer.py --align-field {field_path} "
            f"--results-dir <run> --seed <seed>\n"
        )
    print(f"[info] aligned volumes saved to {os.path.abspath(out_dir)}")


def main():
    parser = argparse.ArgumentParser(description="Neuroglancer viewer for a Probe-EM trace.")
    parser.add_argument("--results-dir", default="trace_results", help="output_root of the trace run")
    parser.add_argument("--seed", type=int, default=1486287284)
    parser.add_argument("--raw-url", default=RAW_URL)
    parser.add_argument("--seg-url", default=SEG_URL)
    parser.add_argument("--use-local", action="store_true",
                        help="use local backup volumes (see backup_local_volumes.py) "
                             "instead of the remote URLs")
    parser.add_argument("--local-dir", default="volumes_local",
                        help="directory with raw/, seg/, traced/, traced_skeletons.json")
    parser.add_argument("--align-field", default=None,
                        help="path to a ZAffineField npz: view the trace in the "
                             "z-affine-ALIGNED space (layers and annotations are "
                             "warped into the reference frame)")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--padding", type=int, default=24, help="mip0 voxels around traced bbox")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--dry-run", action="store_true", help="build layers and print state, no server")
    args = parser.parse_args()

    results_dir = args.results_dir
    result_folder = os.path.join(results_dir, f"{args.seed}_results_sam")
    if not os.path.isdir(result_folder):
        raise SystemExit(f"Trace result folder not found: {result_folder}")

    traced_ids = read_trace_ids(result_folder, args.seed)
    merge_tree = read_merge_tree(result_folder, args.seed)
    print(f"[info] traced segments ({len(traced_ids)}): {traced_ids}")
    print(f"[info] merge tree: {json.dumps(merge_tree)}")

    use_local = args.use_local
    local_dir = args.local_dir
    coord_map = None  # set to field.forward_pts in aligned mode

    if use_local:
        raw_path = os.path.join(local_dir, "raw")
        seg_path = os.path.join(local_dir, "seg")
        if not (os.path.isdir(raw_path) and os.path.isdir(seg_path)):
            raise SystemExit(
                f"Local volumes not found under '{local_dir}'. "
                "Run scripts/backup_local_volumes.py first."
            )
        print(f"[info] using local backup volumes from: {os.path.abspath(local_dir)}")
        raw_src = load_local_layer(raw_path)
        seg_src = load_local_layer(seg_path)

        skel_cache = os.path.join(local_dir, "traced_skeletons.json")
        if os.path.exists(skel_cache):
            print(f"[info] loading skeletons from cache: {skel_cache}")
            skels = load_skeleton_cache(skel_cache, traced_ids)
        else:
            print("[info] skeleton cache missing; fetching from remote seg ...")
            seg_vol = CloudVolume(args.seg_url, mip=2, parallel=False,
                                  fill_missing=True, progress=False)
            skels = fetch_skeletons(seg_vol, traced_ids)

        contacts = collect_merge_contacts(result_folder, traced_ids)

        traced_path = os.path.join(local_dir, "traced")
        if os.path.isdir(traced_path):
            print("[info] using precomputed traced volume from backup")
            traced_data, offset = load_local_array(traced_path)
            label_map = {tid: i + 1 for i, tid in enumerate(traced_ids)}
            vol_contacts = find_nearest_contacts(traced_data, offset)
        else:
            print("[info] traced volume not in backup; building from local seg")
            seg_vol0 = CloudVolume(f"file://{seg_path}", mip=0, parallel=False,
                                   fill_missing=True, progress=False)
            traced_data, label_map, offset, vol_contacts = build_traced_volume(
                seg_vol0, skels, contacts, traced_ids, args.padding
            )
    elif args.align_field:
        from probe_em.z_align import ZAffineField
        field = ZAffineField.load(args.align_field)
        print(f"[info] ALIGNED-space viewer — field z [{field.z0}, {field.z1}], "
              f"ref z={field.ref_z}; layers will be warped into the reference frame")
        coord_map = field.forward_pts

        if use_local:
            raw_path = os.path.join(local_dir, "raw")
            seg_path = os.path.join(local_dir, "seg")
            if not (os.path.isdir(raw_path) and os.path.isdir(seg_path)):
                raise SystemExit(f"Local volumes not found under '{local_dir}'.")
            raw_vol0 = CloudVolume(f"file://{raw_path}", mip=0, parallel=False,
                                   fill_missing=True, progress=False)
            seg_vol0 = CloudVolume(f"file://{seg_path}", mip=0, parallel=False,
                                   fill_missing=True, progress=False)
            print(f"[info] aligned fetch from local backup: {os.path.abspath(local_dir)}")
            skel_cache = os.path.join(local_dir, "traced_skeletons.json")
            if os.path.exists(skel_cache):
                skels = load_skeleton_cache(skel_cache, traced_ids)
            else:
                seg_vol = CloudVolume(args.seg_url, mip=2, parallel=False,
                                      fill_missing=True, progress=False)
                skels = fetch_skeletons(seg_vol, traced_ids)
        else:
            raw_vol0 = CloudVolume(args.raw_url, mip=0, parallel=False,
                                   fill_missing=True, progress=False)
            seg_vol0 = CloudVolume(args.seg_url, mip=0, parallel=False,
                                   fill_missing=True, progress=False)
            print("[info] fetching skeletons ...")
            seg_vol = CloudVolume(args.seg_url, mip=2, parallel=False,
                                  fill_missing=True, progress=False)
            skels = fetch_skeletons(seg_vol, traced_ids)

        contacts = collect_merge_contacts(result_folder, traced_ids)

        raw_data, seg_data, traced_data, label_map, offset, vol_contacts = \
            build_aligned_data(field, raw_vol0, seg_vol0, skels, contacts,
                               traced_ids, args.padding)
        raw_src = neuroglancer.LocalVolume(
            raw_data, dimensions=DIMENSIONS, voxel_offset=[int(v) for v in offset])
        seg_src = neuroglancer.LocalVolume(
            seg_data, dimensions=DIMENSIONS, voxel_offset=[int(v) for v in offset])

        try:
            save_aligned_volumes(local_dir.rstrip("/") + "_aligned",
                                 raw_data, seg_data, traced_data, offset,
                                 label_map, args.raw_url, args.seg_url,
                                 os.path.abspath(args.align_field))
        except Exception as e:
            print(f"[warn] could not persist aligned volumes: {e}")
    else:
        seg_vol = CloudVolume(args.seg_url, mip=2, parallel=False,
                              fill_missing=True, progress=False)
        seg_vol0 = CloudVolume(args.seg_url, mip=0, parallel=False,
                               fill_missing=True, progress=False)
        print("[info] fetching skeletons ...")
        skels = fetch_skeletons(seg_vol, traced_ids)

        contacts = collect_merge_contacts(result_folder, traced_ids)

        print("[info] building traced segmentation volume ...")
        traced_data, label_map, offset, vol_contacts = build_traced_volume(
            seg_vol0, skels, contacts, traced_ids, args.padding
        )
        raw_src = args.raw_url
        seg_src = args.seg_url

    # normalize merge entries: CSV contact rows + nearest-voxel contacts
    merge_entries = []
    for c in contacts:
        if c["target_id"] in label_map and c["neighbor_id"] in label_map:
            pt = np.array([c["point"]])
            if coord_map is not None:
                pt = coord_map(pt)
            merge_entries.append(
                {
                    "label_a": label_map[c["target_id"]],
                    "label_b": label_map[c["neighbor_id"]],
                    "point": [float(v) for v in pt[0]],
                }
            )
    for v in vol_contacts:
        merge_entries.append(
            {
                "label_a": v["label_a"],
                "label_b": v["label_b"],
                "point": v["point"],
                "dist_vox": v.get("dist_vox"),
            }
        )
    print(f"[info] merge annotations: {len(merge_entries)}")

    frag_pts = {}
    for tid, label in label_map.items():
        if skels.get(tid) is None:
            idx = np.argwhere(traced_data == label)
            if len(idx):
                frag_pts[tid] = {
                    "point": offset + idx.mean(axis=0),
                    "n_vox": int(len(idx)),
                }

    traced_vol = neuroglancer.LocalVolume(
        traced_data,
        dimensions=DIMENSIONS,
        voxel_offset=[int(v) for v in offset],
    )

    if args.dry_run:
        # validate the layer construction without starting a server
        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:
            s.dimensions = DIMENSIONS
            ann = make_annotation_layers(skels, merge_entries, label_map, s.dimensions,
                                         frag_pts, coord_map=coord_map)
            s.layers["raw"] = neuroglancer.ImageLayer(source=raw_src)
            s.layers["original_seg"] = neuroglancer.SegmentationLayer(source=seg_src)
            s.layers["traced_seg"] = neuroglancer.SegmentationLayer(source=traced_vol)
            for name, layer in ann.items():
                s.layers[name] = layer
            s.layout = "4panel"
            s.show_slices = False
            seed_pts = skeleton_vox(skels[args.seed]) if skels.get(args.seed) is not None else None
            if seed_pts is not None:
                if coord_map is not None:
                    seed_pts = coord_map(seed_pts)
                s.position = [float(v) for v in seed_pts.mean(axis=0)]
            state = s.to_json()
        print("[dry-run] state JSON:")
        print(state)
        print("[dry-run] OK — viewer layers would be: raw, original_seg, traced_seg, "
              "trace_path_lines, trace_path_points, merge_points")
        return

    neuroglancer.set_server_bind_address(bind_address=args.bind, bind_port=args.port)
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.dimensions = DIMENSIONS
        ann = make_annotation_layers(skels, merge_entries, label_map, s.dimensions,
                                     frag_pts, coord_map=coord_map)
        s.layers["raw"] = neuroglancer.ImageLayer(source=raw_src)
        s.layers["original_seg"] = neuroglancer.SegmentationLayer(source=seg_src)
        s.layers["traced_seg"] = neuroglancer.SegmentationLayer(source=traced_vol)
        for name, layer in ann.items():
            s.layers[name] = layer
        s.layout = "4panel"
        s.show_slices = False
        seed_pts = skeleton_vox(skels[args.seed]) if skels.get(args.seed) is not None else None
        if seed_pts is not None:
            if coord_map is not None:
                seed_pts = coord_map(seed_pts)
            s.position = [float(v) for v in seed_pts.mean(axis=0)]

    try:
        with viewer.config_state.txn() as s:
            s.status_message = (
                f"Probe-EM trace of {args.seed}: {len(traced_ids)} segments. "
                "Toggle layers to compare raw / original_seg / traced_seg."
            )
    except Exception:
        pass  # status_message is not available in older neuroglancer versions

    url = viewer.get_viewer_url()
    print("=" * 72)
    print(f"Neuroglancer viewer: {url}")
    if coord_map is not None:
        print("SPACE: z-affine ALIGNED (layers + annotations in the reference frame)")
    print(f"Data source: {'local backup (' + os.path.abspath(local_dir) + ')' if use_local else 'remote (precomputed URLs)'}")
    print("Layers:")
    print(f"  raw           — raw EM volume ({'local backup' if use_local else 'remote'})")
    print(f"  original_seg  — original segmentation ({'local backup' if use_local else 'remote'})")
    print(f"  traced_seg    — traced segments remapped to labels {min(label_map.values())}..{max(label_map.values())}")
    print("  trace_path_lines  — amber skeleton lines of the traced path")
    print("  trace_path_points — yellow segment centroids (hover for IDs)")
    print("  merge_points      — red contact points of verified merges")
    print(f"label map: {label_map}")
    print("=" * 72)

    if not args.no_browser:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
