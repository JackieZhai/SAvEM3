"""Back up the local region used by a Probe-EM trace as file:// precomputed volumes.

The remote volumes are far too large to mirror wholesale
(raw: 75000x50000x25481 uint8, seg: 72500x50000x25481 uint32). This script
downloads exactly the region the trace touched and stores it locally with the
SAME voxel_offset / resolution, so all mip0 voxel coordinates remain identical
to the remote volumes.

Output layout (default <probe-em>/volumes_local):
  raw/               uint8  jpeg                    precomputed image volume
  seg/               uint32 compressed_segmentation precomputed segmentation volume
  traced/            uint32 traced segments remapped to labels 1..N
  traced_skeletons.json   skeleton vertices/edges of traced segments (offline path)
  raw_info.json / seg_info.json   copies of the remote info files
  README.md          provenance of this backup

The viewer can then use the local copies directly:
  python scripts/open_trace_viewer.py --use-local --results-dir trace_results
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from cloudvolume import CloudVolume

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from open_trace_viewer import (  # noqa: E402
    MIP0_RES,
    collect_merge_contacts,
    fetch_skeletons,
    read_trace_ids,
    skeleton_vox,
)

RAW_URL = "precomputed://https://ng.zebrafish.digital-brain.cn/srv/raw/"
SEG_URL = "precomputed://https://ng.zebrafish.digital-brain.cn/srv/merge_neuro_label/"


def clamp_chunk(n, multiple=8):
    n = int(n)
    return max(multiple, n - n % multiple)


def write_volume(path, data, layer_type, data_type, encoding, offset,
                 chunk=(256, 256, 64), block=(8, 8, 8)):
    nx, ny, nz = data.shape
    chunk = tuple(min(c, s) for c, s in zip(chunk, data.shape))
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type=layer_type,
        data_type=data_type,
        encoding=encoding,
        resolution=[float(v) for v in MIP0_RES],
        voxel_offset=[int(v) for v in offset],
        volume_size=[nx, ny, nz],
        chunk_size=[clamp_chunk(c) for c in chunk],
        compressed_segmentation_block_size=list(block),
    )
    vol = CloudVolume(f"file://{path}", info=info, mip=0, parallel=True, progress=False)
    vol.commit_info()
    vol.commit_provenance()
    print(f"[info] writing {path}: dtype={data_type} encoding={encoding} "
          f"shape={data.shape} offset={offset}")
    vol[vol.bounds] = data
    return vol


def main():
    parser = argparse.ArgumentParser(description="Back up the local region used by a Probe-EM trace.")
    parser.add_argument("--results-dir", default="trace_results")
    parser.add_argument("--seed", type=int, default=1486287284)
    parser.add_argument("--out", default="volumes_local")
    parser.add_argument("--raw-url", default=RAW_URL)
    parser.add_argument("--seg-url", default=SEG_URL)
    parser.add_argument("--margin", type=int, default=32, help="extra mip0 voxels around the region")
    parser.add_argument("--no-slice-extent", action="store_true",
                        help="do not include the slice-extraction boxes (contact +-500 xy)")
    args = parser.parse_args()

    result_folder = os.path.join(args.results_dir, f"{args.seed}_results_sam")
    if not os.path.isdir(result_folder):
        raise SystemExit(f"Trace result folder not found: {result_folder}")

    os.makedirs(args.out, exist_ok=True)

    traced_ids = read_trace_ids(result_folder, args.seed)
    print(f"[info] traced segments: {traced_ids}")

    seg_vol = CloudVolume(args.seg_url, mip=2, parallel=True, fill_missing=True, progress=False)
    seg_vol0 = CloudVolume(args.seg_url, mip=0, parallel=True, fill_missing=True, progress=False)
    raw_vol0 = CloudVolume(args.raw_url, mip=0, parallel=True, fill_missing=True, progress=False)

    print("[info] fetching skeletons ...")
    skels = fetch_skeletons(seg_vol, traced_ids)

    contacts = collect_merge_contacts(result_folder, traced_ids)
    print(f"[info] contact rows: {len(contacts)}")

    # --- region = union(skeleton bboxes, contacts, slice-extraction boxes) + margin
    pts = []
    for tid, skel in skels.items():
        if skel is not None:
            pts.append(skeleton_vox(skel))
    for c in contacts:
        pts.append(np.array([c["point"]]))
    if not pts:
        raise SystemExit("No geometry found; nothing to back up.")
    all_pts = np.vstack(pts)
    lo = np.floor(all_pts.min(axis=0)).astype(int)
    hi = np.ceil(all_pts.max(axis=0)).astype(int) + 1

    if not args.no_slice_extent and contacts:
        ext_lo = all_pts.min(axis=0).astype(int) - np.array([500, 500, 10])
        ext_hi = all_pts.max(axis=0).astype(int) + np.array([501, 501, 11])
        lo = np.minimum(lo, ext_lo)
        hi = np.maximum(hi, ext_hi)

    lo = lo - args.margin
    hi = hi + args.margin
    bounds = np.array(seg_vol0.bounds.maxpt)
    raw_bounds = np.array(raw_vol0.bounds.maxpt)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.minimum(bounds, raw_bounds))
    print(f"[info] backup region (mip0 vox): {lo} .. {hi}  size={hi - lo}")

    # --- download seg cutout, expanding the box while any traced segment
    #     still touches a face (its voxels bulge beyond the skeleton bbox)
    seg_cut = None
    for _round in range(4):
        seg_cut = np.asarray(
            seg_vol0[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).squeeze()
        expand = False
        for tid in traced_ids:
            m = seg_cut == tid
            if not m.any():
                continue
            idx = np.argwhere(m)
            vmin, vmax = idx.min(axis=0), idx.max(axis=0)
            for ax in range(3):
                if vmin[ax] == 0 and lo[ax] > 0:
                    lo[ax] = max(lo[ax] - args.margin, 0)
                    expand = True
                if vmax[ax] == seg_cut.shape[ax] - 1 and hi[ax] < bounds[ax]:
                    hi[ax] = min(hi[ax] + args.margin, bounds[ax])
                    expand = True
        if not expand:
            break
        print(f"[info] traced segments touch the box boundary; expanding to "
              f"{lo} .. {hi} (round {_round + 1})")
    print(f"[info] final backup region (mip0 vox): {lo} .. {hi}  size={hi - lo}")

    # --- download raw cutout (same region)
    print("[info] downloading raw cutout ...")
    raw_cut = np.asarray(raw_vol0[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).squeeze()
    print(f"[info] raw cutout: {raw_cut.shape} {raw_cut.dtype}")
    print(f"[info] seg cutout: {seg_cut.shape} {seg_cut.dtype}")

    # --- write local precomputed volumes (same offset/resolution as remote)
    write_volume(os.path.join(args.out, "raw"), raw_cut, "image", "uint8", "jpeg", lo)
    write_volume(os.path.join(args.out, "seg"), seg_cut, "segmentation", "uint32",
                 "compressed_segmentation", lo)

    # --- traced segments remapped to 1..N
    remapped = np.zeros(seg_cut.shape, dtype=np.uint32)
    label_map = {}
    for label, tid in enumerate(traced_ids, start=1):
        label_map[tid] = label
        mask = seg_cut == tid
        remapped[mask] = label
        print(f"[info] traced label {label} <- seg {tid}: {int(mask.sum())} voxels")
    write_volume(os.path.join(args.out, "traced"), remapped, "segmentation", "uint32",
                 "compressed_segmentation", lo)

    # --- skeleton cache for offline trace-path drawing
    skel_cache = {}
    for tid, skel in skels.items():
        if skel is not None:
            skel_cache[str(tid)] = {
                "vertices": skel.vertices.tolist(),   # nm, mip0 space
                "edges": skel.edges.tolist(),
            }
    with open(os.path.join(args.out, "traced_skeletons.json"), "w", encoding="utf-8") as f:
        json.dump(skel_cache, f, indent=2)
    print(f"[info] saved {len(skel_cache)} skeletons to traced_skeletons.json")

    # --- provenance
    import urllib.request

    def _info(url):
        http_url = url[len("precomputed://"):] if url.startswith("precomputed://") else url
        with urllib.request.urlopen(http_url.rstrip("/") + "/info", timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))

    for name, url in (("raw", args.raw_url), ("seg", args.seg_url)):
        try:
            with open(os.path.join(args.out, f"{name}_info.json"), "w", encoding="utf-8") as f:
                json.dump(_info(url), f, indent=2)
        except Exception as e:
            print(f"[warn] failed to save {name} info: {e}")

    readme = f"""# Local volume backup for Probe-EM trace

Backed up {len(traced_ids)} traced segments of seed {args.seed}:
{traced_ids}

- raw/   : uint8 jpeg precomputed image volume
- seg/   : uint32 compressed_segmentation precomputed segmentation volume
- traced/: traced segments remapped to labels {min(label_map.values())}..{max(label_map.values())}
           mapping: {label_map}
- traced_skeletons.json: skeleton vertices (nm, mip0 space) and edges of traced
           segments that have skeletons.

Region (mip0 voxels): {list(lo)} .. {list(hi)}
Voxel size: (8, 8, 30) nm; voxel_offset preserved, so coordinates are identical
to the remote volumes:

- {args.raw_url}
- {args.seg_url}

View with local volumes:

    python scripts/open_trace_viewer.py --use-local --results-dir {args.results_dir} --seed {args.seed}

"""
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"[info] done. local volumes in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
