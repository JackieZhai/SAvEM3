"""Generate verification figures for the z-alignment + tracing results.

Outputs (saved next to this script in verification_figs/):
  fig_align_field.png    per-slice translations / cumulative displacement of the field
  fig_trace_tree.png     merge tree of the aligned trace run
  fig_trace_overlay.png  raw slices with traced segments overlaid (label 2 seed,
                         other labels = merged fragments)
  fig_sam2_2d.png        montage of the SAM2 2D verification results
  fig_sam2_3d.png        montage of the SAM2 3D collision verification results
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from probe_em.z_align import ZAffineField  # noqa: E402

OUT = Path(__file__).resolve().parent / "verification_figs"
OUT.mkdir(exist_ok=True)

SEED = 1486287284
RESULT_DIR = PROJECT_ROOT / "trace_results_align_test" / f"{SEED}_results_sam"


def main():
    # ------------------------------------------------------------ field plot
    field = ZAffineField.load(PROJECT_ROOT / "trace_results_align_test" / f"align_z_field_seed{SEED}.npz")
    zs = sorted(field.mats)
    tx = np.array([field.mats[z][0, 2] for z in zs])
    ty = np.array([field.mats[z][1, 2] for z in zs])
    disp = np.sqrt(tx ** 2 + ty ** 2)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(zs, tx, lw=1, label="t_x (mip0 vox)")
    axes[0].plot(zs, ty, lw=1, label="t_y (mip0 vox)")
    axes[0].axvline(field.ref_z, color="r", ls="--", lw=1, label=f"ref z={field.ref_z}")
    axes[0].set_ylabel("per-slice translation (vox)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"Z-affine alignment field (motion=translation, {field.z0}..{field.z1}, "
                      f"{len(zs)} slices) — seed {SEED}")

    axes[1].plot(zs, disp, lw=1, color="tab:orange")
    axes[1].axvline(field.ref_z, color="r", ls="--", lw=1)
    axes[1].set_ylabel("|displacement| (vox)")
    axes[1].set_title(f"cumulative displacement vs reference slice "
                      f"(max {disp.max():.1f} vox = {disp.max() * 8 / 1000:.2f} um x/y)")

    steps = np.abs(np.diff(np.vstack([tx, ty]), axis=1)).max(axis=0)
    axes[2].plot(zs[1:], steps, lw=0.8, color="tab:green")
    axes[2].axvline(field.ref_z, color="r", ls="--", lw=1)
    axes[2].set_xlabel("z slice")
    axes[2].set_ylabel("per-step shift (vox)")
    axes[2].set_title("per-slice step magnitude (drift between consecutive sections)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_align_field.png", dpi=110)
    plt.close(fig)
    print("saved", OUT / "fig_align_field.png")

    # ------------------------------------------------------------ merge tree
    tree = json.load(open(RESULT_DIR / f"trace_{SEED}_tree.json"))
    import networkx as nx
    G = nx.DiGraph()
    for parent, children in tree.items():
        G.add_node(parent)
        for c in children:
            G.add_edge(parent, c)
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=1800, node_color="#9ecae1",
            edge_color="gray", font_size=10, arrows=True)
    ax.set_title(f"Probe-EM merge tree (aligned run) — seed {SEED}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_trace_tree.png", dpi=110)
    plt.close(fig)
    print("saved", OUT / "fig_trace_tree.png")

    # ------------------------------------------------------------ overlay
    from cloudvolume import CloudVolume
    raw_vol = CloudVolume(f"file://{PROJECT_ROOT / 'volumes_local' / 'raw'}", mip=0,
                          parallel=False, fill_missing=True, progress=False)
    tr_vol = CloudVolume(f"file://{PROJECT_ROOT / 'volumes_local' / 'traced'}", mip=0,
                         parallel=False, fill_missing=True, progress=False)
    off = np.array(tr_vol.voxel_offset)
    zs_show = [13900, 13963, 14026]  # seed body / reference / fragment z
    labels = {1: ("frag 1486284686", "red"), 2: ("seed 1486287284", "gold"),
              3: ("frag 1504550065", "lime")}
    fig, axes = plt.subplots(1, len(zs_show), figsize=(15, 5.2))
    for ax, z in zip(axes, zs_show):
        zi = z - off[2]
        raw = np.asarray(raw_vol[off[0]:off[0] + 261, off[1]:off[1] + 357, z:z + 1]).squeeze()
        seg = np.asarray(tr_vol[off[0]:off[0] + 261, off[1]:off[1] + 357, z:z + 1]).squeeze()
        raw = np.clip(raw, np.percentile(raw, 1), np.percentile(raw, 99))
        ax.imshow(raw.T, cmap="gray", origin="lower")  # (x, y) -> image (y, x)
        for lab, (name, color) in labels.items():
            m = seg.T == lab
            if m.any():
                ov = np.zeros((*m.shape, 4))
                ov[m] = matplotlib.colors.to_rgba(color, 0.55)
                ax.imshow(ov)
        ax.set_title(f"z={z}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=n)
               for n, c in labels.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Traced segments overlaid on raw (local volumes) — seed {SEED}", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(OUT / "fig_trace_overlay.png", dpi=110)
    plt.close(fig)
    print("saved", OUT / "fig_trace_overlay.png")

    # ------------------------------------------------------------ SAM2 montages
    from PIL import Image

    def montage(paths, cols, max_w):
        ims = [Image.open(p) for p in paths]
        rows = (len(ims) + cols - 1) // cols
        tw = max_w * cols
        th = int(max_w * ims[0].size[1] / ims[0].size[0]) * rows
        canvas = Image.new("RGB", (tw, th), "white")
        for k, im in enumerate(ims):
            ratio = max_w / im.size[0]
            im2 = im.resize((max_w, int(im.size[1] * ratio)))
            canvas.paste(im2, ((k % cols) * max_w, (k // cols) * int(im.size[1] * ratio)))
        return canvas

    two_d = sorted((RESULT_DIR / "temp_vis_1486287284").glob("*_xy_result.jpg"))
    three_d = sorted((RESULT_DIR / "temp_vis3d_1486287284").glob("*_collision.jpg"))
    if two_d:
        montage(two_d, 4, 560).save(OUT / "fig_sam2_2d.png")
        print(f"saved {OUT / 'fig_sam2_2d.png'} ({len(two_d)} panels)")
    if three_d:
        montage(three_d, 3, 640).save(OUT / "fig_sam2_3d.png")
        print(f"saved {OUT / 'fig_sam2_3d.png'} ({len(three_d)} panels)")

    print("done")


if __name__ == "__main__":
    main()
