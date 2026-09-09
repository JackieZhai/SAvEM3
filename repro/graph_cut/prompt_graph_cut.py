"""SAvEM3 prompt graph: 32-channel HQ node means + common-section SAM overlaps."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.volume import load_volume, validate_labels, rag_pairs
from repro.sam import load_predictor, mask_prompt, choose_mask

load_h5 = load_volume


def node_features(fragments, feats):
    """Accumulate means per section; resize feature maps, never integer label IDs."""
    import cv2
    validate_labels(fragments)
    if feats.ndim != 4 or feats.shape[0] != 32 or feats.shape[1] != len(fragments):
        raise ValueError("HQ features must have shape (32,Z,H,W), matching fragment Z")
    ids = np.unique(fragments)
    sums = np.zeros((len(ids), 32), dtype=np.float64)
    counts = np.zeros(len(ids), dtype=np.int64)
    for z, section in enumerate(fragments):
        indices = np.searchsorted(ids, section.ravel())
        counts += np.bincount(indices, minlength=len(ids))
        for channel in range(32):
            plane = feats[channel, z]
            if plane.shape != section.shape:
                plane = cv2.resize(plane, (section.shape[1], section.shape[0]), interpolation=cv2.INTER_LINEAR)
            if not np.isfinite(plane).all():
                raise ValueError("HQ features contain nonfinite values")
            sums[:, channel] += np.bincount(indices, weights=plane.ravel(), minlength=len(ids))
    return {int(label): (sums[i] / counts[i]).astype(np.float32)
            for i, label in enumerate(ids) if label != 0}


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denominator, -1, 1)) if denominator > 0 else 0.0


def edge_cos(fragments, node_feats):
    return {(int(u), int(v)): {'cos': cosine(node_feats[int(u)], node_feats[int(v)])}
            for u, v in rag_pairs(fragments)}


def sample_points_from_mask(mask, num_points=5):
    import cv2
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1].copy()
    points = []
    for _ in range(num_points):
        _, value, _, point = cv2.minMaxLoc(distance)
        if value <= 0:
            break
        points.append(point)
        cv2.circle(distance, point, max(1, int(value)), 0, -1)
    return points


def select_mask(masks, iou_preds):
    return choose_mask((masks, iou_preds))


def predict_one(predictor, prompt):
    if 'mask' in prompt:
        output = predictor.predict(mask_input=mask_prompt(prompt['mask'], predictor), multimask_output=True)
    else:
        points = np.asarray(prompt['points'], dtype=np.float32)
        output = predictor.predict(point_coords=points, point_labels=np.ones(len(points)),
                                   multimask_output=True)
    return choose_mask(output)


def prompt_iou_edges(fragments, img_vol, predictor, erode=1, prompt_mode='mask',
                     num_points=5, use_distance=False):
    """Compare each pair on the SAME z, using prompts from z-1/z/z+1.

    Each image is encoded once. A node is prompted on a target section only if
    it occurs within one section of that target. This avoids comparing masks
    from different physical planes or counting clamped edge sections twice.
    """
    validate_labels(fragments)
    if img_vol.shape[:3] != fragments.shape:
        raise ValueError("Raw image ZYX and fragment ZYX shapes must match")
    if erode < 0 or num_points < 1:
        raise ValueError("erode must be >= 0 and num_points must be positive")
    edges = [(int(u), int(v)) for u, v in rag_pairs(fragments)]
    accumulated = {}
    for z in range(len(fragments)):
        prompts = {}
        sections = sorted(range(max(0, z - 1), min(len(fragments), z + 2)), key=lambda zz: (abs(zz-z), zz))
        for zz in sections:
            for label in np.unique(fragments[zz]):
                label = int(label)
                if label == 0 or label in prompts:
                    continue
                mask = fragments[zz] == label
                if prompt_mode == 'points':
                    prompts[label] = {'points': sample_points_from_mask(mask, num_points)}
                else:
                    eroded = ndimage.binary_erosion(mask, iterations=erode) if erode else mask
                    prompts[label] = {'mask': eroded if eroded.any() else mask}
        active = [(u, v) for u, v in edges if u in prompts and v in prompts]
        if not active:
            continue
        image = img_vol[z]
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        predictor.set_image(np.ascontiguousarray(image))
        predictions = {label: predict_one(predictor, prompts[label])
                       for label in sorted({label for pair in active for label in pair})}
        for u, v in active:
            a, score_a = predictions[u]
            b, score_b = predictions[v]
            weight = score_a * score_b
            if use_distance:
                def center(prompt):
                    if 'points' in prompt:
                        return np.mean(prompt['points'], axis=0)
                    yy, xx = np.nonzero(prompt['mask'])
                    return np.array([xx.mean(), yy.mean()])
                weight /= 1.0 + np.linalg.norm(center(prompts[u]) - center(prompts[v]))
            if weight <= 0:
                continue
            intersection = np.count_nonzero(a & b)
            values = np.array([intersection / max(np.count_nonzero(a | b), 1),
                               intersection / max(np.count_nonzero(a), 1),
                               intersection / max(np.count_nonzero(b), 1)])
            previous = accumulated.setdefault((u, v), [np.zeros(3), 0.0, 0])
            previous[0] += weight * values
            previous[1] += weight
            previous[2] += 1
    return {pair: dict(zip(['iou', 'ioa', 'iob'], (total / weight).tolist()),
                       n_sections=count)
            for pair, (total, weight, count) in accumulated.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fragments', required=True)
    parser.add_argument('--feats', required=True, help='32ZYX HQ features; XY may be at 1/4 resolution')
    parser.add_argument('--feats-key', default='main', help='Use features for an aligned teacher bundle')
    parser.add_argument('--img', help='raw ZYX H5/TIFF/NPY')
    parser.add_argument('--img-key', default='main', help='Use raw for an aligned teacher bundle')
    parser.add_argument('--checkpoint')
    parser.add_argument('--model-type', default='vit_h')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out-dir', default='./graph_feats')
    parser.add_argument('--no-prompt-iou', action='store_true')
    parser.add_argument('--prompt-mode', default='mask', choices=['mask', 'points'])
    parser.add_argument('--num-points', type=int, default=5)
    parser.add_argument('--erode', type=int, default=1)
    parser.add_argument('--use-distance', action='store_true')
    args = parser.parse_args()
    if not args.no_prompt_iou and not (args.img and args.checkpoint):
        parser.error('Provide --img and --checkpoint, or explicitly select --no-prompt-iou for a cosine-only ablation')
    fragments = validate_labels(load_volume(args.fragments))
    nf = node_features(fragments, load_volume(args.feats, args.feats_key))
    evidence = edge_cos(fragments, nf)
    if not args.no_prompt_iou:
        predictor = load_predictor(args.checkpoint, args.model_type, args.device)
        overlaps = prompt_iou_edges(fragments, load_volume(args.img, args.img_key), predictor,
                                   args.erode, args.prompt_mode, args.num_points, args.use_distance)
        for pair, values in overlaps.items():
            evidence[pair].update(values)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    import h5py
    with h5py.File(output / 'node_feats.h5', 'w') as handle:
        for label, vector in nf.items():
            handle.create_dataset(str(label), data=vector)
    (output / 'edge_feats.json').write_text(
        json.dumps({f'{u},{v}': values for (u, v), values in evidence.items()}, indent=2),
        encoding='utf-8')
    print(f'{len(nf)} nodes, {len(evidence)} edges → {output}')


if __name__ == '__main__':
    main()
