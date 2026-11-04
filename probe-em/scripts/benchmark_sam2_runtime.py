"""Measure local SAM2 image/video latency and agreement across encoder runtimes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import load_local
from probe_em.prompt_models import PointModel
from probe_em.propagation import PropagationGrid


def agreement(a, b):
    if a.shape != b.shape:
        raise ValueError('Reference masks must have the same geometry')
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    intersection = int(np.count_nonzero(a & b))
    total = int(a.sum())+int(b.sum())
    return {'dice': 2*intersection/total if total else 1.,
            'different_voxels': int(np.count_nonzero(a != b)), 'voxels': int(a.size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', type=Path, required=True)
    parser.add_argument('--point', type=int, nargs=3, required=True, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-config', default='sam2_hiera_t.yaml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--runtime', choices=['torch', 'onnx'], default='torch')
    parser.add_argument('--onnx-dir', type=Path)
    parser.add_argument('--provider', choices=['cpu', 'coreml', 'cuda', 'directml'], default='cpu')
    parser.add_argument('--feature-cache-mb', type=float, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--slices', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reference', type=Path, help='A completed benchmark directory with identical inputs')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.slices < 3 or args.slices % 2 == 0 or args.repeats < 2:
        parser.error('Use positive threads, an odd section count >=3 and at least two repeats')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'status.json').write_text(json.dumps({'status': 'running'}))
    try:
        run_benchmark(args, output)
    except BaseException as error:
        (output/'status.json').write_text(json.dumps({'status': 'failed', 'error': f'{type(error).__name__}: {error}'}))
        raise


def run_benchmark(args, output):
    torch.set_num_threads(args.threads)
    raw, resolution, offset = load_local(args.local_dir/'raw')
    point = np.asarray(args.point, int)
    local = point-offset
    if np.any(local < 0) or np.any(local >= raw.shape) or local[2] < args.slices//2 or local[2]+args.slices//2 >= raw.shape[2]:
        raise ValueError('Choose an interior point with enough sections around it')
    lo, hi = int(point[2])-args.slices//2, int(point[2])+args.slices//2
    grid = PropagationGrid(raw, offset, lo, hi)
    frames = grid.encode(output/'frames', threading.Event())
    prompts = grid.prompts(np.asarray([point]), np.array([1]))
    model = PointModel('sam2', args.checkpoint, model_config=args.model_config, device=args.device,
                       sam2_runtime=args.runtime, onnx_dir=args.onnx_dir, onnx_provider=args.provider,
                       onnx_threads=args.threads, feature_cache_mb=args.feature_cache_mb)

    def sync():
        if model.device == 'mps':
            torch.mps.synchronize()
        elif model.device.startswith('cuda'):
            torch.cuda.synchronize(model.device)

    def image_prediction(z):
        image = np.repeat(raw[:, :, z-int(offset[2])].T[..., None], 3, axis=2)
        sync(); start = time.perf_counter()
        mask, score = model.predict(image, np.asarray([local[:2]], np.float32), np.array([1]))
        sync()
        elapsed = time.perf_counter()-start
        if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError('Image predictor must return binary masks')
        return elapsed, np.asarray(mask, dtype=bool), score

    cold, _, _ = image_prediction(int(point[2]))
    image_times, image_masks = [], []
    sections = [int(point[2])-1, int(point[2]), int(point[2])+1]
    for _ in range(args.repeats):
        image_masks = []
        for z in sections:
            elapsed, mask, _ = image_prediction(z)
            image_times.append(elapsed); image_masks.append(mask)
    video_times, video_masks, repeat_agreement = [], None, []
    for _ in range(args.repeats):
        sync(); start = time.perf_counter()
        predictions = dict(model.propagate(output/'frames', prompts, cancel=threading.Event()))
        sync(); video_times.append(time.perf_counter()-start)
        if set(predictions) != set(range(args.slices)):
            raise ValueError('Propagation returned an incomplete window')
        current = np.stack([predictions[i] for i in range(args.slices)])
        if video_masks is not None:
            repeat_agreement.append(agreement(current, video_masks))
        video_masks = current
    image_masks = np.stack(image_masks)
    np.savez_compressed(output/'masks.npz', images=image_masks, video=video_masks)
    signature = {'raw_sha256': hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest(),
                 'point_xyz': point.tolist(), 'z_range_inclusive': [lo, hi],
                 'image_sections': sections, 'frame_hashes': [item['sha256'] for item in frames]}
    comparisons = None
    if args.reference:
        reference = json.loads((args.reference/'report.json').read_text())
        if signature != reference['input']:
            raise ValueError('Benchmark reference uses different inputs')
        with np.load(args.reference/'masks.npz', allow_pickle=False) as saved:
            comparisons = {'images': agreement(image_masks, saved['images']),
                           'video': agreement(video_masks, saved['video'])}
    report = {'platform': platform.platform(), 'torch_version': torch.__version__, 'threads': args.threads,
              'model': model.metadata, 'input': signature, 'repeats': args.repeats,
              'timing_seconds': {'cold_image': cold, 'alternating_images': image_times,
                                 'video_runs': video_times, 'warm_video_median': float(np.median(video_times[1:]))},
              'agreement_with_reference': comparisons,
              'agreement_between_video_repeats': repeat_agreement,
              'scope': 'Batch one; image encoder runtime changes only; PyTorch decoder and fresh video memory'}
    (output/'report.json').write_text(json.dumps(report, indent=2))
    (output/'status.json').write_text(json.dumps({'status': 'complete'}))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
