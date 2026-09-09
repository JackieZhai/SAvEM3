"""Run actual SAvEM3 inference for explicit ROI chunks, then stitch probabilities.

ROI JSON: {"chunks": [{"offset": [z,y,x], "size": [64,512,512]}, ...]}.
Offsets index the supplied raw volume. Output coordinates are stored separately;
uncovered voxels have boundary probability 1 and coverage 0, never neuron interior.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.volume import load_volume, save_volume


def stitch(chunks, offsets, overlap=16, return_metadata=False, max_voxels=128_000_000):
    if not chunks or len(chunks) != len(offsets):
        raise ValueError('Expected equal nonempty chunks and offsets')
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.shape != (len(chunks), 3) or any(chunk.ndim != 3 for chunk in chunks):
        raise ValueError('Chunks and offsets must be ZYX')
    origin = offsets.min(axis=0)
    relative = offsets-origin
    shape = np.max(relative + np.array([chunk.shape for chunk in chunks]), axis=0)
    if int(np.prod(shape)) > max_voxels:
        raise ValueError('ROI bounding box exceeds the memory limit; partition the ROI')
    total = np.zeros(tuple(shape), np.float32)
    coverage = np.zeros(tuple(shape), np.uint32)
    for chunk, offset in zip(chunks, relative):
        if not np.isfinite(chunk).all() or np.any(chunk < 0) or np.any(chunk > 1):
            raise ValueError('Chunks must contain boundary probabilities in [0,1]')
        region = tuple(slice(int(start), int(start+size)) for start, size in zip(offset, chunk.shape))
        total[region] += chunk
        coverage[region] += 1
    output = np.divide(total, coverage, out=np.ones_like(total), where=coverage > 0)
    return (output, coverage, origin) if return_metadata else output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roi', required=True)
    parser.add_argument('--raw', help='uint8 raw H5/TIFF/NPY at student resolution')
    parser.add_argument('--raw-key', default='main')
    parser.add_argument('--ckpt')
    parser.add_argument('--out-dir', default='./sparse_out')
    parser.add_argument('--stitch-only', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--patch', nargs=3, type=int, default=[18, 160, 160])
    parser.add_argument('--overlap', nargs=3, type=int, default=[4, 32, 32])
    parser.add_argument('--max-voxels', type=int, default=128_000_000)
    args = parser.parse_args()
    roi = json.loads(Path(args.roi).read_text())
    if not roi.get('chunks'):
        parser.error('ROI must contain at least one chunk')
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = model = device = None
    if not args.stitch_only:
        if not args.raw or not args.ckpt:
            parser.error('Inference needs --raw and --ckpt; use --stitch-only for existing chunks')
        import torch
        from repro.savem3.distill import build_student, predict_volume
        from repro.sam import resolve_device
        raw = load_volume(args.raw, args.raw_key)
        device = resolve_device(args.device)
        model = build_student().to(device)
        state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        weights = state.get('model', state.get('model_weights', state))
        model.load_state_dict({key.removeprefix('module.'): value for key, value in weights.items()}, strict=True)
    chunks, offsets = [], []
    for index, entry in enumerate(roi['chunks']):
        offset, size = entry['offset'], entry['size']
        if len(offset) != 3 or len(size) != 3 or any(int(value) != value for value in offset+size) or min(size) <= 0:
            raise ValueError('Each chunk needs integer ZYX offset and positive size')
        path = output / f'chunk_{index:04d}.h5'
        if args.stitch_only:
            chunk = load_volume(path)
        else:
            if min(offset) < 0 or any(start+extent > bound for start, extent, bound in zip(offset, size, raw.shape)):
                raise ValueError(f'Chunk {index} falls outside raw volume')
            region = tuple(slice(start, start+extent) for start, extent in zip(offset, size))
            chunk = predict_volume(model, raw[region], args.patch, args.overlap, device)
            save_volume(path, chunk, offset_zyx=offset, checkpoint=str(Path(args.ckpt).resolve()))
        if tuple(chunk.shape) != tuple(size):
            raise ValueError(f'Chunk {index} shape differs from ROI manifest')
        chunks.append(chunk)
        offsets.append(offset)
    stitched, coverage, origin = stitch(chunks, offsets, return_metadata=True, max_voxels=args.max_voxels)
    save_volume(output / 'boundary.h5', stitched, offset_zyx=origin)
    save_volume(output / 'coverage.h5', coverage, offset_zyx=origin)
    report = {'offset_zyx': origin.tolist(), 'shape_zyx': list(stitched.shape),
              'covered_voxels': int(np.count_nonzero(coverage)), 'chunks': len(chunks)}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
