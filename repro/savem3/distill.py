"""Portable full-stage distillation on aligned SAEM2 teacher bundles.

Uses the original UNetST_PNI architecture. Training requires no manual labels,
waterz, CUDA monkeypatch, or precomputed affinity/GT weights. See README.md.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.sam import resolve_device
from repro.volume import load_volume, save_volume


def build_student():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'savem3'))
    from model.model_superhuman import UNetST_PNI
    return UNetST_PNI(in_planes=1, out_planes=1, filters=[32, 64, 128, 256],
                     upsample_mode='transposeS', merge_mode='cat', if_sigmoid=True)


def validate_bundle(handle):
    if not handle.attrs.get('complete', False):
        raise ValueError('Teacher bundle is incomplete')
    shape = handle['raw'].shape
    if handle['raw'].dtype != np.uint8 or not all(shape):
        raise ValueError('Teacher raw must be nonempty uint8')
    if len(shape) != 3 or handle['boundary'].shape != shape:
        raise ValueError('raw/boundary must share ZYX shape')
    z, y, x = shape
    if y % 8 or x % 8 or handle['features'].shape != (32, z, y//2, x//2) or handle['embeddings'].shape != (256, z, y//8, x//8):
        raise ValueError('Expected raw:feature:embedding XY sizes 8:4:1 and channels 1:32:256')
    return shape


def sample_batch(handle, rng, patch, batch_size, device):
    shape = validate_bundle(handle)
    if any(p > size for p, size in zip(patch, shape)) or patch[1] % 8 or patch[2] % 8 or min(patch) < 1:
        raise ValueError('Patch must fit the teacher ROI; XY dimensions must be multiples of 8')
    samples = {name: [] for name in ('raw', 'boundary', 'features', 'embeddings')}
    if 'flow' in handle:
        samples['flow'] = []
    for _ in range(batch_size):
        starts = [int(rng.integers(0, shape[0]-patch[0]+1))]
        starts.extend(int(rng.integers(0, (shape[i]-patch[i])//8+1))*8 for i in (1, 2))
        z, y, x = starts
        dz, dy, dx = patch
        for name in ('raw', 'boundary'):
            samples[name].append(handle[name][z:z+dz, y:y+dy, x:x+dx][None])
        for name, scale in (('features', 2), ('embeddings', 8)):
            samples[name].append(handle[name][:, z:z+dz, y//scale:(y+dy)//scale, x//scale:(x+dx)//scale])
        if 'flow' in samples:
            samples['flow'].append(handle['flow'][:, z:z+dz-1, y//2:(y+dy)//2, x//2:(x+dx)//2])
    tensors = {name: torch.as_tensor(np.stack(values), device=device, dtype=torch.float32)
               for name, values in samples.items()}
    tensors['raw'] /= 255.0
    return tensors


def distillation_loss(outputs, targets, feature_weight=1.0, embedding_weight=1.0,
                      secondary_weight=0.0, epsilon=0.1, flow=None):
    boundary, feature, embedding = outputs
    losses = {'boundary': F.binary_cross_entropy(boundary, targets['boundary'])}
    losses['feature'] = F.l1_loss(feature[:, :32], targets['features'])
    losses['embedding'] = F.l1_loss(embedding, targets['embeddings'])
    total = losses['boundary'] + feature_weight * losses['feature'] + embedding_weight * losses['embedding']
    if secondary_weight:
        if feature.shape[2] < 2 or flow is None:
            raise ValueError('Full secondary loss requires >=2 sections and an explicitly supplied optical flow')
        # f is defined so current(x-f_x,y-f_y,z) matches previous(x,y,z-1).
        batch, channels, depth, height, width = feature.shape
        if flow.shape != (batch, 2, depth-1, height, width):
            raise ValueError('Flow shape must be (B,2,Z-1,Hfeature,Wfeature), in feature pixels')
        if not torch.isfinite(flow).all():
            raise ValueError('Flow must contain only finite values')
        yy, xx = torch.meshgrid(torch.arange(height, device=feature.device),
                                torch.arange(width, device=feature.device), indexing='ij')
        grid = torch.stack((xx, yy), -1).float()[None, None] - flow.permute(0, 2, 3, 4, 1)
        valid = (grid[..., 0] >= 0) & (grid[..., 0] <= width-1) & (grid[..., 1] >= 0) & (grid[..., 1] <= height-1)
        grid[..., 0] = 2*grid[..., 0]/max(width-1, 1)-1
        grid[..., 1] = 2*grid[..., 1]/max(height-1, 1)-1
        current = feature[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        warped = F.grid_sample(current, grid.reshape(-1, height, width, 2), align_corners=True)
        previous = feature[:, :, :-1].permute(0, 2, 1, 3, 4).reshape_as(warped)
        difference = torch.linalg.vector_norm(warped-previous, dim=1)
        if not valid.any():
            raise ValueError('Optical flow contains no valid in-bounds correspondences')
        losses['motion'] = difference[valid.reshape(-1, height, width)].mean()
        losses['regularization'] = F.relu(torch.linalg.vector_norm(feature[:, :, 1:]-feature[:, :, :-1], dim=1)-epsilon).mean()
        total = total + secondary_weight * (losses['motion'] + losses['regularization'])
    return total, losses


def train(args):
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build_student().to(device)
    if args.init_encoder:
        initial = torch.load(args.init_encoder, map_location='cpu', weights_only=False)
        initial = initial.get('model', initial.get('model_weights', initial))
        initial = {key.removeprefix('module.'): value for key, value in initial.items()}
        student = model.state_dict()
        expected = [key for key in student if key.split('.')[0] in ('embed_in', 'conv0', 'conv1', 'conv2', 'center')]
        if not all(key in initial and initial[key].shape == student[key].shape for key in expected):
            raise ValueError('Pretraining checkpoint does not cover the complete matching student encoder')
        student.update({key: initial[key] for key in expected})
        model.load_state_dict(student)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Output already has a checkpoint; use --resume or a fresh output directory')
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        for name in ('teacher', 'ablation', 'patch', 'secondary_weight', 'epsilon', 'lr', 'batch'):
            if state['config'].get(name) != getattr(args, name):
                raise ValueError(f'Resume configuration mismatch: {name}')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
    weights = {'bdy': (0, 0), 'bdy_embed': (0, 1), 'full': (1, 1)}[args.ablation]
    if args.steps <= start or args.batch < 1 or min(args.log_every, args.save_every) < 1:
        raise ValueError('steps must exceed the resumed step and batch must be positive')
    with h5py.File(args.teacher, 'r') as handle:
        shape = validate_bundle(handle)
        if args.secondary_weight:
            if 'flow' not in handle or handle['flow'].shape != (2, shape[0]-1, shape[1]//2, shape[2]//2):
                raise ValueError('Secondary losses require flow (2,Z-1,H/2,W/2) in the teacher bundle')
        model.train()
        for step in range(start+1, args.steps+1):
            batch = sample_batch(handle, rng, args.patch, args.batch, device)
            optimizer.zero_grad(set_to_none=True)
            total, losses = distillation_loss(model(batch['raw']), batch, *weights,
                                              args.secondary_weight, args.epsilon, batch.get('flow'))
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            total.backward()
            optimizer.step()
            if step == start+1 or step % args.log_every == 0:
                print(json.dumps({'step': step, 'loss': float(total.detach()),
                                  **{key: float(value.detach()) for key, value in losses.items()}}), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'step': step, 'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                              'config': vars(args), 'teacher': str(Path(args.teacher).resolve()),
                              'architecture': 'UNetST_PNI_32_64_128_256',
                              'teacher_metadata': {key: str(value) for key, value in handle.attrs.items()}}
                temporary = output / 'last.pt.partial'
                torch.save(checkpoint, temporary)
                temporary.replace(output / 'last.pt')


def tile_starts(size, tile, overlap):
    if tile < 1 or overlap < 0 or overlap >= tile:
        raise ValueError('Require 0 <= overlap < tile')
    end = max(0, size-tile)
    return sorted(set(list(range(0, end+1, tile-overlap)) + [end]))


def predict_volume(model, raw, patch, overlap, device):
    if raw.ndim != 3 or raw.dtype != np.uint8:
        raise ValueError('Inference raw must be uint8 ZYX')
    if patch[1] % 8 or patch[2] % 8:
        raise ValueError('Patch XY must be multiples of 8')
    padding = [(0, max(0, p-s)) for s, p in zip(raw.shape, patch)]
    padded = np.pad(raw, padding, mode='edge')
    total, count = np.zeros(padded.shape, np.float32), np.zeros(padded.shape, np.float32)
    starts = [tile_starts(size, tile, margin) for size, tile, margin in zip(padded.shape, patch, overlap)]
    model.eval()
    with torch.inference_mode():
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    region = (slice(z, z+patch[0]), slice(y, y+patch[1]), slice(x, x+patch[2]))
                    image = torch.as_tensor(padded[region].copy(), device=device).float()[None, None]/255
                    prediction = model(image)[0][0, 0].cpu().numpy()
                    total[region] += prediction
                    count[region] += 1
    region = tuple(slice(0, size) for size in raw.shape)
    return (total / count)[region]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--teacher', required=True)
    training.add_argument('--output', required=True)
    training.add_argument('--steps', type=int, default=200000)
    training.add_argument('--batch', type=int, default=4)
    training.add_argument('--lr', type=float, default=1e-4)
    training.add_argument('--ablation', choices=['bdy', 'bdy_embed', 'full'], default='full')
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--resume')
    training.add_argument('--init-encoder', help='Matching self-supervised student encoder checkpoint')
    training.add_argument('--secondary-weight', type=float, default=0.0,
                          help='Use 0.05 for paper secondary losses; requires externally prepared flow')
    training.add_argument('--epsilon', type=float, default=0.1)
    training.add_argument('--save-every', type=int, default=1000)
    training.add_argument('--log-every', type=int, default=100)
    inference = sub.add_parser('infer')
    inference.add_argument('--raw', required=True)
    inference.add_argument('--raw-key', default='main', help='Use raw for an aligned teacher bundle')
    inference.add_argument('--checkpoint', required=True)
    inference.add_argument('--output', required=True)
    inference.add_argument('--overlap', type=int, nargs=3, default=[4, 32, 32])
    for command in (training, inference):
        command.add_argument('--device', default='auto')
        command.add_argument('--patch', type=int, nargs=3, default=[18, 160, 160])
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    else:
        device = resolve_device(args.device)
        model = build_student().to(device)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        weights = state.get('model', state.get('model_weights', state))
        model.load_state_dict({key.removeprefix('module.'): value for key, value in weights.items()}, strict=True)
        output = predict_volume(model, load_volume(args.raw, args.raw_key), args.patch, args.overlap, device)
        save_volume(args.output, output, kind='boundary', checkpoint=str(Path(args.checkpoint).resolve()))


if __name__ == '__main__':
    main()
