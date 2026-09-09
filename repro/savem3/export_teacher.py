"""Export aligned, label-free teacher targets for a raw ZYX ROI.

Requires BOTH an HQ-SAM checkpoint and a trained SAEM2 decoder checkpoint.
The 1024 encoder image, 512 student image/boundary, 256 HQ features and 64 SAM
embeddings cover exactly the same XY field of view. No GT is read.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.volume import load_volume
from repro.sam import load_hq_sam, resolve_device


def load_decoder(checkpoint, model_type, device):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'saem2'))
    from saem2.modeling import MaskDecoderHQ
    decoder = MaskDecoderHQ(model_type, initialize_from_sam=False)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = weights.get('state_dict', weights)
    weights = {key.removeprefix('module.'): value for key, value in weights.items()}
    if any(key.startswith('mask_decoder.') for key in weights):
        weights = {key.removeprefix('mask_decoder.'): value for key, value in weights.items()
                   if key.startswith('mask_decoder.')}
    decoder.load_state_dict(weights, strict=True)
    return decoder.to(device).eval()


def export(raw, sam, decoder, output, resolution_xyz, offset_xyz=(0, 0, 0), batch_size=1,
           provenance=None):
    import h5py
    import torch
    from torch.nn import functional as F
    if raw.ndim != 3 or raw.shape[1:] != (1024, 1024) or raw.dtype != np.uint8:
        raise ValueError('Input must be uint8 (Z,1024,1024); prepare an explicitly calibrated ROI first')
    if batch_size < 1 or not len(raw):
        raise ValueError('Batch size and Z must be positive')
    resolution = np.asarray(resolution_xyz, dtype=float)
    offset = np.asarray(offset_xyz, dtype=float)
    if resolution.shape != (3,) or not np.isfinite(resolution).all() or np.any(resolution <= 0):
        raise ValueError('Expected positive finite XYZ resolution in nm')
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('Expected finite XYZ voxel offset')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    with h5py.File(temporary, 'x') as handle, torch.inference_mode():
        raw_ds = handle.create_dataset('raw', (len(raw), 512, 512), dtype='u1', chunks=(1, 512, 512), compression='gzip')
        bdy_ds = handle.create_dataset('boundary', raw_ds.shape, dtype='f4', chunks=(1, 512, 512), compression='gzip')
        feat_ds = handle.create_dataset('features', (32, len(raw), 256, 256), dtype='f4', chunks=(32, 1, 256, 256), compression='lzf')
        emb_ds = handle.create_dataset('embeddings', (256, len(raw), 64, 64), dtype='f4', chunks=(256, 1, 64, 64), compression='lzf')
        handle.attrs['schema_version'] = 1
        handle.attrs['target_source'] = 'SAEM2 predictions; no manual labels'
        handle.attrs['resolution_xyz_nm'] = resolution * [2, 2, 1]
        handle.attrs['voxel_offset_xyz'] = offset / [2, 2, 1]
        handle.attrs['provenance'] = json.dumps(provenance or {})
        for first in range(0, len(raw), batch_size):
            last = min(first + batch_size, len(raw))
            images = torch.as_tensor(raw[first:last].copy(), device=sam.device).float()[:, None].repeat(1, 3, 1, 1)
            embeddings, intermediate = sam.image_encoder(sam.preprocess(images))
            features = (sam.mask_decoder.embedding_encoder(embeddings)
                        + sam.mask_decoder.compress_vit_feat(intermediate[0].permute(0, 3, 1, 2)))
            boxes = torch.tensor([[0, 0, 1024, 1024]], dtype=torch.float32, device=sam.device).repeat(last-first, 1)
            sparse, dense = sam.prompt_encoder(points=None, boxes=boxes, masks=None)
            # Decoder accepts per-image batches of prompts (each image has one full box).
            _, membrane = decoder(embeddings, sam.prompt_encoder.get_dense_pe().expand(last-first, -1, -1, -1),
                                  sparse[:, None], dense[:, None], False, True, intermediate)
            boundary = F.interpolate(membrane, size=(512, 512), mode='bilinear', align_corners=False).sigmoid()
            student_raw = F.interpolate(images[:, :1], size=(512, 512), mode='area')
            raw_ds[first:last] = student_raw[:, 0].round().clamp(0, 255).byte().cpu().numpy()
            bdy_ds[first:last] = boundary[:, 0].cpu().numpy()
            feat_ds[:, first:last] = features.permute(1, 0, 2, 3).cpu().numpy()
            emb_ds[:, first:last] = embeddings.permute(1, 0, 2, 3).cpu().numpy()
            print(f'Teacher slices {last}/{len(raw)}', flush=True)
        handle.attrs['complete'] = True
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--checkpoint', required=True, help='Full HQ-SAM checkpoint')
    parser.add_argument('--saem2-checkpoint', required=True, help='Trained epoch_N.pth membrane decoder')
    parser.add_argument('--model-type', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_h')
    parser.add_argument('--resolution', type=float, nargs=3, required=True)
    parser.add_argument('--offset', type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = resolve_device(args.device)
    decoder = load_decoder(args.saem2_checkpoint, args.model_type, device)
    sam = load_hq_sam(args.checkpoint, args.model_type, str(device))
    export(load_volume(args.raw), sam, decoder, args.output, args.resolution, args.offset,
           args.batch, {'raw': str(Path(args.raw).resolve()), 'hq_checkpoint': str(Path(args.checkpoint).resolve()),
                        'saem2_checkpoint': str(Path(args.saem2_checkpoint).resolve()), 'model_type': args.model_type})


if __name__ == '__main__':
    main()
