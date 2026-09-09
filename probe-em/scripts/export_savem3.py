"""Export a SAvEM3 ROI (ZYX H5) to Probe-EM/Neuroglancer precomputed volumes.

The export preserves integer instance IDs and physical coordinates, and can
generate per-instance skeletons for Probe-EM's geometric search at mip 0.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.volume import load_volume, validate_labels


def write_precomputed(path, data_xyz, resolution, offset, segmentation=False):
    from cloudvolume import CloudVolume
    info = CloudVolume.create_new_info(num_channels=1,
        layer_type='segmentation' if segmentation else 'image',
        data_type=str(data_xyz.dtype), encoding='raw', resolution=list(resolution),
        voxel_offset=list(offset), chunk_size=[64, 64, 32], volume_size=list(data_xyz.shape))
    volume = CloudVolume(Path(path).resolve().as_uri(), info=info, mip=0,
                         non_aligned_writes=True, parallel=False, progress=False)
    volume.commit_info()
    volume[volume.bounds] = np.asfortranarray(data_xyz[..., None])
    return volume


def export_bundle(raw, labels, output, resolution, offset=(0, 0, 0), skeletonize=False):
    validate_labels(labels)
    if raw.shape != labels.shape or raw.dtype != np.uint8:
        raise ValueError('Raw must be uint8 ZYX and have the same shape as integer labels')
    resolution, offset = np.asarray(resolution, dtype=float), np.asarray(offset)
    if resolution.shape != (3,) or not np.isfinite(resolution).all() or np.any(resolution <= 0):
        raise ValueError('Resolution must contain positive finite XYZ nm/voxel values')
    if offset.shape != (3,) or not np.isfinite(offset).all() or np.any(offset != np.round(offset)):
        raise ValueError('Offset must contain three integer XYZ voxel indices')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new export directory: {output}')
    output.mkdir(parents=True)
    raw_xyz = raw.transpose(2, 1, 0)
    label_xyz = labels.astype(np.uint64).transpose(2, 1, 0)
    write_precomputed(output / 'raw', raw_xyz, resolution, offset)
    segmentation = write_precomputed(output / 'seg', label_xyz, resolution, offset, True)
    ids, counts = np.unique(label_xyz, return_counts=True)
    skeleton_ids = []
    if skeletonize:
        import kimimaro
        skeletons = kimimaro.skeletonize(np.asfortranarray(label_xyz),
                     anisotropy=resolution, dust_threshold=0, fix_branching=True,
                     teasar_params={'scale': 1.5, 'const': 300}, parallel=1, progress=False)
        segmentation.info['skeletons'] = 'skeletons'
        segmentation.commit_info()
        segmentation.skeleton.meta.commit_info()
        cache = {}
        for label, skeleton in skeletons.items():
            skeleton.id = int(label)
            skeleton.vertices += (offset * resolution).astype(np.float32)
            segmentation.skeleton.upload(skeleton)
            cache[str(label)] = {'vertices': skeleton.vertices.tolist(), 'edges': skeleton.edges.tolist()}
            skeleton_ids.append(int(label))
        (output / 'traced_skeletons.json').write_text(json.dumps(cache), encoding='utf-8')
    manifest = {'schema_version': 1, 'source_axes': 'ZYX', 'storage_axes': 'XYZ',
                'resolution_xyz_nm': resolution.tolist(), 'voxel_offset_xyz': offset.tolist(),
                'shape_xyz': list(raw_xyz.shape), 'raw_path': (output / 'raw').as_uri(),
                'seg_path': (output / 'seg').as_uri(), 'target_mip': 0,
                'counts': {str(label): int(count) for label, count in zip(ids, counts) if label},
                'skeleton_ids': [str(label) for label in skeleton_ids]}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--raw-key', default='main', help='Use raw for an aligned teacher bundle')
    parser.add_argument('--seg', required=True)
    parser.add_argument('--seg-key', default='main')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resolution', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--offset', type=int, nargs=3, default=[0, 0, 0], metavar=('X', 'Y', 'Z'))
    parser.add_argument('--skeletonize', action='store_true')
    parser.add_argument('--checkpoint', help='SAM 2/NeuroSAM 2 checkpoint; also writes runnable trace config')
    parser.add_argument('--model-config', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    raw, labels = load_volume(args.raw, args.raw_key), load_volume(args.seg, args.seg_key)
    if args.seed is not None and (args.seed <= 0 or not np.any(labels == args.seed)):
        parser.error('--seed must be a foreground segment present in the ROI')
    if args.checkpoint and not Path(args.checkpoint).is_file():
        parser.error('SAM 2 checkpoint does not exist')
    manifest = export_bundle(raw, labels, args.output, args.resolution, args.offset, args.skeletonize)
    if args.checkpoint and args.seed:
        if str(args.seed) not in manifest['skeleton_ids']:
            raise ValueError('Seed needs a skeleton; export with --skeletonize and choose a nontrivial segment')
        output = Path(args.output).resolve()
        config = {'raw_path': manifest['raw_path'], 'seg_path': manifest['seg_path'],
                  'checkpoint_sam': str(Path(args.checkpoint).resolve()), 'model_cfg_sam': args.model_config,
                  'seed_ids': [args.seed], 'target_mip': 0, 'max_workers': 1, 'device': 'auto',
                  'output_root': str(output / 'trace_results'), 'suffix': 'sam', 'debug_limit': 2,
                  'slice_workers': 1}
        (output / 'trace_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
