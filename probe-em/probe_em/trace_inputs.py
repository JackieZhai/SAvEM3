"""Anchor-aware ASP inputs with explicit, auditable preparation outcomes.

This is a guarded extension of the paper's five-frame ASP, not new learned
evidence. Native labels and consecutive sections are retained without resizing.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
from scipy.ndimage import label


def nearest_component(mask, point_xy):
    """Select the component nearest a physical query's pixel, not crop center."""
    components, count = label(mask, np.ones((3, 3), bool))
    if not count:
        return mask.astype(bool)
    rows, columns = np.where(mask)
    index = np.argmin((columns-point_xy[0])**2 + (rows-point_xy[1])**2)
    return components == components[rows[index], columns[index]]


def prepare_video(seg_vol, raw_vol, center, target_id, neighbor_id, r_vox,
                  output_dir, base_name, num_frames=5, anchor=None,
                  search_distance_nm=500., max_frames=24):
    """Search real source sections across gaps and keep frame numbering dense.

    A five-frame minimum is extended only to reach the nominated candidate plus
    one confirmation section. Impossible/out-of-ROI cases become review items.
    """
    center = np.asarray(center, dtype=int)
    anchor = center if anchor is None else np.asarray(anchor, dtype=float)
    lo, hi = np.asarray(seg_vol.bounds.minpt), np.asarray(seg_vol.bounds.maxpt)
    start, end = np.maximum(center-r_vox, lo), np.minimum(center+r_vox+1, hi)
    radius_z = int(np.ceil(search_distance_nm / seg_vol.resolution[2]))
    start[2], end[2] = max(lo[2], center[2]-radius_z), min(hi[2], center[2]+radius_z+1)
    if np.any(end <= start):
        return {'status': 'deferred', 'reason': 'outside_roi'}
    block = np.asarray(seg_vol[tuple(slice(a, b) for a, b in zip(start, end))])[..., 0]
    points = np.column_stack(np.where(block == target_id)) + start
    if not len(points):
        return {'status': 'deferred', 'reason': 'source_not_in_search_window'}
    # The closest real source voxel selects the source end of the gap and its
    # cross-section. Retain the skeleton anchor separately for review.
    distances = np.sum(((points-center)*seg_vol.resolution)**2, axis=1)
    source = points[np.argmin(distances)]
    source_z = int(source[2])
    if source_z == center[2]:
        return {'status': 'deferred', 'reason': 'coexisting_pair_requires_pec'}
    step = 1 if center[2] > source_z else -1
    required = abs(int(center[2])-source_z)+2
    count = max(num_frames, required)
    if count > max_frames:
        return {'status': 'deferred', 'reason': 'gap_exceeds_frame_cap', 'required_frames': count}
    sequence = [source_z+i*step for i in range(count) if lo[2] <= source_z+i*step < hi[2]]
    if len(sequence) < 2 or int(center[2]) not in sequence:
        return {'status': 'deferred', 'reason': 'insufficient_roi_context'}
    folder = Path(output_dir) / base_name
    if folder.exists():
        # A failed node may have left complete or partial model inputs. Preserve
        # them outside the scanner's input root before rebuilding on resume.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = Path(output_dir).parent / 'input_attempts' / Path(output_dir).name / (base_name+'_'+stamp)
        backup.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(backup)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'video_frames').mkdir()
    (folder / 'segmentation_data').mkdir()
    prompt = None
    for index, z in enumerate(sequence):
        query = (slice(start[0], end[0]), slice(start[1], end[1]), slice(z, z+1))
        seg = np.asarray(seg_vol[query])[:, :, 0, 0].T
        raw = np.asarray(raw_vol[query])[:, :, 0, 0].T
        if raw.dtype != np.uint8:
            raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if not cv2.imwrite(str(folder / 'video_frames' / f'{index:05d}.jpg'), raw):
            raise OSError('Failed to write ASP image')
        np.save(folder / 'segmentation_data' / f'{index:05d}_seg.npy', seg)
        if index == 0:
            prompt = nearest_component(seg == target_id, source[:2]-start[:2])
            if not cv2.imwrite(str(folder / 'prompt_mask_00000.png'), prompt.astype(np.uint8)*255):
                raise OSError('Failed to write ASP prompt')
    metadata = {'schema_version': 1, 'target_id': str(target_id), 'neighbor_id': str(neighbor_id),
                'z_sequence': sequence, 'xy_offset': start[:2].tolist(),
                'resolution_xyz_nm': np.asarray(seg_vol.resolution).tolist(),
                'source_xyz': source.tolist(), 'endpoint_xyz': anchor.tolist(),
                'prompt_pixels': int(prompt.sum()), 'extended_window': len(sequence) > num_frames,
                'roi_truncated': len(sequence) < count}
    (folder / 'case.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return {'status': 'saved', 'path': str(folder), **metadata}
