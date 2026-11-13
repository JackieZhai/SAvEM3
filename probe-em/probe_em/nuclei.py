"""Bounded, center-outward SAM2 tracing on the 32 x 32 x 30 nm overview."""
from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage

from probe_em.propagation import check_cancelled


RESOLUTION = [32, 32, 30]


@dataclass(frozen=True)
class NucleusOptions:
    box_size: int = 384
    max_radius: int = 256
    block_slices: int = 16
    empty_sections: int = 2

    def __post_init__(self):
        for name, low, high in [('box_size', 32, 1024), ('max_radius', 1, 1024),
                                ('block_slices', 2, 64), ('empty_sections', 1, 8)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer from {low} to {high}')
        if self.empty_sections > self.block_slices:
            raise ValueError('empty_sections must not exceed block_slices')


class NucleusRaw:
    """Read only requested blocks; the local HTTP server caches source chunks."""
    def __init__(self, url, info):
        from cloudvolume import CloudVolume
        scale = info['scales'][0]
        if (info['type'] != 'image' or info['data_type'] != 'uint8'
                or info['num_channels'] != 1 or scale['resolution'] != RESOLUTION):
            raise ValueError('Nuclei mode requires single-channel uint8 raw at 32 x 32 x 30 nm')
        self.offset = np.asarray(scale['voxel_offset'], dtype=int)
        self.shape = np.asarray(scale['size'], dtype=int)
        self.resolution = np.asarray(RESOLUTION)
        self.volume = CloudVolume(url.removeprefix('precomputed://'), mip=0, info=info,
                                  parallel=False, progress=False, fill_missing=False)

    def read(self, low, high):
        low, high = np.asarray(low, int), np.asarray(high, int)
        if np.any(low < self.offset) or np.any(high > self.offset+self.shape) or np.any(high <= low):
            raise ValueError('Nucleus block is outside the raw volume')
        value = np.asarray(self.volume[tuple(slice(a, b) for a, b in zip(low, high))])[..., 0]
        if value.dtype != np.uint8 or tuple(value.shape) != tuple(high-low):
            raise ValueError('Raw block has unexpected shape or dtype')
        return value


def center_component(mask, *, point=None, previous=None):
    """Keep the prompted component, or the component overlapping the last section."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError('SAM2 returned an invalid nucleus mask')
    labels, count = ndimage.label(mask)
    if not count:
        return np.zeros(mask.shape, dtype=bool)
    if point is not None:
        x, y = np.floor(point).astype(int)
        selected = labels[y, x]
    else:
        support = ndimage.binary_dilation(previous, iterations=2)
        overlap = np.bincount(labels[support], minlength=count+1)
        overlap[0] = 0
        selected = int(overlap.argmax()) if overlap.any() else 0
    return labels == selected if selected else np.zeros(mask.shape, dtype=bool)


def touches_boundary(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def trace_nucleus(raw, model, center, output, options, cancel, progress):
    """Track two independent directions in small videos, carrying the last mask.

    Each block resets video memory and conditions on the previous boundary mask.
    No synthetic negative points are added. Missing raw chunks fail the run.
    """
    center = np.asarray(center, dtype=float).copy()
    if (center.shape != (3,) or not np.isfinite(center).all()
            or np.any(center < raw.offset) or np.any(center >= raw.offset+raw.shape)):
        raise ValueError('Place the center inside the raw volume')
    center[2] = np.floor(center[2])
    xy_low = np.maximum(raw.offset[:2], np.floor(center[:2]).astype(int)-options.box_size//2)
    xy_high = np.minimum(raw.offset[:2]+raw.shape[:2], xy_low+options.box_size)
    xy_low = np.maximum(raw.offset[:2], xy_high-options.box_size)
    anchor = int(center[2])
    point = center[:2]-xy_low
    check_cancelled(cancel)
    progress(f'Predicting the nucleus center at z={anchor}.')
    image = raw.read([*xy_low, anchor], [*xy_high, anchor+1])[:, :, 0].T
    seed = model.predict_refined(np.repeat(image[..., None], 3, axis=2), point[None], np.array([1]))
    check_cancelled(cancel)
    seed_mask = center_component(seed['mask'], point=point)
    if not seed_mask.any():
        raise ValueError('SAM2 did not include the nucleus center; reposition the center and retry')
    if touches_boundary(seed_mask):
        raise ValueError('Center mask reaches the XY crop edge; increase --nuclei-box-size or reposition the center')
    masks = {anchor: seed_mask}
    stops, blocks = {}, []
    for direction in (1, -1):
        current, conditioning, previous, last_nonempty = anchor, seed_mask, seed_mask, anchor
        boundary = int(raw.offset[2]+raw.shape[2]-1 if direction > 0 else raw.offset[2])
        limit = min(boundary, anchor+options.max_radius) if direction > 0 else max(boundary, anchor-options.max_radius)
        reason, empty = ('volume_boundary' if limit == boundary else 'radius_limit'), 0
        finished = False
        while current != limit and not finished:
            check_cancelled(cancel)
            end = min(limit, current+options.block_slices) if direction > 0 else max(limit, current-options.block_slices)
            first, last = sorted((current, end))
            progress(f'Reading z={first}..{last}; propagating {"up" if direction > 0 else "down"}.')
            volume = raw.read([*xy_low, first], [*xy_high, last+1])
            check_cancelled(cancel)
            block = dict(z_range_inclusive=[first, last], anchor_z=current, direction=direction,
                         raw_sha256=hashlib.sha256(np.ascontiguousarray(volume)).hexdigest())
            blocks.append(block)
            with tempfile.TemporaryDirectory(prefix='video-', dir=output) as temporary:
                for index in range(volume.shape[2]):
                    check_cancelled(cancel)
                    frame = np.repeat(volume[:, :, index].T[..., None], 3, axis=2)
                    Image.fromarray(frame).save(Path(temporary)/f'{index:05d}.jpg', quality=100, subsampling=0)
                start_index = current-first
                seen = []
                with closing(model.propagate(temporary, {}, mask_inputs={start_index: conditioning}, cancel=cancel)) as stream:
                    for index, mask in stream:
                        check_cancelled(cancel)
                        z = first+int(index)
                        expected = current+direction*len(seen)
                        if z != expected or not first <= z <= last:
                            raise ValueError('SAM2 returned unordered, duplicate, or missing nucleus sections')
                        seen.append(z)
                        if z == current:
                            continue
                        mask = center_component(mask, previous=previous)
                        if mask.shape != seed_mask.shape:
                            raise ValueError('SAM2 nucleus section geometry changed')
                        if mask.sum() > max(int(previous.sum())*2, int(previous.sum())+256):
                            reason, finished = 'area_jump', True
                            break
                        masks[z] = mask
                        progress(f'Nucleus propagation at z={z}; {len(masks)} sections evaluated.')
                        if mask.any():
                            previous, empty, last_nonempty = mask, 0, z
                            if touches_boundary(mask):
                                reason, finished = 'xy_boundary', True
                                break
                        else:
                            empty += 1
                            if empty >= options.empty_sections:
                                reason, finished = 'empty', True
                                break
                        conditioning = mask
                if not finished and len(seen) != last-first+1:
                    raise ValueError('SAM2 did not return every section of the nucleus block')
            # If the block ends in an empty section, restart from the actual last
            # foreground section instead of inserting foreground into that empty slice.
            conditioning = previous
            if end == limit:
                finished = True
            current = last_nonempty
            empty = 0
        stops[str(direction)] = dict(reason=reason, limit_z=limit)
    check_cancelled(cancel)
    nonempty = sorted(z for z, mask in masks.items() if mask.any())
    first, last = nonempty[0], nonempty[-1]
    stack = np.stack([masks.get(z, np.zeros_like(seed_mask)) for z in range(first, last+1)])
    occupied = np.argwhere(stack)
    low, high = occupied.min(axis=0), occupied.max(axis=0)+1
    stack = stack[tuple(slice(a, b) for a, b in zip(low, high))].copy()
    offset = np.array([xy_low[0]+low[2], xy_low[1]+low[1], first+low[0]], dtype=int)
    metadata = dict(options=asdict(options), directions=['increasing_z', 'decreasing_z'],
                    stops=stops, blocks=blocks, seed_score=float(seed['score']),
                    partial=any(stop['reason'] != 'empty' for stop in stops.values()),
                    component_policy='center component, then overlap with previous section dilated by two pixels',
                    initialization='center image mask; previous boundary mask for each subsequent block',
                    memory_reset_per_block=True, negative_points=0)
    return dict(mask=stack, offset=offset, center=center, metadata=metadata)
