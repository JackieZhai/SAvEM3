"""Bounded SAM2 video inputs and native-coordinate output geometry.

Each EM section is one video frame. Upstream's legacy JPEG-directory API is used
without modifying SAM2; the exact encoded files are retained for replay. Mask
outputs are always converted back to the original native XYZ voxel grid.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from probe_em.alignment_field import warp_field_slice
from probe_em.demo_alignment import map_points
from probe_em.prompt_models import file_digest


class PropagationCancelled(Exception):
    pass


def check_cancelled(cancel):
    if cancel.is_set():
        raise PropagationCancelled('Propagation cancelled')


class PropagationGrid:
    def __init__(self, raw, offset, z_start, z_end, *, field=None, shape=None, origin=None):
        self.raw, self.offset = raw, np.asarray(offset)
        self.z_start, self.z_end = int(z_start), int(z_end)
        if not self.offset[2] <= z_start < z_end < self.offset[2]+raw.shape[2]:
            raise ValueError('Propagation requires at least two sections inside the cached ROI')
        self.field = field
        self.shape = tuple(shape[:2]) if field is not None else raw.shape[:2]
        self.origin = np.asarray(origin[:2]) if field is not None else self.offset[:2]

    def encode(self, folder, cancel):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        manifest = []
        for index, z in enumerate(range(self.z_start, self.z_end+1)):
            check_cancelled(cancel)
            data = self.raw[:, :, z-int(self.offset[2])]
            data = warp_field_slice(self.field, data, z, self.offset[:2], self.origin, self.shape)
            path = folder/f'{index:05d}.jpg'
            Image.fromarray(np.repeat(data.T[..., None], 3, axis=2)).save(path, quality=100, subsampling=0)
            manifest.append({'frame': index, 'absolute_z': z, 'file': path.name, 'sha256': file_digest(path)})
        return manifest

    def prompts(self, points, labels):
        points = np.asarray(points, float)
        mapped = map_points(self.field, points) if self.field is not None else points.copy()
        mapped[:, :2] -= self.origin
        result = {}
        for z in np.unique(points[:, 2]).astype(int):
            if not self.z_start <= z <= self.z_end:
                raise ValueError('Prompts outside propagation range; expand B/E or clear those points')
            selected = points[:, 2] == z
            if not np.any(labels[selected] == 1):
                raise ValueError(f'Add a positive point on prompted section z={z}')
            result[z-self.z_start] = mapped[selected, :2], labels[selected]
        return result

    def native_mask(self, mask_yx, z):
        mask = np.asarray(mask_yx)
        if mask.shape != self.shape[::-1] or not np.isin(mask, [0, 1]).all():
            raise ValueError('Video predictor returned an invalid binary mask')
        if self.field is None:
            return mask.astype(bool)
        # Gather at transformed native voxel centers. This is the inverse of
        # display resampling, not an affine approximation to an elastic map.
        x, y = np.indices(self.raw.shape[:2])
        points = np.column_stack((x.ravel()+self.offset[0], y.ravel()+self.offset[1], np.full(x.size, z)))
        xy = np.floor(map_points(self.field, points)[:, :2]-self.origin+.5).astype(int)
        valid = ((xy >= 0) & (xy < self.shape)).all(axis=1)
        result = np.zeros(x.size, bool)
        result[valid] = mask[xy[valid, 1], xy[valid, 0]]
        return result.reshape(self.raw.shape[:2]).T

    def video_mask(self, mask_yx, z):
        """Map a native image prediction into the registered video frame."""
        mask = np.asarray(mask_yx)
        if (mask.shape != self.raw.shape[:2][::-1] or not np.isin(mask, [0, 1]).all()
                or not self.z_start <= z <= self.z_end):
            raise ValueError('Conditioning mask must cover a native section inside the video range')
        return warp_field_slice(self.field, mask.T.astype(np.uint8), z,
                                self.offset[:2], self.origin, self.shape, seg=True).T.astype(bool)
