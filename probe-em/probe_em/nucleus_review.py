"""Persist cropped nucleus masks and export their union without a dense overview."""
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from probe_em.nuclei import RESOLUTION
from probe_em.propagation import check_cancelled
from probe_em.review_objects import object_color, write_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class NucleusReview:
    def __init__(self, output, raw, source):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output/'nuclei.json'
        self.context = dict(source=str(source), shape_xyz=list(map(int, raw.shape)),
                            voxel_offset_xyz=list(map(int, raw.offset)), resolution_xyz_nm=RESOLUTION)
        self.objects, self.next_id, self.active, self.final = {}, 1, None, None
        if self.path.is_file():
            data = json.loads(self.path.read_text())
            if data.get('schema_version') != 1 or data.get('context') != self.context:
                raise ValueError('Saved nuclei belong to another dataset or voxel grid; use a new output directory')
            for record in data['objects']:
                key = record['object_id']
                folder = (self.output/record['prediction_dir']).resolve()
                if not folder.is_relative_to(self.output) or type(key) is not int or key < 1 or key in self.objects:
                    raise ValueError('Invalid saved nucleus record')
                with np.load(folder/'prediction.npz', allow_pickle=False) as value:
                    result = dict(mask=value['mask_zyx'].copy(), offset=value['voxel_offset_xyz'].copy(),
                                  center=np.array(record['center']['point']), folder=folder,
                                  center_id=record['center']['id'], object_id=key, color=record['color'])
                    if (not np.array_equal(value['resolution_xyz_nm'], RESOLUTION)
                            or not np.array_equal(value['frame_z'], np.arange(result['offset'][2],
                                                                 result['offset'][2]+len(result['mask'])))):
                        raise ValueError('Saved nucleus section coordinates or resolution differ')
                self._validate(result)
                if hashlib.sha256(result['mask'].tobytes()).hexdigest() != record['mask_sha256']:
                    raise ValueError('Saved nucleus mask checksum differs')
                self.objects[key] = result
            if len({r['center_id'] for r in self.objects.values()}) != len(self.objects):
                raise ValueError('Saved nucleus centers must have unique annotation IDs')
            self.next_id, self.active, self.final = data['next_object_id'], data.get('active'), data.get('final')
            if type(self.next_id) is not int or self.next_id <= max(self.objects, default=0):
                raise ValueError('Invalid next nucleus ID')
            if self.active is not None:
                self.validate_center(self.active['point'])
                if self.active['id'] in {result['center_id'] for result in self.objects.values()}:
                    raise ValueError('Active center already belongs to a saved nucleus')
            if self.final is not None:
                folder = (self.output/self.final['path']).resolve()
                if (not folder.is_relative_to(self.output) or not (folder/'accepted.json').is_file()
                        or not (folder/'precomputed/info').is_file()
                        or not set(self.final['parents']) <= self.objects.keys()):
                    raise ValueError('Invalid saved nuclei union')

    def validate_center(self, center):
        center = np.asarray(center, float)
        low, shape = np.array(self.context['voxel_offset_xyz']), np.array(self.context['shape_xyz'])
        if center.shape != (3,) or not np.isfinite(center).all() or np.any(center < low) or np.any(center >= low+shape):
            raise ValueError('Nucleus center must be inside the 32 nm raw volume')

    def _validate(self, result):
        mask, offset = result['mask'], np.asarray(result['offset'])
        low, shape = np.array(self.context['voxel_offset_xyz']), np.array(self.context['shape_xyz'])
        if (mask.dtype != bool or mask.ndim != 3 or not mask.any()
                or offset.shape != (3,) or not np.isfinite(offset).all() or np.any(offset != np.floor(offset))
                or np.any(offset < low) or np.any(offset+mask.shape[::-1] > low+shape)):
            raise ValueError('Nucleus mask must be nonempty and inside the 32 nm grid')
        self.validate_center(result['center'])
        point = np.floor(result['center']).astype(int)-offset.astype(int)
        if np.any(point < 0) or np.any(point >= mask.shape[::-1]) or not mask[tuple(point[::-1])]:
            raise ValueError('Saved nucleus mask must contain its center')

    def save(self, *, objects=None, next_id=None, active=..., final=...):
        objects = self.objects if objects is None else objects
        next_id = self.next_id if next_id is None else next_id
        active = self.active if active is ... else active
        final = self.final if final is ... else final
        records = [dict(object_id=key, color=result['color'],
                        prediction_dir=result['folder'].relative_to(self.output).as_posix(),
                        center=dict(id=result['center_id'], point=result['center'].tolist()),
                        mask_sha256=hashlib.sha256(result['mask'].tobytes()).hexdigest())
                   for key, result in objects.items()]
        write_json(self.path, dict(schema_version=1, context=self.context, objects=records,
            next_object_id=next_id, active=active, final=final, updated_at=timestamp()))
        self.objects, self.next_id, self.active, self.final = objects, next_id, active, final

    def accept(self, preview):
        self._validate(preview)
        result = dict(preview, object_id=self.next_id, color=object_color(self.next_id))
        write_json(result['folder']/'accepted.json', dict(accepted_at=timestamp(),
            object_id=self.next_id, center_id=result['center_id'], color=result['color']))
        self.save(objects={**self.objects, self.next_id: result}, next_id=self.next_id+1, active=None)
        return result


def export_union(objects, folder, cancel, progress):
    """Write sparse gzip-compressed precomputed chunks, using binary union semantics."""
    folder = Path(folder)
    root = folder/'precomputed'
    scale_key = '32_32_30'
    (root/scale_key).mkdir(parents=True, exist_ok=False)
    values = list(objects.values())
    low = np.min([r['offset'] for r in values], axis=0)
    high = np.max([r['offset']+r['mask'].shape[::-1] for r in values], axis=0)
    chunk_size = np.array([64, 64, 32])
    chunks = set()
    for result in values:
        start = (result['offset']-low)//chunk_size
        end = (result['offset']+result['mask'].shape[::-1]-1-low)//chunk_size
        chunks.update(itertools.product(*(range(int(a), int(b)+1) for a, b in zip(start, end))))
    files = []
    voxels = 0
    for index, chunk in enumerate(sorted(chunks)):
        check_cancelled(cancel)
        first = low+np.array(chunk)*chunk_size
        last = np.minimum(high, first+chunk_size)
        merged = np.zeros(last-first, dtype=np.uint8)
        for result in values:
            begin, end = np.maximum(first, result['offset']), np.minimum(last, result['offset']+result['mask'].shape[::-1])
            if np.any(begin >= end):
                continue
            target = tuple(slice(a, b) for a, b in zip(begin-first, end-first))
            source = tuple(slice(a, b) for a, b in zip(begin-result['offset'], end-result['offset']))
            merged[target] |= result['mask'][source[::-1]].transpose(2, 1, 0)
        if merged.any():
            name = '_'.join(f'{a}-{b}' for a, b in zip(first, last))+'.gz'
            content = gzip.compress(merged.tobytes(order='F'), mtime=0)
            (root/scale_key/name).write_bytes(content)
            files.append(dict(file=name, sha256=hashlib.sha256(content).hexdigest()))
            voxels += int(merged.sum())
        progress(f'Saving nucleus union: {index+1}/{len(chunks)} chunks.')
    check_cancelled(cancel)
    write_json(root/'info', dict(type='segmentation', data_type='uint8', num_channels=1,
        scales=[dict(key=scale_key, resolution=RESOLUTION, voxel_offset=low.tolist(),
                     size=(high-low).tolist(), chunk_sizes=[chunk_size.tolist()], encoding='raw')]))
    write_json(folder/'metadata.json', dict(operation='binary_union', parents=list(objects),
        resolution_xyz_nm=RESOLUTION, missing_chunks='zero', chunks=files, mask_voxels=voxels))
    write_json(folder/'accepted.json', dict(accepted_at=timestamp(), confirmations=2))
    return dict(path=folder.name, parents=list(objects))
