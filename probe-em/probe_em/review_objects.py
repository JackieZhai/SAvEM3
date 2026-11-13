"""Persistent, independently colored review objects on one native voxel grid."""
import colorsys
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np


PALETTE = ('#ffc040', '#42c5f5', '#b388ff', '#66dd88', '#ff8f70', '#80cbc4',
           '#f48fc7', '#c5d86d')


def object_color(object_id):
    if object_id <= len(PALETTE):
        return PALETTE[object_id-1]
    rgb = colorsys.hsv_to_rgb((object_id*.61803398875) % 1, .55, .95)
    return '#' + ''.join(f'{round(channel*255):02x}' for channel in rgb)


def layer_name(object_id):
    return f'sam_object_{object_id:03d}'


def mask_stack(result):
    if result['snapshot']['scope'] == 'volume':
        return int(result['z_start']), result['mask']
    return int(result['snapshot']['z']), result['mask'][None]


def write_json(path, data):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ReviewObjects:
    def __init__(self, output, shape, offset, resolution, source=None):
        self.output = Path(output).resolve()
        self.path = self.output/'review_objects.json'
        self.context = dict(shape_xyz=list(map(int, shape)), voxel_offset_xyz=list(map(int, offset)),
                            resolution_xyz_nm=list(map(float, resolution)), source=source)
        self.objects = {}
        self.next_id = 1
        self.final = None
        if self.path.exists():
            manifest = json.loads(self.path.read_text())
            if manifest.get('schema_version') != 1 or manifest.get('context') != self.context:
                raise ValueError('Saved review objects belong to a different source or voxel grid; use another output directory')
            for record in manifest['objects']:
                object_id = record['object_id']
                folder = (self.output/record['prediction_dir']).resolve()
                if not folder.is_relative_to(self.output) or type(object_id) is not int or object_id < 1:
                    raise ValueError('Invalid saved review object')
                if object_id in self.objects or not (folder/'accepted.json').is_file():
                    raise ValueError('Duplicate or unaccepted saved review object')
                with np.load(folder/'prediction.npz', allow_pickle=False) as saved:
                    if record['scope'] == 'volume':
                        mask = saved['mask_zyx'].copy()
                        if not np.array_equal(saved['frame_z'], np.arange(record['z_start'], record['z_end']+1)):
                            raise ValueError('Saved review object section coordinates differ')
                    elif record['scope'] == 'image':
                        mask = saved['mask_yx'].copy()
                    else:
                        raise ValueError('Unknown review object scope')
                result = dict(mask=mask, folder=folder, object_id=object_id, color=record['color'],
                              snapshot=dict(scope=record['scope'], z=record['z_start']), z_start=record['z_start'],
                              annotations=json.loads((folder/'accepted.json').read_text()).get('annotations_native', []))
                self._validate(result)
                digest = hashlib.sha256(mask.tobytes()).hexdigest()
                if digest != record['mask_sha256']:
                    raise ValueError('Saved review object mask checksum differs')
                self.objects[object_id] = result
            self.next_id = manifest['next_object_id']
            if type(self.next_id) is not int or self.next_id <= max(self.objects, default=0):
                raise ValueError('Invalid next review object ID')
            final = manifest.get('final')
            if final is not None:
                folder = (self.output/final['prediction_dir']).resolve()
                if not folder.is_relative_to(self.output):
                    raise ValueError('Invalid final review path')
                with np.load(folder/'prediction.npz', allow_pickle=False) as saved:
                    masks = saved['mask_zyx'].copy()
                    frames = saved['frame_z'].copy()
                self.final = dict(mask=masks, folder=folder, z_start=int(frames[0]),
                                  snapshot=dict(scope='volume', z=None), color='#ffffff', parents=final['parents'])
                self._validate(self.final)
                if (not np.array_equal(frames, np.arange(frames[0], frames[0]+len(masks)))
                        or hashlib.sha256(masks.tobytes()).hexdigest() != final['mask_sha256']):
                    raise ValueError('Final review geometry or checksum differs')

    @property
    def color(self):
        return object_color(self.next_id)

    def _validate(self, result):
        start, masks = mask_stack(result)
        x, y, depth = self.context['shape_xyz']
        first = self.context['voxel_offset_xyz'][2]
        if (masks.dtype != bool or masks.ndim != 3 or masks.shape[1:] != (y, x)
                or not masks.any() or start < first or start+len(masks) > first+depth):
            raise ValueError('Review object must be a nonempty binary mask on the native ROI grid')

    def _persist(self, objects, next_id, final=None):
        records = []
        for object_id, result in objects.items():
            start, masks = mask_stack(result)
            records.append(dict(object_id=object_id, color=result['color'],
                prediction_dir=str(Path(result['folder']).resolve().relative_to(self.output)),
                scope=result['snapshot']['scope'], z_start=start, z_end=start+len(masks)-1,
                mask_sha256=hashlib.sha256(result['mask'].tobytes()).hexdigest()))
        self.output.mkdir(parents=True, exist_ok=True)
        final = self.final if final is None else final
        final_record = (dict(prediction_dir=str(final['folder'].resolve().relative_to(self.output)),
                             parents=final['parents'], mask_sha256=hashlib.sha256(final['mask'].tobytes()).hexdigest())
                        if final is not None else None)
        write_json(self.path, dict(schema_version=1, context=self.context, next_object_id=next_id,
                                  objects=records, final=final_record,
                                  updated_at=datetime.now(timezone.utc).isoformat()))
        self.objects, self.next_id = objects, next_id
        self.final = final

    def accept(self, result, annotations=()):
        self._validate(result)
        result.update(object_id=self.next_id, color=self.color, annotations=list(annotations))
        write_json(result['folder']/'accepted.json', dict(
            accepted_at=datetime.now(timezone.utc).isoformat(), object_id=self.next_id,
            color=self.color, prediction='prediction.npz', metadata='metadata.json',
            annotations_native=result['annotations']))
        self._persist({**self.objects, self.next_id: result}, self.next_id+1)
        return result

    def merge(self):
        if not self.objects:
            raise ValueError('Finish at least one object before accepting the final result')
        entries = [mask_stack(result) for result in self.objects.values()]
        first = min(z for z, masks in entries)
        end = max(z+len(masks) for z, masks in entries)
        x, y, _ = self.context['shape_xyz']
        merged = np.zeros((end-first, y, x), dtype=bool)
        for z, masks in entries:
            merged[z-first:z-first+len(masks)] |= masks
        folder = self.output/('merge_' + uuid.uuid4().hex)
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder/'prediction.npz', mask_zyx=merged, frame_z=np.arange(first, end))
        parents = [dict(object_id=key, color=value['color'],
                        prediction_dir=str(value['folder'].resolve().relative_to(self.output)))
                   for key, value in self.objects.items()]
        write_json(folder/'metadata.json', dict(schema_version=1, scope='accepted_object_union',
            operation='binary_union', confirmations=2, source=self.context['source'],
            resolution_xyz_nm=self.context['resolution_xyz_nm'],
            voxel_offset_xyz=[*self.context['voxel_offset_xyz'][:2], first], mask_axes='ZYX',
            z_range_inclusive=[first, end-1], color='#ffffff',
            mask_pixels=int(merged.sum()), parents=parents,
            created_at=datetime.now(timezone.utc).isoformat()))
        result = dict(mask=merged, snapshot=dict(scope='volume', z=None), z_start=first,
                      folder=folder, color='#ffffff', parents=list(self.objects))
        write_json(folder/'accepted.json', dict(accepted_at=datetime.now(timezone.utc).isoformat(),
            confirmations=2, color='#ffffff', prediction='prediction.npz', metadata='metadata.json'))
        self._persist(self.objects, self.next_id, final=result)
        return result
