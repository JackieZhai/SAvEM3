"""Install portable example reviews into a fresh local review directory."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from probe_em.demo_alignment import map_points
from probe_em.review_objects import ReviewObjects, write_json


def seed_review(source, destination, raw, offset, resolution, raw_source):
    """Copy a validated starter review once; existing user work always takes priority."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        return None
    state = json.loads((source/'session.json').read_text(encoding='utf-8'))
    if state.get('schema_version') != 1:
        raise ValueError('Unsupported starter session format')
    if hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest() != state['raw_sha256']:
        raise ValueError('Starter session belongs to different raw data')
    review = ReviewObjects(source, raw.shape, offset, resolution, source=None)
    low, high = int(offset[2]), int(offset[2])+raw.shape[2]-1
    start, end = state['propagation_range']
    if not low <= start <= end <= high:
        raise ValueError('Starter propagation range is outside the example')
    position = np.asarray(state['position_native_xyz'], float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Starter view position must contain three finite coordinates')
    for annotations in [state['active_annotations'], *(r['annotations'] for r in review.objects.values())]:
        ids = set()
        for annotation in annotations:
            point = np.asarray(annotation['point'], float)
            if (annotation['id'] in ids or annotation['label'] not in (0, 1)
                    or point.shape != (3,) or not np.isfinite(point).all()
                    or np.any(point-offset < 0) or np.any(point-offset >= raw.shape)):
                raise ValueError('Invalid native starter annotation')
            ids.add(annotation['id'])
    # Validate everything before publishing the new directory. Prediction masks
    # and native points remain unchanged; only the local source path is rebound.
    manifest = json.loads((source/'review_objects.json').read_text(encoding='utf-8'))
    manifest['context']['source'] = str(Path(raw_source).resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.starter-', dir=destination.parent) as temporary:
        staging = Path(temporary)/'review'
        staging.mkdir()
        records = list(manifest['objects'])
        if manifest.get('final') is not None:
            records.append(manifest['final'])
        for record in records:
            folder = (source/record['prediction_dir']).resolve()
            relative = folder.relative_to(source)
            record['prediction_dir'] = relative.as_posix()
            target = staging/relative
            target.mkdir(parents=True)
            for name in ('prediction.npz', 'accepted.json', 'metadata.json'):
                shutil.copyfile(folder/name, target/name)
        write_json(staging/'review_objects.json', manifest)
        write_json(staging/'session.json', state)
        if destination.exists():
            if any(destination.iterdir()):
                return None
            destination.rmdir()
        staging.rename(destination)
    return state


def restore_starter_view(session, saved):
    """Apply the captured view and active points after the default window setup."""
    if saved is None:
        return False
    import neuroglancer
    from probe_em.interactive import POINT_LAYERS
    session.propagation_range = list(saved['propagation_range'])
    with session.viewer.txn() as state:
        position = saved['position_native_xyz']
        state.position = (map_points(session.field, [position])[0]
                          if session.mode['aligned'] else position)
        properties = {'crossSectionScale': 'cross_section_scale', 'projectionScale': 'projection_scale',
                      'crossSectionOrientation': 'cross_section_orientation',
                      'projectionOrientation': 'projection_orientation'}
        for key, attribute in properties.items():
            if key in saved['view']:
                setattr(state, attribute, saved['view'][key])
        for label, name in POINT_LAYERS.items():
            annotations = [a for a in saved['active_annotations'] if a['label'] == label]
            points = np.asarray([a['point'] for a in annotations], float).reshape(-1, 3)
            if len(points) and session.mode['aligned']:
                points = map_points(session.field, points)
            state.layers[name].annotations = [neuroglancer.PointAnnotation(
                id=a['id'], point=p, description=a.get('description', '')) for a, p in zip(annotations, points)]
        properties = {'visible': 'visible', 'segments': 'segments', 'segmentColors': 'segment_colors',
                      'selectedAlpha': 'selected_alpha', 'objectAlpha': 'object_alpha',
                      'notSelectedAlpha': 'not_selected_alpha', 'annotationColor': 'annotation_color'}
        for name, settings in saved['layers'].items():
            if name in state.layers:
                for key, attribute in properties.items():
                    if key in settings:
                        value = settings[key]
                        if key == 'segments':
                            # Neuroglancer serializes uint64 IDs as decimal strings.
                            value = [int(segment) for segment in value]
                        setattr(state.layers[name], attribute, value)
    session.status(f'Loaded {len(session.review_objects.objects)} saved example objects and their points. '
                   'Add points for the next object.')
    return True
