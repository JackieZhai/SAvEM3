"""Unregistered multiscale overview and physical bounds of cached test ROIs."""
import json
from pathlib import Path

import neuroglancer
import numpy as np

from probe_em.viewer_status import set_status


def geometry(info):
    scale = info['scales'][0]
    resolution = np.asarray(scale['resolution'], dtype=float)
    offset = np.asarray(scale.get('voxel_offset', [0, 0, 0]), dtype=float)
    size = np.asarray(scale['size'], dtype=float)
    if (any(value.shape != (3,) or not np.isfinite(value).all() for value in (resolution, offset, size))
            or np.any(resolution <= 0) or np.any(size <= 0) or np.any(size != np.floor(size))
            or np.any(offset != np.floor(offset))):
        raise ValueError('Precomputed geometry requires finite XYZ resolution, integer size and offset')
    return resolution, offset, size


def roi_boxes(windows, overview_info):
    """Return half-open bounds in the overview's native voxel coordinates."""
    resolution, offset, size = geometry(overview_info)
    boxes = []
    for index, item in enumerate(windows):
        if item.get('kind', 'roi') != 'roi':
            continue
        info = json.loads((Path(item['local_dir'])/'raw/info').read_text(encoding='utf-8'))
        roi_resolution, roi_offset, roi_size = geometry(info)
        low = roi_offset*roi_resolution/resolution
        high = (roi_offset+roi_size)*roi_resolution/resolution
        if np.any(low < offset-1e-6) or np.any(high > offset+size+1e-6):
            raise ValueError(f'Test window {item["name"]!r} is outside the overview volume')
        boxes.append({'id': f'window_{index}', 'name': item['name'], 'window_index': index,
                      'point_a': low.tolist(), 'point_b': high.tolist(),
                      'bounds_nm': [(roi_offset*roi_resolution).tolist(),
                                    ((roi_offset+roi_size)*roi_resolution).tolist()]})
    return boxes


def build_overview(item, windows, raw_info, seg_info, raw_url, seg_url):
    resolution, offset, size = geometry(raw_info)
    seg_resolution, seg_offset, seg_size = geometry(seg_info)
    if not (np.allclose(offset*resolution, seg_offset*seg_resolution)
            and np.allclose((offset+size)*resolution, (seg_offset+seg_size)*seg_resolution)):
        raise ValueError('Overview raw and labels must cover the same physical volume')
    boxes = roi_boxes(windows, raw_info)
    dimensions = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=resolution)
    segments = item.get('segments', sorted({entry['seed'] for entry in windows if entry.get('kind', 'roi') == 'roi'}))
    if any(type(value) is not int or not 0 < value < 2**64 for value in segments):
        raise ValueError('Overview segments must be positive uint64 integers')
    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.dimensions = dimensions
        state.layers['raw'] = neuroglancer.ImageLayer(source=raw_url)
        state.layers['lc'] = neuroglancer.SegmentationLayer(source=seg_url, segments=segments,
                                                          selected_alpha=.25, object_alpha=.75,
                                                          hover_highlight=False)
        state.layers['test_windows'] = neuroglancer.LocalAnnotationLayer(
            dimensions=dimensions, annotation_color='#ffcb52',
            annotations=[neuroglancer.AxisAlignedBoundingBoxAnnotation(
                id=box['id'], description=box['name'], point_a=box['point_a'], point_b=box['point_b'])
                for box in boxes])
        state.position = offset+size/2
        state.cross_section_scale = max(size*resolution)/(resolution[0]*600)
        state.projection_scale = 1.6*max(size*resolution)/resolution[0]
        state.projection_orientation = [.2, .3, 0, .9327379053088815]
        state.layout = '4panel'
        state.show_slices = False
        state.show_axis_lines = False
        state.show_default_annotations = False

    def navigate(action, step):
        with viewer.txn() as state:
            position = np.asarray(state.position).copy()
            position[2] = np.clip(np.floor(position[2])+step, offset[2], offset[2]+size[2]-1)+.5
            state.position = position
        set_status(viewer, 'overview-navigation',
                   f'Current z={int(position[2])}. D/F: previous/next slice. Select a test window for SAM2 review.')

    viewer.actions.add('overview-previous-section', lambda action: navigate(action, -1))
    viewer.actions.add('overview-next-section', lambda action: navigate(action, 1))
    with viewer.config_state.txn() as state:
        for scope in (state.input_event_bindings.viewer, state.input_event_bindings.slice_view):
            scope['keyd'] = 'overview-previous-section'
            scope['keyf'] = 'overview-next-section'
    set_status(viewer, 'overview',
               f'{item["name"]}: full raw and lc volume. Local alignment: OFF. '
               f'{len(boxes)} test-window boxes in original coordinates. Data cached as viewed.')
    set_status(viewer, 'overview-navigation',
               'D/F: previous/next slice. Select a test window for point prediction and SAM2 propagation.')
    report = {'kind': 'overview', 'alignment': {'enabled': False}, 'show_slices': False,
              'resolution_xyz_nm': resolution.tolist(), 'voxel_offset_xyz': offset.tolist(),
              'shape_xyz': size.astype(int).tolist(), 'boxes': boxes, 'segments': segments,
              'raw_source': item['raw_path'], 'seg_source': item['seg_path'],
              'meshes': 'Existing unregistered precomputed lc meshes; no new whole-volume meshing'}
    return viewer, report
