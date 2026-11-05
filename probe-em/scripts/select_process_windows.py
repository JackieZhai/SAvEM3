"""Cache neurite windows that exclude the annotated soma regions of every target."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_local_gt import interior_point, sample_gt, volume, write_json
from export_savem3 import export_bundle, write_precomputed
from probe_em.demo_alignment import prepare_field


def soma_regions(labels, resolution, ids, core_radius_nm=1300):
    """Estimate thick soma cores; retain their physical positions for audit."""
    resolution = np.asarray(resolution, float)
    somata, radii, report = np.zeros(labels.shape, bool), {}, {}
    for label in ids:
        mask = labels == label
        radius = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=resolution)[1:-1,1:-1,1:-1]
        center = np.asarray(np.unravel_index(radius.argmax(), labels.shape))
        if radius.max() < core_radius_nm:
            raise ValueError(f'No resolved thick soma core for label {label}; use a finer locator')
        components, _ = ndimage.label(radius >= max(core_radius_nm, .55*radius.max()))
        core = components == components[tuple(center)]
        soma = ndimage.binary_dilation(core, iterations=5) & mask
        somata |= soma
        radii[label] = radius.astype(np.float32)
        report[label] = dict(center_local_xyz=center.tolist(), max_radius_nm=float(radius.max()),
                             soma_voxels=int(soma.sum()))
    return somata, radii, report


def process_candidates(labels, resolution, offset, ids, distances_um, roi_shape, roi_resolution, bounds_nm):
    resolution, offset = np.asarray(resolution), np.asarray(offset)
    extent = np.asarray(roi_shape)*roi_resolution
    somata, radii, report = soma_regions(labels, resolution, ids)
    # The entire box, with an extra locator-cell guard, must avoid every soma.
    footprint = 2*np.ceil(extent/2/resolution).astype(int)+3
    forbidden = ndimage.maximum_filter(somata, size=tuple(footprint), mode='constant')
    candidates = {}
    low, high = np.asarray(bounds_nm)
    for label, requested in zip(ids, distances_um):
        center_nm = (np.asarray(report[label]['center_local_xyz'])+offset+.5)*resolution
        report[label]['center_nm'] = center_nm.tolist()
        radius = radii[label]
        coordinates = np.argwhere((labels == label) & (radius >= 350) & (radius <= 850) & ~forbidden)
        positions = (coordinates+offset+.5)*resolution
        distances = np.linalg.norm(positions-center_nm, axis=1)
        interior = np.all(positions-extent/2 >= low+resolution, axis=1) & np.all(positions+extent/2 <= high-resolution, axis=1)
        indices = np.flatnonzero(interior)
        scores = np.abs(distances[indices]-requested*1000) + 3*np.abs(radius[tuple(coordinates[indices].T)]-550)
        chosen = []
        for index in indices[np.argsort(scores)]:
            if all(np.linalg.norm(positions[index]-entry['center_nm']) > 2500 for entry in chosen):
                chosen.append(dict(center_nm=positions[index].tolist(),
                                   distance_to_soma_center_um=float(distances[index]/1000),
                                   locator_radius_nm=float(radius[tuple(coordinates[index])]),
                                   requested_distance_um=requested))
            if len(chosen) == 8:
                break
        if not chosen:
            raise ValueError(f'No soma-free process candidates for label {label}')
        candidates[label] = chosen
    return candidates, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-path', type=Path, required=True)
    parser.add_argument('--gt-path', type=Path, required=True)
    parser.add_argument('--crop-manifest', type=Path, required=True, help='Shared bounds_nm in original coordinates')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', type=int, nargs='+', default=[30001,30002,30003,30004,30005])
    parser.add_argument('--distances-um', type=float, nargs='+', default=[8,14,22,30,38])
    parser.add_argument('--xy', type=int, default=512)
    parser.add_argument('--slices', type=int, default=63)
    parser.add_argument('--raw-mip', type=int, default=1)
    args = parser.parse_args()
    if len(args.ids) != len(args.distances_um) or args.slices < 3 or args.slices % 2 == 0 or args.xy < 64:
        parser.error('Use one distance per ID, XY >=64, and an odd section count >=3')
    raw, gt = volume(args.raw_path, args.raw_mip), volume(args.gt_path)
    locator = volume(args.gt_path, len(gt.info['scales'])-1)
    if np.prod(locator.bounds.size(), dtype=object) > 64_000_000 or args.xy**2*args.slices > 64_000_000:
        raise ValueError('Locator or requested ROI exceeds the 64-million voxel limit')
    ratio = raw.resolution/gt.resolution
    if np.any(ratio < 1) or not np.allclose(ratio, np.rint(ratio)):
        raise ValueError('Raw must be an integer coarsening of GT')
    ratio = np.rint(ratio).astype(int)
    if args.xy**2*args.slices*np.prod(ratio, dtype=object) > 256_000_000:
        raise ValueError('Fine GT read exceeds the 256-million voxel limit')
    shape = np.array([args.xy,args.xy,args.slices])
    bounds = json.loads(args.crop_manifest.read_text())['bounds_nm']
    candidates, somata = process_candidates(np.asarray(locator[:])[...,0], locator.resolution, locator.voxel_offset,
                                            args.ids, args.distances_um, shape, raw.resolution, bounds)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'selection.json', dict(candidates=candidates, somata=somata, bounds_nm=bounds,
               method='Thick-core soma exclusion over entire ROI; varying Euclidean soma-center distances',
               scope='Geometry-based process selection; no axon/dendrite classification or path-length estimate'))
    windows = []
    for label in args.ids:
        for candidate in candidates[label]:
            center = np.floor(np.array(candidate['center_nm'])/raw.resolution).astype(int)
            low, high = center-shape//2, center-shape//2+shape
            fine = np.asarray(gt[tuple(slice(int(a),int(b)) for a,b in zip(low*ratio,high*ratio))])[...,0]
            labels = sample_gt(fine, ratio)
            del fine
            target = labels == label
            anchor = args.slices//2
            components, _ = ndimage.label(target[:,:,anchor])
            points = np.argwhere(target[:,:,anchor])
            if not len(points):
                continue
            nearest = points[np.linalg.norm(points-shape[:2]//2,axis=1).argmin()]
            component = components == components[tuple(nearest)]
            px, py = interior_point(component)
            radius_nm = float(ndimage.distance_transform_edt(np.pad(component,1), sampling=raw.resolution[:2]).max())
            area_um2 = float(component.sum()*np.prod(raw.resolution[:2])/1e6)
            occupied = int(np.count_nonzero(target.any(axis=(0,1))))
            if radius_nm > 1100 or area_um2 > 8 or occupied < .8*args.slices:
                continue
            break
        else:
            raise ValueError(f'No candidate passed fine-grid process checks for label {label}')
        image = np.asarray(raw[tuple(slice(int(a),int(b)) for a,b in zip(low,high))])[...,0]
        folder = args.output/f'neuron_{label}'
        export_bundle(image.transpose(2,1,0), labels.transpose(2,1,0), folder/'cache', raw.resolution, low)
        write_precomputed(folder/'cache/target', target.astype(np.uint8), raw.resolution, low, True)
        point = [int(px+low[0]),int(py+low[1]),int(anchor+low[2])]
        _, alignment = prepare_field(image, raw.resolution, low, cache_dir=folder/'alignment', ref_z=point[2], backend='ecc')
        fields = list((folder/'alignment').glob('field_*.npz'))
        if len(fields) != 1:
            raise ValueError('Expected exactly one cached alignment field')
        report = dict(label=label, **candidate, point_xyz=point, offset_xyz=low.tolist(), shape_xyz=shape.tolist(),
                      resolution_xyz_nm=list(map(int,raw.resolution)), central_component_area_um2=area_um2,
                      central_component_radius_nm=radius_nm, occupied_sections=occupied, alignment=alignment,
                      soma_excluded_from_entire_box=True, raw_source=str(args.raw_path.resolve()),
                      gt_source=str(args.gt_path.resolve()))
        write_json(folder/'selection.json', report)
        windows.append(dict(name=f'Neuron {label} process ({candidate["distance_to_soma_center_um"]:.1f} um from soma)',
                            seed=label, local_dir=str((folder/'cache').resolve()), position_xyz=point,
                            align_field=str(fields[0].resolve()), image_layers={},
                            label_layers={'ground_truth':str((folder/'cache/target').resolve())},
                            visible_layers=['ground_truth'], colors={'ground_truth':'#2a83ff'}))
        write_json(args.output/'windows.json', dict(schema_version=1, initial_index=0, windows=windows))
        print(json.dumps({key:report[key] for key in ('label','point_xyz','distance_to_soma_center_um','central_component_area_um2','occupied_sections')}),flush=True)


if __name__ == '__main__':
    main()
