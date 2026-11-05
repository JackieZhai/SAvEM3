"""Clip legacy precomputed mesh surfaces in physical coordinates."""
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np

from probe_em.working_volume import write_json


def clip_mesh(vertices, faces, bounds_nm):
    """Clip triangles to an AABB, retaining orientation and open crop surfaces."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32).reshape(-1,3)
    bounds = np.asarray(bounds_nm, dtype=np.float64)
    if bounds.shape != (2,3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError('Mesh bounds must be a finite nonempty physical XYZ box')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Mesh vertices must be finite XYZ points')
    if faces.size and faces.max() >= len(vertices):
        raise ValueError('Mesh face references an absent vertex')
    triangles = vertices[faces]
    inside = np.all((triangles >= bounds[0]) & (triangles <= bounds[1]), axis=(1,2))
    possible = np.all(triangles.max(axis=1) >= bounds[0], axis=1) & np.all(triangles.min(axis=1) <= bounds[1], axis=1)
    kept = faces[inside]
    used, remapped = np.unique(kept, return_inverse=True)
    output_vertices = vertices[used]
    output_faces = remapped.astype(np.uint32).reshape(-1,3)
    added_vertices, added_faces = [], []
    for triangle in triangles[possible & ~inside]:
        polygon = [point.astype(np.float64) for point in triangle]
        for axis in range(3):
            for side, sign in ((0,1),(1,-1)):
                if not polygon:
                    break
                limit = bounds[side,axis]
                clipped = []
                previous = polygon[-1]
                previous_inside = sign*(previous[axis]-limit) >= 0
                for point in polygon:
                    point_inside = sign*(point[axis]-limit) >= 0
                    if point_inside != previous_inside:
                        fraction = (limit-previous[axis])/(point[axis]-previous[axis])
                        intersection = previous+fraction*(point-previous)
                        intersection[axis] = limit
                        clipped.append(intersection)
                    if point_inside:
                        clipped.append(point)
                    previous, previous_inside = point, point_inside
                polygon = clipped
        if len(polygon) < 3:
            continue
        for index in range(1,len(polygon)-1):
            triangle = np.array([polygon[0],polygon[index],polygon[index+1]],dtype=np.float32)
            if np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])) <= 1e-8:
                continue
            base = len(output_vertices)+len(added_vertices)
            added_vertices.extend(triangle)
            added_faces.append([base,base+1,base+2])
    if added_vertices:
        output_vertices = np.concatenate((output_vertices,np.asarray(added_vertices,dtype=np.float32)))
        output_faces = np.concatenate((output_faces,np.asarray(added_faces,dtype=np.uint32)))
    return output_vertices, output_faces


def crop_legacy_meshes(source, destination, bounds_nm, source_archive=None):
    """Export clipped fragments/manifests without copying stale spatial indices."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Mesh source and output must be separate directories')
    if source_archive:
        with zipfile.ZipFile(source_archive) as archive:
            prefix=f'{source.parent.name}/{source.name}/'
            info=json.loads(archive.read(prefix+'info'))
            manifests=[name[len(prefix):] for name in archive.namelist()
                       if name.startswith(prefix) and name.endswith(':0') and '/' not in name[len(prefix):]]
            def read(name):
                key=prefix+name
                return gzip.decompress(archive.read(key+'.gz')) if key+'.gz' in archive.NameToInfo else archive.read(key)
            return _write_meshes(source,destination,bounds_nm,info,manifests,read)
    info=json.loads((source/'info').read_text())
    def read(name):
        path=source/name;compressed=path.with_name(path.name+'.gz')
        return gzip.decompress(compressed.read_bytes()) if compressed.is_file() else path.read_bytes()
    return _write_meshes(source,destination,bounds_nm,info,[path.name for path in source.glob('*:0')],read)


def _write_meshes(source,destination,bounds_nm,info,manifests,read):
    from cloudvolume import Mesh
    if info.get('@type') != 'neuroglancer_legacy_mesh':
        raise ValueError('Only legacy precomputed mesh fragments are supported')
    destination.mkdir(parents=True,exist_ok=False)
    records = []
    for manifest in sorted(manifests):
        segment = manifest[:-2]
        if not segment.isdecimal():
            raise ValueError('Legacy mesh manifests must use numeric segment IDs')
        fragments = json.loads(read(manifest))['fragments']
        saved, input_faces, output_faces = [], 0, 0
        for index, name in enumerate(fragments):
            if not isinstance(name,str) or Path(name).name != name or name in ('.','..') or '\\' in name:
                raise ValueError('Mesh fragments must be local filenames')
            payload = read(name)
            mesh = Mesh.from_precomputed(payload)
            input_faces += len(mesh.faces)
            vertices, faces = clip_mesh(mesh.vertices,mesh.faces,bounds_nm)
            if not len(faces):
                continue
            output_faces += len(faces)
            name = f'{segment}:0:crop_{index}'
            payload = Mesh(vertices,faces).to_precomputed()
            (destination/(name+'.gz')).write_bytes(gzip.compress(payload,compresslevel=1,mtime=0))
            saved.append(name)
        write_json(destination/manifest,dict(fragments=saved))
        records.append(dict(segment=segment,input_faces=input_faces,output_faces=output_faces,fragments=len(saved)))
    write_json(destination/'info',{'@type':'neuroglancer_legacy_mesh'})
    report = dict(status='complete',source=str(source),bounds_nm=bounds_nm,segments=records,
                  source_metadata=info,method='Triangle clipping in physical nm; original interior geometry retained',
                  crop_surfaces='Open; no new caps, mesh extraction or smoothing')
    write_json(destination/'crop_report.json',report)
    return report
