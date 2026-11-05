"""Clipping must retain crossing triangles, winding and physical bounds."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'probe-em'))
from probe_em.mesh_crop import clip_mesh


def test_clipped_triangle_area_and_orientation():
    vertices = np.array([[-1,0,1],[3,0,1],[1,4,1],[20,20,20]],np.float32)
    points,faces = clip_mesh(vertices,[[0,1,2]],[[0,0,0],[2,2,2]])
    assert np.all(points >= 0) and np.all(points <= 2)
    triangles = points[faces]
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    assert np.all(normals[:,2]>0)
    assert np.isclose(normals[:,2].sum()/2,4)


def test_triangle_crossing_box_with_no_inside_vertices_is_retained():
    points,faces = clip_mesh([[-4,0,0],[4,0,0],[0,4,0]],[[0,1,2]],[[-1,.5,-1],[1,1.5,1]])
    assert len(faces)>0
    assert np.all(points[:,0]>=-1) and np.all(points[:,0]<=1)
    assert np.all(points[:,1]>=.5) and np.all(points[:,1]<=1.5)


def test_outside_vertices_are_removed_without_moving_interior_geometry():
    vertices = np.array([[1,1,1],[2,1,1],[1,2,1],[10,10,10],[11,10,10],[10,11,10]],np.float32)
    points,faces = clip_mesh(vertices,[[0,1,2],[3,4,5]],[[0,0,0],[3,3,3]])
    np.testing.assert_array_equal(points,vertices[:3])
    np.testing.assert_array_equal(faces,[[0,1,2]])


def test_meshes_can_be_cropped_directly_from_original_zip(tmp_path):
    import json,gzip,zipfile
    from cloudvolume import Mesh
    from probe_em.mesh_crop import crop_legacy_meshes
    vertices=np.array([[1,1,1],[2,1,1],[1,2,1]],np.float32)
    data=Mesh(vertices,np.array([[0,1,2]],np.uint32)).to_precomputed()
    source=tmp_path/'source/mesh';archive_path=tmp_path/'source.zip'
    with zipfile.ZipFile(archive_path,'w') as archive:
        archive.writestr('source/mesh/info',json.dumps({'@type':'neuroglancer_legacy_mesh'}))
        archive.writestr('source/mesh/30001:0',json.dumps({'fragments':['triangle']}))
        archive.writestr('source/mesh/triangle.gz',gzip.compress(data))
    result=crop_legacy_meshes(source,tmp_path/'cropped',[[0,0,0],[3,3,3]],source_archive=archive_path)
    assert result['segments'][0]['output_faces'] == 1
    mesh=Mesh.from_precomputed(gzip.decompress((tmp_path/'cropped/30001:0:crop_0.gz').read_bytes()))
    np.testing.assert_array_equal(mesh.vertices,vertices)
