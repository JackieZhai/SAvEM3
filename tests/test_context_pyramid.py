"""Context display reduction must preserve Z and correctly average partial edges."""
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe-em' / 'scripts'))
from build_context_pyramid import build, mean_two, reduced_jpeg


def test_jpeg_reduction_preserves_individual_sections():
    values = np.empty((32,24,3),np.uint8)
    values[:,:,0], values[:,:,1], values[:,:,2] = 24,96,224
    stream = io.BytesIO()
    Image.fromarray(values.transpose(2,1,0).reshape(72,32)).save(stream,format='JPEG',quality=100)
    reduced = reduced_jpeg(stream.getvalue(),values.shape)
    assert reduced.shape == (4,3,3)
    for z in range(3):
        np.testing.assert_array_equal(reduced[:,:,z], values[::8,::8,z])


def test_partial_box_means_use_available_pixels():
    values = np.arange(5*3*7,dtype=np.uint8).reshape(5,3,7)
    actual = mean_two(values)
    for x,y,z in np.ndindex(actual.shape):
        expected = np.rint(values[x*2:x*2+2,y*2:y*2+2,z*2:z*2+2].mean())
        assert actual[x,y,z] == expected
    assert actual[-1,-1,-1] == values[-1,-1,-1]


def test_pyramid_resume_repairs_incomplete_chunks_and_preserves_geometry(tmp_path):
    from cloudvolume import CloudVolume
    import pytest
    source=tmp_path/'source'
    info=CloudVolume.create_new_info(num_channels=1,layer_type='image',data_type='uint8',encoding='jpeg',
        resolution=[4,4,30],voxel_offset=[0,0,0],chunk_size=[8,8,2],volume_size=[16,16,5])
    volume=CloudVolume(source.resolve().as_uri(),info=info,parallel=False)
    volume.commit_info()
    values=np.broadcast_to(np.array([24,64,96,128,224],np.uint8),(16,16,5)).copy()
    volume[:]=values[...,None]
    bounds=[[0,0,0],[64,64,150]]
    build(source,info,bounds,tmp_path/'pyramid',workers=2)
    first=CloudVolume((tmp_path/'pyramid').resolve().as_uri(),fill_missing=False,parallel=False)
    expected=np.asarray(first[:]).copy()
    assert expected.shape == (2,2,5,1)
    assert [s['resolution'] for s in first.info['scales']] == [[32,32,30],[64,64,60],[128,128,120]]
    next((tmp_path/'pyramid/32_32_30').iterdir()).write_bytes(b'truncated gzip')
    build(source,info,bounds,tmp_path/'pyramid',workers=2,resume=True)
    np.testing.assert_array_equal(np.asarray(first[:]),expected)
    with pytest.raises(FileExistsError,match='identical saved plan'):
        build(source,info,[[0,0,0],[32,64,150]],tmp_path/'pyramid',resume=True)
