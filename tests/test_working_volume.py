"""Working-volume coverage, exact labels, codec boundaries and resumability."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe-em'))
from probe_em.working_volume import crop_info, crop_precomputed, largest_rectangle, padding_mask


def test_largest_rectangle_matches_exhaustive_search():
    rng = np.random.default_rng(29)
    for _ in range(30):
        valid = rng.random((7, 6)) > .25
        low, high = largest_rectangle(valid)
        assert valid[low[0]:high[0], low[1]:high[1]].all()
        expected = max((b-a)*(d-c) for a in range(7) for b in range(a+1, 8)
                       for c in range(6) for d in range(c+1, 7) if valid[a:b, c:d].all())
        assert np.prod(high-low) == expected


def test_padding_retains_dark_tissue_and_excludes_slanted_margin():
    section = np.full((30, 25), 100, np.uint8)
    section[0:3, 4:8] = 0  # A dark membrane touches an unrelated image edge.
    for y in range(25):
        section[25-y//5:, y] = 0
    section[:, -2:] = 0
    missing = padding_mask(section)
    assert missing[-1, -1] and missing[:, -1].all()
    assert not missing[1, 5] and not missing[0, 0]
    with pytest.raises(ValueError, match='padding seed'):
        padding_mask(np.ones((5, 5))*100)


def make_source(path, values, encoding='raw'):
    from cloudvolume import CloudVolume
    info = CloudVolume.create_new_info(num_channels=1, layer_type='segmentation' if values.dtype == np.uint64 else 'image',
        data_type=str(values.dtype), encoding=encoding, resolution=[4,4,30], voxel_offset=[16,24,5],
        chunk_size=[32,32,3], volume_size=list(values.shape))
    source = CloudVolume(path.resolve().as_uri(), info=info, parallel=False)
    source.commit_info();source[source.bounds] = values[..., None]
    return source


def test_crop_retiles_offset_and_preserves_large_integer_ids(tmp_path):
    from cloudvolume import CloudVolume
    values = np.zeros((48,48,6), np.uint64)
    values[12:36,8:42] = 2**63+37
    source = make_source(tmp_path/'source', values)
    bounds = [[(16+7)*4,(24+5)*4,5*30],[(16+43)*4,(24+45)*4,11*30]]
    result = crop_precomputed(tmp_path/'source', tmp_path/'working', bounds, workers=2)
    output = CloudVolume((tmp_path/'working').resolve().as_uri(), parallel=False, fill_missing=False)
    np.testing.assert_array_equal(output[:][...,0], values[7:43,5:45])
    assert list(output.voxel_offset) == [23,29,5]
    assert all(scale['verified'] == scale['chunks'] for scale in result['scales'])
    np.testing.assert_array_equal(source[:][...,0], values)
    with pytest.raises(FileExistsError):
        crop_precomputed(tmp_path/'source', tmp_path/'working', bounds)
    resumed = crop_precomputed(tmp_path/'source', tmp_path/'working', bounds, resume=True)
    assert resumed['scales'][0]['resumed'] == result['scales'][0]['chunks']
    with pytest.raises(FileExistsError):
        crop_precomputed(tmp_path/'source', tmp_path/'working', [[64,96,150],[240,260,330]], resume=True)


def test_jpeg_partial_blocks_use_lossless_raw_and_never_fill_missing(tmp_path):
    from cloudvolume import CloudVolume
    source = make_source(tmp_path/'source', np.random.default_rng(4).integers(0,256,(48,48,6),dtype=np.uint8), 'jpeg')
    bounds = [[92,116,150],[236,276,330]]
    target = crop_info(source.info, bounds, jpeg_crop='optional-helper')
    assert target['scales'][0]['encoding'] == 'raw'
    crop_precomputed(tmp_path/'source', tmp_path/'working', bounds, workers=2)
    output = CloudVolume((tmp_path/'working').resolve().as_uri(), parallel=False)
    np.testing.assert_array_equal(output[:], source[23:59,29:69,5:11])
    next((tmp_path/'source'/'4_4_30').iterdir()).unlink()
    with pytest.raises(FileNotFoundError):
        crop_precomputed(tmp_path/'source', tmp_path/'missing', bounds, workers=2)
    assert not (tmp_path/'missing'/'info').exists()
    assert json.loads((tmp_path/'missing'/'crop_status.json').read_text())['status'] == 'failed'


@pytest.mark.skipif(not os.environ.get('JPEG_CROP_EXECUTABLE'), reason='Optional libjpeg coefficient helper')
def test_jpeg_coefficient_retile_preserves_every_decoded_pixel(tmp_path):
    from cloudvolume import CloudVolume
    source = make_source(tmp_path/'source', np.random.default_rng(5).integers(0,256,(64,64,6),dtype=np.uint8), 'jpeg')
    bounds = [[96,128,150],[288,320,330]]
    report = crop_precomputed(tmp_path/'source', tmp_path/'working', bounds,
                             jpeg_crop=os.environ['JPEG_CROP_EXECUTABLE'], workers=2)
    output = CloudVolume((tmp_path/'working').resolve().as_uri(), parallel=False)
    np.testing.assert_array_equal(output[:], source[24:72,32:80,5:11])
    assert report['scales'][0]['encoding'] == 'jpeg'


def test_resume_verification_detects_corruption(tmp_path):
    import gzip
    source = make_source(tmp_path/'source', np.ones((48,48,6),np.uint64))
    bounds = [[64,96,150],[256,288,330]]
    crop_precomputed(tmp_path/'source',tmp_path/'working',bounds)
    path = next((tmp_path/'working'/'4_4_30').glob('*.gz'))
    original = gzip.decompress(path.read_bytes())
    path.write_bytes(gzip.compress(bytes(len(original))))
    with pytest.raises(ValueError,match='differs from source'):
        crop_precomputed(tmp_path/'source',tmp_path/'working',bounds,resume=True,verify_resume=True)


def test_uniform_compressed_labels_preserve_zero_and_large_ids(tmp_path):
    from cloudvolume import CloudVolume
    values = np.zeros((64,64,6),np.uint64)
    values[35:50,35:50,3:] = 2**63+19
    make_source(tmp_path/'source',values,'compressed_segmentation')
    bounds = [[96,128,150],[288,320,330]]
    report = crop_precomputed(tmp_path/'source',tmp_path/'working',bounds,workers=2)
    output = CloudVolume((tmp_path/'working').resolve().as_uri(),fill_missing=False,parallel=False)
    np.testing.assert_array_equal(output[:][...,0],values[8:56,8:56])
    assert report['scales'][0]['uniform_verified'] > 0
    assert report['scales'][0]['verified'] > 0


def test_jpeg_pipeline_creates_no_temporary_images_even_on_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import probe_em.working_volume as working
    make_source(tmp_path/'source',np.ones((64,64,6),np.uint8),'jpeg')
    bounds = [[96,128,150],[288,320,330]]
    scratch = tmp_path/'internal-scratch'
    observed = []

    def failed_helper(command, **kwargs):
        assert command[1] == '--stream'
        assert 1 <= int.from_bytes(os.read(kwargs['stdin'],4),'little') <= 4
        assert kwargs['env']['TMPDIR'] == str(scratch)
        assert not list(scratch.iterdir())
        observed.append(command)
        return SimpleNamespace(returncode=1,stderr=b'intentional helper failure')

    monkeypatch.setattr(working.subprocess,'run',failed_helper)
    with pytest.raises(ValueError,match='intentional helper failure'):
        crop_precomputed(tmp_path/'source',tmp_path/'working',bounds,jpeg_crop='helper',
                         scratch_dir=scratch,workers=1)
    assert observed and not list(scratch.iterdir())
    assert not (tmp_path/'working'/'info').exists()
    assert not list((tmp_path/'working').glob('*jpeg-*'))
    with pytest.raises(ValueError,match='outside the working dataset'):
        crop_precomputed(tmp_path/'source',tmp_path/'other',bounds,scratch_dir=tmp_path/'other'/'scratch')


@pytest.mark.parametrize('identity',['sha256','central-directory'])
def test_direct_zip_source_preserves_labels_without_extracted_files(tmp_path,identity):
    import shutil
    import zipfile
    from cloudvolume import CloudVolume
    values=np.zeros((48,48,6),np.uint64)
    values[9:40,11:39,2:]=2**63+29
    make_source(tmp_path/'source',values,'compressed_segmentation')
    archive_path=tmp_path/'source.zip'
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (tmp_path/'source').rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(tmp_path).as_posix())
    shutil.rmtree(tmp_path/'source')
    bounds=[[92,116,150],[236,276,330]]
    crop_precomputed(tmp_path/'source',tmp_path/'working',bounds,source_archive=archive_path,
                     archive_identity=identity,workers=2)
    output=CloudVolume((tmp_path/'working').resolve().as_uri(),fill_missing=False,parallel=False)
    np.testing.assert_array_equal(output[:][...,0],values[7:43,5:45])
    plan=json.loads((tmp_path/'working/crop_plan.json').read_text())
    key='sha256' if identity=='sha256' else 'central_directory_sha256'
    assert len(plan['source_archive'][key]) == 64


@pytest.mark.parametrize('dtype',['uint32','uint64'])
def test_zero_palette_certificate_matches_decoded_constant_blocks(dtype):
    from cloudvolume import chunks
    from probe_em.working_volume import certified_zero_segmentation
    shape=(19,13,3,1);block=[8,8,8]
    values=np.zeros(shape,dtype=dtype,order='F')
    encoded=chunks.encode(values,'compressed_segmentation',block_size=block)
    assert certified_zero_segmentation(encoded,shape,block,dtype)
    values[4:12,3:9]=np.iinfo(dtype).max
    encoded=chunks.encode(values,'compressed_segmentation',block_size=block)
    assert not certified_zero_segmentation(encoded,shape,block,dtype)
    values[:]=np.iinfo(dtype).max
    encoded=chunks.encode(values,'compressed_segmentation',block_size=block)
    assert not certified_zero_segmentation(encoded,shape,block,dtype)
    assert not certified_zero_segmentation(b'bad',shape,block,dtype)
