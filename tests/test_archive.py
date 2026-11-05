"""Independent ZIP parts must round-trip exactly and reject corrupt or unsafe input."""
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe-em'))
from probe_em.archive import digest, package_dataset, partition, restore_dataset, safe_relative


def dataset(path):
    path.mkdir()
    (path/'scale').mkdir()
    (path/'mesh').mkdir()
    (path/'info').write_text(json.dumps(dict(scales=[dict(key='scale')], mesh='mesh')))
    (path/'crop_status.json').write_text('{"status":"complete"}')
    (path/'crop_plan.json').write_text('{"private_path":"/local/private/source"}')
    (path/'scale'/'0-1_0-1_0-1.gz').write_bytes(bytes(range(256))*3)
    (path/'scale'/'1-2_0-1_0-1.gz').write_bytes(bytes(range(256))*3)
    (path/'mesh'/'30001:0').write_text('{"fragments": ["triangle"]}')
    (path/'mesh'/'triangle').write_bytes(b'actual mesh payload')
    return path


def test_independent_zip_parts_round_trip_and_resume(tmp_path):
    source = dataset(tmp_path/'data')
    report = package_dataset(source,tmp_path/'package',max_bytes=1600)
    assert len(report['parts']) > 1 and report['status'] == 'complete'
    names = []
    for part in report['parts']:
        path = tmp_path/'package'/part['file']
        assert path.stat().st_size <= 1600
        assert digest(path) == part['sha256']
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            names.extend(archive.namelist())
    assert len(names) == len(set(names)) == report['source_files']
    assert not any('crop_plan' in name or 'crop_status' in name for name in names)
    restore_dataset(tmp_path/'package',tmp_path/'restored')
    for name in names:
        assert (tmp_path/'restored'/name).read_bytes() == (tmp_path/name).read_bytes()
    restore_dataset(tmp_path/'package',tmp_path/'restored')
    assert package_dataset(source,tmp_path/'package',max_bytes=1600,resume=True) == report
    (tmp_path/'restored/data/info').write_text('different')
    with pytest.raises(FileExistsError):
        restore_dataset(tmp_path/'package',tmp_path/'restored')


def test_corrupt_part_is_rejected_before_extraction(tmp_path):
    source = dataset(tmp_path/'data')
    report = package_dataset(source,tmp_path/'package',max_bytes=1600)
    path = tmp_path/'package'/report['parts'][-1]['file']
    path.write_bytes(path.read_bytes()[:-10]+b'corruption')
    with pytest.raises(ValueError,match='checksum mismatch'):
        restore_dataset(tmp_path/'package',tmp_path/'restored')
    assert not (tmp_path/'restored').exists()
    with pytest.raises(ValueError,match='resume verification'):
        package_dataset(source,tmp_path/'package',max_bytes=1600,resume=True)


@pytest.mark.parametrize('name',['../escape','a/../escape','/absolute','C:/windows','a\\b','a//b'])
def test_unsafe_archive_paths_are_rejected(name):
    with pytest.raises(ValueError):
        safe_relative(name)


def test_incomplete_source_and_oversized_member_are_rejected(tmp_path):
    source = dataset(tmp_path/'data')
    (source/'crop_status.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError,match='completed working crop'):
        package_dataset(source,tmp_path/'package')
    with pytest.raises(ValueError,match='exceeds'):
        partition([dict(path='data/large',size=1000)],1024)


def test_symlink_cannot_redirect_extraction(tmp_path):
    source = dataset(tmp_path/'data')
    package_dataset(source,tmp_path/'package')
    (tmp_path/'restored').mkdir()
    (tmp_path/'outside').mkdir()
    (tmp_path/'restored/data').symlink_to(tmp_path/'outside',target_is_directory=True)
    with pytest.raises(ValueError,match='escape'):
        restore_dataset(tmp_path/'package',tmp_path/'restored')
    assert not list((tmp_path/'outside').iterdir())
