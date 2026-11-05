"""Save a shared physical crop of local precomputed datasets, without pixel loss."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_em.working_volume import crop_precomputed, scan_coverage, write_json
from probe_em.mesh_crop import crop_legacy_meshes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path,
                        help='JSON with datasets [{name, source, optional info_path}] and bounds_nm [[x,y,z],[x,y,z]]')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--jpeg-crop', type=Path, help='Optional compiled jpeg_crop.c helper; otherwise lossless gzip/raw')
    parser.add_argument('--scratch-dir', type=Path,
                        help='Scratch environment outside the output dataset; JPEG requests use memory pipes')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-resume', action='store_true', help='Compare existing chunks against source pixels again')
    parser.add_argument('--scan-raw', type=Path, help='Derive XY bounds from bottom-right padding in every section')
    parser.add_argument('--scan-mip', type=int, default=3, help='Must retain the original Z resolution')
    parser.add_argument('--threshold', type=int, default=5)
    parser.add_argument('--guard', type=int, default=1, help='Inward padding guard in scan-grid XY voxels')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    bounds = manifest.get('bounds_nm')
    if args.scan_raw:
        from cloudvolume import CloudVolume
        import numpy as np
        raw = CloudVolume(args.scan_raw.resolve().as_uri(), mip=args.scan_mip, fill_missing=False, parallel=False)
        if raw.resolution[2] != raw.info['scales'][0]['resolution'][2]:
            raise ValueError('Coverage scan must inspect every original Z section')
        common, report = scan_coverage(raw, args.threshold, args.guard)
        np.save(output/'coverage.npy', common)
        write_json(output/'coverage.json', report)
        native = raw.info['scales'][0]
        z0 = native.get('voxel_offset', [0,0,0])[2] * native['resolution'][2]
        z1 = z0 + native['size'][2] * native['resolution'][2]
        bounds = [report['bounds_xy_nm'][0]+[z0], report['bounds_xy_nm'][1]+[z1]]
    if bounds is None:
        raise ValueError('Provide bounds_nm or --scan-raw')
    collection = dict(status='running', bounds_nm=bounds, datasets=manifest['datasets'])
    write_json(output/'collection.json', collection)
    try:
        for item in manifest['datasets']:
            name = item['name']
            if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
                raise ValueError('Dataset names must be single directory names')
            def local(value):
                path = Path(value)
                return path.resolve() if path.is_absolute() else (args.manifest.parent/path).resolve()
            source = local(item['source'])
            archive_path=local(item['archive_path']) if item.get('archive_path') else None
            if archive_path and not item.get('info_path'):
                with zipfile.ZipFile(archive_path) as archive:
                    info=json.loads(archive.read(source.name+'/info'))
            else:
                info = json.loads((local(item['info_path']) if item.get('info_path') else source/'info').read_text())
            crop_precomputed(source, output/name, bounds, info=info,
                             jpeg_crop=args.jpeg_crop.resolve() if args.jpeg_crop else None,
                             workers=args.workers, resume=args.resume, verify_resume=args.verify_resume,
                             scratch_dir=args.scratch_dir,
                             source_archive=archive_path,
                             archive_identity=item.get('archive_identity','sha256'),
                             progress=lambda value: print(json.dumps(value), flush=True))
            if info.get('mesh'):
                mesh_folder = output/name/'mesh'
                report_path = mesh_folder/'crop_report.json'
                if args.resume and report_path.is_file():
                    report = json.loads(report_path.read_text())
                    if (report.get('status') != 'complete' or report.get('bounds_nm') != bounds
                            or report.get('source') != str((source/info['mesh']).resolve())):
                        raise ValueError('Existing mesh crop does not match this source and crop')
                else:
                    crop_legacy_meshes(source/info['mesh'],mesh_folder,bounds,source_archive=archive_path)
                metadata = json.loads((output/name/'info').read_text())
                metadata['mesh'] = 'mesh'
                write_json(output/name/'info',metadata)
                print(json.dumps(dict(dataset=name,meshes='complete')),flush=True)
        collection['status'] = 'voxel_exports_complete'
        write_json(output/'collection.json', collection)
    except BaseException as error:
        collection.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'collection.json', collection)
        raise


if __name__ == '__main__':
    main()
