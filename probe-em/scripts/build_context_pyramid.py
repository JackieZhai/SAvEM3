"""Build bounded display pyramids for native-only grayscale JPEG context volumes.

The native working copy is independent and pixel-exact. These additional display
scales use libjpeg's 1/8 DCT reduction, followed by valid-voxel box means; they are
not instance labels, probabilities, or an exact area average of native pixels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import gzip
import io
import itertools
import json
from pathlib import Path
import sys
import threading
import zlib

import numpy as np
from PIL import Image
from cloudvolume import CloudVolume

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_em.working_volume import chunk_bounds, chunk_name, write_json


def reduced_jpeg(payload, shape):
    """Reduce XY only; JPEG stacks place complete Y planes along image rows."""
    shape = np.asarray(shape, int)
    if np.any(shape[:2] % 8):
        raise ValueError('DCT reduction requires input XY dimensions divisible by eight')
    with Image.open(io.BytesIO(payload)) as image:
        if image.mode != 'L' or image.size != (int(shape[0]), int(shape[1]*shape[2])):
            raise ValueError('Expected a grayscale precomputed JPEG stack')
        expected = (int(shape[0]//8), int(shape[1]*shape[2]//8))
        image.draft('L', expected)
        if image.size != expected:
            raise ValueError('JPEG decoder did not provide the requested 1/8 reduction')
        return np.asarray(image).reshape(int(shape[2]),int(shape[1]//8),int(shape[0]//8)).transpose(2,1,0).copy()


def mean_two(values):
    """Mean over 2x2x2 cells; partial edge cells use only available voxels."""
    total = values.astype(np.uint32)
    count = np.ones(values.shape, np.uint8)
    for axis in range(3):
        indices = np.arange(0,total.shape[axis],2)
        total = np.add.reduceat(total,indices,axis=axis)
        count = np.add.reduceat(count,indices,axis=axis)
    return np.rint(total/count).astype(np.uint8)


def build(source, info, bounds_nm, output, workers=4, resume=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    plan = dict(source=str(source),info=info,bounds_nm=bounds_nm)
    if output.exists():
        if not resume or not (output/'plan.json').is_file() or json.loads((output/'plan.json').read_text()) != plan:
            raise FileExistsError('Context resume requires an identical saved plan')
    else:
        output.mkdir(parents=True)
        write_json(output/'plan.json',plan)
    native = info['scales'][0]
    resolution = np.asarray(native['resolution'],int)
    offset = np.asarray(native.get('voxel_offset',[0,0,0]),int)
    end = offset+native['size']
    chunk = np.asarray(native['chunk_sizes'][0],int)
    factor = np.array([8,8,1])
    bounds = np.asarray(bounds_nm)/resolution
    if (native['encoding'] != 'jpeg' or info['data_type'] != 'uint8' or info['num_channels'] != 1
            or not np.allclose(bounds/factor,np.rint(bounds/factor))
            or np.any(offset % factor) or np.any(chunk % factor)):
        raise ValueError('Context pyramid needs a grayscale uint8 JPEG crop on the 8x8 XY grid')
    low, high = np.rint(bounds/factor).astype(int)
    res = resolution*factor
    scale = dict(key='_'.join(map(str,res)),encoding='raw',resolution=res.tolist(),
                 voxel_offset=low.tolist(),size=(high-low).tolist(),chunk_sizes=[[256,256,int(chunk[2])]])
    metadata = dict(type='image',data_type='uint8',num_channels=1,scales=[scale])
    (output/scale['key']).mkdir(exist_ok=resume)
    locks = [threading.Lock() for _ in range(64)]

    @lru_cache(maxsize=96)
    def read(index):
        start = offset+np.array(index)*chunk
        stop = np.minimum(start+chunk,end)
        path = source/native['key']/chunk_name(start,stop)
        compressed = path.with_name(path.name+'.gz')
        payload = gzip.decompress(compressed.read_bytes()) if compressed.is_file() else path.read_bytes()
        return reduced_jpeg(payload,stop-start)

    def write_chunk(box):
        a,b = box
        path = output/scale['key']/(chunk_name(a,b)+'.gz')
        if resume and valid_chunk(path, b-a):
            return
        values = np.empty(tuple(b-a),np.uint8)
        first = (a*factor-offset)//chunk
        last = ((b*factor-1)-offset)//chunk
        for index in itertools.product(*(range(int(x),int(y)+1) for x,y in zip(first,last))):
            with locks[hash(index)%len(locks)]:
                tile = read(index)
            start = (offset+np.array(index)*chunk)//factor
            stop = start+tile.shape
            u,v = np.maximum(a,start),np.minimum(b,stop)
            values[tuple(slice(int(x),int(y)) for x,y in zip(u-a,v-a))] = tile[
                tuple(slice(int(x),int(y)) for x,y in zip(u-start,v-start))]
        save_chunk(path,values)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number,_ in enumerate(pool.map(write_chunk,chunk_bounds(scale)),1):
            if number % 256 == 0:
                print(json.dumps(dict(dataset=output.name,scale=scale['key'],chunks=number)),flush=True)
    read.cache_clear()
    write_json(output/'info',metadata)
    for _ in range(2):
        previous = metadata['scales'][-1]
        parent = CloudVolume(output.as_uri(),mip=len(metadata['scales'])-1,fill_missing=False,parallel=False)
        p0 = np.array(previous['voxel_offset']);p1=p0+previous['size']
        if np.any(p0%2):
            raise ValueError('Box-mean pyramid requires an even parent offset')
        low,high = p0//2,(p1+1)//2
        res = np.array(previous['resolution'])*2
        scale = dict(key='_'.join(map(str,res)),encoding='raw',resolution=res.tolist(),
                     voxel_offset=low.tolist(),size=(high-low).tolist(),chunk_sizes=[[256,256,32]])
        (output/scale['key']).mkdir(exist_ok=resume)
        for a,b in chunk_bounds(scale):
            path=output/scale['key']/(chunk_name(a,b)+'.gz')
            if resume and valid_chunk(path,b-a):
                continue
            stop=np.minimum(b*2,p1)
            values=np.asarray(parent[tuple(slice(int(x),int(y)) for x,y in zip(a*2,stop))])[...,0]
            result=mean_two(values)
            save_chunk(path,result)
        metadata['scales'].append(scale)
        write_json(output/'info',metadata)
    write_json(output/'provenance.json',dict(source=str(source),bounds_nm=bounds_nm,
               first_scale='Grayscale JPEG 1/8 DCT reduction in XY; original Z resolution',
               subsequent_scales='2x2x2 valid-voxel arithmetic means rounded to uint8',
               purpose='Display context intensities; native crop remains unchanged',status='complete'))


def valid_chunk(path, shape):
    if not path.is_file():
        return False
    try:
        return len(gzip.decompress(path.read_bytes())) == int(np.prod(shape))
    except (OSError,EOFError,zlib.error):
        return False


def save_chunk(path, values):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_bytes(gzip.compress(values.tobytes(order='F'),compresslevel=1,mtime=0))
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--info',required=True,type=Path)
    parser.add_argument('--crop-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true',help='Reuse gzip-validated chunks under the identical saved plan')
    args=parser.parse_args()
    build(args.source,json.loads(args.info.read_text()),json.loads(args.crop_manifest.read_text())['bounds_nm'],
          args.output,args.workers,args.resume)


if __name__ == '__main__':
    main()
