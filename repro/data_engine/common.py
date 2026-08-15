"""data_engine 共享工具：数据集位置配置 + CloudVolume 读写封装。

数据集配置统一取自 HQ-SAM/sam-hq/train/utils/location.py：
    data_location_list  : {数据集名: [img, seg, mem, dst, msk] 相对路径}
    data_range_list     : {数据集名: [[xs,ys,zs],[xe,ye,ze]] 体数据范围}
    data_label_range_list: {数据集名: 同上（标签范围，可能不同）}
    data_resolution_list: {数据集名: [x,y,z] nm/体素}
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DATA_ROOT = os.environ.get('SAVEM3_DATA_ROOT', REPO_ROOT)

_HQ_TRAIN = None
for _candidate in (os.path.join(DATA_ROOT, 'sam-hq', 'train'), os.path.join(DATA_ROOT, 'HQ-SAM', 'sam-hq', 'train')):
    if os.path.isdir(_candidate):
        _HQ_TRAIN = _candidate
        break
if _HQ_TRAIN is None:
    raise RuntimeError('未找到 sam-hq/train；请设置 SAVEM3_DATA_ROOT。')
sys.path.insert(0, _HQ_TRAIN)

from utils.location import (  # noqa: E402
    data_location_list, data_range_list, data_label_range_list, data_resolution_list)

# 成像法分组（论文 Tab.1 分类）
IMAGING_METHODS = {
    'ATUM-SEM': ['snemi', 'ac3'] + [k for k in data_location_list if k.startswith('axonem-h')],
    'FIB-SEM':  ['fib25'] + [k for k in data_location_list if k.startswith('hemibrain')],
    'ssTEM':    [k for k in data_location_list if k.startswith(('axonem-m', 'cremi'))],
    'SBEM':     [k for k in data_location_list if k.startswith(('j0126', 'segem'))],
}
METHOD_OF = {name: m for m, names in IMAGING_METHODS.items() for name in names}

# 已知黑名单标签（髓鞘/胶质/缺陷），出自 HQ-SAM/Ours/seg2mem_*.py
BLACKLIST = {
    'snemi':  [401],                # SNEMI3D 髓鞘
    'ac3':    [401],                # AC3 髓鞘（同源标注）
    'axonem-h': [],                 # AxonEM-H：髓鞘已单独标注，按需再补
    'axonem-m': [],
    'fib25':  [],
    'hemibrain': [317, 326, 370, 340],   # 胶质（eb/fb/pb 各块见 seg2mem_hemibrain.py）
    'cremi':  [],                   # 缺陷掩膜走 msk 图层（cremi 的 msk 已含缺陷权重）
    'j0126':  [],                   # 髓鞘标签按需补充（seg2mem_j0126.py 提及）
    'segem':  [],
}


def cv_path(dataset, layer, root=None):
    """返回某数据集某图层的本地绝对路径（file:// 用）。"""
    root = root or DATA_ROOT
    rel = data_location_list[dataset][['img', 'seg', 'mem', 'dst', 'msk'].index(layer)]
    if rel is None:
        return None
    for base in ('data', 'HQ-SAM', 'Ours'):
        p = os.path.join(root, base, rel)
        if os.path.exists(p):
            return p
    return os.path.join(root, 'data', rel)

def open_cv(dataset, layer, mip=0, root=None):
    from cloudvolume import CloudVolume
    p = cv_path(dataset, layer, root=root)
    if p is None:
        return None
    return CloudVolume('file://' + p, mip=mip, fill_missing=True)


def ranges_of(dataset):
    """返回 (数据范围, 标签范围, 分辨率)。"""
    return (data_range_list[dataset], data_label_range_list.get(dataset, data_range_list[dataset]),
            data_resolution_list[dataset])
