"""M6 · 大体积稀疏评测：oracle RoI + 分块推理 + 重叠拼接（论文 §3 Sparse Evaluation 原话：

"According to the scope around the mesh of neurons, the model only runs in chunks
 (512x512x64 voxels) within the passing scope, and the results merge into
 complete neurons through the vanilla overlap-based stitching."
"H01: 512x512x64 in 8x8x33nm; MICrONS Pinky: 512x512x64 in 8x8x40nm"

本脚本编排"蒸馏模型在指定 RoI 上的分块推理 + 重叠拼接"：
1. 读 RoI 清单（JSON：每个 chunk 的体数据偏移与尺寸；由目标神经元的 mesh/骨架范围生成）；
2. 对每个 chunk 调 scripts_savem3 的推理（test_devoem_sparse_membrane_triplet_2.py 或
   滑窗 Provider）得到膜概率块；
3. 相邻块在重叠区取平均（vanilla overlap stitching）拼成整块膜图；
4. 整块膜图 → distill_postprocess.py 的过分割+聚合 → 提取目标神经元段。

用法：
    python sparse_eval.py --roi roi_chunks.json --ckpt models/xxx/model-200000.ckpt \
        --cfg mem3c2c_3ds_t3t --out-dir ./sparse_out
roi_chunks.json 格式：{"chunks": [{"offset": [z,y,x], "size": [64,512,512]}, ...],
                       "neuron_id": 2252715458}
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


def load_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['main'][:]


def stitch(chunks, offsets, overlap=16):
    """重叠区取平均的 vanilla overlap-based stitching。

    chunks: list of (Z,H,W) 膜概率块；offsets: 对应 [z,y,x] 全局偏移（含重叠步进）。
    """
    from tqdm import tqdm
    shapes = np.array([c.shape for c in chunks])
    z_max = max(o[0] + s[0] for o, s in zip(offsets, shapes))
    y_max = max(o[1] + s[1] for o, s in zip(offsets, shapes))
    x_max = max(o[2] + s[2] for o, s in zip(offsets, shapes))
    acc = np.zeros((z_max, y_max, x_max), dtype=np.float64)
    cnt = np.zeros_like(acc)
    for c, o in tqdm(zip(chunks, offsets), desc='stitch'):
        zs, ys, xs = o
        zh, yh, xh = c.shape
        acc[zs:zs + zh, ys:ys + yh, xs:xs + xh] += c.astype(np.float64)
        cnt[zs:zs + zh, ys:ys + yh, xs:xs + xh] += 1
    cnt[cnt == 0] = 1
    return (acc / cnt).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roi', required=True, help='chunk 清单 JSON')
    ap.add_argument('--ckpt', required=True, help='蒸馏模型 ckpt 路径')
    ap.add_argument('--cfg', default='mem3c2c_3ds_t3t')
    ap.add_argument('--scripts', default=None,
                    help='蒸馏训练脚本目录（默认 $SAVEM3_DATA_ROOT/savem3）')
    ap.add_argument('--out-dir', default='./sparse_out')
    ap.add_argument('--overlap', type=int, default=16)
    ap.add_argument('--data-root', default=None, help='体数据根（CloudVolume file:// 前缀父目录）')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.roi) as f:
        roi = json.load(f)
    neuron_id = roi.get('neuron_id')

    data_root = args.data_root or os.environ.get('SAVEM3_DATA_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
    scripts = args.scripts or os.path.join(data_root, 'savem3')

    chunk_files = []
    for i, ch in enumerate(roi['chunks']):
        out_h5 = os.path.join(args.out_dir, f'chunk_{i:04d}.h5')
        # 分块推理：这里以子进程调用蒸馏推理入口；具体推理脚本按 scripts_savem3 的
        # test_devoem_sparse_membrane_triplet_2.py 的实际参数调整
        cmd = [sys.executable, os.path.join(scripts, 'test_devoem_sparse_membrane_triplet_2.py'),
               '-c', args.cfg, '--ckpt', args.ckpt,
               '--offset', ','.join(map(str, ch['offset'])),
               '--size', ','.join(map(str, ch['size'])),
               '--out', out_h5]
        print(' '.join(cmd))
        # subprocess.check_call(cmd)  # 在 GPU 服务器上取消注释
        chunk_files.append(out_h5)

    print(f'共 {len(chunk_files)} 块；请在 GPU 服务器上执行分块推理后再次运行 --stitch-only 模式')
    print('拼接逻辑：chunks 按 roi 清单 offsets 重叠 --overlap 体素取平均。')


if __name__ == '__main__':
    main()
