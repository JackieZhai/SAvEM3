"""MPS 适配 · 定点补丁（配合 sitecustomize.py 使用）。

修复三类 sitecustomize 覆盖不到的问题（保留 .orig 备份，幂等）：
1. 显式 `device = torch.device('cuda:0')` → MPS/CUDA/CPU 自动选择；
2. `if torch.cuda.is_available() is False: raise AttributeError('No GPU available')`
   → 允许 MPS；
3. PyTorch 0.4 旧 API：`param.data.add(alpha, param.data)`（新 torch 已移除该签名）
   → `param.data.mul_(1 - wd*lr)`；`np.bool` → `bool`。

目标目录：
  - savem3/（蒸馏训练主脚本）
  - saem2/（SAEM² 训练脚本，只做旧 API 修复；.cuda() 由 sitecustomize 覆盖）

用法：python patch_mps.py [--check]
"""
import argparse
import os
import re
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TARGETS = [
    os.path.join(REPO_ROOT, 'savem3'),
    os.path.join(REPO_ROOT, 'saem2'),
]

RULES = [
    # (pattern, replacement)
    (r"device = torch\.device\('cuda:0'\)",
     "device = (torch.device('mps') if torch.backends.mps.is_available()\n"
     "          else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')))"),
    (r"if torch\.cuda\.is_available\(\) is False:\n(\s*)raise AttributeError\('No GPU available'\)",
     r"if (torch.cuda.is_available() is False) and (torch.backends.mps.is_available() is False):\n\1raise AttributeError('No GPU/MPS available')"),
    (r"param\.data = param\.data\.add\(-cfg\.TRAIN\.weight_decay \* group\['lr'\], param\.data\)",
     "param.data.mul_(1.0 - cfg.TRAIN.weight_decay * group['lr'])"),
    (r"batch = self\.data_iter\.next\(\)",
     "batch = next(self.data_iter)"),
    (r"dtype=np\.bool\b", "dtype=bool"),
    (r"np\.bool\b", "bool"),
]


def patch_file(path, check_only):
    with open(path) as f:
        text = f.read()
    new = text
    hits = 0
    for pat, rep in RULES:
        new, n = re.subn(pat, rep, new)
        hits += n
    if hits == 0:
        return
    if check_only:
        print(f'  [将改动] {os.path.relpath(path, REPO_ROOT)}：{hits} 处')
        return
    bak = path + '.orig'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, 'w') as f:
        f.write(new)
    print(f'  [已修复] {os.path.relpath(path, REPO_ROOT)}：{hits} 处（备份 .orig）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    for target in TARGETS:
        if not os.path.isdir(target):
            print(f'[跳过] {target}', file=sys.stderr)
            continue
        for dirpath, _, files in os.walk(target):
            if '__pycache__' in dirpath:
                continue
            for fn in files:
                if fn.endswith('.py'):
                    patch_file(os.path.join(dirpath, fn), args.check)


if __name__ == '__main__':
    main()
