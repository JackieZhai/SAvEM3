"""Targeted MPS compatibility patches, used together with sitecustomize.py.

Handle issues that sitecustomize cannot intercept. Keep .orig backups and make
patching idempotent:
1. Replace explicit `device = torch.device('cuda:0')` with MPS/CUDA/CPU selection.
2. `if torch.cuda.is_available() is False: raise AttributeError('No GPU available')`
   is relaxed to allow MPS.
3. Replace the removed PyTorch 0.4 `param.data.add(alpha, param.data)` signature
   with `param.data.mul_(1 - wd*lr)`, and replace `np.bool` with `bool`.

Target directories:
  - savem3/: main distillation training scripts.
  - saem2/: SAEM² training scripts; patch legacy APIs, with .cuda() handled by
    sitecustomize.

Usage: python patch_mps.py [--check]
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
        print(f'  [would patch] {os.path.relpath(path, REPO_ROOT)}: {hits} matches')
        return
    bak = path + '.orig'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, 'w') as f:
        f.write(new)
    print(f'  [patched] {os.path.relpath(path, REPO_ROOT)}: {hits} matches (.orig backup)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    for target in TARGETS:
        if not os.path.isdir(target):
            print(f'[skip] {target}', file=sys.stderr)
            continue
        for dirpath, _, files in os.walk(target):
            if '__pycache__' in dirpath:
                continue
            for fn in files:
                if fn.endswith('.py'):
                    patch_file(os.path.join(dirpath, fn), args.check)


if __name__ == '__main__':
    main()
