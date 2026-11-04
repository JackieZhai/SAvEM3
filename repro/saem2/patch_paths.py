"""M2 preparation: replace hard-coded SAEM2 cluster paths with local/configurable paths.

Rules:
1. Respect CUDA_VISIBLE_DEVICES from the launcher/environment instead of a fixed value.
2. /home/zhaih/.../pretrained_checkpoint/... and /home/guojy2/.../pretrained_checkpoint/...
   become 'pretrained_checkpoint/...' relative to the training directory.
3. /home/guojy2/.../work_dirs/... becomes relative 'work_dirs/...'.
4. Resolve the dataloader roots prepared_embedding, prepared_segments_mul_2d, and
   prepared_segments_mul_mem under os.environ.get('SAVEM3_DATA_ROOT', '.').
   SAVEM3_EMBED_ROOT, SAVEM3_LABEL_ROOT, and SAVEM3_MEM_ROOT override each root.
5. Use relative './record2d_*.json' paths.

Idempotent: keep a .orig backup on first modification without replacing it later.
Usage: python patch_paths.py [--check]. The --check option prints changes without writing files.
"""
import argparse
import fileinput
import os
import re
import shutil
import sys

TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'saem2')

RULES = [
    # Apply (pattern, replacement) rules in order; skip unmatched patterns.
    (r"os\.environ\['CUDA_VISIBLE_DEVICES'\] = '\d+'",
     "os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')"),
    (r"['\"](/home/(zhaih|guojy2)/[^'\"]*?/)?pretrained_checkpoint/([^'\"]+)['\"]",
     r"'pretrained_checkpoint/\3'"),
    (r"['\"]/home/guojy2/[^'\"]*?/work_dirs/([^'\"]+)['\"]",
     r"'work_dirs/\1'"),
    (r"self\.embed_root_path = '/home/[^']*prepared_embedding/'",
     "self.embed_root_path = os.environ.get('SAVEM3_EMBED_ROOT',\n"
     "    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_embedding/'))"),
    (r"self\.label_root_path = '/home/[^']*prepared_segments_mul_2d/'",
     "self.label_root_path = os.environ.get('SAVEM3_LABEL_ROOT',\n"
     "    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_2d/'))"),
    (r"self\.mem_root_path = '/home/[^']*prepared_segments_mul_mem/'",
     "self.mem_root_path = os.environ.get('SAVEM3_MEM_ROOT',\n"
     "    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_mem/'))"),
    (r"label_root = '/home/[^']*prepared_segments_mul_mem/'",
     "label_root = os.environ.get('SAVEM3_MEM_ROOT',\n"
     "    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_mem/'))"),
    (r"label_root_2d = '/home/[^']*prepared_segments_mul_2d/'",
     "label_root_2d = os.environ.get('SAVEM3_LABEL_ROOT',\n"
     "    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_2d/'))"),
    (r"['\"]/home/[^'\"]*?/record2d_(train|test)\.json['\"]",
     r"'./record2d_\1.json'"),
]

FILES = [
    'trainMemProISBI_2d_v4.py', 'test_SAEM_2d_v4.py', 'trainMemProISBIPB_2d_v4.py',
    'trainMemProISBI_2d_v4_iter.py',
    'trainMemProISBI_bbbc_v3.py', 'trainMemProISBI_weak_v3.py', 'trainMemProISBI_elong_v3.py',
    'trainNonMemProISBI_2d_v4.py', 'trainNonMemProISBI_bbbc_v3.py',
    'trainNonMemProISBI_weak_v3.py', 'trainNonMemProISBI_elong_v3.py',
    'trainSAMfineISBI_2d_v4.py', 'trainSAMfineISBI_bbbc_v3.py',
    'trainSAMfineISBI_weak_v3.py', 'trainSAMfineISBI_elong_v3.py',
    'test_HQ_2d.py', 'test_HQ_fine_2d_v4.py', 'test_SAM_2d.py', 'test_SAM_fine_2d_v4.py',
    'test_micro_SAM_2d.py', 'test_micro_SAM_bbbc_v3.py', 'test_micro_SAM_weak_v3.py',
    'test_micro_SAM_elong_v3.py',
    'json2d_create.py', 'json2d_lm_create.py',
    'utils/dataloader_isbi_2d_v4.py', 'utils/dataloader_isbi_test_2d.py',
]


def patch_file(path, check_only):
    with open(path) as f:
        lines = f.readlines()
    changed = []
    for i, line in enumerate(lines):
        new = line
        for pat, rep in RULES:
            if re.search(pat, new):
                new = re.sub(pat, rep, new)
        if new != line:
            changed.append((i + 1, line.rstrip(), new.rstrip()))
            lines[i] = new
    if not changed:
        print(f'  [unchanged] {os.path.basename(path)}')
        return
    if check_only:
        print(f'  [would patch] {os.path.basename(path)}')
        for ln, old, new in changed[:8]:
            print(f'    L{ln}: {old[:90]}')
            print(f'        -> {new[:90]}')
        return
    bak = path + '.orig'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, 'w') as f:
        f.writelines(lines)
    print(f'  [patched] {os.path.basename(path)}: {len(changed)} changes (.orig backup)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    if not os.path.isdir(TRAIN_DIR):
        print(f'Not found: {TRAIN_DIR}', file=sys.stderr)
        sys.exit(1)
    for name in FILES:
        path = os.path.join(TRAIN_DIR, name)
        if os.path.isfile(path):
            patch_file(path, args.check)
        else:
            print(f'  [missing] {name}')


if __name__ == '__main__':
    main()
