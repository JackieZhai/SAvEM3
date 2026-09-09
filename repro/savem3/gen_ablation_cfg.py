"""Generate supported historical ablations with explicit active loss weights.

ab1: boundary; ab3: boundary + embedding; ab4: boundary + embedding + HQ feature;
ab5: ab4 + motion/smooth (requires AC4_flows.h5). The base YAML has NO secondary
losses. The old BDY+DST placeholder is deliberately not generated.
For label-free teacher bundles use distill.py instead of the legacy loader.
"""
import argparse
import os

import yaml

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                    'savem3', 'config', 'mem3c2c_3ds_t3t.yaml')

# Per-ablation overrides; other fields inherit the base configuration.
ABLATIONS = {
    'ab1_bdy_only': {
        'loss_lambda_func': None,      # Disable L_feat/L_embed distillation
        'loss_motion_func': None,
        'loss_smooth_func': None,
    },
    'ab2_bdy_dst': {
        # Unsupported DST placeholder: the base model has output_nc=1 (boundary only).
        'loss_lambda_func': None,
        'loss_motion_func': None,
        'loss_smooth_func': None,
    },
    'ab3_bdy_embed': {
        'loss_lambda_embed': 1.0,      # Enable embedding and disable feature loss
        'loss_lambda_feat': 0.0,
        'loss_motion_func': None,
        'loss_smooth_func': None,
    },
    'ab4_bdy_embed_feat': {
        'loss_lambda_feat': 1.0,
        'loss_lambda_embed': 1.0,
        'loss_motion_func': None,
        'loss_smooth_func': None,
    },
    'ab5_bdy_embed_feat_2nd': {
        'loss_lambda_feat': 1.0,
        'loss_lambda_embed': 1.0,
        'loss_motion_func': 'MotionLoss2',
        'loss_smooth_func': 'WeightedSmoothLoss',
        'loss_motion_iter': 20000,
        'loss_smooth_iter': 20000,
        'loss_motion_feat': 0.05,
        'loss_motion_embed': 0.0,
        'loss_smooth_feat': 0.05,
        'loss_smooth_embed': 0.0,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=str, default=None)
    ap.add_argument('--base', type=str, default=BASE)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.base), 'ablation')
    os.makedirs(out_dir, exist_ok=True)

    with open(args.base) as f:
        base = yaml.safe_load(f)

    # The base uses three primary targets; ablations explicitly override secondary losses.
    base['TRAIN']['resume'] = False
    base['TRAIN']['load_path'] = None

    for name, overrides in ABLATIONS.items():
        if name == 'ab2_bdy_dst':
            print('Skipping unsupported BDY+DST: the historical two-channel placeholder is not an implemented two-head experiment.')
            continue
        cfg = yaml.safe_load(yaml.safe_dump(base))  # Deep copy
        cfg['NAME'] = name
        if name == 'ab5_bdy_embed_feat_2nd':
            cfg['DATA']['dataset_name'] = 'ac4-triplet-flow'
        for k, v in overrides.items():
            cfg['TRAIN'][k] = v
        path = os.path.join(out_dir, f'{name}.yaml')
        with open(path, 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f'-> {path}')

    print('\nFull secondary losses require AC4_flows.h5. For label-free aligned targets use repro/savem3/distill.py.')


if __name__ == '__main__':
    main()
