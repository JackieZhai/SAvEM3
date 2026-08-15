"""M4 准备 · 由基础配置生成 6 档消融配置（论文 Table 3 下半部分阶梯）。

基础配置：savem3/config/mem3c2c_3ds_t3t.yaml（= 完整 L_bdy+L_feat+L_embed+L_2nd）
消融阶梯（每档是上一档的子集）：
    1  bdy_only        : 只 L_bdy（关 loss_lambda_func / motion / smooth）
    2  bdy_dst         : + L_dst（DST 辅助输出；论文后续弃用，见 5）
    3  bdy_embed       : L_bdy + L_embed（关 feat/motion/smooth）
    4  bdy_embed_feat  : L_bdy + L_embed + L_feat
    5  bdy_embed_feat_2nd : 4 + motion/smooth（= 基础配置）
    6  +graph-cut      : 5 的产物在评测端启用提示图割（见 graph_cut/，无训练差异）

实现：把基础 yaml 作为模板，按档位开关字段；resume 一律置 False（新训练）。
用法：
    python gen_ablation_cfg.py --out-dir /path/to/savem3/config/ablation
"""
import argparse
import os
import shutil

import yaml

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                    'savem3', 'config', 'mem3c2c_3ds_t3t.yaml')

# 每档：需要置 None 的字段（其余沿用基础配置）
ABLATIONS = {
    'ab1_bdy_only': {
        'loss_lambda_func': None,      # 关 L_feat/L_embed 蒸馏
        'loss_motion_func': None,
        'loss_smooth_func': None,
    },
    'ab2_bdy_dst': {
        # DST 分支：基础模型 output_nc=1（仅 BDY）；DST 版需改模型/输出，见注释
        'loss_lambda_func': None,
        'loss_motion_func': None,
        'loss_smooth_func': None,
    },
    'ab3_bdy_embed': {
        'loss_lambda_embed': 1.0,      # 开 embed、关 feat
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
        'loss_motion_feat': 1.0,
        'loss_smooth_feat': 0.1,
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

    # 基础配置即第 5 档（含 motion/smooth），确保字段与论文一致
    base['TRAIN']['resume'] = False
    base['TRAIN']['load_path'] = None

    for name, overrides in ABLATIONS.items():
        cfg = yaml.safe_load(yaml.safe_dump(base))  # 深拷贝
        if name == 'ab2_bdy_dst':
            # DST 版：output_nc=2（BDY+DST），loss 需在 loss.py 扩展；此处仅生成占位配置并注释
            cfg['MODEL']['output_nc'] = 2
        cfg['NAME'] = name
        for k, v in overrides.items():
            cfg['TRAIN'][k] = v
        path = os.path.join(out_dir, f'{name}.yaml')
        with open(path, 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f'-> {path}')

    print('\n注：ab2_bdy_dst 需要先扩展模型输出头与损失（论文中 DST 收益微小，可跳过 2 直接做 1→3→4→5→6）。')


if __name__ == '__main__':
    main()
