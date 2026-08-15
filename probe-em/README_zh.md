# Probe-EM

[English](README.md) | [中文](README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2607.04696-b31b1b.svg)](https://arxiv.org/abs/2607.04696)

这是 MICCAI 2026 论文的官方代码仓库：

**Probe-EM: Targeted Neuron Tracing via Training-Free Semantic Verification**

Probe-EM 是一个面向大规模电子显微镜体数据的目标神经元追踪框架。它从 seed segment ID 出发，基于几何线索搜索可能相连的候选 segment，并使用基于 SAM 2 的语义验证模块判断候选连接是否成立。

论文：[arXiv:2607.04696](https://arxiv.org/abs/2607.04696)

## 仓库结构

```text
Probe-EM/
  configs/              示例配置文件
  probe_em/             核心追踪和语义验证模块
  scripts/              追踪、评估、人工校验入口脚本
  finetune_sam2/        NeuroSAM 2 的 SAM 2 adaptation 代码
  INSTALL.md            环境安装说明
  requirements.txt      Python 依赖
```

## 安装

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

Probe-EM 依赖官方 SAM 2 实现。运行追踪流程前，需要先安装 SAM 2，并准备对应的模型配置文件和 checkpoint。

我们实验使用的环境安装流程见 [INSTALL.md](INSTALL.md)。

## 配置文件

复制示例配置文件，然后修改本地路径：

```bash
cp configs/config.example.json configs/config.json
```

关键字段说明：

- `raw_path`: 原始 EM 图像体数据路径，或者 CloudVolume URL。
- `seg_path`: 分割结果体数据路径，或者 CloudVolume URL。
- `checkpoint_sam`: SAM 2 checkpoint 路径。
- `model_cfg_sam`: SAM 2 模型配置文件路径。
- `seed_ids`: 要追踪的初始 segment ID 列表。
- `seed_list_file`: 可选 txt 文件，每行一个 seed ID。
- `output_root`: 追踪结果输出目录。
- `max_workers`: 并行追踪的 seed ID 数量。
- `slice_workers`: 每个 seed 内部切片提取使用的 worker 数量。
- `debug_limit`: 每个 seed 最多处理多少个图节点，调试时可用。
- `sam2_num_frames`: 3D propagation 使用的轴向帧数。

路径可以写成本地路径，例如 `/path/to/...`，也可以写成 `file:///path/to/...`。

## 运行追踪

```bash
python scripts/run_probe_em.py --config configs/config.json
```

每个 seed 的结果会写入：

```text
<output_root>/<seed_id>_results_<suffix>/
```

## SAM 2 微调

用于将 SAM 2 适配到神经元分割 prompt 的训练脚本位于 `finetune_sam2/`。

EMNeuron 数据集引用、2D image/mask 数据格式和训练命令见 [finetune_sam2/README.md](finetune_sam2/README.md)。

主追踪流程使用微调后的 checkpoint 做语义验证；追踪阶段不会继续更新模型参数。

## 结果评估

`scripts/evaluate_tracing.py` 用于将追踪得到的 segment ID 集合和人工标注的 GT segment 集合进行比较。

GT 文件夹中应包含类似下面命名的文件：

```text
<seed_id>_gt_segments.txt
```

运行示例：

```bash
python scripts/evaluate_tracing.py \
  --seg-path /path/to/segmentation/precomputed \
  --gt-folder /path/to/gt_labels \
  --results-dir trace_results \
  --suffix sam \
  --output-dir quantification_results \
  --excel-name evaluation_results.xlsx
```

默认评估时会排除 seed segment 本身。如果想把 seed segment 也计入指标，可以加：

```bash
--include-seed
```

## Neuroglancer 人工校验

`scripts/verify_traces_neuroglancer.py` 会打开一个 Neuroglancer viewer，用于离线查看 Probe-EM 的追踪结果，并进行人工修正和保存。

```bash
cp configs/verification_config.example.json configs/verification_config.json
python scripts/verify_traces_neuroglancer.py --config configs/verification_config.json
```

这个工具会在 `results_folder` 下查找追踪结果目录，目录名形如 `<seed_id>_results...`。

人工保存的标注会写入 `gt_folder`，文件名形如 `<seed_id>_gt_segments.txt`。

Neuroglancer viewer 中常用快捷键：

- `q`: 加载鼠标当前指向 segment 的追踪结果。
- `a`: 高亮所有已经生成追踪结果的 seed ID。
- `t`: 加载当前选中 ID 已保存的 GT 标注。
- `w`: 显示所有已经标注过的 GT segment ID。
- `g`: 将当前显示的 segment 集合保存为 GT。
- `b`: 跳转到配置中的 `home_position`，如果提供了的话。
- `c`: 清空当前显示的 segment 和日志标记。

## 引用

如果这个仓库对你有帮助，请引用我们的论文：

```bibtex
@article{jiang2026probeem,
  title={Probe-EM: Targeted Neuron Tracing via Training-Free Semantic Verification},
  author={Jiang, Liuyun and Zhang, Yanchao and Guo, Jinyue and Chen, Chuanyue and Yan, Haiyang and Yuan, Ye and Liu, Jing and Han, Hua},
  journal={arXiv preprint arXiv:2607.04696},
  year={2026}
}
```

MICCAI 2026 proceedings 正式版本发布后，会再补充官方引用格式。
