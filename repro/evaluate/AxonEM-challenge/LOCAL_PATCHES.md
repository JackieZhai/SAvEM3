# 本地适配说明

上游：https://github.com/PytorchConnectomics/AxonEM-challenge（depth 1 clone）。

本目录用于 `../erl.py` 的官方 ERL 实现。相对于上游做了最小修补，以便 Python 3.9 / 本机运行：

1. `erl_wrapper/eval_erl.py`：上游 `compute_segment_lut_tile_combine` 函数定义缺少冒号，已补。
2. `erl_wrapper/data_io.py`：上游缺少 `write_vol`，已按 h5/tif 两种格式补实现。
3. 当前官方仓库 master 的 `test_volume.py` 调用了不存在的
   `compute_node_segment_lut`，因此本复现没有使用该入口，而是按
   `test_axonEM.py::test_AxonEM` 的官方流程重新封装到 `../erl.py`。

运行环境：`SAvEM3/repro/env/run_postprocess.sh ../evaluate/erl.py ...`

补充（本次 repo 整理）：
4. `erl_wrapper/test_j1026.py`：修复上游若干 Python 语法错误（缺失冒号、引号），
   仅保证源码可编译；官方 ERL 入口请使用 `../erl.py`，不要使用该脚本直接评测。
