# Local compatibility patches

Upstream: https://github.com/PytorchConnectomics/AxonEM-challenge (depth-1 clone).

This directory provides the official ERL implementation used by `../erl.py`.
Minimal changes were made for Python 3.9 and local execution:

1. `erl_wrapper/eval_erl.py`: added the missing colon in the upstream
   `compute_segment_lut_tile_combine` function definition.
2. `erl_wrapper/data_io.py`: implemented the missing `write_vol` function for H5/TIFF.
3. In the upstream revision used here, `test_volume.py` calls the nonexistent
   `compute_node_segment_lut`. This reproduction therefore wraps the official
   `test_axonEM.py::test_AxonEM` workflow in `../erl.py` instead.
4. `erl_wrapper/test_j1026.py`: fixed upstream Python syntax errors, including
   missing colons and quotes. Only source compilation was checked; use `../erl.py`
   for official ERL evaluation, not this historical example script.

From the repository root, use `./repro/env/run_postprocess.sh repro/evaluate/erl.py ...`.
