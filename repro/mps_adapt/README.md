# MPS compatibility for historical trainers

`run_mps.sh`, `sitecustomize.py`, and `patch_mps.py` adapt historical `.cuda()` calls
and provide CPU fallback for unsupported operators. Use them only for old scripts
that do not yet manage devices explicitly.

~~~bash
cd savem3
../repro/mps_adapt/run_mps.sh main_devoem_sparse_membrane_triplet_2.py \
  -c mem3c2c_3ds_t3t --fresh --no-valid --num-workers 0 -m train
~~~

The standalone `repro/savem3/distill.py` uses `--device auto/cpu/mps/cuda` directly
and does not need this global monkeypatch. If a PyTorch operator is unavailable on
MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` before launching. CPU fallback throughput
does not represent the paper's V100 training speed.

Probe-EM's SAM 2 MPS support resides in `probe-em/probe_em/device.py` and
`mps_patch.py`; do not additionally patch its PyTorch behavior with this historical wrapper.
