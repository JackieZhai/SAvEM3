# Environments and postprocessing launcher

Training uses Python 3.10+ (3.12 recommended) and `requirements-train.txt`.
Probe-EM uses a separate Python 3.10+/SAM 2
environment. ELF/nifty/vigra/WaterZ use `environment-post.yml` and `requirements-post.txt`.

~~~bash
./repro/env/setup_postprocess.sh
./repro/env/run_postprocess.sh repro/savem3/distill_postprocess.py --help
~~~

`run_postprocess.sh` first looks for `.conda-envs/savem3-post` inside the repository,
then in its parent workspace. Set `SAVEM3_POST_ENV=/absolute/env` to select an
environment. macOS SDK/C++ flags are added only on Darwin, not Linux; existing
`CXXFLAGS` and `LDFLAGS` are retained. WaterZ JIT artifacts go under repository `.cache/`.

The validation setup used three separate environments:

- Training: PyTorch, HQ-SAM, SAEM², SAvEM³.
- Postprocessing: ELF, nifty, WaterZ.
- Probe-EM: SAM 2, CloudVolume, Neuroglancer, Kimimaro.

Local environment paths are not portable dependency lockfiles. Record package
versions, CUDA, GPU, and configuration for quantitative experiments. The first
WaterZ run may compile C++ templates. `setup.sh` creates a Python 3.12 training
environment without activating it in the calling shell. It defaults to the
official CUDA 12.6 wheel index; set
`SAVEM3_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` for a Linux CPU setup.
Inspect server/driver compatibility before running it. Use a fresh environment
name for upgrades; ELF/WaterZ remain separate from the Torch upgrade.
