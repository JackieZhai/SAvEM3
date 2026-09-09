# Environments and postprocessing launcher

Training uses `requirements-train.txt`. Probe-EM uses a separate Python 3.10+/SAM 2
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
WaterZ run may compile C++ templates. `setup.sh` is a historical server helper;
inspect its platform assumptions before running it.
