# Evaluation and sparse ROI inference

## Dense instance metrics

~~~bash
./repro/env/run_postprocess.sh repro/evaluate/dense_eval.py \
  --gt gt.h5 --preds post/seg_lmc.hdf post/seg_waterz_t0.5.hdf --names MC WaterZ
~~~

The H5 dataset is `main`. GT and prediction must have equal shapes and nonnegative
integer labels; GT label 0 is ignored. Outputs are VoI split, merge, total, and
adapted Rand error; lower is better. `--preds` and `--names` accept spaces or commas.
Paper reference values printed afterward are comparison targets, not computed results.

## Sparse ROIs

`sparse_eval.py` runs tiled inference with the original student and stitches local coordinates.

~~~json
{"chunks": [{"offset": [0, 0, 0], "size": [64, 512, 512]},
            {"offset": [48, 0, 0], "size": [64, 512, 512]}]}
~~~

~~~bash
python repro/evaluate/sparse_eval.py --roi chunks.json --raw raw_student.h5 \
  --ckpt outputs/student/last.pt --out-dir outputs/sparse
python repro/evaluate/sparse_eval.py --roi chunks.json --stitch-only --out-dir outputs/sparse
~~~

Offsets and sizes are ZYX indices into the supplied raw array. Outputs include
`chunk_NNNN.h5`, `boundary.h5`, `coverage.h5`, and `report.json`. The stitched
origin is the component-wise minimum chunk offset. Uncovered voxels have
boundary probability 1 and coverage 0; do not evaluate them as neuron interiors.

The current implementation reads local raw ROIs and stitches in memory, with a
default 128M-voxel limit. It is not a remote petabyte executor. Oracle ROI generation,
global instance stitching, and target-neuron extraction still require experiment-specific
data and strategies.

## ERL

`erl.py` uses the official AxonEM-challenge wrapper and trusted
`gt_*_skel_stats.p` files. The full upstream project is not bundled. Set
`SAVEM3_AXONEM_ROOT` to an installed `erl_wrapper` directory in a standalone clone.
A historical sibling-workspace checkout is also supported as a fallback.

~~~bash
export SAVEM3_AXONEM_ROOT=/absolute/path/to/AxonEM-challenge/erl_wrapper
./repro/env/run_postprocess.sh repro/evaluate/erl.py \
  --seg instances.h5 --gt-stats gt_skel_stats.p
~~~

GT resolution is ZYX in nm. Converted skeleton voxel coordinates remain int64
to avoid wrapping coordinates above 65535. `erl_approx.py` is a historical
sanity-check approximation and must not be used for paper ERL reports.
