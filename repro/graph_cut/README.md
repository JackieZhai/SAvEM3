# Prompt-guided graph cut

This module implements the EM-fragment workflow associated with SAvEM³ Section 2.3.
DST point selection, predicted IoU, and mask selection reference
[SAM-Graph](https://github.com/zju3dv/SAM-Graph); the inputs here are oversegmented
EM sections, not a multi-view scene point cloud.

~~~bash
python repro/graph_cut/prompt_graph_cut.py \
  --fragments outputs/post/fragments.h5 --feats outputs/teacher.h5 --feats-key features \
  --img outputs/teacher.h5 --img-key raw --checkpoint sam-hq/pretrained_checkpoint/sam_hq_vit_h.pth \
  --device auto --out-dir outputs/graph
./repro/env/run_postprocess.sh repro/graph_cut/lmc_prompt.py \
  --fragments outputs/post/fragments.h5 --affs outputs/post/affinities.h5 \
  --edge-feats outputs/graph/edge_feats.json --out-dir outputs/prompt
./repro/env/run_postprocess.sh repro/graph_cut/waterz_iou.py \
  --fragments outputs/post/fragments.h5 --affs outputs/post/affinities.h5 \
  --edge-feats outputs/graph/edge_feats.json --out-dir outputs/prompt
~~~

## Interfaces and algorithm

All steps must use the same globally identified `ZYX` fragment volume.
Features have shape `32ZYX` and may have lower XY resolution but must cover the
same ROI. Node means accumulate section by section and channel by channel,
avoiding a whole-volume scan for every instance. Features are interpolated to the
label grid; integer labels are never converted to floating point. The example
reads one teacher H5 directly; standalone H5 files default to dataset `main`.

Mask prompts are transformed through image resize/padding into `1×256×256`
floating-point logits. Prediction uses the native `(masks, scores, logits)` result.
`--prompt-mode points` selects DST points; default `mask` erodes by one pixel
and falls back to the original mask if erosion removes a thin target.

For each target section, seeds may originate in that section or an adjacent one.
Both predictions for an edge are compared on the same target section and weighted
by the product of prediction confidences. Image encoding occurs once per section;
`n_sections` records the actual number of evidence sections.

`edge_feats.json` uses `"u,v"` keys and contains cosine plus optional IoU/IoA/IoB.
Dataset names in `node_feats.h5` are original fragment IDs. Without model weights,
explicitly select `--no-prompt-iou` for a cosine-only ablation; missing weights are
not silently treated as completed prompt inference.

## Aggregation semantics

Positive ELF costs penalize cutting an edge; high similarity therefore increases
the cost. No extra lifted edges are currently added, so this local multicut is
not a complete lifted-multicut experiment. `--w-cos` and `--w-iou` default to 1;
IoA/IoB weights default to 0. These are tunable implementation parameters.

WaterZ boosts only the directional affinity at actual u/v interfaces, not every
channel throughout dilated regions. This is a voxel-interface approximation of
edge-score modification, using the 50% histogram quantile. Select thresholds on
a proper validation set, not the test GT.
