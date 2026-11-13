# Evaluation against local neuron labels

`scripts/evaluate_local_gt.py` selects bounded windows from local precomputed
raw and neuron labels, runs SAM2 propagation, and scores the resulting masks.
Use the [Fish-X LC-NE Test release](../docs/LC_NE_TEST.md) for the supplied test data.

## Run an evaluation

```bash
python probe-em/scripts/evaluate_local_gt.py \
  --raw-path /data/Fish-X-LC-NE-Test/zfishlc_raw \
  --gt-path /data/Fish-X-LC-NE-Test/zfishlc_lc \
  --raw-mip 0 --gt-mip 0 --locator-mip 5 \
  --xy 512 --slices 63 --labels 30001 30002 30003 30004 30005 \
  --checkpoint /models/sam2_hiera_tiny.pt \
  --model-config sam2_hiera_t.yaml --device auto \
  --methods unregistered ecc --output-dir /data/evaluation
```

Use a new output directory. Add `sofima` to `--methods` and specify
`--sofima-python` to include the [SOFIMA worker](README_sofima.md).
Here, `unregistered` means no additional local field; the Fish-X LC-NE Test input already
has whole-volume alignment.

The locator chooses a central occupied section and an interior point for each
label. This command selects evaluation windows independently of the five
supplied demo process windows. Thin structures may require a finer locator mip.
For interactive review of the supplied windows, use [demo_windows.py](README_windows.md).

## Coordinates and scoring

Raw and GT must share a physical origin. The selected raw grid must be an integer
coarsening of the GT grid. Instance labels are sampled with nearest neighbors.
Each method receives the same central positive point, starts fresh SAM2 memory,
and maps masks back to the native raw grid before scoring.

Outputs include `cache/raw`, `cache/seg`, `cache/target`, registration metadata,
per-method masks and metrics, and an aggregate `report.json`. `status.json`
records whether processing completed. Prompted and unprompted sections are scored
separately. GT assists region and point selection, so these scores describe
point-guided segmentation.

Fish-X LC-NE Test labels cover selected neurons. Target Dice/IoU assumes that target's mask
is complete within the evaluated window. Overlap with another annotated neuron
can identify a merge; absence of overlap cannot rule out a merge into an
unannotated structure. Inspect truncated processes and use geometric checks
alongside segmentation scores.
