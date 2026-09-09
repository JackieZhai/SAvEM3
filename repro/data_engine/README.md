# Data engine

Dataset locations come from `sam-hq/train/utils/location.py`; `SAVEM3_DATA_ROOT`
selects the physical data root. The configuration is loaded under a unique module
name to avoid collisions with training scripts' `utils` packages. CloudVolume
reads are strict: missing chunks raise errors instead of becoming zero-valued background.

| Script | Purpose | Decisions still required |
| --- | --- | --- |
| `resample_to_8nm.py` | Resample XY while retaining Z resolution | Target resolution and ROI |
| `realign_cremi.py` | CREMI registration helper | Misalignment/defect identification and validation |
| `mask_myelin_glia.py` | Apply myelin/glia/defect exclusions | Dataset blacklists are not complete manual annotations |
| `sam_refine.py` | Eroded mask prompts, SAM refinement, manual-review list | Review low-IoU masks |
| `unify_iter.py` | Replace boundaries above IoU 0.7 | Retraining and acceptance criteria for the two-round loop |
| `select_training_set.py` | Group by imaging method and produce sampling lists | Same-superset exclusion and verification of the actual 188K samples |

~~~bash
export SAVEM3_DATA_ROOT=/absolute/path/to/data-bank
python repro/data_engine/resample_to_8nm.py --datasets snemi,ac3 --target 8
python repro/data_engine/sam_refine.py --datasets snemi \
  --checkpoint /absolute/path/to/sam_hq_vit_h.pth --device auto
python repro/data_engine/unify_iter.py --datasets snemi --seg-layer seg \
  --pred-layer seg_refine --out-layer seg_unified --round 1
~~~

Refinement explicitly converts CloudVolume XY planes to image YX. Overlaps are
assigned by prediction confidence; ties preserve the earlier assignment. Unification
removes the old boundary of replaced labels and protects labels that fail the threshold.
Masks in a discrete label map do not overlap. NMS on overlapping mask proposals
belongs in the model-output stage; label-map filtering is not a full SAM AMG NMS implementation.

Historical `precompute_teacher.py` and `make_record2d.py` must use the same `--xy-nm`
and center-cropped 1024² field of view. The default 4 nm follows the historical cache
convention. For an 8 nm HQ/SAEM experiment, explicitly pass 8 to both tools and
provide a sufficiently large physical ROI; small ROIs are not silently stretched.
`seg_refine` and `seg_unified` map to sibling layers suffixed `_refine` and `_unified`
relative to the original segmentation path.
