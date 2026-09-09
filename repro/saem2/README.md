# SAEM² data preparation and launcher

The entry points are `link_pretrained.sh`, `make_record2d.py`,
`saem2/json2d_create.py`, `repro/savem3/precompute_teacher.py --write-tif`,
and `run_train.sh`. See [saem2/README.md](../../saem2/README.md) for training arguments.

Despite its historical name, `make_record2d.py` exports membrane TIFFs, not record
JSON. `json2d_create.py` derives connected instances and records from membranes.
Verify matching physical fields of view and section indices across membranes,
instances, and embeddings. The `snemi` cache directory maps to `SNEMI`.

~~~bash
export SAVEM3_DATA_ROOT=/absolute/path/to/data-bank
python repro/saem2/make_record2d.py --datasets snemi --xy-nm 4
python repro/savem3/precompute_teacher.py --datasets snemi --xy-nm 4 \
  --write-tif --checkpoint /absolute/path/to/sam_hq_vit_h.pth --device auto
DEVICE=cuda ./repro/saem2/run_train.sh --record-train /absolute/train.json \
  --record-valid /absolute/held_out.json
~~~

The launcher uses batch 8, 12 epochs, and learning rate 1e-3, and forwards additional
CLI arguments. Supply an explicit train/validation split: the default reuse of
training records is not independent validation. `patch_paths.py` is a historical
migration helper, not a script to repeatedly run during normal training.
