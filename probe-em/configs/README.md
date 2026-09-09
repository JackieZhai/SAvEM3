# Probe-EM configuration

Copy `config.example.json` to your own configuration and edit paths there.
Machine-specific configurations are local-only and are not rewritten by the demo.

| Field | Contract |
| --- | --- |
| `raw_path`, `seg_path` | CloudVolume URI or local precomputed path; grids must match |
| `checkpoint_sam` | SAM 2 / NeuroSAM 2 checkpoint matching the model configuration |
| `model_cfg_sam` | Installed Hydra resource, e.g. `sam2_hiera_t.yaml` or `configs/sam2.1/sam2.1_hiera_l.yaml` |
| `seed_ids` | Positive integer instance IDs; duplicates are removed and 0 is invalid |
| `seed_list_file` | Optional one-ID-per-line file; a specified missing file raises an error |
| `target_mip` | Existing mip used to convert skeleton coordinates; newly exported ROIs use 0 |
| `device` | `auto` / `cpu` / `mps` / `cuda:0` |
| `max_workers` | Seed-level processes; use 1 for a single GPU/MPS demo to avoid duplicate models |
| `slice_workers` | Slice I/O/cropping threads; 1–4 is suitable for local demos |
| `debug_limit` | Maximum nodes processed per invocation; 0/null means unlimited; a limit yields `limited` |
| `random_seed` | Default 42; repeatable per-segment prompt sampling |
| `output_root`, `suffix` | Output location and experiment suffix; use a new directory for different data/models |
| `resume` | Default false; also available as CLI `--resume` |
| `align_z` | Tracing default false; the historical automatic estimator assumes zebrafish resolution |
| `align_z_field` | Explicit cached registration NPZ for tracing; use a new result directory when changing it |

~~~bash
python scripts/run_probe_em.py --config configs/my_run.json
python scripts/run_probe_em.py --config configs/my_run.json --resume
~~~

Relative paths resolve from the launch working directory. Use absolute paths after
copying a configuration to a specific machine to avoid ambiguity. The generated
`trace_config.json` from `export_savem3.py` contains absolute paths.

`verification_config.example.json` belongs to the historical interactive verifier
and uses different fields. Demo display registration is independently enabled by
default; it does not change tracing's `align_z` setting.
