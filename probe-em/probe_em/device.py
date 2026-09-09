"""Device selection helper for Probe-EM.

The upstream pipeline targets CUDA. On Apple Silicon we select MPS when
available so the SAM 2 image/video predictors can run locally.
"""

import os
from contextlib import nullcontext
from pathlib import Path

import torch


def resolve_device(explicit=None, gpu_id=None):
    """Return a torch device string suitable for SAM 2.

    Priority:
    1. explicit device (`cuda`, `cuda:0`, `mps`, `cpu`, ...)
    2. CUDA if available
    3. MPS if available
    4. CPU
    """
    if explicit and str(explicit).lower() != "auto":
        explicit = str(explicit)
        if explicit.startswith("mps"):
            from probe_em.mps_patch import patch_sam2_for_mps
            patch_sam2_for_mps()
        return explicit

    if torch.cuda.is_available():
        return f'cuda:{gpu_id}' if gpu_id is not None else 'cuda'

    if torch.backends.mps.is_available():
        from probe_em.mps_patch import patch_sam2_for_mps
        patch_sam2_for_mps()
        return "mps"

    return "cpu"


def autocast_dtype(device):
    """Use bfloat16 only on CUDA when supported; otherwise float16."""
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def inference_autocast(device):
    """CUDA AMP accepts a device TYPE; CPU/MPS inference stays in float32."""
    device = torch.device(device)
    if device.type != 'cuda':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=autocast_dtype(device))


def sam2_config_name(config_file):
    """Return a Hydra config name for SAM 2.

    The SAM 2 ``build_sam2`` entry point expects a config name such as
    ``sam2_hiera_l.yaml``, while user configs may provide an absolute path or a
    relative path like ``configs/sam2.1/sam2.1_hiera_l.yaml``.
    """
    import sam2
    import importlib.util
    root = Path(sam2.__file__).resolve().parent
    name = str(config_file).replace('\\', '/')
    # SAM 2 (2024) initializes Hydra from the sibling sam2_configs package.
    # SAM 2.1 uses configs/... inside sam2. Keep both resource layouts intact.
    legacy = importlib.util.find_spec('sam2_configs')
    if legacy is not None and legacy.origin:
        legacy_root = Path(legacy.origin).parent
        candidate = legacy_root / Path(name).name
        if candidate.is_file():
            return candidate.name
    if Path(name).is_absolute():
        try:
            name = Path(name).resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError('SAM 2 config must be a Hydra resource inside the installed sam2 package') from exc
    candidates = [name, 'configs/' + name, name.removeprefix('configs/'),
                  'configs/' + Path(name).name]
    for candidate in candidates:
        if (root / candidate).is_file():
            return candidate
    raise FileNotFoundError(f'SAM 2 config {config_file!r} not found in {root}; match SAM 2 / SAM 2.1 and checkpoint size')
