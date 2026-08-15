"""Device selection helper for Probe-EM.

The upstream pipeline targets CUDA. On Apple Silicon we select MPS when
available so the SAM 2 image/video predictors can run locally.
"""

import os

import torch


def resolve_device(explicit=None, gpu_id=None):
    """Return a torch device string suitable for SAM 2.

    Priority:
    1. explicit device (`cuda`, `cuda:0`, `mps`, `cpu`, ...)
    2. CUDA if available
    3. MPS if available
    4. CPU
    """
    if explicit:
        explicit = str(explicit)
        if explicit.startswith("mps"):
            from probe_em.mps_patch import patch_sam2_for_mps
            patch_sam2_for_mps()
        return explicit

    if torch.cuda.is_available():
        if gpu_id:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
        return "cuda"

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


def sam2_config_name(config_file):
    """Return a Hydra config name for SAM 2.

    The SAM 2 ``build_sam2`` entry point expects a config name such as
    ``sam2_hiera_l.yaml``, while user configs may provide an absolute path or a
    relative path like ``configs/sam2.1/sam2.1_hiera_l.yaml``.
    """
    return os.path.basename(str(config_file))
