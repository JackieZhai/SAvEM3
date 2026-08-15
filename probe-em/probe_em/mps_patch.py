"""Optional MPS compatibility patch for SAM 2.

The upstream SAM 2 rotary-position function uses ``Tensor.repeat`` on complex
tensors, which is not implemented for MPS. This module replaces the rotary
function with an equivalent real-arithmetic implementation when running on MPS.
"""

import torch


def _apply_rotary_enc_real(xq, xk, freqs_cis, repeat_freqs_k=False):
    """Real-valued equivalent of sam2.modeling.position_encoding.apply_rotary_enc."""
    from sam2.modeling.position_encoding import reshape_for_broadcast

    xq_pair = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_pair = xk.float().reshape(*xk.shape[:-1], -1, 2) if xk.shape[-2] != 0 else None

    xq_complex = torch.view_as_complex(xq_pair)
    xk_complex = torch.view_as_complex(xk_pair) if xk_pair is not None else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)
    cos, sin = freqs_cis.real, freqs_cis.imag

    xq = xq_pair
    xk = xk_pair

    xq_real = xq[..., 0]
    xq_imag = xq[..., 1]
    xq_out = torch.stack(
        [xq_real * cos - xq_imag * sin, xq_real * sin + xq_imag * cos], dim=-1
    ).flatten(3)

    if xk is None:
        return xq_out.type_as(xq).to(xq.device), xk

    if repeat_freqs_k:
        r = xk_complex.shape[-2] // xq_complex.shape[-2]
        cos = cos.repeat(*([1] * (cos.ndim - 2)), r, 1)
        sin = sin.repeat(*([1] * (sin.ndim - 2)), r, 1)

    xk_real = xk[..., 0]
    xk_imag = xk[..., 1]
    xk_out = torch.stack(
        [xk_real * cos - xk_imag * sin, xk_real * sin + xk_imag * cos], dim=-1
    ).flatten(3)

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def patch_sam2_for_mps():
    """Replace SAM 2's rotary function with the MPS-compatible implementation."""
    try:
        import sam2.modeling.sam.transformer as transformer
        transformer.apply_rotary_enc = _apply_rotary_enc_real
        return True
    except Exception:
        return False
