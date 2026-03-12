# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton fused RoPE kernel adapted for vLLM Stable Audio interface.

Operates on [B, S, H, D] tensor layout with pre-computed (cos, sin) tensors.
Fuses: fp32 cast + rotate_half + element-wise multiply + cast back
into a single kernel, eliminating intermediate allocations.

Input shape:  (B, S, H, D) — q or k tensor
cos/sin shape: (S, rotary_dim) — pre-computed rotation matrices
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_cossin_fwd_kernel(
    T, Cos, Sin, Out,
    S, H,
    D: tl.constexpr,
    rotary_dim: tl.constexpr,
    HALF: tl.constexpr,
    UNROT: tl.constexpr,
    STORE_FP16: tl.constexpr,
):
    """
    One program per (b, s, h) row.  Applies RoPE with pre-computed cos/sin.
    T layout: [B, S, H, D] contiguous — row stride = D, total rows = B*S*H.
    Cos/Sin layout: [S, rotary_dim] contiguous.
    """
    pid = tl.program_id(0)
    s_idx = (pid // H) % S

    T_row = T + pid * D
    Out_row = Out + pid * D
    cs_off = s_idx * rotary_dim

    c0 = tl.arange(0, HALF)
    c1 = HALF + tl.arange(0, HALF)

    t1 = tl.load(T_row + c0).to(tl.float32)
    t2 = tl.load(T_row + c1).to(tl.float32)

    cos1 = tl.load(Cos + cs_off + c0).to(tl.float32)
    sin1 = tl.load(Sin + cs_off + c0).to(tl.float32)
    cos2 = tl.load(Cos + cs_off + c1).to(tl.float32)
    sin2 = tl.load(Sin + cs_off + c1).to(tl.float32)

    # rotate_half semantics: [-x_imag, x_real]
    out1 = t1 * cos1 - t2 * sin1
    out2 = t2 * cos2 + t1 * sin2

    if STORE_FP16:
        tl.store(Out_row + c0, out1.to(tl.float16))
        tl.store(Out_row + c1, out2.to(tl.float16))
    else:
        tl.store(Out_row + c0, out1)
        tl.store(Out_row + c1, out2)

    if UNROT > 0:
        c_unrot = rotary_dim + tl.arange(0, UNROT)
        x_unrot = tl.load(T_row + c_unrot)
        tl.store(Out_row + c_unrot, x_unrot)


def apply_rope_fused(
    hidden_states: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Fused RoPE for [B, S, H, D] layout with pre-computed (cos, sin).

    hidden_states: [B, S, H, D] contiguous, fp16 or bf16
    freqs_cis:     (cos[S, rotary_dim], sin[S, rotary_dim])
    """
    assert hidden_states.ndim == 4
    cos, sin = freqs_cis
    assert cos.ndim == 2 and sin.ndim == 2

    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    B, S, H, D = hidden_states.shape
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    unrot = D - rotary_dim

    assert rotary_dim % 2 == 0

    out = torch.empty_like(hidden_states)
    store_fp16 = hidden_states.dtype == torch.float16

    grid = (B * S * H,)
    _rope_cossin_fwd_kernel[grid](
        hidden_states, cos, sin, out,
        S=S, H=H, D=D,
        rotary_dim=rotary_dim, HALF=half, UNROT=unrot,
        STORE_FP16=store_fp16,
        num_warps=2, num_stages=2,
    )
    return out
