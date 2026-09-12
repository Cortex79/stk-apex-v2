"""
STKApex core v2 — 2026-os kutatások alapján.

Változások v1-hez képest:
  RMSNorm         LayerNorm helyett (gyorsabb, stabilabb)
  SwiGLU          SiLU Sequential helyett (F.silu(gate) * up)
  GQA             MHA helyett (n_kv_heads=2, 4× kisebb KV-cache)
  KV-cache        generáláshoz (RoPEAttnBlock.forward visszaad past_kv-t)
  RoPE base       10k → 500k (hosszú kontextus)
  Fine-grained    16 routed expert, top-3, d_ff=384
  Shared expert   mindig aktív expert d_ff=384 (DeepSeek-V4)
  HyperResidual   tanult alpha/beta reziduális (DeepSeek-V4 Hyper-Connections)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_ckpt

from .config import STKApexConfig


def _ckpt(fn, *args):
    return _torch_ckpt(fn, *args, use_reentrant=False)


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm (PyTorch 2.4+ native, fallback ha régebbi)
# ─────────────────────────────────────────────────────────────────────────────

try:
    RMSNorm = nn.RMSNorm
except AttributeError:
    class RMSNorm(nn.Module):
        def __init__(self, d: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = eps
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU — kapuzott FFN (GPT-4 / LLaMA / Mistral)
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """F.silu(gate(x)) * up(x) → down → [B, L, D]."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up   = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight,   std=0.02)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ─────────────────────────────────────────────────────────────────────────────
# HyperResidual — tanult reziduális kapcsolat (DeepSeek-V4)
# ─────────────────────────────────────────────────────────────────────────────

class HyperResidual(nn.Module):
    """
    x_out = alpha * x + beta * f(x)

    Ahelyett hogy x_out = x + f(x) — a súlyok inicializálása identity-re
    (alpha=1, beta=1), de a gradiens tanulja a helyes mértéket.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta * fx


# ─────────────────────────────────────────────────────────────────────────────
# RoPE (base=500k a hosszú kontextushoz)
# ─────────────────────────────────────────────────────────────────────────────

def _rope_angles(positions: torch.Tensor, d_head: int,
                 base: float = 500_000.0):
    half = d_head // 2
    inv  = base ** (-torch.arange(half, dtype=torch.float32,
                                  device=positions.device) / half)
    ang  = positions.float().unsqueeze(-1) * inv
    return torch.cos(ang), torch.sin(ang)


def _apply_rope(x: torch.Tensor, cos, sin) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SGS — Szelektív Gátolt Állapot (változatlan, Multi-Scale chunk)
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveGatedState(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0,
                 chunk_size: int = 16):
        super().__init__()
        self.chunk_size = chunk_size
        self.W_a = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)
        nn.init.constant_(self.W_a.bias, 2.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.W_a(x)).clamp(1e-2, 1 - 1e-3)
        v = self.W_v(x)
        B, L, D = x.shape
        C   = self.chunk_size
        pad = (-L) % C
        af  = a.float()
        wf  = ((1 - a) * v).float()
        if pad:
            af = F.pad(af, (0, 0, 0, pad), value=1.0)
            wf = F.pad(wf, (0, 0, 0, pad))
        N     = af.shape[1] // C
        Ac    = torch.log(af).view(B, N, C, D).cumsum(dim=2)
        decay = torch.exp(Ac)
        # exp(-Ac) önmagában float32-ben túlcsordul: Ac a chunk végén C*log(a_min),
        # ami C=32, a=0.01 mellett -147 -> exp(147)=inf -> 0*inf = NaN. A chunk
        # közepére normalizálva az exponens |arg| <= C/2*|log a_min| = 73.7 marad.
        # Az M additív konstans kiesik: exp(Ac_t-M) * sum(w*exp(M-Ac_s)) azonos.
        M     = Ac[:, :, C // 2 : C // 2 + 1]
        Acn   = Ac - M
        inner = (wf.view(B, N, C, D) * torch.exp(-Acn)).cumsum(dim=2)
        h_loc = torch.exp(Acn) * inner
        P     = decay[:, :, -1]
        q     = h_loc[:, :, -1]
        carry = torch.zeros(B, D, device=x.device, dtype=torch.float32)
        starts = []
        for j in range(N):
            starts.append(carry)
            carry = P[:, j] * carry + q[:, j]
        starts = torch.stack(starts, dim=1)
        h = (h_loc + decay * starts.unsqueeze(2)).reshape(B, N * C, D)[:, :L]
        return self.dropout(h.to(x.dtype) * F.silu(self.W_g(x)))


class SpatialDynamicKernel(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5,
                 dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv      = nn.Conv1d(d_model, d_model, kernel_size,
                                   dilation=dilation, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x.transpose(1, 2), (self.pad, 0))
        return self.dropout(self.pointwise(F.silu(self.conv(h))).transpose(1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Fine-grained MoE + Shared Expert (DeepSeek-V4 minta)
# ─────────────────────────────────────────────────────────────────────────────

class MoEFeedForward(nn.Module):
    """
    Tokenenkénti top-k routing (16 routed expert, top-3, d_ff=384)
    + 1 mindig-aktív shared expert (d_ff=384).

    Aktív compute / token:
        shared:   384
        routed:   3 × 384 = 1152
        összesen: 1536  ==  sűrű FFN (d_ff=1536)  ✓
    """

    def __init__(self, d_model: int, expert_d_ff: int, n_experts: int = 16,
                 top_k: int = 3, dropout: float = 0.0, noise_std: float = 0.3,
                 shared_d_ff: Optional[int] = None):
        super().__init__()
        self.n_experts  = n_experts
        self.top_k      = min(top_k, n_experts)
        self.noise_std  = noise_std
        self.d_model    = d_model
        self.expert_d_ff = expert_d_ff

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.w_in   = nn.Parameter(torch.empty(n_experts, d_model, expert_d_ff))
        self.w_out  = nn.Parameter(torch.empty(n_experts, expert_d_ff, d_model))
        self.b_in   = nn.Parameter(torch.zeros(n_experts, expert_d_ff))
        self.b_out  = nn.Parameter(torch.zeros(n_experts, d_model))
        self.drop   = nn.Dropout(dropout)

        for e in range(n_experts):
            nn.init.normal_(self.w_in[e],  std=0.02)
            nn.init.normal_(self.w_out[e], std=0.02)
        nn.init.normal_(self.router.weight, std=0.02)

        # Shared expert (mindig aktív — SwiGLU)
        self.shared = SwiGLU(d_model, shared_d_ff, dropout) if shared_d_ff else None

        self.aux_loss: Optional[torch.Tensor] = None
        self.register_buffer("load_counts", torch.zeros(n_experts), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xf  = x.reshape(-1, D)

        logits = self.router(xf)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        gates = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(xf)
        for e in range(self.n_experts):
            sel = (topi == e)
            if not sel.any():
                continue
            tok = sel.any(-1).nonzero(as_tuple=True)[0]
            g   = (gates * sel).sum(-1)[tok].unsqueeze(-1)
            h   = F.silu(xf[tok] @ self.w_in[e] + self.b_in[e])
            y   = self.drop(h) @ self.w_out[e] + self.b_out[e]
            out.index_add_(0, tok, g * y)

        # Switch Transformer load balancing
        with torch.no_grad():
            cnt = torch.zeros(self.n_experts, device=x.device)
            cnt.index_add_(0, topi.reshape(-1),
                           torch.ones(topi.numel(), device=x.device))
            self.load_counts = cnt / max(1, topi.numel())
        f = torch.zeros(self.n_experts, device=x.device, dtype=probs.dtype)
        f.index_add_(0, topi.reshape(-1),
                     torch.ones(topi.numel(), device=x.device, dtype=probs.dtype))
        f        = f / max(1, topi.numel())
        self.aux_loss = self.n_experts * (f * probs.mean(0)).sum()

        routed = self.drop(out).view(B, L, D)
        if self.shared is not None:
            return routed + self.shared(x)
        return routed


def collect_moe_aux_loss(blocks: nn.ModuleList) -> torch.Tensor:
    losses = [b.mlp.aux_loss for b in blocks
              if isinstance(getattr(b, "mlp", None), MoEFeedForward)
              and b.mlp.aux_loss is not None]
    if not losses:
        return torch.zeros(())
    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# SGS-blokk — RMSNorm + SwiGLU + HyperResidual
# ─────────────────────────────────────────────────────────────────────────────

class SGSBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, kernel_size: int = 5,
                 dilation: int = 1, dropout: float = 0.0,
                 chunk_size: int = 16, use_moe: bool = False,
                 n_experts: int = 16, top_k: int = 3,
                 expert_d_ff: int = 384, moe_noise_std: float = 0.3,
                 shared_d_ff: Optional[int] = None,
                 use_hyper: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.tdk   = SpatialDynamicKernel(d_model, kernel_size, dilation, dropout)
        self.sgs   = SelectiveGatedState(d_model, dropout, chunk_size)
        self.norm2 = RMSNorm(d_model)

        if use_moe:
            self.mlp = MoEFeedForward(
                d_model, expert_d_ff, n_experts, top_k, dropout,
                moe_noise_std, shared_d_ff=shared_d_ff)
        else:
            self.mlp = SwiGLU(d_model, d_ff, dropout)

        self.res1 = HyperResidual(d_model) if use_hyper else None
        self.res2 = HyperResidual(d_model) if use_hyper else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        fx = self.sgs(h + self.tdk(h))
        x  = self.res1(x, fx) if self.res1 else x + fx
        fx2 = self.mlp(self.norm2(x))
        return self.res2(x, fx2) if self.res2 else x + fx2


# ─────────────────────────────────────────────────────────────────────────────
# GQA RoPE-Attention blokk — KV-cache + HyperResidual
# ─────────────────────────────────────────────────────────────────────────────

KVCache = Tuple[torch.Tensor, torch.Tensor]   # (K, V)


class RoPEAttnBlock(nn.Module):
    """
    Grouped Query Attention (GQA): n_heads Q, n_kv_heads K+V.
    Prefix-LM maszk. KV-cache generáláshoz.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, dropout: float = 0.0,
                 rope_base: float = 500_000.0, use_hyper: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # Q heads per KV head
        self.d_head     = d_model // n_heads
        self.rope_base  = rope_base
        assert self.d_head % 2 == 0

        self.norm1    = RMSNorm(d_model)
        self.q_proj   = nn.Linear(d_model, n_heads    * self.d_head, bias=False)
        self.kv_proj  = nn.Linear(d_model, 2 * n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2    = RMSNorm(d_model)
        self.mlp      = SwiGLU(d_model, d_ff, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.res1     = HyperResidual(d_model) if use_hyper else None
        self.res2     = HyperResidual(d_model) if use_hyper else None
        self._mask_cache: Dict[tuple, torch.Tensor] = {}

    def to(self, *args, **kwargs):
        self._mask_cache.clear()
        return super().to(*args, **kwargs)

    def _prefix_mask(self, L: int, S: int, device,
                     prefix_len: Optional[int]) -> torch.Tensor:
        """Kauzális maszk [L, S] — prefix kétirányú, generálás kauzális."""
        key = (L, S, prefix_len if prefix_len is not None else -1)
        cached = self._mask_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        # Q-pozíciók: offset S-L..S-1; K-pozíciók: 0..S-1
        mask = torch.zeros(L, S, device=device)
        for qi in range(L):
            q_pos = S - L + qi
            for ki in range(S):
                if ki > q_pos:
                    mask[qi, ki] = float("-inf")
        if prefix_len is not None and prefix_len > 0:
            p = min(prefix_len, S)
            mask[:, :p] = mask[:, :p].masked_fill(
                mask[:, :p] == float("-inf"), 0.0)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor,
                prefix_len: Optional[int] = None,
                past_kv: Optional[KVCache] = None
                ) -> Tuple[torch.Tensor, KVCache]:
        B, L, D = x.shape
        H, Hkv, rep, dh = self.n_heads, self.n_kv_heads, self.n_rep, self.d_head

        h  = self.norm1(x)
        q  = self.q_proj(h).view(B, L, H, dh).permute(0, 2, 1, 3)      # [B,H,L,dh]
        kv = self.kv_proj(h).view(B, L, 2, Hkv, dh).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                              # [B,Hkv,L,dh]

        # KV-cache hozzáfűzés
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv: KVCache = (k, v)
        S = k.shape[2]   # teljes szekvencia-hossz (cache + jelen)

        # RoPE
        pos = torch.arange(S, device=x.device)
        cos, sin = _rope_angles(pos, dh, self.rope_base)
        cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
        # csak a jelen L pozícióra alkalmazzuk a Q-ra
        q_cos, q_sin = cos[:, :, -L:, :], sin[:, :, -L:, :]
        q = _apply_rope(q, q_cos, q_sin)
        k = _apply_rope(k, cos, sin)

        # GQA: K,V ismétlése Q-fejekhez
        k = k.repeat_interleave(rep, dim=1)   # [B,H,S,dh]
        v = v.repeat_interleave(rep, dim=1)

        att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        att = att + self._prefix_mask(L, S, x.device, prefix_len)
        att = self.attn_drop(F.softmax(att, dim=-1))
        o   = (att @ v).permute(0, 2, 1, 3).reshape(B, L, D)

        fx = self.out_proj(o)
        x  = self.res1(x, fx) if self.res1 else x + fx
        fx2 = self.mlp(self.norm2(x))
        x   = self.res2(x, fx2) if self.res2 else x + fx2
        return x, new_kv


# ─────────────────────────────────────────────────────────────────────────────
# STKApexTrunk v2
# ─────────────────────────────────────────────────────────────────────────────

class STKApexTrunk(nn.Module):

    def __init__(self, cfg: STKApexConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_norm = RMSNorm(cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)

        blocks: List[nn.Module] = []
        sgs_i = 0
        for layer_i in range(cfg.n_layers):
            is_attn = (cfg.attn_every > 0
                       and (layer_i + 1) % cfg.attn_every == 0)
            if is_attn:
                blocks.append(RoPEAttnBlock(
                    cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                    cfg.d_ff, cfg.dropout, cfg.rope_base,
                    use_hyper=cfg.use_hyper_connections))
            else:
                chunk_size  = cfg.sgs_chunk_sizes[sgs_i % len(cfg.sgs_chunk_sizes)]
                dilation    = cfg.tdk_dilations[sgs_i % len(cfg.tdk_dilations)]
                use_moe     = (cfg.use_moe
                               and sgs_i % cfg.moe_every == cfg.moe_every - 1)
                shared_d_ff = cfg.shared_expert_d_ff if (use_moe and cfg.use_shared_expert) else None
                blocks.append(SGSBlock(
                    cfg.d_model, cfg.d_ff,
                    kernel_size=cfg.tdk_kernel_size,
                    dilation=dilation,
                    dropout=cfg.dropout,
                    chunk_size=chunk_size,
                    use_moe=use_moe,
                    n_experts=cfg.n_experts,
                    top_k=cfg.top_k,
                    expert_d_ff=cfg.expert_d_ff,
                    moe_noise_std=cfg.moe_noise_std,
                    shared_d_ff=shared_d_ff,
                    use_hyper=cfg.use_hyper_connections,
                ))
                sgs_i += 1

        self.blocks = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(cfg.d_model)
        self.grad_checkpointing = False

        nn.init.normal_(self.embedding.weight, std=0.02)
        self._init_weights()
        for m in self.modules():
            if isinstance(m, SelectiveGatedState):
                nn.init.constant_(m.W_a.bias, 2.0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor,
                prefix_len: Optional[int] = None,
                past_kvs: Optional[List[Optional[KVCache]]] = None,
                ) -> Tuple[torch.Tensor, List[KVCache]]:
        """
        Visszaad: (hidden [B,L,D], new_kvs list)
        past_kvs: lista, egy elem per RoPE-Attention réteg (None = nincs cache)
        """
        x = self.embedding(input_ids) * (self.cfg.d_model ** 0.5)
        x = self.embed_drop(self.embed_norm(x))
        use_ckpt = (self.grad_checkpointing and self.training
                    and torch.is_grad_enabled())

        new_kvs: List[KVCache] = []
        attn_idx = 0
        for blk in self.blocks:
            if isinstance(blk, RoPEAttnBlock):
                past = (past_kvs[attn_idx]
                        if past_kvs is not None and attn_idx < len(past_kvs)
                        else None)
                if use_ckpt:
                    # checkpointing + KV-cache: cache kikapcsol tanításkor
                    x, nkv = _ckpt(
                        lambda t, b=blk, p=prefix_len: b(t, prefix_len=p),
                        x)
                else:
                    x, nkv = blk(x, prefix_len=prefix_len, past_kv=past)
                new_kvs.append(nkv)
                attn_idx += 1
            else:
                x = _ckpt(blk, x) if use_ckpt else blk(x)

        return self.norm_f(x), new_kvs

    def moe_aux_loss(self) -> torch.Tensor:
        return collect_moe_aux_loss(self.blocks)

    @property
    def n_params_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_params_active(self) -> int:
        dormant = 0
        for blk in self.blocks:
            moe = getattr(blk, "mlp", None)
            if not isinstance(moe, MoEFeedForward):
                continue
            per_e = moe.d_model * moe.expert_d_ff * 2
            dormant += (moe.n_experts - moe.top_k) * per_e
        return self.n_params_total - dormant
