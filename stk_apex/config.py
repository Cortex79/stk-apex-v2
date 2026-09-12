"""
STKApexConfig v2 — 2026-os kutatások alapján frissítve.

Új elemek vs v1:
  GQA          n_kv_heads=2 (8 Q-fej, 2 KV-fej → 4× kisebb KV-cache)
  RoPE base    500 000 (volt: 10 000 → hosszú kontextus)
  Fine-grained 16 routed expert, top-3, d_ff_per=384 (volt: 8 expert top-2 d_ff=768)
  Shared expert mindig aktív expert d_ff=384 (DeepSeek-V4 minta)
  SwiGLU       kapuzott FFN (volt: SiLU Sequential)
  RMSNorm      (volt: LayerNorm)
  Hyper-Conn   tanult reziduális kapcsolat (DeepSeek-V4 Hyper-Connections)
  KV-cache     generáláshoz (patch: model.py)
  TIAR         trajektória-súlyozott RLVR (train_utils.py)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class STKApexConfig:
    # ── alap ──────────────────────────────────────────────────────────────
    vocab_size:   int   = 24576   # stk_apex/tokenizer/superbrain_tokenizer_24k_v2
    d_model:      int   = 512
    n_layers:     int   = 12
    n_heads:      int   = 8
    n_kv_heads:   int   = 2          # GQA: 4 Q-fej / KV-fej
    d_ff:         int   = 1536       # SwiGLU gate/up dim (single projection)
    max_seq_len:  int   = 2048
    dropout:      float = 0.05

    # ── SGS + TDK ─────────────────────────────────────────────────────────
    attn_every:       int   = 4
    tdk_kernel_size:  int   = 5
    rope_base:        float = 500_000.0   # YaRN-kompatibilis hosszú kontextus

    # ── Multi-Scale SGS ───────────────────────────────────────────────────
    sgs_chunk_sizes: Optional[Tuple[int, ...]] = None
    tdk_dilations:   Optional[Tuple[int, ...]] = None

    # ── Fine-grained MoE + shared expert ──────────────────────────────────
    use_moe:               bool  = True
    moe_every:             int   = 3
    n_experts:             int   = 16       # routed (volt: 8)
    top_k:                 int   = 3        # top-3 / 16 (volt: top-2 / 8)
    expert_d_ff_ratio:     float = 0.25     # 384 / expert (volt: 0.5 → 768)
    use_shared_expert:     bool  = True     # mindig aktív shared expert
    shared_expert_d_ff_ratio: float = 0.25  # shared d_ff = 384
    moe_noise_std:         float = 0.3
    moe_aux_loss_weight:   float = 0.01

    # ── Hyper-Connections (DeepSeek-V4) ───────────────────────────────────
    use_hyper_connections: bool  = True

    # ── Kalibrált tartózkodás (STK-6 / Abstain-R1) ───────────────────────
    abstention_threshold: float = 0.75
    use_abstention:       bool  = True

    # ── RAG ───────────────────────────────────────────────────────────────
    use_rag:          bool = True
    rag_index_path:   str  = r"C:\Users\tibor\Desktop\SUPERBRAIN-STK2-274M\data\knowledge_layer"
    rag_top_k:        int  = 5
    rag_max_tokens:   int  = 512

    # ── Kognitív modulok ──────────────────────────────────────────────────
    use_temporal_brain:  bool = True
    use_reason_compiler: bool = True
    use_reality_anchor:  bool = True
    temporal_slots:      int  = 8
    reason_nodes:        int  = 4
    reason_node_dim:     int  = 32

    # ── Multimodal (STK-4) ────────────────────────────────────────────────
    use_multimodal:   bool = False
    byte_patch_size:  int  = 8
    mm_local_d:       int  = 128

    # ── Generálás ─────────────────────────────────────────────────────────
    pad_token_id:   int   = 0
    eos_token_id:   int   = 2
    max_new_tokens: int   = 256
    temperature:    float = 0.8
    top_p:          float = 0.9

    def __post_init__(self):
        n = self._n_sgs_blocks()
        if self.sgs_chunk_sizes is None:
            if n == 9:
                self.sgs_chunk_sizes = (8, 16, 32, 16, 32, 16, 8, 16, 8)
            else:
                base = (8, 16, 32)
                self.sgs_chunk_sizes = tuple(base[i % 3] for i in range(n))
        if self.tdk_dilations is None:
            if n == 9:
                self.tdk_dilations = (1, 2, 4, 1, 2, 4, 1, 2, 4)
            else:
                base = (1, 2, 4)
                self.tdk_dilations = tuple(base[i % 3] for i in range(n))

    def _n_sgs_blocks(self) -> int:
        if self.attn_every <= 0:
            return self.n_layers
        return self.n_layers - self.n_layers // self.attn_every

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def wrong_penalty(self) -> float:
        t = self.abstention_threshold
        return -t / (1.0 - t)

    @property
    def expert_d_ff(self) -> int:
        return max(8, int(round(self.d_ff * self.expert_d_ff_ratio)))

    @property
    def shared_expert_d_ff(self) -> int:
        return max(8, int(round(self.d_ff * self.shared_expert_d_ff_ratio)))

    def validate(self) -> None:
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.d_head % 2 == 0
        n = self._n_sgs_blocks()
        assert len(self.sgs_chunk_sizes) == n
        assert len(self.tdk_dilations) == n
        assert 0 < self.abstention_threshold < 1


# ── Előre definiált méretek ────────────────────────────────────────────────

def apex_60m() -> STKApexConfig:
    """~70M teljes / ~52M aktív (SwiGLU +16% FFN params, shared expert)."""
    return STKApexConfig(
        d_model=512, n_layers=12, n_heads=8, n_kv_heads=2, d_ff=1536,
        n_experts=16, top_k=3, expert_d_ff_ratio=0.25,
        use_shared_expert=True, use_hyper_connections=True,
    )


def apex_mini() -> STKApexConfig:
    """~15M aktív — kísérletezéshez."""
    return STKApexConfig(
        d_model=256, n_layers=6, n_heads=4, n_kv_heads=2, d_ff=768,
        n_experts=8, top_k=2, expert_d_ff_ratio=0.25,
        use_shared_expert=True, use_hyper_connections=True,
        use_rag=False,
    )
