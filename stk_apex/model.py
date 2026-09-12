"""
STKApex v2 — fő modell, KV-cache generálással.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .config import STKApexConfig
from .core import STKApexTrunk, RMSNorm, SwiGLU, KVCache
from .abstention import is_abstention


# ─────────────────────────────────────────────────────────────────────────────
# TemporalBrain
# ─────────────────────────────────────────────────────────────────────────────

class TemporalBrain(nn.Module):
    def __init__(self, d_model: int, n_slots: int = 8, dropout: float = 0.05):
        super().__init__()
        self.n_slots = n_slots
        self.gru  = nn.GRUCell(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, n_slots)
        self.drop = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)
        self.register_buffer("_slots", torch.zeros(1, n_slots, d_model))

    def reset(self) -> None:
        self._slots.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        slots   = self._slots.expand(B, -1, -1)
        ctx     = x.mean(dim=1)
        weights = F.softmax(
            self.gate(torch.cat([ctx, slots.mean(1)], dim=-1)), dim=-1)
        read    = (weights.unsqueeze(-1) * slots).sum(1)
        new_vec = self.gru(ctx[:1], slots[:1, 0])
        self._slots = self._slots.clone()
        self._slots[0, 0] = new_vec.detach()[0]
        inject = read.unsqueeze(1).expand(-1, L, -1)
        return self.norm(x + self.drop(inject))


# ─────────────────────────────────────────────────────────────────────────────
# ReasonCompiler
# ─────────────────────────────────────────────────────────────────────────────

class ReasonCompiler(nn.Module):
    def __init__(self, d_model: int, n_nodes: int = 4, node_dim: int = 32,
                 dropout: float = 0.05):
        super().__init__()
        self.n_nodes  = n_nodes
        self.proj_in  = nn.Linear(d_model, n_nodes * node_dim)
        self.gnn      = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2), nn.SiLU(),
            nn.Linear(node_dim * 2, node_dim))
        self.proj_out = nn.Linear(n_nodes * node_dim, d_model)
        self.norm     = RMSNorm(d_model)
        self.drop     = nn.Dropout(dropout)
        self._cache: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        ctx   = x.mean(1)
        nodes = self.proj_in(ctx).view(B, self.n_nodes, -1)
        nodes = self.gnn(nodes)
        self._cache = nodes.detach()
        out   = self.proj_out(nodes.reshape(B, -1))
        return self.norm(x + self.drop(out.unsqueeze(1).expand(-1, L, -1)))


# ─────────────────────────────────────────────────────────────────────────────
# RealityAnchor — geometry-calibrated conformal abstention (2604.27914 minta)
# ─────────────────────────────────────────────────────────────────────────────

class RealityAnchor(nn.Module):
    """
    Konfidencia-skalár p(confident) ∈ [0,1].

    v2: a konfidencia nemcsak a pool-olt mean-ből, hanem a szórásból is
    számítódik — alacsony szórás = magabiztos (conformal inspiration).
    """
    def __init__(self, d_model: int, threshold: float = 0.75):
        super().__init__()
        self.threshold = threshold
        # mean + std → konfidencia
        self.conf_head = nn.Linear(d_model * 2, 1)
        nn.init.zeros_(self.conf_head.weight)
        nn.init.constant_(self.conf_head.bias, 1.0)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1)                              # [B, D]
        std  = x.std(dim=1, correction=0).clamp(min=1e-6)  # [B, D]
        feat = torch.cat([mean, std], dim=-1)             # [B, 2D]
        conf = torch.sigmoid(self.conf_head(feat)).squeeze(-1)  # [B]
        return x, conf


# ─────────────────────────────────────────────────────────────────────────────
# STKApex v2
# ─────────────────────────────────────────────────────────────────────────────

class STKApex(nn.Module):

    def __init__(self, cfg: STKApexConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.trunk   = STKApexTrunk(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.trunk.embedding.weight   # weight tying

        self.temporal_brain:  Optional[TemporalBrain]   = None
        self.reason_compiler: Optional[ReasonCompiler]  = None
        self.reality_anchor:  Optional[RealityAnchor]   = None

        if cfg.use_temporal_brain:
            self.temporal_brain = TemporalBrain(
                cfg.d_model, cfg.temporal_slots, cfg.dropout)
        if cfg.use_reason_compiler:
            self.reason_compiler = ReasonCompiler(
                cfg.d_model, cfg.reason_nodes, cfg.reason_node_dim, cfg.dropout)
        if cfg.use_reality_anchor:
            self.reality_anchor = RealityAnchor(
                cfg.d_model, cfg.abstention_threshold)

        self._rag = None

    # ── RAG ───────────────────────────────────────────────────────────────

    def _get_rag(self):
        if self._rag is None and self.cfg.use_rag:
            from .rag import RAGRetriever
            self._rag = RAGRetriever(
                self.cfg.rag_index_path, self.cfg.rag_top_k,
                self.cfg.rag_max_tokens)
        return self._rag

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset_memory(self) -> None:
        if self.temporal_brain:   self.temporal_brain.reset()
        if self.reason_compiler:  self.reason_compiler.reset()

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:  torch.Tensor,
        prefix_len: Optional[int]           = None,
        labels:     Optional[torch.Tensor]  = None,
        past_kvs:   Optional[List[Optional[KVCache]]] = None,
    ) -> Dict:
        x, new_kvs = self.trunk(input_ids, prefix_len=prefix_len,
                                past_kvs=past_kvs)

        if self.temporal_brain is not None:
            x = self.temporal_brain(x)
        if self.reason_compiler is not None:
            x = self.reason_compiler(x)

        conf = None
        if self.reality_anchor is not None:
            x, conf = self.reality_anchor(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)

        moe_aux = self.trunk.moe_aux_loss()
        if loss is not None:
            moe_aux = moe_aux.to(loss.device)

        return {
            "logits":       logits,
            "loss":         loss,
            "moe_aux_loss": moe_aux,
            "confidence":   conf,
            "past_kvs":     new_kvs,
        }

    # ── Generálás KV-cache-sel ────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids:       torch.Tensor,
        *,
        max_new_tokens:  Optional[int]   = None,
        temperature:     Optional[float] = None,
        top_p:           Optional[float] = None,
        prefix_len:      Optional[int]   = None,
        rag_query:       Optional[str]   = None,
        use_abstention:  Optional[bool]  = None,
    ) -> Dict:
        prev_training = self.training
        self.eval()

        cfg     = self.cfg
        max_new = max_new_tokens or cfg.max_new_tokens
        temp    = temperature    or cfg.temperature
        p_top   = top_p          or cfg.top_p
        use_abs = (use_abstention if use_abstention is not None
                   else cfg.use_abstention)

        rag_context = None
        if rag_query and cfg.use_rag:
            rag = self._get_rag()
            if rag and rag.is_available:
                rag_context = rag.format_context(rag_query)

        device = input_ids.device
        seq    = input_ids.clone()   # [B, L]

        # Prefill: az egész prompt egy menetben, KV-cache feltöltése
        out      = self.forward(seq, prefix_len=prefix_len)
        past_kvs = out["past_kvs"]

        for _ in range(max_new):
            # Csak az utolsó tokent feldolgozza, cache-sel
            last_tok = seq[:, -1:]
            out      = self.forward(last_tok, past_kvs=past_kvs)
            past_kvs = out["past_kvs"]
            logits   = out["logits"][:, -1, :]
            conf     = out["confidence"]

            # Tartózkodás-ellenőrzés
            if use_abs and conf is not None:
                if conf.min().item() < cfg.abstention_threshold:
                    eos = torch.full((seq.shape[0], 1), cfg.eos_token_id,
                                    dtype=torch.long, device=device)
                    seq = torch.cat([seq, eos], dim=1)
                    break

            # Mintavételezés
            if temp > 0:
                logits = logits / temp
            if p_top < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                rm = cp - F.softmax(sl, dim=-1) > p_top
                sl[rm] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, si, sl)
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            seq      = torch.cat([seq, next_tok], dim=1)

            if (next_tok == cfg.eos_token_id).all():
                break

        self.train(prev_training)
        return {"sequences": seq, "rag_context": rag_context}

    # ── Statisztikák ──────────────────────────────────────────────────────

    def param_report(self) -> Dict:
        total  = sum(p.numel() for p in self.parameters())
        active = self.trunk.n_params_active
        cog    = sum(p.numel() for n, p in self.named_parameters()
                     if any(k in n for k in
                            ("temporal_brain", "reason_compiler", "reality_anchor")))
        return {"total": total, "active": active,
                "dormant": total - active, "cognitive_modules": cog,
                "ratio": active / max(1, total)}
