"""
STKApex tréning v2 — TIAR-alapú RLVR + kombinált veszteség.

Új v1-hez képest:
  TIAR  Trajectory-Informed Advantage Reweighting (arXiv:2605.25850)
        — az egész generálási trajektória kap súlyt, nem csak az utolsó token.
        A helyes trajektóriák korai tokenjei magasabb advantaget kapnak.
  Abstain-R1 minta (arXiv:2604.17073): tartózkodás is jutalmaz, büntet.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstention import rlvr_reward
from .model import STKApex


# ─────────────────────────────────────────────────────────────────────────────
# Kombinált veszteség
# ─────────────────────────────────────────────────────────────────────────────

def combined_loss(
    model: STKApex,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    moe_weight: float = 0.01,
    prefix_len: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    out = model(input_ids, labels=labels, prefix_len=prefix_len)
    lm  = out["loss"]
    aux = out["moe_aux_loss"].to(lm.device)
    return {"loss": lm + moe_weight * aux, "lm": lm, "moe_aux": aux}


# ─────────────────────────────────────────────────────────────────────────────
# TIAR — Trajectory-Informed Advantage Reweighting
# ─────────────────────────────────────────────────────────────────────────────

def _tiar_weights(seq_len: int, reward: float,
                  gamma: float = 0.99) -> torch.Tensor:
    """
    Exponenciálisan csökkenő súlyok a trajektória tokenjein.

    Helyes trajektória (reward=+1): korai tokenek magasabb súlyt kapnak
    — megerősíti a korai, helyes elköteleződést.
    Helytelen (reward<0): egyenletes súly — minden tokent egyforma mértékben
    büntet.

    gamma: diszkont-faktor (0.99 → az utolsó token ~0.99^(L-1) súlya)
    """
    if seq_len == 0:
        return torch.zeros(1)
    if reward > 0:
        w = torch.tensor([gamma ** (seq_len - 1 - i) for i in range(seq_len)])
        w = w / w.sum()
    else:
        w = torch.ones(seq_len) / seq_len
    return w * abs(reward)


def tiar_rlvr_loss(
    model: STKApex,
    input_ids: torch.Tensor,         # [B, L_prompt + L_gen]
    prompt_lens: List[int],           # prompt hossza minden batch-elemben
    generated_texts: List[str],       # dekódolt generálások
    target_texts: List[str],
    domains: List[str],
    *,
    threshold: float = 0.75,
    gamma: float = 0.99,
    moe_weight: float = 0.01,
    entropy_bonus: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    TIAR-alapú policy gradient.

    1. Jutalmak kiszámítása (RLVR + tartózkodás)
    2. TIAR-súlyok generálása per trajektória
    3. Súlyozott token-szintű log-valószínűség veszteség
    4. Entrópia bónusz (explorációhoz)
    5. MoE aux loss
    """
    B = input_ids.shape[0]
    device = input_ids.device

    rewards = torch.tensor(
        [rlvr_reward(d, p, t, threshold=threshold)
         for d, p, t in zip(domains, generated_texts, target_texts)],
        dtype=torch.float32, device=device,
    )

    # Baseline (running mean)
    baseline   = rewards.mean().detach()
    advantages = rewards - baseline          # [B]

    # Forward pass a teljes input_ids-en
    out    = model(input_ids)
    logits = out["logits"]                   # [B, L, V]
    log_p  = F.log_softmax(logits, dim=-1)   # [B, L, V]

    # TIAR: per-token súlyozás
    pg_loss = torch.zeros(1, device=device)
    for i in range(B):
        p_len  = prompt_lens[i]
        gen_len = input_ids.shape[1] - p_len
        if gen_len <= 0:
            continue
        r = rewards[i].item()
        a = advantages[i].item()

        # TIAR súlyok a generálási ablakra
        w = _tiar_weights(gen_len, r, gamma).to(device)   # [gen_len]

        # log-valószínűség a generált tokenekre
        gen_ids   = input_ids[i, p_len:]                  # [gen_len]
        gen_logp  = log_p[i, p_len - 1: -1]              # [gen_len, V]
        tok_logp  = gen_logp.gather(1, gen_ids.unsqueeze(1)).squeeze(1)  # [gen_len]

        pg_loss = pg_loss - (a * (w * tok_logp)).sum()

    pg_loss = pg_loss / max(B, 1)

    # Entrópia bónusz (exploráció)
    probs   = F.softmax(logits, dim=-1)
    entropy = -(probs * log_p).sum(-1).mean()
    ent_loss = -entropy_bonus * entropy

    aux   = out["moe_aux_loss"].to(device)
    total = pg_loss + ent_loss + moe_weight * aux

    return {
        "loss":        total,
        "pg_loss":     pg_loss,
        "entropy":     entropy,
        "moe_aux":     aux,
        "rewards":     rewards,
        "mean_reward": rewards.mean(),
        "advantages":  advantages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer + LR scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: STKApex,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas,
    )


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: STKApex, optimizer: torch.optim.Optimizer,
                    step: int, path: str) -> None:
    import dataclasses
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config":    dataclasses.asdict(model.cfg),
    }, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    return torch.load(path, map_location=device)
