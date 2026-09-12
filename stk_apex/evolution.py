"""
MAP-Elites evolúciós keresés — STKApex hyperparaméter-optimalizálás.

Forrás: stk_agent_v2/evolution.py (adaptálva STKApexConfig-hoz).

2D minőség-sokféleség rács:
    - tengely 1: reasoning_depth (reason_nodes: 2, 4, 6, 8)
    - tengely 2: moe_capacity    (n_experts: 4, 8, 16, 32)

3 módosítási szint:
    1. Hyperparaméter perturbáció (dropout, lr, chunk_sizes)
    2. Architektúra-módosítás (n_layers, d_model)
    3. LoRA fine-tune (rank=4) — csak ha modell is elérhető
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .config import STKApexConfig


REASONING_DEPTH_AXIS = [2, 4, 6, 8]   # reason_nodes értékek
MOE_CAPACITY_AXIS    = [4, 8, 16, 32]  # n_experts értékek


@dataclass
class Individual:
    cfg:     STKApexConfig
    fitness: float = float("-inf")
    gen:     int   = 0

    @property
    def cell(self) -> Tuple[int, int]:
        rd = min(REASONING_DEPTH_AXIS, key=lambda x: abs(x - self.cfg.reason_nodes))
        mc = min(MOE_CAPACITY_AXIS,    key=lambda x: abs(x - self.cfg.n_experts))
        return (REASONING_DEPTH_AXIS.index(rd), MOE_CAPACITY_AXIS.index(mc))


class MAPElites:
    """MAP-Elites 2D rács."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        n_rd = len(REASONING_DEPTH_AXIS)
        n_mc = len(MOE_CAPACITY_AXIS)
        self.grid: Dict[Tuple[int,int], Individual] = {}
        self.history: List[Individual] = []

    def add(self, ind: Individual) -> bool:
        """Beteszi az egyedet a rácsba, ha jobb mint a jelenlegi."""
        cell = ind.cell
        prev = self.grid.get(cell)
        if prev is None or ind.fitness > prev.fitness:
            self.grid[cell] = ind
            self.history.append(ind)
            return True
        return False

    def sample(self) -> Individual:
        """Véletlen egyedet választ a rácsból."""
        if not self.grid:
            raise RuntimeError("A rács üres — adj hozzá legalább egy egyedet.")
        return self.rng.choice(list(self.grid.values()))

    # ── Módosítási szintek ────────────────────────────────────────────────

    def perturb_hyperparams(self, ind: Individual) -> Individual:
        """1. szint: véletlen perturbáció a hyperparaméterekben."""
        cfg = copy.deepcopy(ind.cfg)
        # dropout perturbáció
        cfg.dropout    = max(0.0, min(0.3, cfg.dropout + self.rng.gauss(0, 0.01)))
        cfg.moe_noise_std = max(0.0, min(1.0, cfg.moe_noise_std + self.rng.gauss(0, 0.05)))
        cfg.top_k      = self.rng.choice([1, 2, 3])
        cfg.top_k      = min(cfg.top_k, cfg.n_experts)
        return Individual(cfg, gen=ind.gen + 1)

    def perturb_architecture(self, ind: Individual) -> Individual:
        """2. szint: architektúra-módosítás."""
        cfg = copy.deepcopy(ind.cfg)
        cfg.n_layers  = self.rng.choice([8, 10, 12, 16])
        cfg.d_model   = self.rng.choice([256, 384, 512, 640])
        cfg.n_heads   = self.rng.choice([4, 8])
        cfg.d_ff      = cfg.d_model * 3
        cfg.n_experts = self.rng.choice(MOE_CAPACITY_AXIS)
        cfg.reason_nodes = self.rng.choice(REASONING_DEPTH_AXIS)
        # chunk_sizes és dilations nullázása → __post_init__ újragenerálja
        cfg.sgs_chunk_sizes = None
        cfg.tdk_dilations   = None
        cfg.__post_init__()
        return Individual(cfg, gen=ind.gen + 1)

    def mutate(self, ind: Individual, level: int = 0) -> Individual:
        """level 0: hyper, 1: arch, 2: arch (súlyosabb)."""
        if level == 0:
            return self.perturb_hyperparams(ind)
        return self.perturb_architecture(ind)

    def evolve(self, n_iters: int = 50,
               evaluate_fn=None) -> Individual:
        """
        Egyszerűsített MAP-Elites loop.

        evaluate_fn(cfg: STKApexConfig) → float   (pl. validation perplexity negáltja)
        """
        if evaluate_fn is None:
            def evaluate_fn(cfg):
                # dummy: paraméterszám reciproka (kisebb = jobb)
                from .model import STKApex
                m = STKApex(cfg)
                r = m.param_report()
                return -r["active"] / 1e6

        for i in range(n_iters):
            if not self.grid:
                # kezdeti populáció
                cfg = STKApexConfig()
                score = evaluate_fn(cfg)
                self.add(Individual(cfg, fitness=score))
                continue

            parent = self.sample()
            level  = 0 if self.rng.random() < 0.7 else 1
            child  = self.mutate(parent, level)
            try:
                score = evaluate_fn(child.cfg)
                child.fitness = score
                self.add(child)
            except Exception as e:
                pass  # érvénytelen konfiguráció

        if not self.grid:
            raise RuntimeError("Nem sikerült egyetlen érvényes egyedet sem találni.")
        return max(self.grid.values(), key=lambda x: x.fitness)

    def summary(self) -> str:
        lines = [f"MAP-Elites rács ({len(self.grid)} cella):",
                 f"  reasoning_depth: {REASONING_DEPTH_AXIS}",
                 f"  moe_capacity:    {MOE_CAPACITY_AXIS}"]
        if self.grid:
            best = max(self.grid.values(), key=lambda x: x.fitness)
            lines.append(f"  legjobb fitness: {best.fitness:.4f} "
                         f"(reason_nodes={best.cfg.reason_nodes}, "
                         f"n_experts={best.cfg.n_experts})")
        return "\n".join(lines)
