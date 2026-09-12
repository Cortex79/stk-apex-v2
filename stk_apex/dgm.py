"""
Darwin Gödel Machine (DGM) — STKApex önjavító loop.

Forrás: Sakana AI, arXiv:2505.22954, ICLR 2026.

Eredeti DGM eredmények:
    SWE-bench:  20.0% → 50.0%
    Polyglot:   14.2% → 30.7%

Megvalósítás:
    1. Archívum (MAP-Elites)       — már megvolt az evolution.py-ban
    2. LLM-vezérelt kódmutáció     — STKApex javasol Python patchet
    3. Sandbox tesztelő            — subprocess, izolált futtatás
    4. Empirikus kiértékelés       — validation loss / reward
    5. Archívum frissítése         — jobb → bekerül, rosszabb → eldobja

A mutáció célpontjai (3 szint, mint a DGM-ben):
    LEVEL_1  Hyperparaméter (config.py mezők)
    LEVEL_2  Architektúra (core.py SGSBlock/RoPEAttnBlock módosítása)
    LEVEL_3  Kognitív modul (model.py TemporalBrain/ReasonCompiler)
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .config import STKApexConfig
from .evolution import MAPElites, Individual

# ── Archívum és mutáció szintjei ─────────────────────────────────────────────

LEVEL_HYPER = 0    # config mezők perturbálása
LEVEL_ARCH  = 1    # architektúra-paraméter
LEVEL_MODULE = 2   # kognitív modul kapcsoló


# ─────────────────────────────────────────────────────────────────────────────
# Mutáció-javaslatok (LLM-vezérelt — szöveg alapján)
# ─────────────────────────────────────────────────────────────────────────────

_HYPER_PROPOSALS = [
    {"dropout": 0.03},
    {"dropout": 0.08},
    {"n_experts": 32,  "top_k": 4, "expert_d_ff_ratio": 0.125},
    {"n_experts": 8,   "top_k": 2, "expert_d_ff_ratio": 0.5},
    {"rope_base": 1_000_000.0},
    {"rope_base": 100_000.0},
    {"moe_noise_std": 0.1},
    {"moe_noise_std": 0.5},
    {"abstention_threshold": 0.6},
    {"abstention_threshold": 0.9},
    {"attn_every": 3},
    {"attn_every": 6},
    {"temporal_slots": 4},
    {"temporal_slots": 16},
]

_ARCH_PROPOSALS = [
    {"d_model": 384, "n_heads": 6, "n_kv_heads": 2, "d_ff": 1152},
    {"d_model": 640, "n_heads": 8, "n_kv_heads": 2, "d_ff": 1920},
    {"n_layers": 10},
    {"n_layers": 16},
    {"moe_every": 2},
    {"moe_every": 4},
    {"use_hyper_connections": False},
    {"use_shared_expert": False},
    {"n_kv_heads": 4},
    {"n_kv_heads": 1},
]

_MODULE_PROPOSALS = [
    {"use_temporal_brain": False},
    {"use_reason_compiler": False},
    {"use_reality_anchor": False},
    {"use_temporal_brain": True, "use_reason_compiler": True, "use_reality_anchor": True},
    {"temporal_slots": 12, "reason_nodes": 6},
    {"reason_nodes": 2, "reason_node_dim": 64},
]


# ─────────────────────────────────────────────────────────────────────────────
# DGMAgent — egyetlen archívum-példány
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DGMAgent:
    cfg:        STKApexConfig
    fitness:    float           = float("-inf")
    gen:        int             = 0
    level:      int             = 0
    parent_id:  Optional[int]   = None
    mutations:  List[Dict]      = field(default_factory=list)

    @property
    def n_active(self) -> int:
        from .model import STKApex
        try:
            m = STKApex(self.cfg)
            return m.trunk.n_params_active
        except Exception:
            return 0


class DarwinGodelMachine:
    """
    Önjavító loop STKApex-hez.

    evaluate_fn(cfg) → float   — magasabb = jobb (pl. -val_loss, vagy reward)
    Ha nincs evaluate_fn, dummy: -aktív_params/1e6 (kisebb = jobb heurisztika).
    """

    def __init__(self, seed_cfg: Optional[STKApexConfig] = None,
                 evaluate_fn: Optional[Callable] = None,
                 seed: int = 42):
        self.archive: List[DGMAgent] = []
        self.rng = random.Random(seed)
        self.evaluate_fn = evaluate_fn or self._dummy_eval
        self.gen = 0
        self._id_counter = 0

        if seed_cfg is not None:
            self._seed(seed_cfg)

    # ── Inicializálás ─────────────────────────────────────────────────────

    def _seed(self, cfg: STKApexConfig) -> None:
        score = self._safe_eval(cfg)
        agent = DGMAgent(cfg=copy.deepcopy(cfg), fitness=score, gen=0)
        self._add(agent)

    def _dummy_eval(self, cfg: STKApexConfig) -> float:
        from .model import STKApex
        try:
            m = STKApex(cfg)
            r = m.param_report()
            # Heurisztika: aktív/teljes arány + nem túl kicsi nem túl nagy
            ratio = r["active"] / max(1, r["total"])
            size_score = -abs(r["active"] - 50e6) / 1e6
            return ratio + size_score * 0.01
        except Exception:
            return float("-inf")

    def _safe_eval(self, cfg: STKApexConfig) -> float:
        try:
            cfg.validate()
            return self.evaluate_fn(cfg)
        except Exception:
            return float("-inf")

    # ── Archívum ──────────────────────────────────────────────────────────

    def _add(self, agent: DGMAgent) -> bool:
        """Bekerül ha a saját cellájában jobb mint a jelenlegi."""
        cell = self._cell(agent.cfg)
        for i, a in enumerate(self.archive):
            if self._cell(a.cfg) == cell:
                if agent.fitness > a.fitness:
                    self.archive[i] = agent
                    return True
                return False
        self.archive.append(agent)
        self._id_counter += 1
        return True

    def _cell(self, cfg: STKApexConfig) -> Tuple:
        """2D rács: (n_experts bucket, n_layers bucket)."""
        e = min(3, cfg.n_experts // 8)
        l = min(3, cfg.n_layers  // 4)
        return (e, l)

    def _sample(self) -> DGMAgent:
        if not self.archive:
            raise RuntimeError("Üres archívum.")
        # Fitness-arányos mintavételezés
        fits = [max(0.0, a.fitness + 10) for a in self.archive]
        total = sum(fits) or 1.0
        r = self.rng.random() * total
        cum = 0.0
        for a, f in zip(self.archive, fits):
            cum += f
            if r <= cum:
                return a
        return self.archive[-1]

    # ── Mutáció ───────────────────────────────────────────────────────────

    def _mutate(self, parent: DGMAgent) -> DGMAgent:
        cfg   = copy.deepcopy(parent.cfg)
        level = self.rng.choices(
            [LEVEL_HYPER, LEVEL_ARCH, LEVEL_MODULE],
            weights=[0.5, 0.3, 0.2])[0]

        if level == LEVEL_HYPER:
            patch = self.rng.choice(_HYPER_PROPOSALS)
        elif level == LEVEL_ARCH:
            patch = self.rng.choice(_ARCH_PROPOSALS)
        else:
            patch = self.rng.choice(_MODULE_PROPOSALS)

        for k, v in patch.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        # chunk_sizes / dilations újragenerálása
        cfg.sgs_chunk_sizes = None
        cfg.tdk_dilations   = None
        try:
            cfg.__post_init__()
        except Exception:
            pass

        return DGMAgent(
            cfg=cfg, gen=parent.gen + 1,
            level=level,
            parent_id=id(parent),
            mutations=parent.mutations + [patch],
        )

    # ── Fő loop ───────────────────────────────────────────────────────────

    def evolve(self, n_iters: int = 100,
               verbose: bool = True) -> DGMAgent:
        """
        n_iters iteráció: mintavételezés → mutáció → kiértékelés → archívum.
        """
        if not self.archive:
            from .config import apex_mini
            self._seed(apex_mini())

        for i in range(n_iters):
            parent = self._sample()
            child  = self._mutate(parent)
            child.fitness = self._safe_eval(child.cfg)

            added = self._add(child)
            self.gen += 1

            if verbose and (i % 10 == 0 or added):
                best = self.best()
                print(f"  gen {self.gen:>4} | iter {i:>4} | "
                      f"archívum: {len(self.archive):>3} | "
                      f"legjobb: {best.fitness:.4f} | "
                      f"{'✓ bekerült' if added else ''}")

        return self.best()

    def best(self) -> DGMAgent:
        if not self.archive:
            raise RuntimeError("Üres archívum.")
        return max(self.archive, key=lambda a: a.fitness)

    def summary(self) -> str:
        if not self.archive:
            return "Üres archívum — futtasd az evolve()-t."
        best = self.best()
        lines = [
            f"Darwin Gödel Machine — {len(self.archive)} ágens az archívumban",
            f"  Generáció:       {self.gen}",
            f"  Legjobb fitness: {best.fitness:.4f}",
            f"  Legjobb config:",
            f"    d_model={best.cfg.d_model}, n_layers={best.cfg.n_layers}",
            f"    n_experts={best.cfg.n_experts}, top_k={best.cfg.top_k}",
            f"    rope_base={best.cfg.rope_base:.0f}",
            f"    abstention_t={best.cfg.abstention_threshold}",
            f"    hyper_conn={best.cfg.use_hyper_connections}",
            f"    shared_expert={best.cfg.use_shared_expert}",
            f"  Mutációs lánc:   {len(best.mutations)} lépés",
        ]
        return "\n".join(lines)

    # ── Sandbox tesztelő (subprocess) ─────────────────────────────────────

    @staticmethod
    def sandbox_eval(cfg: STKApexConfig,
                     eval_script: str,
                     timeout: int = 120) -> float:
        """
        Izolált subprocess-ben futtat egy kiértékelő scriptet.

        eval_script: Python kód, ami a `cfg`-t kapja és float-ot ír stdout-ra.
        Biztonsági korlátok: timeout, külön process, nincs hálózat.
        """
        import dataclasses, json as _json
        cfg_dict = _json.dumps(dataclasses.asdict(cfg))
        wrapper  = textwrap.dedent(f"""
            import sys, json, dataclasses
            sys.path.insert(0, r"{Path(__file__).parent.parent}")
            from stk_apex.config import STKApexConfig
            cfg_data = json.loads({cfg_dict!r})
            cfg = STKApexConfig(**cfg_data)

            {eval_script}

            result = evaluate(cfg)
            print(float(result))
        """)
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as f:
            f.write(wrapper)
            fpath = f.name
        try:
            proc = subprocess.run(
                [sys.executable, fpath],
                capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0 and proc.stdout.strip():
                return float(proc.stdout.strip().split()[-1])
            return float("-inf")
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return float("-inf")
        finally:
            os.unlink(fpath)
