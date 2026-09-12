"""
Ablation — helyettesíti-e a súlymegosztott rekurrens mélység a paramétert?

A kérdés: a Huginn-vonal (Geiping 2025) 3,5B-en mutatta, hogy a látens
rekurrencia paraméter helyett compute-tal vásárol mélységet. Lejön-e ez
~1M skálára, CPU-n? Ha igen, a 60M modell magja 3× kisebb lehet.

Karok:
  A   6 blokk × 1 iteráció   baseline, független rétegek
  B   2 blokk × 3 iteráció   ~1/3 paraméter, AZONOS effektív mélység
  C   6 blokk × 3 iteráció   iso-paraméter, 3× effektív mélység

  A vs B  -> helyettesíti-e a rekurrencia a paramétert?
  A vs C  -> vásárol-e a compute többletképességet fix paraméter mellett?

Feladatok (mélység- és kapacitás-igény szétválasztására):
  recall  asszociatív felidézés     kapacitás-igényes -> a rekurrencia elvileg NEM segít
  chain   iterált x=(x*a+b) mod p   szekvenciális mélység -> elvileg segít
  dyck    Dyck-2 stack-teteje       stack-mélység -> elvileg segít

Ha B ~ A a mélység-feladatokon, de B << A a recall-on, akkor pontosan tudjuk,
mit vásárol a rekurrencia — és mit nem.

Futtatás:
  python experiments/ablation_recurrent.py --quick        # ~3 perc, tájékozódó
  python experiments/ablation_recurrent.py --steps 3000   # teljes
  python experiments/ablation_recurrent.py --task chain --steps 4000
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stk_apex.core import RMSNorm, RoPEAttnBlock, SGSBlock  # noqa: E402

torch.set_num_threads(4)

Batch = Tuple[torch.Tensor, torch.Tensor]   # (ids [B,L], target [B])


# ─────────────────────────────────────────────────────────────────────────────
# Szintetikus feladatok — mind on-the-fly generált, így nincs overfitting
# ─────────────────────────────────────────────────────────────────────────────

def gen_recall(bs: int, g: torch.Generator,
               n_pairs: int = 8, n_keys: int = 16, n_vals: int = 16) -> Batch:
    """k1 v1 k2 v2 ... kn vn SEP kq  ->  vq

    Kapacitás-igényes: a modellnek n_pairs kulcs-érték kötést kell egyszerre
    tartania. Ez az a képesség, amiben a tiszta SSM-ek elbuknak (Mamba-gap).
    """
    SEP = n_keys + n_vals
    seqs, tgts = [], []
    for _ in range(bs):
        keys = torch.randperm(n_keys, generator=g)[:n_pairs]
        vals = torch.randint(0, n_vals, (n_pairs,), generator=g) + n_keys
        s: List[int] = []
        for k, v in zip(keys.tolist(), vals.tolist()):
            s += [k, v]
        qi = int(torch.randint(0, n_pairs, (1,), generator=g))
        s += [SEP, int(keys[qi])]
        seqs.append(s)
        tgts.append(int(vals[qi]))
    return torch.tensor(seqs), torch.tensor(tgts)


def gen_chain(bs: int, g: torch.Generator,
              n_steps: int = 6, p: int = 7) -> Batch:
    """x0 a1 b1 a2 b2 ... SEP  ->  x_n,  ahol x_{i+1} = (x_i*a_i + b_i) mod p

    Az affin leképezés NEM kommutatív, tehát a lánc nem bontható párhuzamos
    redukcióra — minden lépés egy sorosított számítási mélységet igényel.
    """
    SEP = p
    seqs, tgts = [], []
    for _ in range(bs):
        x = int(torch.randint(0, p, (1,), generator=g))
        s: List[int] = [x]
        for _ in range(n_steps):
            a = int(torch.randint(1, p, (1,), generator=g))
            b = int(torch.randint(0, p, (1,), generator=g))
            s += [a, b]
            x = (x * a + b) % p
        s.append(SEP)
        seqs.append(s)
        tgts.append(x)
    return torch.tensor(seqs), torch.tensor(tgts)


def gen_dyck(bs: int, g: torch.Generator,
             length: int = 16, n_types: int = 2) -> Batch:
    """Dyck-2 prefix SEP  ->  a stack tetejét lezáró zárójel.

    A stack sosem ürül ki (csak len>1 esetén zárunk), így a válasz mindig
    definiált és a szekvenciahossz fix.
    """
    SEP = 2 * n_types
    seqs, tgts = [], []
    for _ in range(bs):
        stack: List[int] = []
        s: List[int] = []
        for _ in range(length):
            close = len(stack) > 1 and float(torch.rand(1, generator=g)) < 0.45
            if close:
                s.append(n_types + stack.pop())
            else:
                t = int(torch.randint(0, n_types, (1,), generator=g))
                stack.append(t)
                s.append(t)
        s.append(SEP)
        seqs.append(s)
        tgts.append(n_types + stack[-1])
    return torch.tensor(seqs), torch.tensor(tgts)


# A feladat nehézségét kalibrálni kell: padlón (a modell semmit nem tanul) és
# plafonon (mindenki megoldja) a karok nem különböztethetők meg. A cél, hogy a
# baseline kar a 60–85% sávba essen — ott van a legnagyobb felbontás.
TASK_DEFAULTS = {
    "recall": {"n_pairs": 8, "n_keys": 16, "n_vals": 16},
    "chain":  {"n_steps": 6, "p": 7},
    "dyck":   {"length": 16, "n_types": 2},
}
GENERATORS = {"recall": gen_recall, "chain": gen_chain, "dyck": gen_dyck}

# Kalibrációs rács — 1. kör eredménye (1200 lépés, bs=32, A kar):
#   recall  3 pár -> 100%,  5 pár -> 29,3%,  8 pár -> 25,4%
#           Éles fázisátmenet 3 és 5 pár között; a felbontás n_pairs=4 körül van.
#   chain   a legkönnyebb (2 lépés, p=5) is padlón: 23,0% vs 20,0% véletlen.
#           Moduláris aritmetika = grokking-feladat (Power 2022), ~10^5 lépést
#           igényelne. KIVÉVE a rácsból: ezen a compute-budgeten nem mérhető.
#   dyck    48 hossznál is 93,8% — a "stack teteje" kérdés asszociatív keresés,
#           nem mély számítás. Hosszabb szekvencia kell a plafon alá.
DIFFICULTY = {
    "recall": [{"n_pairs": 4, "n_keys": 10, "n_vals": 10},
               {"n_pairs": 4, "n_keys": 14, "n_vals": 14},
               {"n_pairs": 5, "n_keys": 10, "n_vals": 10}],
    "dyck":   [{"length": 64, "n_types": 3},
               {"length": 96, "n_types": 4},
               {"length": 128, "n_types": 4}],
}


def make_task(name: str, **kw):
    """-> (generátor bs+g-vel hívható, vocab_size, véletlen szintű pontosság)"""
    d = dict(TASK_DEFAULTS[name])
    d.update(kw)
    fn = partial(GENERATORS[name], **d)
    if name == "recall":
        return fn, d["n_keys"] + d["n_vals"] + 1, 1 / d["n_vals"]
    if name == "chain":
        return fn, d["p"] + 1, 1 / d["p"]
    return fn, 2 * d["n_types"] + 1, 1 / d["n_types"]


# ─────────────────────────────────────────────────────────────────────────────
# Modell — a mag blokkjai N-szer iterálva, input-injektálással
# ─────────────────────────────────────────────────────────────────────────────

class AblationModel(nn.Module):
    """STK-APEX mag-blokkok minimális burkolóban.

    n_iter > 1 esetén UGYANAZOK a blokkok futnak le többször (súlymegosztás).
    Minden iterációban visszainjektáljuk az embeddinget — enélkül a rekurrens
    változat nem tanul, mert az eredeti bemenet elvész a mélységben
    (Universal Transformer, Dehghani 2019).
    """

    def __init__(self, vocab: int, d_model: int = 128, n_blocks: int = 6,
                 n_iter: int = 1, d_ff: int = 384, attn_every: int = 3,
                 chunk_size: int = 8):
        super().__init__()
        self.n_iter = n_iter
        self.embed = nn.Embedding(vocab, d_model)

        blocks: List[nn.Module] = []
        for i in range(n_blocks):
            # Az UTOLSÓ blokk mindig attention. Enélkül a 2-blokkos karokba
            # egyetlen attention sem kerülne ((i+1)%3 sosem 0), így tiszta SSM-et
            # hasonlítanánk hibridhez — és épp az attention hiánya dönti el az
            # asszociatív felidézést (Mamba-gap).
            if (attn_every > 0 and (i + 1) % attn_every == 0) or i == n_blocks - 1:
                blocks.append(RoPEAttnBlock(
                    d_model, n_heads=4, n_kv_heads=2, d_ff=d_ff,
                    dropout=0.0, rope_base=10_000.0, use_hyper=True))
            else:
                blocks.append(SGSBlock(
                    d_model, d_ff, kernel_size=5, dilation=1, dropout=0.0,
                    chunk_size=chunk_size, use_moe=False, use_hyper=True))
        self.blocks = nn.ModuleList(blocks)

        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        # Tied súly mellett az nn.Embedding N(0,1) alapértelmezése a logitokat
        # ~sqrt(d_model)-szeresére fújja (CE 100+ ln(V)~2 helyett), ami a
        # rekurrens karokat aránytalanul bünteti.
        nn.init.normal_(self.embed.weight, std=0.02)
        self.head.weight = self.embed.weight          # tied

    def _core(self, h: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if isinstance(blk, RoPEAttnBlock):
                h, _ = blk(h)
            else:
                h = blk(h)
        return h

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x0 = self.embed(ids)
        h = x0
        for i in range(self.n_iter):
            h = self._core(h if i == 0 else h + x0)
        return self.head(self.norm(h))


ARMS = {
    #  név : (n_blocks, n_iter, leírás)
    "A": (6, 1, "6 blokk x 1 iter  — baseline"),
    "B": (2, 3, "2 blokk x 3 iter  — 1/3 param, azonos mélység"),
    "C": (6, 3, "6 blokk x 3 iter  — iso-param, 3x mélység"),
    # D nélkül B eredménye értelmezhetetlen: ugyanannyi paraméter, de rekurrencia
    # nélkül. B-D különbség = a rekurrencia hatása. A-D különbség = a méreté.
    "D": (2, 1, "2 blokk x 1 iter  — B kontrollja, rekurrencia nélkül"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Tanítás / kiértékelés
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, gen_fn: Callable, bs: int, n_batches: int,
             seed: int = 12345) -> float:
    model.eval()
    g = torch.Generator().manual_seed(seed)      # fix eval-halmaz minden karra
    correct = total = 0
    for _ in range(n_batches):
        ids, tgt = gen_fn(bs, g)
        pred = model(ids)[:, -1].argmax(-1)
        correct += int((pred == tgt).sum())
        total += tgt.numel()
    model.train()
    return correct / total


def train_arm(arm: str, task: str, task_spec: tuple, steps: int, bs: int,
              lr: float, seed: int) -> dict:
    gen_fn, vocab, chance = task_spec
    n_blocks, n_iter, desc = ARMS[arm]

    torch.manual_seed(seed)
    model = AblationModel(vocab, n_blocks=n_blocks, n_iter=n_iter)
    n_param = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) *                  # warmup
                       (0.5 * (1 + math.cos(math.pi * s / steps))))  # cosine

    g = torch.Generator().manual_seed(seed + 999)
    t0 = time.perf_counter()
    losses: List[float] = []

    for step in range(steps):
        ids, tgt = gen_fn(bs, g)
        loss = F.cross_entropy(model(ids)[:, -1], tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))

        if (step + 1) % max(1, steps // 4) == 0:
            recent = sum(losses[-50:]) / len(losses[-50:])
            print(f"      {step+1:5d}/{steps}  loss {recent:.4f}", flush=True)

    acc = evaluate(model, gen_fn, bs, n_batches=8)
    return {
        "arm": arm, "desc": desc, "task": task, "param": n_param,
        "acc": acc, "chance": chance,
        "loss": sum(losses[-50:]) / 50, "sec": time.perf_counter() - t0,
    }


def calibrate(steps: int, bs: int, lr: float, seed: int) -> None:
    """Csak az A kar, minden nehézségi fokon — hol esik a 60–85% sávba?

    Ez a mérőműszer hitelesítése. Padlón és plafonon a karok közti különbség
    nem mérhető, tehát az ablationnek csak kalibrált feladaton van értelme.
    """
    print(f"\nKalibráció — A kar, {steps} lépés, bs={bs}")
    print(f"{'feladat':8s} {'beállítás':34s} {'véletlen':>9s} {'A kar':>8s}  sáv")
    print("-" * 72)
    for task, presets in DIFFICULTY.items():
        for p in presets:
            spec = make_task(task, **p)
            r = train_arm("A", task, spec, steps, bs, lr, seed)
            acc, chance = r["acc"], r["chance"]
            zone = ("PADLO" if acc < chance + 0.10 else
                    "PLAFON" if acc > 0.92 else
                    "HASZNALHATO" if acc > 0.55 else "gyenge")
            print(f"{task:8s} {str(p):34s} {chance*100:8.1f}% "
                  f"{acc*100:7.1f}%  {zone}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=[*TASK_DEFAULTS, "all"])
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="rövid tájékozódó futás (600 lépés, bs=32)")
    ap.add_argument("--calibrate", action="store_true",
                    help="nehézségi rács végigmérése az A karral")
    ap.add_argument("--set", default="",
                    help="feladat-paraméterek, pl. length=32,n_types=3")
    args = ap.parse_args()

    if args.quick:
        args.steps, args.bs = 600, 32

    if args.calibrate:
        calibrate(args.steps, args.bs, args.lr, args.seed)
        return

    overrides = {}
    for kv in filter(None, args.set.split(",")):
        k, v = kv.split("=")
        overrides[k] = int(v)

    tasks = list(TASK_DEFAULTS) if args.task == "all" else [args.task]
    arms = args.arms.split(",")
    results: List[dict] = []

    print(f"\nAblation — rekurrens mélység vs paraméter")
    print(f"lépés={args.steps}  batch={args.bs}  lr={args.lr}  szál=4\n")

    for task in tasks:
        spec = make_task(task, **overrides)
        print(f"  [{task}]  (véletlen szint: {spec[2]*100:.1f}%)")
        for arm in arms:
            print(f"    {arm}: {ARMS[arm][2]}")
            r = train_arm(arm, task, spec, args.steps, args.bs, args.lr,
                          args.seed)
            results.append(r)
            print(f"      -> pontosság {r['acc']*100:5.1f}%   "
                  f"{r['param']/1e6:.3f}M param   {r['sec']:.0f}s\n", flush=True)

    print("\n" + "=" * 72)
    print(f"{'feladat':8s} {'kar':4s} {'param':>9s} {'pontosság':>10s} "
          f"{'vs véletlen':>12s} {'idő':>7s}")
    print("-" * 72)
    for r in results:
        lift = (r["acc"] - r["chance"]) / (1 - r["chance"])
        print(f"{r['task']:8s} {r['arm']:4s} {r['param']/1e6:8.3f}M "
              f"{r['acc']*100:9.1f}% {lift*100:11.1f}% {r['sec']:6.0f}s")
    print("=" * 72)

    for task in tasks:
        rs = {r["arm"]: r for r in results if r["task"] == task}
        print(f"\n[{task}]")

        # A REKURRENCIA hatása: azonos paraméter, csak az iterációszám tér el.
        # Ez az egyetlen összehasonlítás, ami a rekurrenciát izolálja.
        if "B" in rs and "D" in rs:
            d = (rs["B"]["acc"] - rs["D"]["acc"]) * 100
            print(f"  B-D  (2 blokk, 3 iter vs 1 iter, azonos param): {d:+.1f}pp"
                  f"  ->  {'a rekurrencia vásárol képességet' if d > 3 else 'a rekurrencia nem számít'}")
        if "C" in rs and "A" in rs:
            d = (rs["C"]["acc"] - rs["A"]["acc"]) * 100
            print(f"  C-A  (6 blokk, 3 iter vs 1 iter, azonos param): {d:+.1f}pp"
                  f"  ->  {'a rekurrencia vásárol képességet' if d > 3 else 'a rekurrencia nem számít'}")

        # A MÉRET hatása: azonos iterációszám, eltérő blokkszám.
        if "A" in rs and "D" in rs:
            d = (rs["A"]["acc"] - rs["D"]["acc"]) * 100
            print(f"  A-D  (6 blokk vs 2 blokk, mindkettő 1 iter): {d:+.1f}pp"
                  f"  ->  {'a paraméter számít' if d > 3 else 'a paraméter NEM köt ezen a skálán'}")

        # Csak ha a rekurrencia és a méret hatása is izolált, értelmezhető:
        if all(k in rs for k in "ABD"):
            rec = (rs["B"]["acc"] - rs["D"]["acc"]) * 100
            siz = (rs["A"]["acc"] - rs["D"]["acc"]) * 100
            if siz <= 3:
                print("  => A modellméret nem köt ezen a skálán; a B>A különbség "
                      "optimalizálási hatás, nem kapacitásé.")
            elif rec >= siz - 3:
                print("  => A rekurrencia a paraméter-növelés ÉRDEMI helyettesítője.")
            else:
                print("  => A rekurrencia csak részben pótolja a paramétert.")


if __name__ == "__main__":
    main()
