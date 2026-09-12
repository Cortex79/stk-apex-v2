"""
STK-APEX 1. fázisú pretraining — Kaggle T4/P100-ra méretezve.

Korpusz: SUPERBRAIN `stage1_v6/tokens.bin`
    1 968 644 824 token, uint16, 768 hosszú packed szekvenciák (fill 99,96%)
    tokenizer: superbrain_tokenizer_24k_v2 — azonos a stk_apex/tokenizer/-rel

Futtatás:
    python training/train_stage1.py --benchmark
    python training/train_stage1.py --steps 50000 --bs 16 --accum 4
    python training/train_stage1.py --resume ckpt/last.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stk_apex import build_apex                      # noqa: E402
from stk_apex.config import apex_60m                 # noqa: E402

SEQ_LEN = 768


# ─────────────────────────────────────────────────────────────────────────────
# Adatbetöltő
# ─────────────────────────────────────────────────────────────────────────────

class PackedTokens:
    """memmap-elt uint16 token-mátrix, [rows, SEQ_LEN].

    A korpusz sorrendje tanítójel: a tények szándékosan visszatérnek benne
    (spaced repetition). Globális keverés ezt megsemmisítené, ezért csak egy
    csúszó ablakon belül keverünk — a nagyléptékű sorrend megmarad, de a
    batch-ek nem lesznek tökéletesen korreláltak.
    """

    def __init__(self, path: str, seq_len: int = SEQ_LEN,
                 window: int = 8192, seed: int = 0):
        p = Path(path)
        n_tok = p.stat().st_size // 2
        self.rows = n_tok // seq_len
        self.seq_len = seq_len
        self.data = np.memmap(p, dtype=np.uint16, mode="r",
                              shape=(self.rows, seq_len))
        self.window = window
        self.rng = np.random.default_rng(seed)

        # SEO-spam sorok kihagyása (training/filter_corpus.py állítja elő).
        # Kulcsszóhalmazok, nem folyó szöveg — rosszabbul hasznosulnak a
        # tanításban, mint bármely természetes szöveg.
        # Kaggle-n a korpusz read-only /kaggle/input alatt van, ezért a lista
        # a kód mellől is betölthető.
        self.skip = np.zeros(self.rows, dtype=bool)
        for sp in (p.parent / "skip_rows.npy",
                   Path(__file__).parent / "skip_rows.npy"):
            if sp.exists():
                idx = np.load(sp)
                self.skip[idx[idx < self.rows]] = True
                print(f"szűrés : {int(self.skip.sum()):,} spam-sor kihagyva "
                      f"({100*self.skip.mean():.2f}%)  [{sp.name}]")
                break
        else:
            print("FIGYELEM: nincs skip_rows.npy — a spam-szűrés kimarad")

    def __len__(self) -> int:
        return self.rows

    def batches(self, bs: int, start_row: int = 0) -> Iterator[torch.Tensor]:
        """Végtelen batch-folyam, ablakon belül keverve."""
        pos = start_row
        while True:
            if pos + self.window > self.rows:
                pos = 0
            idx = np.arange(pos, pos + self.window)
            idx = idx[~self.skip[idx]]
            self.rng.shuffle(idx)
            for i in range(0, len(idx) - bs + 1, bs):
                sel = np.sort(idx[i:i + bs])          # rendezve: gyorsabb memmap
                chunk = np.asarray(self.data[sel], dtype=np.int64)
                yield torch.from_numpy(chunk)
            pos += self.window


def make_xy(seq: torch.Tensor, device: str):
    seq = seq.to(device, non_blocking=True)
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Tanítás
# ─────────────────────────────────────────────────────────────────────────────

def lr_lambda(step: int, warmup: int, total: int, floor: float = 0.1):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


class Holdout:
    """A korpusz melletti holdout.jsonl — NEM a tanítóanyag vége.

    A tanítókorpuszból mintázni azért hibás, mert a betöltő sorban olvassa:
    a "holdout" előbb-utóbb tanítóadat lesz. A stage1_v6 külön holdoutot ad,
    nyelvenként kiegyensúlyozva (hu 1500 / en 1500 / hu+en 500).
    """

    def __init__(self, path: Path, seq_len: int = SEQ_LEN, max_rows: int = 1024):
        from stk_apex.tokenizer import STKTokenizer
        tok = STKTokenizer()
        self.pad = tok.pad_id
        rows = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(rows) >= max_rows:
                    break
                try:
                    txt = json.loads(line).get("input", "")
                except json.JSONDecodeError:
                    continue
                ids = tok.encode(txt)[:seq_len]
                if len(ids) >= 64:
                    rows.append(ids)
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def batches(self, bs: int):
        for i in range(0, len(self.rows) - bs + 1, bs):
            chunk = self.rows[i:i + bs]
            L = max(len(r) for r in chunk)
            x = torch.full((len(chunk), L), self.pad, dtype=torch.long)
            y = torch.full((len(chunk), L), -100, dtype=torch.long)
            for j, r in enumerate(chunk):
                t = torch.tensor(r, dtype=torch.long)
                x[j, :len(r)] = t
                y[j, :len(r) - 1] = t[1:]        # shift; a pad -100 marad
            yield x[:, :-1].contiguous(), y[:, :-1].contiguous()


@torch.no_grad()
def evaluate(model, holdout: Optional[Holdout], device: str, bs: int,
             max_batches: int = 32) -> float:
    if holdout is None or len(holdout) < bs:
        return float("nan")
    model.eval()
    tot, n = 0.0, 0
    for x, y in holdout.batches(bs):
        out = model(x.to(device), labels=y.to(device))
        tot += float(out["loss"])
        n += 1
        if n >= max_batches:
            break
    model.train()
    return tot / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/stage1_v6/tokens.bin")
    ap.add_argument("--holdout", default="",
                    help="alapból a tokens.bin melletti holdout.jsonl")
    ap.add_argument("--size", default="60m")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    # A mért korlát: 6+ rétegnél 3e-3 divergál, 1e-3-tól lefelé konvergál.
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--resume", default="")
    ap.add_argument("--benchmark", action="store_true",
                    help="60 s átbocsátás-mérés, majd kilép")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    if device == "cpu":
        torch.set_num_threads(4)

    cfg = apex_60m()
    model = build_apex(args.size, use_rag=False).to(device)
    n_param = sum(p.numel() for p in model.parameters())

    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    gpu_names = " + ".join(torch.cuda.get_device_name(i) for i in range(n_gpu)) if n_gpu else ""
    print(f"eszköz: {device}" + (f" ({gpu_names})" if gpu_names else ""))
    print(f"GPU-k: {n_gpu}")
    print(f"modell: {n_param/1e6:.2f}M paraméter, vocab={cfg.vocab_size}")

    data = PackedTokens(args.tokens)
    print(f"korpusz: {data.rows:,} sor x {SEQ_LEN} = {data.rows*SEQ_LEN/1e9:.2f}B token")

    # A holdout a korpusz MELLETT van; a tanítóanyag végéből mintázni hibás,
    # mert a betöltő sorban olvas és előbb-utóbb belelép.
    hpath = Path(args.holdout) if args.holdout else \
        Path(args.tokens).parent / "holdout.jsonl"
    holdout = None
    if hpath.exists():
        holdout = Holdout(hpath)
        print(f"holdout: {hpath.name}, {len(holdout)} sor")
    else:
        print(f"FIGYELEM: nincs holdout ({hpath}) — a kiértékelés kimarad, "
              f"NEM helyettesítjük a tanítóanyag végével")

    # Lépésszám sorban is, ne csak tokenben (slot != valódi token).
    tok_per_step = args.bs * args.accum * (SEQ_LEN - 1)
    full_epoch = data.rows * (SEQ_LEN - 1) / tok_per_step
    print(f"ütem   : {tok_per_step:,} token/lépés, "
          f"a teljes korpusz = {full_epoch:,.0f} lépés "
          f"({args.steps/full_epoch:.2f} epoch a --steps={args.steps} mellett)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup, args.steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    step0, row0 = 0, 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step0, row0 = ck["step"], ck.get("row", 0)
        print(f"folytatás: {args.resume} (lépés {step0})")

    gen = data.batches(args.bs, start_row=row0)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Benchmark ────────────────────────────────────────────────────────
    if args.benchmark:
        model.train()
        for _ in range(3):
            x, y = make_xy(next(gen), device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(x, labels=y)
                loss = out["loss"].mean() + cfg.moe_aux_loss_weight * out["moe_aux_loss"].mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()

        n_steps, seen, t0 = 0, 0, time.perf_counter()
        while time.perf_counter() - t0 < 60:
            x, y = make_xy(next(gen), device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(x, labels=y)
                loss = out["loss"].mean() + cfg.moe_aux_loss_weight * out["moe_aux_loss"].mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            n_steps += 1
            seen += x.numel()            # független számláló, ld. lent
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # A teljesítményszámot két külön úton számoljuk: a névleges alakból
        # (lépés x batch x hossz) és a ténylegesen átvitt elemekből. Eltérés
        # esetén valamelyik faktor hibás — ez a hiba korábban 20x-os és
        # 2x-es téves tok/s-t okozott.
        tps = n_steps * args.bs * (SEQ_LEN - 1) / dt
        tps_check = seen / dt
        print(f"\n  {n_steps} lépés / {dt:.1f}s   bs={args.bs}")
        print(f"  átbocsátás : {tps:,.0f} token/s"
              f"   (független számítás: {tps_check:,.0f})")
        if abs(tps - tps_check) / max(tps, 1) > 0.02:
            print("  FIGYELEM: a két számítás eltér — ellenőrizd a faktorokat!")
        for lbl, n in (("teljes korpusz (1,97B)", 1.97e9),
                       ("Chinchilla 20x (1,35B)", 1.35e9)):
            print(f"  {lbl:24s}: {n/tps/3600:6.1f} óra")
        if device == "cuda":
            print(f"  GPU memória: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        return

    # ── Tanítási ciklus ──────────────────────────────────────────────────
    model.train()
    t0, seen, running = time.perf_counter(), 0, []

    for step in range(step0, args.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            x, y = make_xy(next(gen), device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(x, labels=y)
                loss = out["loss"].mean() + cfg.moe_aux_loss_weight * out["moe_aux_loss"].mean()
            scaler.scale(loss / args.accum).backward()
            seen += x.numel()
            running.append(float(out["loss"].mean().detach()))

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if (step + 1) % 50 == 0:
            avg = sum(running[-200:]) / len(running[-200:])
            el = time.perf_counter() - t0
            print(f"  {step+1:7d}/{args.steps}  loss {avg:.4f}  "
                  f"ppl {math.exp(min(20, avg)):8.1f}  "
                  f"{seen/el:,.0f} tok/s  lr {sched.get_last_lr()[0]:.2e}",
                  flush=True)

        if (step + 1) % args.eval_every == 0:
            ev = evaluate(model, holdout, device, args.bs)
            if ev == ev:                                  # nem NaN
                print(f"  >> holdout loss {ev:.4f}  "
                      f"ppl {math.exp(min(20, ev)):.1f}", flush=True)

        if (step + 1) % args.ckpt_every == 0:
            path = ckpt_dir / "last.pt"
            m = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "step": step + 1, "row": 0, "cfg": vars(cfg)}, path)
            print(f"  >> mentve: {path} ({step+1})", flush=True)

    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save({"model": m.state_dict(), "step": args.steps,
                "cfg": vars(cfg)}, ckpt_dir / "final.pt")
    print("kész")


if __name__ == "__main__":
    main()
