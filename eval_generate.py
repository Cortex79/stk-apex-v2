"""
eval_generate.py — STK-APEX szöveg-generálás kiértékelés

Futtatás:
  python eval_generate.py --ckpt ckpt/step6000.pt
  python eval_generate.py --ckpt ckpt/step6000.pt --max-new 200 --top-p 0.9

Kimenetek:
  - HU/EN perplexitás (holdout.jsonl alapján)
  - Szöveg-generálás fix promptokból
  - distinct-1, distinct-2, rep-rate-4g metrikák
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stk_apex import build_apex
from stk_apex.config import apex_60m
from stk_apex.tokenizer import STKTokenizer

HOLDOUT = Path(__file__).parent / "data" / "stage1_v6" / "holdout.jsonl"
SUPERBRAIN_HOLDOUT = Path(r"C:\Users\tibor\Desktop\SUPERBRAIN-STK2-274M\data\stage1_v6\holdout.jsonl")

HU_PROMPTS = [
    "A mesterséges intelligencia fejlődése azt mutatja,",
    "Magyarország fővárosa Budapest, ahol",
    "Az időjárás-előrejelzés szerint holnap",
    "Régen az emberek úgy éltek, hogy",
    "A tudomány legújabb eredményei szerint",
]
EN_PROMPTS = [
    "Artificial intelligence has transformed the way",
    "The history of Europe shows that",
    "In recent years, scientists have discovered",
    "Language models are trained on large",
    "Once upon a time, in a land far away,",
]


def detect_lang(text: str) -> str:
    hu = sum(text.count(c) for c in "áéíóöőúüűÁÉÍÓÖŐÚÜŰ")
    return "hu" if hu > len(text) * 0.01 else "en"


def top_p_sample(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    mask = (cum - sorted_probs) >= top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1)
    return int(sorted_idx[idx])


@torch.no_grad()
def generate(model, tok: STKTokenizer, prompt: str, max_new: int,
             top_p: float, temperature: float, device: str) -> str:
    ids = tok.encode(prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        out = model(x)
        logits = out["logits"][:, -1, :]
        nxt = top_p_sample(logits[0], top_p, temperature)
        if nxt in (tok.eos_id,):
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if x.shape[1] > 768:
            break
    generated_ids = x[0, len(ids):].tolist()
    return tok.decode(generated_ids)


def diversity_metrics(texts: list[str], tok: STKTokenizer) -> dict:
    all_uni, all_bi, all_4g = Counter(), Counter(), Counter()
    for t in texts:
        ids = tok.encode(t)
        all_uni.update(ids)
        all_bi.update(zip(ids, ids[1:]))
        all_4g.update(zip(ids, ids[1:], ids[2:], ids[3:]))
    n_tok = sum(all_uni.values())
    rep4 = sum(c for c in all_4g.values() if c > 1) / max(1, sum(all_4g.values()))
    return {
        "distinct-1": len(all_uni) / max(1, n_tok),
        "distinct-2": len(all_bi) / max(1, sum(all_bi.values())),
        "rep-rate-4g": rep4,
        "avg_tokens": n_tok / max(1, len(texts)),
    }


@torch.no_grad()
def eval_perplexity(model, tok: STKTokenizer, holdout_path: Path,
                    device: str, max_rows: int = 512) -> dict:
    if not holdout_path.exists():
        return {}
    losses = {"hu": [], "en": [], "other": []}
    with open(holdout_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            try:
                txt = json.loads(line).get("input", "")
            except json.JSONDecodeError:
                continue
            ids = tok.encode(txt)[:767]
            if len(ids) < 16:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            out = model(x, labels=y)
            lang = detect_lang(txt)
            losses[lang].append(float(out["loss"]))

    result = {}
    for lang, ls in losses.items():
        if ls:
            avg = sum(ls) / len(ls)
            result[lang] = {"loss": avg, "ppl": math.exp(min(20, avg)), "n": len(ls)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--holdout", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = STKTokenizer()

    ckpt_path = Path(args.ckpt)
    print(f"checkpoint : {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ck.get("step", "?")
    print(f"lépés      : {step}")

    model = build_apex("60m", use_rag=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"paraméter  : {n_param/1e6:.2f}M\n")

    # ── Perplexitás ──────────────────────────────────────────────────────────
    hpath = Path(args.holdout) if args.holdout else (
        HOLDOUT if HOLDOUT.exists() else SUPERBRAIN_HOLDOUT)
    print(f"holdout    : {hpath}")
    ppl = eval_perplexity(model, tok, hpath, device)
    if ppl:
        print("\n── Perplexitás ─────────────────────────────────")
        for lang, d in ppl.items():
            print(f"  {lang:5s}  loss {d['loss']:.4f}  ppl {d['ppl']:7.1f}  ({d['n']} sor)")
    else:
        print("  (nincs holdout)")

    # ── Generálás ────────────────────────────────────────────────────────────
    print("\n── Magyar generálás ────────────────────────────")
    hu_texts = []
    for prompt in HU_PROMPTS:
        gen = generate(model, tok, prompt, args.max_new, args.top_p,
                       args.temperature, device)
        full = prompt + gen
        hu_texts.append(gen)
        print(f"\n[PROMPT] {prompt}")
        print(f"[GEN]    {gen[:300]}")

    print("\n── Angol generálás ─────────────────────────────")
    en_texts = []
    for prompt in EN_PROMPTS:
        gen = generate(model, tok, prompt, args.max_new, args.top_p,
                       args.temperature, device)
        en_texts.append(gen)
        print(f"\n[PROMPT] {prompt}")
        print(f"[GEN]    {gen[:300]}")

    # ── Diverzitás ───────────────────────────────────────────────────────────
    print("\n── Diverzitás metrikák ─────────────────────────")
    for label, texts in [("HU", hu_texts), ("EN", en_texts)]:
        m = diversity_metrics(texts, tok)
        print(f"  {label}  distinct-1={m['distinct-1']:.3f}  "
              f"distinct-2={m['distinct-2']:.3f}  "
              f"rep-rate-4g={m['rep-rate-4g']:.3f}  "
              f"avg_tok={m['avg_tokens']:.0f}")


if __name__ == "__main__":
    main()
