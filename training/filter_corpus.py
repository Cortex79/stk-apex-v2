"""
Spam-sorok kiszűrése a packed korpuszból — dekódolás nélkül.

A tokens.bin packed formátumú, újragenerálni drága lenne. Helyette egy
kihagyandó-sor listát készítünk, amit a betöltő figyelembe vesz.

A szűrés token-id szinten megy: a kulcsszavakat NEM a korpuszban keressük,
hanem a 24 576 elemű szótárban — így a 2,56M sor átvizsgálása vektorizált
numpy-művelet, nem 2,56M BPE-dekódolás.

Futtatás:
    python training/filter_corpus.py --tokens PATH            # elemzés
    python training/filter_corpus.py --tokens PATH --write    # skip.npy írása
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stk_apex import STKTokenizer            # noqa: E402

SEQ_LEN = 768

# SEO-spam és felnőtt kulcsszavak. A cél nem tartalmi cenzúra, hanem a
# kulcsszóhalmazok kiszűrése: azok nem természetes nyelv, tehát a tanításban
# rosszabbul hasznosulnak, mint bármely folyó szöveg.
KEYWORDS = re.compile(
    r"(szex|pornó|porno|punci|fasz|geci|kurva|anál|maszti|csöcs|"
    r"meztelen|xxx|bdsm|erotik|sexvideo|szexvideo|sexfilm|szexfilm)",
    re.IGNORECASE)


def bad_token_ids(tok: STKTokenizer) -> np.ndarray:
    ids = [i for s, i in tok.token2id.items() if KEYWORDS.search(s)]
    return np.array(sorted(ids), dtype=np.uint16)


def scan(path: Path, bad: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    rows = path.stat().st_size // 2 // SEQ_LEN
    data = np.memmap(path, dtype=np.uint16, mode="r", shape=(rows, SEQ_LEN))
    counts = np.zeros(rows, dtype=np.int16)
    for i in range(0, rows, chunk):
        blk = np.asarray(data[i:i + chunk])
        counts[i:i + chunk] = np.isin(blk, bad).sum(axis=1)
        if (i // chunk) % 20 == 0:
            print(f"    {i:,}/{rows:,}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--threshold", type=int, default=12,
                    help="ennyi spam-token/sor fölött kihagyjuk")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    tok = STKTokenizer()
    bad = bad_token_ids(tok)
    print(f"  szótár-találat: {len(bad)} token a {tok.vocab_size}-ből")
    print(f"  példák: {[tok.id2token[int(i)] for i in bad[:12]]}")

    path = Path(args.tokens)
    counts = scan(path, bad)

    print(f"\n  sorok: {len(counts):,}")
    print(f"  {'küszöb':>8s} {'kihagyott sor':>14s} {'arány':>8s} {'token':>10s}")
    for t in (4, 8, 12, 20, 32):
        n = int((counts >= t).sum())
        print(f"  {t:>8d} {n:>14,} {100*n/len(counts):7.2f}% "
              f"{n*SEQ_LEN/1e6:9.0f}M")

    skip = counts >= args.threshold
    print(f"\n  választott küszöb {args.threshold}: "
          f"{int(skip.sum()):,} sor ({100*skip.mean():.2f}%), "
          f"{int(skip.sum())*SEQ_LEN/1e6:.0f}M token kihagyva")

    if args.write:
        out = path.parent / "skip_rows.npy"
        np.save(out, np.flatnonzero(skip).astype(np.int64))
        print(f"  mentve: {out}")
    else:
        print("  (--write nélkül nem írtam fájlt)")


if __name__ == "__main__":
    main()
