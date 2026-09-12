"""
Kaggle-csomag készítése.

Két külön feltöltendő darab:
  1. kód       — stk_apex csomag + tanítóscript (~2 MB), Kaggle Dataset
  2. korpusz   — tokens.bin, Kaggle Dataset

Az első körben elég egy korpusz-MINTA: a T4 átbocsátásának méréséhez és a
pipeline validálásához nem kell a teljes 3,94 GB feltöltése. A teljes korpusz
csak akkor megy fel, ha a benchmark rendben volt.

Futtatás:
    python training/make_kaggle_bundle.py --sample-mb 200
    python training/make_kaggle_bundle.py --full
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEQ_LEN = 768
DEFAULT_TOKENS = Path(
    r"C:\Users\tibor\Desktop\SUPERBRAIN-STK2-274M\data\stage1_v6\tokens.bin")


def build_code_zip(out: Path) -> Path:
    """stk_apex csomag + tanítóscript egyetlen zipbe."""
    zip_path = out / "stk_apex_code.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT / "stk_apex").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(ROOT))
        for name in ("train_stage1.py", "filter_corpus.py", "skip_rows.npy"):
            f = ROOT / "training" / name
            if f.exists():
                z.write(f, f"training/{name}")
    return zip_path


def slice_tokens(src: Path, dst: Path, mb: int | None) -> tuple[int, int]:
    """Az első N sor kimásolása; mb=None esetén teljes másolat."""
    n_tok = src.stat().st_size // 2
    rows = n_tok // SEQ_LEN
    if mb is None:
        shutil.copy2(src, dst)
        return rows, src.stat().st_size

    keep = min(rows, (mb * 1_000_000) // (SEQ_LEN * 2))
    data = np.memmap(src, dtype=np.uint16, mode="r", shape=(rows, SEQ_LEN))
    with open(dst, "wb") as f:
        for i in range(0, keep, 4096):                # darabolva, hogy a RAM bírja
            np.asarray(data[i:min(i + 4096, keep)]).tofile(f)
    return keep, dst.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=str(DEFAULT_TOKENS))
    ap.add_argument("--out", default=str(ROOT / "kaggle_bundle"))
    ap.add_argument("--sample-mb", type=int, default=200)
    ap.add_argument("--full", action="store_true",
                    help="teljes korpusz másolása minta helyett")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    code = build_code_zip(out)
    print(f"  kód      : {code}  ({code.stat().st_size/1e6:.1f} MB)")

    src = Path(args.tokens)
    if not src.exists():
        print(f"  HIBA: nincs meg a korpusz: {src}")
        return

    dst = out / "tokens.bin"
    mb = None if args.full else args.sample_mb
    rows, size = slice_tokens(src, dst, mb)
    print(f"  korpusz  : {dst}  ({size/1e6:.1f} MB, {rows:,} sor, "
          f"{rows*SEQ_LEN/1e6:.0f}M token)")

    print(f"""
  Kaggle lepesek
  --------------
  1. kaggle.com/datasets -> New Dataset -> töltsd fel:
       {code.name}   (nevezd el: stk-apex-code)
       {dst.name}    (nevezd el: stk-apex-corpus)

  2. New Notebook -> Add Data -> a két dataset
     Settings -> Accelerator: GPU T4 x2  (vagy P100)

  3. A notebook első cellája:

     !unzip -q /kaggle/input/stk-apex-code/stk_apex_code.zip -d /kaggle/working
     %cd /kaggle/working
     !python training/train_stage1.py --benchmark --bs 16 \\
         --tokens /kaggle/input/stk-apex-corpus/tokens.bin

  4. Ha a benchmark rendben: --full csomag feltöltése és éles tanítás:

     !python training/train_stage1.py --steps 50000 --bs 16 --accum 4 \\
         --tokens /kaggle/input/stk-apex-corpus/tokens.bin \\
         --ckpt-dir /kaggle/working/ckpt

     A session 12 óra után leáll -> a ckpt/last.pt-t mentsd le Kaggle
     Output-ként, és a következő futásban add meg: --resume ckpt/last.pt
""")


if __name__ == "__main__":
    main()
