# STK-APEX v2

Hibrid SSM + Attention nyelvi modell, magyar-angol fókusszal, CPU-ra méretezve.

**Állapot: az infrastruktúra kész, a modell NINCS betanítva.** A súlyok
véletlenszerűek — a modell ma nem generál értelmes szöveget. A pretraining
Kaggle GPU-n futtatható (ld. [Tanítás](#tanítás)).

---

## Architektúra

67,74M paraméter (~53M aktív), 12 réteg:

```
  9 x SGSBlock        Szelektív Gátolt Állapot (SSM) + Spatial Dynamic Kernel
  3 x RoPEAttnBlock   GQA: 8 Q-fej / 2 KV-fej, RoPE base 500 000
  3 rétegben MoE      16 routed expert, top-3, + 1 mindig aktív shared
  minden blokkban     RMSNorm, SwiGLU, HyperResidual
```

Kognitív modulok a törzs fölött: `temporal_brain` (1,59M), `reason_compiler`
(0,14M), `reality_anchor` (konfidenciabecslés, abstention küszöb 0,75).
Együtt 1,72M — a modell 2,5%-a.

Miért hibrid, és nem tiszta SSM: az attention-rétegek nem kompromisszumot
jelentenek, hanem szükségesek. A tiszta SSM-ek asszociatív felidézésben
elbuknak (Mamba-gap), ezért **minden** 2026-os hibrid megtartja őket.

## Tokenizáló

`stk_apex/tokenizer/superbrain_tokenizer_24k_v2.json`, vocab **24 576**.
Betöltés a saját `STKTokenizer`-rel — a HF `tokenizers` NEM tudja beolvasni
(saját `token2id` formátum), és a `_hf.json` változat rossz konverzió.

| tokenizáló | vocab | HU tok/szó | EN tok/szó | HU kar/token |
|---|---|---|---|---|
| cl100k (GPT-4) | 100 277 | 4,27 | 1,13 | 2,32 |
| **24k_v2** | **24 576** | **1,96** | 1,21 | **5,04** |

2,18× tömörebb magyaron a szótár negyedével. Ez közvetlenül sebesség: a
felhasználó szót olvas, nem tokent.

Ismert hibái (`SUPERBRAIN/TANULSAGOK.md` 3.5): az alap-ábécé kimaradt a
szótárból (12% byte-fallback), és a produktív magyar toldalékok nincsenek
önálló tokenként. Javításuk új szótárt, tehát nulláról tanítást igényelne.

## Mért teljesítmény

Ryzen 3 PRO 2200G (4 mag), 16 GB RAM, CPU-only:

| | érték |
|---|---|
| Modell fp32 | 271 MB |
| Generálás | 22,2 tok/s |
| **Magyar szó/s** | **11,3** |
| Tanítás | 190 token/s |
| Memória-sávszélesség | 15,0 GB/s |

Az inferencia **memória-sávszélesség-korlátos**, nem compute-korlátos (a
modell 71× nagyobb a 4 MB-os L3-nál). Ebből következik, hogy az INT8
kvantálás **nem gyorsít** (mérve: 17,0 tok/s < 19,2 FP32) — a PyTorch
dinamikus kvantálás csak `nn.Linear`-t céloz, a szűk keresztmetszet viszont
az SGS scan.

CPU-n a teljes pretraining 82–207 nap lenne, ezért megy GPU-ra.

---

## Struktúra

```
stk_apex/
  config.py         STKApexConfig, apex_60m(), apex_mini()
  core.py           SGSBlock, RoPEAttnBlock, MoEFeedForward, SelectiveGatedState
  model.py          STKApex — forward(labels=...), generate() KV-cache-sel
  train_utils.py    TIAR trajektória-súlyozott RLVR
  dgm.py            Darwin-Gödel önjavító ciklus
  abstention.py     kalibrált tartózkodás
  rag.py            RAGRetriever (CSAK retriever, nincs cross-attention)
  tokenizer/        STKTokenizer + 24k_v2 szótár
  assistant/        ReAct ágens: 20 eszköz, SQLite epizodikus memória
training/
  train_stage1.py       pretraining, Kaggle-re méretezve
  make_kaggle_bundle.py csomagoló
experiments/
  ablation_recurrent.py rekurrens mélység ablation (ág lezárva, ld. lent)
tests/test_apex.py      43 teszt
```

## Használat

```bash
pytest tests/ -q                      # 43 teszt
python training/train_stage1.py --benchmark
```

```python
from stk_apex import build_apex
from stk_apex.tokenizer import STKTokenizer
model = build_apex("60m", use_rag=False)
ids   = STKTokenizer().encode("Szia!")
```

Az asszisztens (`AssistantAgent`) betanított modellt igényel — ma a
tool-réteg működik, a generálás nem.

---

## Tanítás

Korpusz: `SUPERBRAIN-STK2-274M/data/stage1_v6/`

```
tokens.bin      1 968 644 824 token, uint16, 768-as packed szekvenciák
holdout.jsonl   3 500 sor (hu 1500 / en 1500 / hu+en 500)
tokenizer       superbrain_tokenizer_24k_v2 — azonos a bekötöttel
```

1,97B token a 68M modellhez = a Chinchilla-optimum 1,45-szerese, ami kis
modellnél előnyös tartomány.

**A korpusz sorrendje tanítójel** — a tények szándékosan visszatérnek benne.
A betöltő ezért csak egy 8192 soros csúszó ablakon belül kever, globálisan
soha.

**Spam-szűrés.** A korpusz ~2,3%-a SEO-spam (felnőtt kulcsszóhalmazok,
11%-os kulcsszó-sűrűséggel) — nem folyó szöveg, tehát rosszabbul hasznosul a
tanításban, mint bármely természetes szöveg. A `training/filter_corpus.py`
token-id szinten azonosítja (a 24 576-os szótárban 53 érintett token), és
sor-listát ír; a betöltő ezeket kihagyja: **59 500 sor, 46M token**.

Az ellenőrzés két független módszerrel egyezett (mintavétel: 2,20% / 43M,
teljes szkennelés: 2,32% / 46M). A packed sorok vegyesek lehetnek — egy
kidobott sor eleje néha legitim szöveg —, de ez a korpusz ~0,7%-a.

Tanítás előtt érdemes ellenőrizni, hogy a korpusz és a tokenizáló egyezik:
dekódolj pár sort a `tokens.bin`-ből. Ha értelmes szöveget ad, egyeznek; ha
zagyvaságot, a tanítás csendben rosszat tanulna.

Kaggle (30 GPU-óra/hét ingyen, 12 órás session):

1. `kaggle_bundle/stk_apex_code.zip` → Dataset
2. `kaggle_bundle/stk_apex_train.ipynb` → Import Notebook
3. Accelerator: GPU T4 ×2 vagy P100
4. A benchmark-cella megadja a valódi átbocsátást; onnan tervezhető a kvóta
5. Folytatás: `--resume ckpt/last.pt`

### Tanítási korlátok (mérve)

| | |
|---|---|
| **Learning rate** | `≤ 1e-3` 6+ rétegnél. `3e-3`-nál a 6 rétegű modell **nem tanul** (41,8% vs 100%). Ha egy futás nem konvergál, az lr az első gyanúsított, ne az architektúra. |
| **Holdout** | Mindig a `holdout.jsonl`-ből, soha a tanítókorpusz végéből — a betöltő sorban olvas, és előbb-utóbb belelép. |
| **Kétfázisú tanítás** | **Ne.** A 274M-nél 6× nyereség volt, itt mérve 1,00× — ott a kognitív modulok 136M-et vittek, itt 1,72M-et. |

---

## Lezárt ágak

**Rekurrens mélység** (súlymegosztott iteráció, Huginn-vonal). Négy kísérleti
kör után sem sikerült érvényesen eldönteni; minden kör metodológiai hibát
tárt fel, nem eredményt (kalibrálatlan feladatok → hiányzó attention az egyik
karban → hiányzó kontroll kar → lr-konfundálás). Nem költséghatékony ezen a
hardveren folytatni, és a fő terv nem függ tőle. Ld. `experiments/`.

## Ami hiányzik

- **Betanított súlyok** — ez az egyetlen dolog, ami a modellt használhatóvá teszi
- **RETRO-integrált retrieval** — a `rag.py` ma csak retriever. A tudás
  kivétele a súlyokból (chunked cross-attention) az egyetlen ismert út, amivel
  egy 68M modell egy 10× nagyobb közelébe kerülhet: a különbség érdemi
  tartalma ~135 MB memorizált tény, ami indexbe való, nem paraméterbe.
- **GGUF export** — llama.cpp backend, AVX2-vel ~3× a PyTorch-hoz képest
- **Eszközhasználati tanítóadat** — `glaive_function_calling.jsonl` (120 MB)
  megvan a korpuszban, de nincs bekötve

## Kapcsolódó

`../SUPERBRAIN-STK2-274M/` — korábbi 274M-es projekt: korpusz, tokenizáló,
és a `TANULSAGOK.md` (39 KB). **Olvasd el tanítás előtt**, de a benne lévő
számokat mérd újra: a 274M-es architektúrára érvényesek, és léptékváltásnál
megbuknak.
