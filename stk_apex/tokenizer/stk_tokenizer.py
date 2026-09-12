"""
SUPERBRAIN-STK2-274M — egyedi BPE tokenizáló betöltő és kódoló.

Betölti a `superbrain_tokenizer_24k.json` egyedi formátumot (token2id / merges /
byte_fallback_base), és teljes encode/decode felületet biztosít a modellhez.

Formátum: karakterszintű BPE, </w> szóvégi jelölővel, szóhatár-hasítás nélkül.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# A v2 a használandó változat: HU 1,96 tok/szó (a v1 2,46, a _hf konverzió 4,12).
_DEFAULT_PATH = Path(__file__).with_name("superbrain_tokenizer_24k_v2.json")

_EOW = "</w>"


class STKTokenizer:
    """Minimál interfész a SuperBrain modelltanításhoz."""

    # Kanonikus speciális token nevek
    PAD_TOKEN  = "<pad>"
    UNK_TOKEN  = "<unk>"
    BOS_TOKEN  = "<bos>"
    EOS_TOKEN  = "<eos>"

    def __init__(self, path: Union[str, Path] = _DEFAULT_PATH) -> None:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.token2id: Dict[str, int] = data["token2id"]
        self.id2token: Dict[int, str] = {v: k for k, v in self.token2id.items()}
        self._byte_base: int          = data.get("byte_fallback_base", 40)

        # Merge-ek: [(a, b, rank)] — rangsorolt, kisebb rangszám = korábban alkalmazni
        raw_merges = data["merges"]
        self._merges: Dict[Tuple[str, str], int] = {}
        for entry in raw_merges:
            if isinstance(entry, list) and len(entry) >= 3:
                self._merges[(entry[0], entry[1])] = entry[2]
            elif isinstance(entry, list) and len(entry) == 2:
                self._merges[(entry[0], entry[1])] = len(self._merges)

        # Speciális token id-k
        self.pad_id  = self.token2id.get(self.PAD_TOKEN, 0)
        self.unk_id  = self.token2id.get(self.UNK_TOKEN, 1)
        self.bos_id  = self.token2id.get(self.BOS_TOKEN, 2)
        self.eos_id  = self.token2id.get(self.EOS_TOKEN, 3)

        # ── SZÓ-SZINTŰ GYORSÍTÓTÁR ────────────────────────────────────
        # A BPE merge-elés a legdrágább lépés, és tiszta Pythonban fut.
        # Természetes szövegben a szavak erősen ismétlődnek, ezért a
        # szó → id-lista leképezés gyorsítótárazása nagyságrendi
        # gyorsulást ad. Enélkül a 3.5M soros korpusz tokenizálása
        # 4.7 óra/epoch (mérve, 208 szekvencia/s).
        self._word_cache: Dict[str, List[int]] = {}
        self._cache_limit = 400_000        # ~40 MB, bőven elég a fedéshez

    # ------------------------------------------------------------------
    # Alapadatok
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def __len__(self) -> int:
        return self.vocab_size

    # ------------------------------------------------------------------
    # BPE kódolás
    # ------------------------------------------------------------------

    def _pretokenize(self, text: str) -> List[str]:
        """Szavak szétválasztása — szóközök megtartásával a szókezdeten."""
        # "hello world" → ["hello</w>", " world</w>"]
        # Regex: minden szó: (szóközök?) + nem-szóköz karakterek
        words: List[str] = []
        for m in re.finditer(r'[ \t]*\S+|\n', text):
            token = m.group()
            # Az utolsó karakter elé </w> kerül, majd a szót karakterekre bontjuk
            words.append(token)
        return words

    def _word_to_chars(self, word: str) -> List[str]:
        """Szót karakterlistává alakít, a legutolsó karakterhez </w>-t fűz."""
        if not word:
            return []
        chars = list(word)
        chars[-1] = chars[-1] + _EOW
        return chars

    def _bpe_merge(self, symbols: List[str]) -> List[str]:
        """Egy szó BPE merge-elése a rangsorolt merge-lista szerint."""
        if len(symbols) <= 1:
            return symbols

        while True:
            best_rank = float("inf")
            best_idx  = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx  = i
            if best_idx == -1:
                break
            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2:]
        return symbols

    def _token_to_ids(self, token: str) -> List[int]:
        """
        Token → id-lista. Ismeretlen tokennél teljes UTF-8 byte-fallback.

        A szótár mind a 256 byte-tokent tartalmazza (<0x00>=40 … <0xFF>=295),
        ezért BÁRMILYEN Unicode karakter ábrázolható a UTF-8 bájtjaival.
        <unk> csak akkor keletkezhet, ha a byte-tokenek hiányoznak a szótárból.
        """
        if token in self.token2id:
            return [self.token2id[token]]

        # Az eow jelölőt levágjuk, mielőtt bájtokra bontanánk
        raw = token[:-len(_EOW)] if token.endswith(_EOW) else token
        if not raw:
            return []

        ids: List[int] = []
        for byte_val in raw.encode("utf-8"):
            byte_tok = f"<0x{byte_val:02X}>"
            bid = self.token2id.get(byte_tok)
            ids.append(bid if bid is not None else self.unk_id)
        return ids

    def _token_to_id(self, token: str) -> int:
        """Visszafelé kompatibilis egy-id változat (első bájt)."""
        ids = self._token_to_ids(token)
        return ids[0] if ids else self.unk_id

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Szöveg → token id lista."""
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)

        cache = self._word_cache
        for word in self._pretokenize(text):
            hit = cache.get(word)
            if hit is not None:
                ids.extend(hit)
                continue
            merged   = self._bpe_merge(self._word_to_chars(word))
            word_ids: List[int] = []
            for piece in merged:
                word_ids.extend(self._token_to_ids(piece))
            if len(cache) < self._cache_limit:
                cache[word] = word_ids
            ids.extend(word_ids)

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    # ------------------------------------------------------------------
    # Dekódolás
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Id-lista → szöveg. A byte-tokeneket (<0x00>…<0xFF>) összegyűjti és
        egyben dekódolja UTF-8-ként, hogy a többbájtos karakterek
        (héber, CJK, emoji) helyreálljanak.
        """
        pieces: List[str] = []
        byte_buf = bytearray()
        special_set = {self.pad_id, self.bos_id, self.eos_id}

        def flush_bytes() -> None:
            if byte_buf:
                pieces.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tok_id in ids:
            if skip_special and tok_id in special_set:
                continue
            tok = self.id2token.get(tok_id, self.UNK_TOKEN)

            # Byte-token: pufferbe, mert több bájt adhat ki egy karaktert
            if self._byte_base <= tok_id < self._byte_base + 256:
                byte_buf.append(tok_id - self._byte_base)
                continue

            flush_bytes()
            # </w> levágjuk — jelzi a szóvéget, de a szövegben nincs ott
            if tok.endswith(_EOW):
                tok = tok[: -len(_EOW)]
            pieces.append(tok)

        flush_bytes()
        return "".join(pieces)

    # ------------------------------------------------------------------
    # Kényelmi metódusok (HuggingFace-like interfész)
    # ------------------------------------------------------------------

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token2id.get(t, self.unk_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id2token.get(i, self.UNK_TOKEN) for i in ids]

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token2id)

    def __repr__(self) -> str:
        return (f"STKTokenizer(vocab={self.vocab_size:,}, "
                f"merges={len(self._merges):,})")


# ------------------------------------------------------------------
# Gyors önálló teszt
# ------------------------------------------------------------------
if __name__ == "__main__":
    tok = STKTokenizer()
    print(tok)

    tests = [
        ("HU", "A mesterséges intelligencia fejlődése megállíthatatlan."),
        ("EN", "Artificial intelligence is advancing rapidly."),
        ("HU2", "Én egy kísérlet vagyok."),
    ]
    for lang, text in tests:
        ids  = tok.encode(text)
        back = tok.decode(ids)
        fert = len(ids) / len(text.split())
        print(f"\n[{lang}] {text!r}")
        print(f"  ids ({len(ids)}): {ids[:12]}...")
        print(f"  fertility: {fert:.2f} token/szó")
        print(f"  vissza: {back!r}")
