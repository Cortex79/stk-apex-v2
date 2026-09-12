"""
Kalibrált tartózkodás — STK-6 / Kalai & Nachum (arXiv:2509.04664).

Jutalom-struktúra:
    helyes válasz    : +1
    tartózkodás      : 0
    helytelen válasz : −t/(1−t)   [t=0.75 → −3.0]

Empirikus eredmény (SUPERBRAIN-STK2-274M/rlvr/abstention.py mérése):
    0% hallucinációs arány (vs 82% ha mindig válaszol)

A 2026-os Nature-cikk megerősítette: a helytelen jutalom ösztönzési
problémát okoz — a fix az aszimmetrikus jutalom-struktúra.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AbstentionKind(str, Enum):
    CORRECT    = "correct"
    ABSTAINED  = "abstained"
    WRONG      = "wrong"


@dataclass
class AbstentionOutcome:
    kind:    AbstentionKind
    reward:  float
    verdict: str


_ABSTENTION_PATTERNS = re.compile(
    r"(?i)\b("
    r"nem tudom|nem vagyok biztos|nincs elegend[oő] inform|"
    r"i don't know|i do not know|i'm not sure|i am not sure|"
    r"cannot (determine|say|answer)|not enough (info|information|context)|"
    r"insufficient (information|data|context)|"
    r"unclear|uncertain|bizonytalan"
    r")\b"
)


def is_abstention(text: str) -> bool:
    return bool(_ABSTENTION_PATTERNS.search(text))


def wrong_penalty(t: float) -> float:
    """Helytelen válasz jutalma: −t/(1−t)."""
    assert 0 < t < 1
    return -t / (1.0 - t)


def abstention_reward(
    prediction: str,
    target: str,
    *,
    threshold: float = 0.75,
    exact: bool = False,
) -> AbstentionOutcome:
    """
    Szöveg-szintű jutalomszámítás.

    exact=True: szó szerinti egyezést vár (pl. ARC/GSM8K célnál).
    exact=False: normált egyezés (kisbetűsített, strip).
    """
    if is_abstention(prediction):
        return AbstentionOutcome(AbstentionKind.ABSTAINED, 0.0,
                                 "tartózkodás — semleges jutalom")
    pred = prediction.strip().lower() if not exact else prediction.strip()
    tgt  = target.strip().lower()     if not exact else target.strip()
    if pred == tgt:
        return AbstentionOutcome(AbstentionKind.CORRECT, 1.0, "helyes")
    return AbstentionOutcome(AbstentionKind.WRONG,
                             wrong_penalty(threshold),
                             f"helytelen (büntetés={wrong_penalty(threshold):.1f})")


# ─────────────────────────────────────────────────────────────────────────────
# Verifikátorok (ARC, GSM8K, math)
# ─────────────────────────────────────────────────────────────────────────────

import json
from dataclasses import dataclass as _dc


@_dc
class Verdict:
    ok:     bool
    score:  float
    parsed: Optional[str]
    detail: str


_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_GSM8K = re.compile(r"####\s*([-\d,. ]+)")


def _extract_number(text: str) -> Optional[str]:
    m = _BOXED.search(text) or _GSM8K.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    toks = re.findall(r"-?\d+(?:\.\d+)?", text)
    return toks[-1] if toks else None


def parse_grid(text: str) -> Optional[list]:
    """JSON-tömb vagy Python-lista szövegből."""
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def verify_arc(prediction: str, target: str) -> Verdict:
    p = parse_grid(prediction)
    t = parse_grid(target)
    if p is None:
        return Verdict(False, 0.0, None, "nincs grid a válaszban")
    ok = (p == t)
    return Verdict(ok, 1.0 if ok else 0.0, str(p),
                   "helyes" if ok else "grid nem egyezik")


def verify_math(prediction: str, target: str) -> Verdict:
    p = _extract_number(prediction)
    t = _extract_number(target)
    if p is None:
        return Verdict(False, 0.0, None, "nincs szám a válaszban")
    try:
        ok = abs(float(p) - float(t)) < 1e-6
    except ValueError:
        ok = p.strip() == (t or "").strip()
    return Verdict(ok, 1.0 if ok else 0.0, p, "helyes" if ok else "szám nem egyezik")


def verify(domain: str, prediction: str, target: str) -> Verdict:
    if domain in ("arc", "fluid"):
        return verify_arc(prediction, target)
    return verify_math(prediction, target)


def rlvr_reward(domain: str, prediction: str, target: str,
                threshold: float = 0.75) -> float:
    """
    Teljes RLVR jutalom: tartózkodás-vizsgálat + domain verifikátor.
    """
    if is_abstention(prediction):
        return 0.0
    v = verify(domain, prediction, target)
    if v.ok:
        return 1.0
    return wrong_penalty(threshold)
