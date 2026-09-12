"""
ReAct ágens-loop — Gondolat → Eszköz → Megfigyelés → Válasz.

Lépések:
  1. Rendszer-prompt összeállítása (eszközlista + memória-kontextus)
  2. Modell generál: <thought>...</thought><action>{...}</action>
     VAGY végső szöveges választ
  3. Ha action → eszköz futtatása → <observation>...</observation> injektálás
  4. Ismétlés (max max_steps lépés)
  5. Epizód mentése memóriába
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from ..model import STKApex
from .config import AssistantConfig
from .memory.episodic import EpisodicMemory
from .tools.registry import REGISTRY, parse_action, parse_thought, is_final_answer

# Eszközök regisztrálása (importálással aktiválódnak)
from .tools import web, shopping, travel, smarthome, office  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizáló — magyar-angol BPE, 24 576 token (HU 1,96 tok/szó)
# ─────────────────────────────────────────────────────────────────────────────

def _load_tokenizer():
    from ..tokenizer import STKTokenizer
    return STKTokenizer()


# ─────────────────────────────────────────────────────────────────────────────
# AgentStep — egy lépés nyomkövetése
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    thought:     str  = ""
    action:      Dict = field(default_factory=dict)
    observation: str  = ""
    raw:         str  = ""


# ─────────────────────────────────────────────────────────────────────────────
# AssistantAgent
# ─────────────────────────────────────────────────────────────────────────────

class AssistantAgent:
    """
    STK-APEX v2 alapú személyes asszisztens.

    Példa:
        agent = AssistantAgent.from_config()
        resp  = agent.chat("Mi az időjárás Budapesten?")
        print(resp.answer)
    """

    SYSTEM_PROMPT = """\
Te egy intelligens személyes asszisztens vagy. Magyar és angol nyelvű kéréseket egyaránt kezelsz.
Képességeid:
{tool_schema}

Gondolkodási protokoll:
  Ha eszközt kell használnod, írj egy <thought>rövid terv</thought> blokkot,
  majd egy <action>{{"tool": "...", "param": "..."}}</action> blokkot.
  A megfigyelés után folytasd. Ha kész vagy, írj közvetlen választ (nincs <action> blokk).

Fontos:
  - Csak a felsorolt eszközöket használd
  - Maximum 5 lépés egy válaszon belül
  - Mindig végezz a végső, ember-olvasható válasszal
  - Ha bizonytalanságot érzed, tartózkodj a találgatástól
"""

    def __init__(self, model: STKApex, cfg: AssistantConfig,
                 tok=None, verbose: bool = False):
        self.model   = model.eval()
        self.cfg     = cfg
        self.tok     = tok if tok is not None else _load_tokenizer()
        self.memory  = EpisodicMemory(cfg.memory_db_path)
        self.verbose = verbose
        self._system = self.SYSTEM_PROMPT.format(
            tool_schema=REGISTRY.schema_text())

    @classmethod
    def from_config(cls, cfg: Optional[AssistantConfig] = None,
                    verbose: bool = False) -> "AssistantAgent":
        if cfg is None:
            cfg = AssistantConfig.load()
        from .. import build_apex
        tok   = _load_tokenizer()
        model = build_apex(cfg.model_size, use_rag=False,
                           vocab_size=tok.vocab_size)
        return cls(model, cfg, tok, verbose)

    # ── Tokenizálás ───────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text)

    def decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.tok.decode(ids)

    # ── Generálás ─────────────────────────────────────────────────────────

    def _generate_text(self, prompt: str,
                       max_new: int = 200,
                       stop_on_action: bool = True) -> str:
        ids  = self.encode(prompt)
        # Kontextus-ablak csonkítása
        max_ctx = self.cfg.max_context_tokens - max_new
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]
        inp = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model.generate(
                inp,
                max_new_tokens=max_new,
                temperature=self.cfg.temperature,
                use_abstention=False,
            )
        new_ids = out["sequences"][0, len(ids):].tolist()
        text    = self.decode(new_ids)

        # Vágás az action blokk után (ha stop_on_action)
        if stop_on_action:
            m = re.search(r'</action>', text)
            if m:
                text = text[:m.end()]
        return text

    # ── Fő chat loop ──────────────────────────────────────────────────────

    def chat(self, user_msg: str,
             max_steps: int = 5,
             stream_callback: Optional[Callable[[str], None]] = None
             ) -> "AgentResponse":
        t0 = time.perf_counter()

        # Emlékezet-kontextus
        mem_ctx = self.memory.format_context(user_msg, self.cfg.memory_top_k)

        # Kontextus összeállítása
        context = self._system
        if mem_ctx:
            context += f"\n\n{mem_ctx}"
        context += f"\n\nFelhasználó: {user_msg}\nAsszisztens:"

        steps: List[AgentStep] = []
        tools_used: List[str]  = []
        full_answer            = ""

        for step_i in range(max_steps):
            raw = self._generate_text(context, max_new=self.cfg.max_new_tokens)

            thought = parse_thought(raw)
            action  = parse_action(raw)

            if self.verbose:
                print(f"\n--- Lépés {step_i+1} ---")
                if thought:
                    print(f"Gondolat: {thought}")
                if action:
                    print(f"Eszköz: {json.dumps(action, ensure_ascii=False)}")

            if action is None:
                # Végső válasz
                full_answer = re.sub(
                    r'<thought>.*?</thought>', '', raw,
                    flags=re.DOTALL).strip()
                steps.append(AgentStep(thought=thought, raw=raw))
                break

            # Eszköz futtatása
            result = REGISTRY.dispatch(action)
            obs    = result.to_text()[:800]   # kontextus-korlát

            if stream_callback:
                stream_callback(f"[{action.get('tool')}] → {obs[:100]}…")

            tools_used.append(action.get("tool", ""))
            steps.append(AgentStep(
                thought=thought, action=action,
                observation=obs, raw=raw))

            # Kontextus frissítése
            context += raw + f"\n<observation>{obs}</observation>\n"

        else:
            # Kimerült a lépések száma
            full_answer = "Sajnálom, nem sikerült teljesen feldolgozni a kérést."

        elapsed = time.perf_counter() - t0

        # Memória mentése
        self.memory.save_episode(
            user_msg=user_msg,
            assistant=full_answer[:500],
            tools=tools_used,
            tags=",".join(set(tools_used))
        )

        return AgentResponse(
            answer=full_answer,
            steps=steps,
            tools_used=tools_used,
            elapsed_s=elapsed,
        )

    # ── Emlékeztetők ──────────────────────────────────────────────────────

    def check_reminders(self) -> List[str]:
        pending = self.memory.pending_reminders()
        msgs = []
        for r in pending:
            msgs.append(f"Emlékeztető: {r['text']}")
            if r["repeat"] == "none":
                self.memory.mark_done(r["id"])
        return msgs


@dataclass
class AgentResponse:
    answer:     str
    steps:      List[AgentStep]
    tools_used: List[str]
    elapsed_s:  float

    def __str__(self) -> str:
        return self.answer

    def debug(self) -> str:
        lines = [f"Válasz ({self.elapsed_s:.2f}s, {len(self.steps)} lépés):"]
        for i, s in enumerate(self.steps, 1):
            if s.thought:
                lines.append(f"  [{i}] Gondolat: {s.thought[:80]}")
            if s.action:
                lines.append(f"       Eszköz:   {s.action}")
            if s.observation:
                lines.append(f"       Obs:      {s.observation[:80]}")
        lines.append(f"  Válasz: {self.answer[:200]}")
        return "\n".join(lines)
