"""
Eszköz-regiszter — minden tool definíciója és a dispatcher.

Tool hívás formátuma (modell kimenetből parsolt JSON):
  {"tool": "web_search", "query": "legjobb hotel Budapest 2025"}
  {"tool": "ha_control", "domain": "light", "service": "turn_on", "entity_id": "light.nappali"}

Minden tool egy ToolResult-ot ad vissza: {"ok": bool, "data": ..., "error": str|None}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDef:
    name:        str
    description: str
    parameters:  Dict[str, str]   # {"param_name": "leírás"}
    fn:          Callable


@dataclass
class ToolResult:
    ok:    bool
    data:  Any    = None
    error: Optional[str] = None

    def to_text(self) -> str:
        if not self.ok:
            return f"[HIBA] {self.error}"
        if isinstance(self.data, str):
            return self.data
        return json.dumps(self.data, ensure_ascii=False, indent=2)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}

    def register(self, name: str, description: str,
                 parameters: Dict[str, str]) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._tools[name] = ToolDef(name, description, parameters, fn)
            return fn
        return decorator

    def dispatch(self, call: Dict[str, Any]) -> ToolResult:
        name = call.get("tool")
        if not name or name not in self._tools:
            return ToolResult(ok=False, error=f"Ismeretlen eszköz: '{name}'")
        kwargs = {k: v for k, v in call.items() if k != "tool"}
        try:
            result = self._tools[name].fn(**kwargs)
            return ToolResult(ok=True, data=result)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

    def schema_text(self) -> str:
        """Rendszer-prompt részlete: elérhető eszközök listája."""
        lines = ["Elérhető eszközök (JSON formátumban hívhatók):"]
        for t in self._tools.values():
            params = ", ".join(f'"{k}": {v}' for k, v in t.parameters.items())
            lines.append(f'  {{"tool": "{t.name}", {params}}}')
            lines.append(f'    → {t.description}')
        return "\n".join(lines)

    def names(self) -> List[str]:
        return list(self._tools.keys())


# Globális regiszter (minden tool ebbe regisztrál)
REGISTRY = ToolRegistry()


# ── Tool-hívás parseolása a modell kimenetéből ───────────────────────────────

_ACTION_RE = re.compile(
    r'<action>\s*(\{.*?\})\s*</action>',
    re.DOTALL | re.IGNORECASE
)
_THOUGHT_RE = re.compile(
    r'<thought>(.*?)</thought>',
    re.DOTALL | re.IGNORECASE
)


def parse_action(text: str) -> Optional[Dict]:
    """Kinyeri az első <action>{...}</action> blokkot."""
    m = _ACTION_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def parse_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip() if m else ""


def is_final_answer(text: str) -> bool:
    """Nincs több tool-hívás — a modell kész válaszolt."""
    return parse_action(text) is None
