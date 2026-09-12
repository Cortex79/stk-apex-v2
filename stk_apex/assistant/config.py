"""
Asszisztens konfigurációja — API kulcsok, Home Assistant URL, preferenciák.
Értékek betöltési sorrendje: environment variables → ~/.stk_assistant.json → default
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from pathlib import Path

_CFG_PATH = Path.home() / ".stk_assistant.json"


@dataclass
class AssistantConfig:
    # ── Modell ────────────────────────────────────────────────────
    model_size:         str   = "60m"
    max_context_tokens: int   = 512
    max_new_tokens:     int   = 256
    temperature:        float = 0.7

    # ── Memória ───────────────────────────────────────────────────
    memory_db_path:     str   = str(Path.home() / ".stk_memory.db")
    memory_top_k:       int   = 5          # mennyi epizódot injektálunk kontextusba

    # ── Web / kutatás ─────────────────────────────────────────────
    search_max_results: int   = 5
    scrape_timeout_s:   float = 8.0
    brave_api_key:      str   = ""         # opcionális (jobb mint DDG)

    # ── Okosotthon ────────────────────────────────────────────────
    ha_url:             str   = "http://homeassistant.local:8123"
    ha_token:           str   = ""         # Home Assistant Long-Lived Token

    # ── Iroda ─────────────────────────────────────────────────────
    caldav_url:         str   = ""
    caldav_user:        str   = ""
    caldav_password:    str   = ""
    email_imap_host:    str   = ""
    email_imap_user:    str   = ""
    email_imap_pass:    str   = ""
    email_smtp_host:    str   = ""
    email_smtp_port:    int   = 587

    # ── TTS ───────────────────────────────────────────────────────
    tts_enabled:        bool  = False      # edge-tts vagy pyttsx3
    tts_voice:          str   = "hu-HU-NoemiNeural"

    # ── Felhasználói profil ───────────────────────────────────────
    user_name:          str   = ""
    user_language:      str   = "hu"
    user_timezone:      str   = "Europe/Budapest"

    @classmethod
    def load(cls) -> "AssistantConfig":
        data: dict = {}
        if _CFG_PATH.exists():
            try:
                data = json.loads(_CFG_PATH.read_text("utf-8"))
            except Exception:
                pass
        # Environment variables felülírják
        env_map = {
            "STK_HA_URL":         "ha_url",
            "STK_HA_TOKEN":       "ha_token",
            "STK_BRAVE_KEY":      "brave_api_key",
            "STK_CALDAV_URL":     "caldav_url",
            "STK_CALDAV_USER":    "caldav_user",
            "STK_CALDAV_PASS":    "caldav_password",
            "STK_EMAIL_HOST":     "email_imap_host",
            "STK_EMAIL_USER":     "email_imap_user",
            "STK_EMAIL_PASS":     "email_imap_pass",
        }
        for env_key, field_name in env_map.items():
            val = os.environ.get(env_key)
            if val:
                data[field_name] = val
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})

    def save(self) -> None:
        import dataclasses
        _CFG_PATH.write_text(
            json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8")
