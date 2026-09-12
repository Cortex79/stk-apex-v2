"""
Epizodikus memória — SQLite alapú, tartós, session-független.

Három tábla:
  episodes    — párbeszéd-epizódok (kérdés + válasz + eszközök)
  preferences — felhasználói preferenciák kulcs-érték párokban
  reminders   — emlékeztetők és ár-figyelmek

Keresés: egyszerű kulcsszavas (FTS5 ha elérhető, egyébként LIKE).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class EpisodicMemory:

    def __init__(self, db_path: str = "~/.stk_memory.db"):
        self.db_path = str(Path(db_path).expanduser())
        self._conn   = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        c = self._conn
        c.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT NOT NULL,
            user_msg  TEXT NOT NULL,
            assistant TEXT NOT NULL,
            tools     TEXT DEFAULT '[]',
            tags      TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS preferences (
            key       TEXT PRIMARY KEY,
            value     TEXT NOT NULL,
            updated   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reminders (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            text      TEXT NOT NULL,
            due       TEXT,
            repeat    TEXT DEFAULT 'none',
            done      INTEGER DEFAULT 0,
            created   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS price_alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            product       TEXT NOT NULL,
            target_price  INTEGER NOT NULL,
            url           TEXT DEFAULT '',
            triggered     INTEGER DEFAULT 0,
            created       TEXT NOT NULL
        );
        """)
        c.commit()

    # ── Epizódok ─────────────────────────────────────────────────────────────

    def save_episode(self, user_msg: str, assistant: str,
                     tools: List[str] = [], tags: str = "") -> int:
        cur = self._conn.execute(
            "INSERT INTO episodes (ts, user_msg, assistant, tools, tags) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), user_msg, assistant,
             json.dumps(tools, ensure_ascii=False), tags)
        )
        self._conn.commit()
        return cur.lastrowid

    def search_episodes(self, query: str, k: int = 5) -> List[sqlite3.Row]:
        words = query.lower().split()[:5]
        if not words:
            return []
        like_clauses = " AND ".join(
            ["(LOWER(user_msg) LIKE ? OR LOWER(assistant) LIKE ?)"] * len(words))
        params = []
        for w in words:
            params.extend([f"%{w}%", f"%{w}%"])
        params.append(k)
        rows = self._conn.execute(
            f"SELECT * FROM episodes WHERE {like_clauses} "
            f"ORDER BY id DESC LIMIT ?",
            params
        ).fetchall()
        return rows

    def recent_episodes(self, k: int = 3) -> List[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (k,)
        ).fetchall()

    def format_context(self, query: str, k: int = 4) -> str:
        """Releváns epizódok formázva kontextus-injektáláshoz."""
        rows = self.search_episodes(query, k)
        if not rows:
            rows = self.recent_episodes(2)
        if not rows:
            return ""
        lines = ["[Emlékezet — korábbi párbeszédek]"]
        for r in reversed(rows):
            ts_short = r["ts"][:16]
            lines.append(f"[{ts_short}] Te: {r['user_msg'][:120]}")
            lines.append(f"         Asszisztens: {r['assistant'][:200]}")
        return "\n".join(lines)

    # ── Preferenciák ─────────────────────────────────────────────────────────

    def set_preference(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value, updated) VALUES (?,?,?)",
            (key, value, datetime.now().isoformat())
        )
        self._conn.commit()

    def get_preference(self, key: str,
                       default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM preferences WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def all_preferences(self) -> dict:
        rows = self._conn.execute("SELECT key, value FROM preferences").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ── Emlékeztetők ─────────────────────────────────────────────────────────

    def add_reminder(self, text: str, due: str = "",
                     repeat: str = "none") -> int:
        cur = self._conn.execute(
            "INSERT INTO reminders (text, due, repeat, created) VALUES (?,?,?,?)",
            (text, due, repeat, datetime.now().isoformat())
        )
        self._conn.commit()
        return cur.lastrowid

    def pending_reminders(self) -> List[sqlite3.Row]:
        now = datetime.now().isoformat()
        return self._conn.execute(
            "SELECT * FROM reminders WHERE done=0 AND (due IS NULL OR due<=?) "
            "ORDER BY due",
            (now,)
        ).fetchall()

    def mark_done(self, reminder_id: int) -> None:
        self._conn.execute(
            "UPDATE reminders SET done=1 WHERE id=?", (reminder_id,))
        self._conn.commit()

    # ── Ár-figyelmek ─────────────────────────────────────────────────────────

    def add_price_alert(self, product: str, target_price: int,
                        url: str = "") -> int:
        cur = self._conn.execute(
            "INSERT INTO price_alerts (product, target_price, url, created) "
            "VALUES (?,?,?,?)",
            (product, target_price, url, datetime.now().isoformat())
        )
        self._conn.commit()
        return cur.lastrowid

    def active_price_alerts(self) -> List[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM price_alerts WHERE triggered=0"
        ).fetchall()

    def close(self) -> None:
        self._conn.close()
