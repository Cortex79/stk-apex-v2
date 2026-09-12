"""
Web eszközök: keresés, tartalom-lekérés, mély kutatás.

Keresési backend sorrend:
  1. Brave Search API (ha STK_BRAVE_KEY be van állítva)
  2. DuckDuckGo (duckduckgo_search csomag, API-kulcs nélkül)
  3. Dummy fallback (ha egyik sem elérhető)
"""
from __future__ import annotations

import re
import textwrap
from typing import List, Optional
from urllib.parse import quote_plus

from .registry import REGISTRY

# Lazy importok (opcionális függőségek)
def _ddg():
    try:
        from duckduckgo_search import DDGS
        return DDGS()
    except ImportError:
        return None

def _httpx():
    try:
        import httpx
        return httpx
    except ImportError:
        return None

def _bs4():
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup
    except ImportError:
        return None


# ── Eszközök ─────────────────────────────────────────────────────────────────

@REGISTRY.register(
    name="web_search",
    description="Webes keresés — rövid összefoglalók + URL-ek visszaadása",
    parameters={"query": "keresési kifejezés", "n": "eredmények száma (opcionális, default 5)"}
)
def web_search(query: str, n: int = 5) -> str:
    n = min(int(n), 10)

    # 1. Brave API
    import os
    brave_key = os.environ.get("STK_BRAVE_KEY", "")
    if brave_key:
        hx = _httpx()
        if hx:
            try:
                r = hx.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n, "lang": "hu"},
                    headers={"Accept": "application/json",
                             "X-Subscription-Token": brave_key},
                    timeout=8.0
                )
                items = r.json().get("web", {}).get("results", [])
                parts = []
                for it in items[:n]:
                    parts.append(f"[{it['title']}]({it['url']})\n{it.get('description','')}")
                return "\n\n".join(parts) if parts else "Nincs találat."
            except Exception:
                pass

    # 2. DuckDuckGo
    ddg = _ddg()
    if ddg:
        try:
            results = list(ddg.text(query, max_results=n))
            if results:
                parts = []
                for r in results:
                    parts.append(
                        f"[{r['title']}]({r['href']})\n{r.get('body', '')[:200]}")
                return "\n\n".join(parts)
        except Exception:
            pass

    return f"Keresési eredmény: '{query}' — telepítsd a duckduckgo_search csomagot vagy adj meg Brave API kulcsot."


@REGISTRY.register(
    name="web_fetch",
    description="Weboldal tartalmának lekérése és szöveg-kivonása",
    parameters={"url": "teljes URL", "max_chars": "max karakterek (opcionális, default 2000)"}
)
def web_fetch(url: str, max_chars: int = 2000) -> str:
    hx = _httpx()
    if not hx:
        return "httpx csomag szükséges: pip install httpx"
    BS = _bs4()
    if not BS:
        return "beautifulsoup4 csomag szükséges: pip install beautifulsoup4"

    try:
        r = hx.get(url, timeout=8.0, follow_redirects=True,
                   headers={"User-Agent": "STK-Apex-Assistant/2.0"})
        soup = BS(r.text, "html.parser")
        # Felesleges elemek eltávolítása
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "form", "button"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Whitespace tisztítás
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text[:int(max_chars)]
    except Exception as e:
        return f"Lekérési hiba: {e}"


@REGISTRY.register(
    name="deep_research",
    description="Mély kutatás — több forrás keresése és összefoglalása egy témáról",
    parameters={"topic": "kutatási téma", "depth": "mélység: 'quick'|'thorough' (default: quick)"}
)
def deep_research(topic: str, depth: str = "quick") -> str:
    n_search = 3 if depth == "quick" else 6
    chars_per_page = 1000 if depth == "quick" else 1500

    # 1. Keresés
    search_result = web_search(topic, n=n_search)
    lines = search_result.split("\n\n")

    # 2. URL-ek kinyerése
    url_re = re.compile(r'\]\((https?://[^\)]+)\)')
    urls = []
    for line in lines:
        m = url_re.search(line)
        if m:
            urls.append(m.group(1))

    # 3. Tartalom lekérése (első 2-3 URL)
    fetched = []
    for url in urls[:3]:
        content = web_fetch(url, max_chars=chars_per_page)
        if content and "hiba" not in content.lower():
            fetched.append(f"Forrás: {url}\n{content[:chars_per_page]}")

    summary = f"Kutatás: '{topic}'\n\n"
    summary += "=== Keresési találatok ===\n" + search_result + "\n\n"
    if fetched:
        summary += "=== Forrástartalom ===\n" + "\n\n---\n\n".join(fetched)
    return summary
