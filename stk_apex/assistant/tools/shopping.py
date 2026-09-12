"""
Vásárlási eszközök — termékkeresés, ár-összehasonlítás.

Implementáció: web_search + strukturált kinyerés.
Opcionális: affiliate/partner API-k (pl. Árukereső, Amazon PA-API).
"""
from __future__ import annotations
import re
from .registry import REGISTRY
from .web import web_search, web_fetch


@REGISTRY.register(
    name="product_search",
    description="Termék keresése és ár-összehasonlítás webáruházakban",
    parameters={
        "product":  "keresett termék neve/leírása",
        "max_price": "maximális ár HUF-ban (opcionális)",
        "site":     "konkrét webáruház (opcionális, pl. 'emag.hu', 'alza.hu')"
    }
)
def product_search(product: str, max_price: str = "", site: str = "") -> str:
    query = product
    if site:
        query += f" site:{site}"
    elif not site:
        # Főbb magyar webáruházak
        query += " (emag.hu OR alza.hu OR mediamarkt.hu OR euronics.hu) ár"
    if max_price:
        query += f" max {max_price} Ft"

    raw = web_search(query, n=6)

    # Ár-minták kinyerése
    price_re = re.compile(r'(\d[\d\s]*)\s*(?:Ft|HUF|forint)', re.IGNORECASE)
    prices = price_re.findall(raw)
    prices_clean = []
    for p in prices:
        try:
            val = int(p.replace(" ", "").replace("\xa0", ""))
            prices_clean.append(val)
        except ValueError:
            pass

    result = f"Termékkeresés: '{product}'\n\n{raw}"
    if prices_clean:
        result += f"\n\n--- Talált árak ---\n"
        prices_clean.sort()
        result += f"Legolcsóbb: {prices_clean[0]:,} Ft\n"
        result += f"Legdrágább: {prices_clean[-1]:,} Ft\n"
        if len(prices_clean) > 1:
            avg = sum(prices_clean) // len(prices_clean)
            result += f"Átlagár:    {avg:,} Ft\n"
    return result


@REGISTRY.register(
    name="price_alert_set",
    description="Ár-figyelés beállítása egy termékre (memóriába menti)",
    parameters={
        "product":      "termék neve",
        "target_price": "kívánt ár HUF-ban",
        "url":          "termék URL (opcionális)"
    }
)
def price_alert_set(product: str, target_price: str, url: str = "") -> str:
    # Memóriában tárolja (episodic store-ba kerül az agent loop-on keresztül)
    return {
        "action":        "price_alert_registered",
        "product":       product,
        "target_price":  target_price,
        "url":           url,
        "message": (f"Beállítva: értesítés ha '{product}' ára "
                    f"{target_price} Ft alá esik.")
    }


@REGISTRY.register(
    name="shopping_cart_suggest",
    description="Bevásárlólista alapján online kosár összeállítása ajánlott boltokkal",
    parameters={
        "items":  "vesszővel elválasztott termékek listája",
        "budget": "teljes keret HUF-ban (opcionális)"
    }
)
def shopping_cart_suggest(items: str, budget: str = "") -> str:
    item_list = [i.strip() for i in items.split(",") if i.strip()]
    results = []
    for item in item_list[:5]:   # max 5 tétel egyszerre
        r = web_search(f"{item} ár webáruház", n=2)
        # Csak az első sor / URL
        first = r.split("\n")[0]
        results.append(f"• {item}: {first}")

    out = "Bevásárlólista javaslatok:\n" + "\n".join(results)
    if budget:
        out += f"\n\nKeret: {budget} Ft"
    return out
