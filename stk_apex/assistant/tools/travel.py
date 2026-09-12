"""
Utazási eszközök — repjegy, szálloda, transzfer keresése.

Implementáció: webes keresés + strukturált kinyerés.
Opcionális: Amadeus API, Booking.com Affiliate API (ha STK_AMADEUS_KEY be van állítva).
"""
from __future__ import annotations
import os, re
from datetime import datetime
from .registry import REGISTRY
from .web import web_search, web_fetch


def _amadeus_token() -> str:
    return os.environ.get("STK_AMADEUS_KEY", "")


@REGISTRY.register(
    name="flight_search",
    description="Repülőjegy keresés indulási helytől, célállomásig, dátummal",
    parameters={
        "origin":       "indulási város vagy repülőtér IATA kód (pl. BUD)",
        "destination":  "célváros vagy IATA kód (pl. BCN)",
        "date":         "indulás dátuma ÉÉÉÉ-HH-NN formátumban",
        "return_date":  "visszaút dátuma (opcionális)",
        "passengers":   "utasok száma (opcionális, default 1)"
    }
)
def flight_search(origin: str, destination: str, date: str,
                  return_date: str = "", passengers: str = "1") -> str:
    # Amadeus API (ha van kulcs)
    if _amadeus_token():
        try:
            import httpx
            # Amadeus OAuth token
            auth = httpx.post(
                "https://test.api.amadeus.com/v1/security/oauth2/token",
                data={"grant_type": "client_credentials",
                      "client_id": _amadeus_token(),
                      "client_secret": os.environ.get("STK_AMADEUS_SECRET", "")},
                timeout=8.0
            )
            token = auth.json().get("access_token", "")
            if token:
                params = {
                    "originLocationCode":      origin.upper(),
                    "destinationLocationCode": destination.upper(),
                    "departureDate":           date,
                    "adults":                  int(passengers),
                    "currencyCode":            "HUF",
                    "max":                     5
                }
                if return_date:
                    params["returnDate"] = return_date
                r = httpx.get(
                    "https://test.api.amadeus.com/v2/shopping/flight-offers",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params, timeout=10.0
                )
                offers = r.json().get("data", [])
                if offers:
                    lines = [f"Repülőjegyek: {origin} → {destination}, {date}"]
                    for o in offers[:5]:
                        price = o["price"]["total"]
                        seg   = o["itineraries"][0]["segments"][0]
                        dep   = seg["departure"]["at"]
                        arr   = seg["arrival"]["at"]
                        lines.append(f"  {dep} → {arr}  |  {price} HUF")
                    return "\n".join(lines)
        except Exception:
            pass

    # Fallback: webes keresés
    rt = f" visszaút {return_date}" if return_date else ""
    query = f"repülőjegy {origin} {destination} {date}{rt} {passengers} fő ár"
    raw   = web_search(query, n=5)
    return f"Repülőjegy keresés: {origin} → {destination}, {date}\n\n{raw}"


@REGISTRY.register(
    name="hotel_search",
    description="Szálloda keresés városban, dátummal és éjszakák számával",
    parameters={
        "city":        "célváros",
        "check_in":    "érkezés dátuma ÉÉÉÉ-HH-NN",
        "check_out":   "távozás dátuma ÉÉÉÉ-HH-NN",
        "guests":      "vendégek száma (opcionális, default 2)",
        "max_price":   "max ár/éj HUF-ban (opcionális)",
        "stars":       "minimum csillagok száma (opcionális, 1-5)"
    }
)
def hotel_search(city: str, check_in: str, check_out: str,
                 guests: str = "2", max_price: str = "",
                 stars: str = "") -> str:
    # Próbálkozás Booking.com affiliate API-val (ha elérhető)
    booking_key = os.environ.get("STK_BOOKING_KEY", "")

    # Webes keresés fallback
    q = f"szálloda {city} {check_in} {check_out} {guests} fő"
    if max_price:
        q += f" max {max_price} Ft"
    if stars:
        q += f" {stars} csillagos"
    q += " booking.com OR szallas.hu OR hotels.com ár"

    raw = web_search(q, n=5)

    # Éjszakák kiszámítása
    try:
        d1 = datetime.strptime(check_in,  "%Y-%m-%d")
        d2 = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (d2 - d1).days
    except Exception:
        nights = None

    result = f"Szálloda: {city}, {check_in} → {check_out}"
    if nights:
        result += f" ({nights} éjszaka, {guests} vendég)"
    result += f"\n\n{raw}"
    return result


@REGISTRY.register(
    name="transfer_search",
    description="Reptéri transzfer vagy taxi keresés",
    parameters={
        "from_loc": "indulási helyszín",
        "to_loc":   "célhelyszín",
        "date":     "dátum ÉÉÉÉ-HH-NN (opcionális)",
        "time":     "időpont ÓÓ:PP (opcionális)"
    }
)
def transfer_search(from_loc: str, to_loc: str,
                    date: str = "", time: str = "") -> str:
    q = f"transzfer taxi {from_loc} {to_loc}"
    if date:
        q += f" {date}"
    if time:
        q += f" {time}"
    q += " ár foglalas"
    raw = web_search(q, n=4)
    return f"Transzfer: {from_loc} → {to_loc}\n\n{raw}"
