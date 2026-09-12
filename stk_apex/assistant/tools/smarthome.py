"""
Okosotthon eszközök — Home Assistant REST API integráció.

Beállítás: STK_HA_URL és STK_HA_TOKEN environment változók,
           vagy ~/.stk_assistant.json fájlban.

Home Assistant entitás példák:
  light.nappali, switch.kávéfőző, climate.hálószoba,
  media_player.nappalitv, sensor.külső_hőmérséklet
"""
from __future__ import annotations
import os
from typing import Any, Dict, Optional
from .registry import REGISTRY


def _ha_get(path: str) -> Dict:
    """GET kérés a Home Assistant API-hoz."""
    import httpx
    url   = os.environ.get("STK_HA_URL", "http://homeassistant.local:8123")
    token = os.environ.get("STK_HA_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Home Assistant token hiányzik. Állítsd be: STK_HA_TOKEN=<long-lived token>")
    r = httpx.get(f"{url}/api/{path}",
                  headers={"Authorization": f"Bearer {token}",
                           "Content-Type": "application/json"},
                  timeout=5.0)
    r.raise_for_status()
    return r.json()


def _ha_post(path: str, data: Dict = {}) -> Dict:
    """POST kérés a Home Assistant API-hoz."""
    import httpx
    url   = os.environ.get("STK_HA_URL", "http://homeassistant.local:8123")
    token = os.environ.get("STK_HA_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Home Assistant token hiányzik. Állítsd be: STK_HA_TOKEN=<long-lived token>")
    r = httpx.post(f"{url}/api/{path}",
                   headers={"Authorization": f"Bearer {token}",
                            "Content-Type": "application/json"},
                   json=data, timeout=5.0)
    r.raise_for_status()
    return r.json()


@REGISTRY.register(
    name="ha_control",
    description="Okosotthon eszköz vezérlése (lámpa, kapcsoló, termosztát, média)",
    parameters={
        "domain":    "entitás domain: light|switch|climate|media_player|cover|fan|scene",
        "service":   "szolgáltatás: turn_on|turn_off|toggle|set_temperature|media_play|media_pause",
        "entity_id": "entitás azonosítója (pl. light.nappali, switch.kávéfőző)",
        "extra":     "extra paraméterek JSON-ban (opcionális, pl. {\"brightness\": 128, \"color_temp\": 4000})"
    }
)
def ha_control(domain: str, service: str, entity_id: str,
               extra: str = "") -> str:
    import json as _json
    data: Dict[str, Any] = {"entity_id": entity_id}
    if extra:
        try:
            data.update(_json.loads(extra))
        except Exception:
            pass
    try:
        _ha_post(f"services/{domain}/{service}", data)
        action_hu = {
            "turn_on":  "bekapcsolta",
            "turn_off": "kikapcsolta",
            "toggle":   "átváltotta",
        }.get(service, service)
        return f"OK — {action_hu}: {entity_id}"
    except Exception as e:
        return f"Home Assistant hiba: {e}\nEllenőrizd: STK_HA_URL, STK_HA_TOKEN"


@REGISTRY.register(
    name="ha_status",
    description="Okosotthon eszköz állapotának lekérdezése",
    parameters={
        "entity_id": "entitás azonosítója, vagy 'all' az összes eszközhöz"
    }
)
def ha_status(entity_id: str = "all") -> str:
    try:
        if entity_id == "all":
            states = _ha_get("states")
            if not isinstance(states, list):
                return str(states)
            lines = []
            for s in states[:30]:  # max 30
                eid   = s.get("entity_id", "")
                state = s.get("state", "")
                attrs = s.get("attributes", {})
                name  = attrs.get("friendly_name", eid)
                lines.append(f"  {name}: {state}")
            return "Eszközök:\n" + "\n".join(lines)
        else:
            s     = _ha_get(f"states/{entity_id}")
            state = s.get("state", "ismeretlen")
            attrs = s.get("attributes", {})
            name  = attrs.get("friendly_name", entity_id)
            detail = ""
            if "temperature" in attrs:
                detail += f", hőmérséklet: {attrs['temperature']}°C"
            if "brightness" in attrs:
                detail += f", fényerő: {attrs['brightness']}"
            return f"{name}: {state}{detail}"
    except Exception as e:
        return f"Home Assistant lekérdezési hiba: {e}"


@REGISTRY.register(
    name="ha_scene",
    description="Jelenet aktiválása (pl. este, film, reggel, vendég)",
    parameters={
        "scene_id": "jelenet azonosítója (pl. scene.este, scene.film)"
    }
)
def ha_scene(scene_id: str) -> str:
    try:
        _ha_post("services/scene/turn_on", {"entity_id": scene_id})
        return f"Jelenet aktiválva: {scene_id}"
    except Exception as e:
        return f"Jelenet hiba: {e}"


@REGISTRY.register(
    name="ha_climate",
    description="Termosztát beállítása szobában",
    parameters={
        "entity_id":   "termosztát azonosítója (pl. climate.nappali)",
        "temperature": "kívánt hőmérséklet Celsius-ban",
        "mode":        "mód: heat|cool|auto|off (opcionális)"
    }
)
def ha_climate(entity_id: str, temperature: str, mode: str = "") -> str:
    try:
        data: Dict[str, Any] = {
            "entity_id":          entity_id,
            "temperature":        float(temperature)
        }
        if mode:
            data["hvac_mode"] = mode
        _ha_post("services/climate/set_temperature", data)
        return f"Termosztát beállítva: {entity_id} → {temperature}°C"
    except Exception as e:
        return f"Termosztát hiba: {e}"
