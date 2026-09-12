"""
Irodai eszközök — naptár (CalDAV), e-mail (IMAP/SMTP), fájlok, időjárás.

Beállítás environment változókkal:
  STK_CALDAV_URL, STK_CALDAV_USER, STK_CALDAV_PASS
  STK_EMAIL_HOST, STK_EMAIL_USER, STK_EMAIL_PASS
  STK_SMTP_HOST (opcionális, default = EMAIL_HOST)
"""
from __future__ import annotations
import imaplib, email, os, smtplib, textwrap
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import List
from .registry import REGISTRY
from .web import web_search


# ── Naptár ───────────────────────────────────────────────────────────────────

@REGISTRY.register(
    name="calendar_list",
    description="Közelgő naptáreseményeket listáz (CalDAV)",
    parameters={
        "days": "hány napra előre (opcionális, default 7)"
    }
)
def calendar_list(days: str = "7") -> str:
    try:
        import caldav
        url  = os.environ.get("STK_CALDAV_URL", "")
        user = os.environ.get("STK_CALDAV_USER", "")
        pwd  = os.environ.get("STK_CALDAV_PASS", "")
        if not url:
            return "CalDAV nem konfigurálva. Állítsd be: STK_CALDAV_URL, STK_CALDAV_USER, STK_CALDAV_PASS"
        client    = caldav.DAVClient(url=url, username=user, password=pwd)
        principal = client.principal()
        calendars = principal.calendars()
        if not calendars:
            return "Nincs elérhető naptár."
        now  = datetime.now()
        end  = now + timedelta(days=int(days))
        events = []
        for cal in calendars[:3]:
            for ev in cal.date_search(start=now, end=end, expand=True):
                comp = ev.vobject_instance.vevent
                dtstart = comp.dtstart.value
                summary = str(comp.summary.value) if hasattr(comp, "summary") else "(névtelen)"
                if hasattr(dtstart, "strftime"):
                    events.append((dtstart, summary))
        events.sort(key=lambda x: x[0])
        if not events:
            return f"Nincs esemény a következő {days} napban."
        lines = [f"Közelgő {days} nap eseményei:"]
        for dt, name in events[:15]:
            lines.append(f"  {dt.strftime('%m.%d %H:%M')}  {name}")
        return "\n".join(lines)
    except ImportError:
        return "caldav csomag szükséges: pip install caldav"
    except Exception as e:
        return f"Naptár lekérdezési hiba: {e}"


@REGISTRY.register(
    name="calendar_add",
    description="Esemény hozzáadása a naptárhoz",
    parameters={
        "title":      "esemény neve",
        "start":      "kezdet: ÉÉÉÉ-HH-NN ÓÓ:PP",
        "end":        "vég: ÉÉÉÉ-HH-NN ÓÓ:PP (opcionális, default: 1 óra)",
        "description":"leírás (opcionális)"
    }
)
def calendar_add(title: str, start: str, end: str = "",
                 description: str = "") -> str:
    try:
        import caldav
        from icalendar import Calendar, Event
        import uuid
        url  = os.environ.get("STK_CALDAV_URL", "")
        user = os.environ.get("STK_CALDAV_USER", "")
        pwd  = os.environ.get("STK_CALDAV_PASS", "")
        if not url:
            return "CalDAV nem konfigurálva."
        dt_start = datetime.strptime(start, "%Y-%m-%d %H:%M")
        dt_end   = (datetime.strptime(end, "%Y-%m-%d %H:%M") if end
                    else dt_start + timedelta(hours=1))
        cal = Calendar()
        ev  = Event()
        ev.add("summary",     title)
        ev.add("dtstart",     dt_start)
        ev.add("dtend",       dt_end)
        ev.add("description", description)
        ev.add("uid",         str(uuid.uuid4()))
        cal.add_component(ev)
        client    = caldav.DAVClient(url=url, username=user, password=pwd)
        principal = client.principal()
        calendars = principal.calendars()
        if not calendars:
            return "Nincs elérhető naptár."
        calendars[0].add_event(cal.to_ical().decode())
        return f"Esemény hozzáadva: '{title}' — {start}"
    except ImportError:
        return "caldav és icalendar csomagok szükségesek: pip install caldav icalendar"
    except Exception as e:
        return f"Naptár írási hiba: {e}"


# ── E-mail ────────────────────────────────────────────────────────────────────

@REGISTRY.register(
    name="email_read",
    description="Legutóbbi e-mailek listázása a beérkező levelek mappából",
    parameters={
        "n":      "e-mailek száma (opcionális, default 5)",
        "folder": "mappa neve (opcionális, default INBOX)"
    }
)
def email_read(n: str = "5", folder: str = "INBOX") -> str:
    host = os.environ.get("STK_EMAIL_HOST", "")
    user = os.environ.get("STK_EMAIL_USER", "")
    pwd  = os.environ.get("STK_EMAIL_PASS", "")
    if not host:
        return "E-mail nem konfigurálva. Állítsd be: STK_EMAIL_HOST, STK_EMAIL_USER, STK_EMAIL_PASS"
    try:
        mail = imaplib.IMAP4_SSL(host)
        mail.login(user, pwd)
        mail.select(folder)
        _, data = mail.search(None, "ALL")
        ids = data[0].split()[-int(n):]
        lines = [f"Legutóbbi {n} e-mail ({folder}):"]
        for uid in reversed(ids):
            _, msg_data = mail.fetch(uid, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            subject = email.header.decode_header(msg["Subject"])[0]
            subj_str = (subject[0].decode(subject[1] or "utf-8")
                        if isinstance(subject[0], bytes) else subject[0])
            sender  = msg["From"][:50]
            date    = msg["Date"][:20] if msg["Date"] else ""
            lines.append(f"  [{date}] {sender}\n  {subj_str}")
        mail.logout()
        return "\n".join(lines)
    except Exception as e:
        return f"E-mail lekérdezési hiba: {e}"


@REGISTRY.register(
    name="email_send",
    description="E-mail küldése",
    parameters={
        "to":      "címzett e-mail cím",
        "subject": "tárgy",
        "body":    "üzenet szövege"
    }
)
def email_send(to: str, subject: str, body: str) -> str:
    user      = os.environ.get("STK_EMAIL_USER", "")
    pwd       = os.environ.get("STK_EMAIL_PASS", "")
    smtp_host = os.environ.get("STK_SMTP_HOST",
                os.environ.get("STK_EMAIL_HOST", ""))
    smtp_port = int(os.environ.get("STK_SMTP_PORT", "587"))
    if not smtp_host:
        return "SMTP nem konfigurálva."
    try:
        msg           = MIMEText(body, "plain", "utf-8")
        msg["From"]   = user
        msg["To"]     = to
        msg["Subject"]= subject
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(user, pwd)
            s.send_message(msg)
        return f"E-mail elküldve: '{subject}' → {to}"
    except Exception as e:
        return f"E-mail küldési hiba: {e}"


# ── Időjárás ─────────────────────────────────────────────────────────────────

@REGISTRY.register(
    name="weather",
    description="Időjárás lekérdezése városra (wttr.in — API-kulcs nélkül)",
    parameters={
        "city": "város neve (pl. Budapest)",
        "days": "előrejelzés napok száma 1-3 (opcionális, default 1)"
    }
)
def weather(city: str, days: str = "1") -> str:
    try:
        import httpx
        # wttr.in API — ingyenes, API-kulcs nélkül
        r = httpx.get(
            f"https://wttr.in/{city}",
            params={"format": "j1", "lang": "hu"},
            timeout=8.0,
            headers={"User-Agent": "curl/8.0"}
        )
        data = r.json()
        current = data["current_condition"][0]
        temp    = current["temp_C"]
        feels   = current["FeelsLikeC"]
        desc    = current["lang_hu"][0]["value"] if current.get("lang_hu") else current["weatherDesc"][0]["value"]
        wind    = current["windspeedKmph"]
        humid   = current["humidity"]
        result  = f"Időjárás: {city}\n"
        result += f"  {temp}°C (érzet: {feels}°C), {desc}\n"
        result += f"  Szél: {wind} km/h, Páratartalom: {humid}%\n"
        n_days = min(int(days), 3)
        if n_days > 1 and "weather" in data:
            result += "\nElőrejelzés:\n"
            for day in data["weather"][:n_days]:
                date    = day["date"]
                max_t   = day["maxtempC"]
                min_t   = day["mintempC"]
                desc_d  = day["hourly"][4]["lang_hu"][0]["value"] if day["hourly"][4].get("lang_hu") else ""
                result += f"  {date}: {min_t}–{max_t}°C  {desc_d}\n"
        return result.strip()
    except ImportError:
        return "httpx csomag szükséges: pip install httpx"
    except Exception as e:
        # fallback: web keresés
        return web_search(f"időjárás {city} ma", n=2)


# ── Fájlkezelés ───────────────────────────────────────────────────────────────

@REGISTRY.register(
    name="file_read",
    description="Helyi szöveges fájl olvasása (txt, md, csv)",
    parameters={
        "path":      "fájl elérési útja",
        "max_chars": "max karakterek (opcionális, default 3000)"
    }
)
def file_read(path: str, max_chars: str = "3000") -> str:
    try:
        content = open(path, encoding="utf-8", errors="replace").read()
        return content[:int(max_chars)]
    except FileNotFoundError:
        return f"Fájl nem található: {path}"
    except Exception as e:
        return f"Fájl olvasási hiba: {e}"


@REGISTRY.register(
    name="reminder_set",
    description="Emlékeztető beállítása (memóriában tárolja)",
    parameters={
        "text":     "emlékeztető szövege",
        "datetime": "mikor: ÉÉÉÉ-HH-NN ÓÓ:PP (opcionális)",
        "repeat":   "ismétlés: daily|weekly|none (opcionális)"
    }
)
def reminder_set(text: str, datetime: str = "", repeat: str = "none") -> str:
    return {
        "action":   "reminder_registered",
        "text":     text,
        "datetime": datetime,
        "repeat":   repeat,
        "message":  f"Emlékeztető beállítva: '{text}'" + (f" — {datetime}" if datetime else "")
    }
