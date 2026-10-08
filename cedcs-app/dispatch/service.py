"""Dispatch orchestration: who gets told, in what order, and the confirmation loop.

  send_dispatch   family + nearest hospital + nearest ambulance (needs explicit confirmation; one dispatch per case)
  record_ack      a hospital's answer (link, e-mail reply, or simulated in dry-run); ACCEPT reserves a bed, DECLINE escalates
  escalate        tell the next-ranked hospital when the first declined or stayed silent past the timeout
  send_update     family status update ("updated health condition")
  poll_imap       optional: read replies from a mailbox and turn them into acks
"""

from __future__ import annotations

import email
import imaplib
import logging
import os
import re
import time
from typing import Optional

from dispatch import ambulances, compose, mailer, store
from services import history

logger = logging.getLogger("cedcs.dispatch")

RESPONSES = {"accept": "ACCEPTED", "accepted": "ACCEPTED", "decline": "DECLINED", "declined": "DECLINED", "eta": "ETA_UPDATE"}


def public_base() -> str:
    return (os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "http://127.0.0.1:8001").rstrip("/")


def ack_urls(token: str) -> dict:
    base = f"{public_base()}/ack/{token}"
    return {"token": token, "page": base, "accept": f"{base}?response=accept", "decline": f"{base}?response=decline"}


def _road_minutes(origin, dests):
    from services import locationiq

    return locationiq.route_minutes(origin, dests)


def choose_ambulance(snap: dict) -> Optional[dict]:
    loc = snap["location"]
    return ambulances.nearest(loc["lat"], loc["lng"], road_minutes=_road_minutes)


def hospital_address(hospital_id: str) -> Optional[str]:
    """Where an alert to this hospital goes. The fictional network has synthetic desks; a REAL hospital has an address only if an
    administrator registered one (its console contact). Never guessed."""
    from services import registry

    if registry.is_real_id(hospital_id):
        c = store.contact_for(hospital_id)
        return (c or {}).get("email") or None
    return ambulances.hospital_email(hospital_id)


def _bed_key(snap: dict) -> str:
    return "icu_beds" if "ICU" in snap.get("needs", []) else "emergency_beds"


# ---------------------------------------------------------------- preview (nothing is stored or sent)
def preview(snap: dict) -> dict:
    ambulance = choose_ambulance(snap)
    msgs = [compose.family_email(snap, c, ambulance) for c in store.list_contacts()]
    if snap["hospitals"]:
        h = snap["hospitals"][0]
        m = compose.hospital_email(snap, h, ambulance, ack_urls("PREVIEWTOKEN" + "0" * 20))
        m["to_email"] = hospital_address(h["id"])
        msgs.append(m)
    if ambulance:
        msgs.append(compose.ambulance_email(snap, ambulance, snap["hospitals"][0] if snap["hospitals"] else None))
    return {"messages": [{k: m[k] for k in ("kind", "to_name", "to_email", "subject", "body_text")} for m in msgs],
            "ambulance": ambulance, "mail": mailer.describe(), "family_contacts": len(store.list_contacts())}


# ---------------------------------------------------------------- sending
def _deliver(mid: str, msg: dict) -> dict:
    res = mailer.send(msg)
    store.update_message(mid, status=res["status"], actual_to=res["actual_to"], subject=res["subject"], sent_at=mailer.now(), error=res["error"])
    return res


def _add_hospital_message(did: str, snap: dict, hospital: dict, ambulance: Optional[dict], backup: bool = False) -> str:
    to = hospital_address(hospital["id"])
    stub = store.add_message(did, "HOSPITAL", hospital["name"], to, "(composing)", "", "", hospital_id=hospital["id"])
    msg = compose.hospital_email(snap, hospital, ambulance, ack_urls(stub["token"]), backup=backup)
    msg["to_email"] = to
    store.update_message(stub["id"], subject=msg["subject"], body_text=msg["body_text"], body_html=msg["body_html"])
    if to is None:  # a real hospital with no registered address: nothing is sent; a person must phone them
        phone = hospital.get("phone")
        store.update_message(stub["id"], status="NO_ADDRESS", error="No dispatch email on file for this hospital: " + (f"call {phone}" if phone else "find their emergency number and call"))
        return stub["id"]
    _deliver(stub["id"], msg)
    return stub["id"]


def send_dispatch(snap: dict, *, confirmed: bool, include_family: bool = True) -> dict:
    """Notify family, the nearest (best-ranked) hospital and the nearest ambulance. Refuses without confirmation, and a case
    is only ever dispatched once (asking again returns the existing dispatch instead of sending duplicates)."""
    if not confirmed:
        raise PermissionError("confirmation required: sending emergency alerts must be an explicit decision")
    existing = store.dispatch_for_case(snap["case_id"])
    if existing:
        return status(existing, tick=False)
    if not snap["hospitals"]:
        raise ValueError("no recommended hospital to notify")

    ambulance = choose_ambulance(snap)
    did = store.create_dispatch(snap["case_id"], mailer.mode(), {"snapshot": snap, "ambulance": ambulance,
                                                                "notified": [snap["hospitals"][0]["id"]], "escalated_from": []})
    if include_family:
        for c in store.list_contacts():
            m = compose.family_email(snap, c, ambulance)
            mid = store.add_message(did, "FAMILY", c["name"], c["email"], m["subject"], m["body_text"], m["body_html"])["id"]
            _deliver(mid, m)
    _add_hospital_message(did, snap, snap["hospitals"][0], ambulance)
    if ambulance:
        a = compose.ambulance_email(snap, ambulance, snap["hospitals"][0])
        mid = store.add_message(did, "AMBULANCE", ambulance["name"], ambulance["email"], a["subject"], a["body_text"], a["body_html"])["id"]
        _deliver(mid, a)
    out = status(did, tick=False)
    history.event("dispatch_sent", case_id=snap["case_id"], hospital_id=snap["hospitals"][0]["id"], dispatch_id=did, mode=out["mode"],
                  hospital=snap["hospitals"][0]["name"], ambulance=(ambulance or {}).get("name"), family_alerts=sum(m["kind"] == "FAMILY" for m in out["messages"]),
                  messages=[{"kind": m["kind"], "status": m["status"]} for m in out["messages"]])
    return out


def send_update(did: str, note: str) -> dict:
    d = store.get_dispatch(did)
    if d is None:
        raise KeyError("unknown dispatch")
    note = (note or "").strip()
    if not note:
        raise ValueError("an update needs some text")
    snap, ambulance = d["payload"]["snapshot"], d["payload"].get("ambulance")
    for c in store.list_contacts():
        m = compose.family_email(snap, c, ambulance, is_update=True, note=note[:1000])
        mid = store.add_message(did, "FAMILY", c["name"], c["email"], m["subject"], m["body_text"], m["body_html"])["id"]
        _deliver(mid, m)
    history.event("family_update", case_id=snap["case_id"], dispatch_id=did, note=note[:200])
    return status(did, tick=False)


# ---------------------------------------------------------------- confirmation loop
def escalate(did: str, reason: str = "manual") -> Optional[dict]:
    """Notify the next-ranked hospital that has not been contacted yet. Returns its info, or None if there is nobody left."""
    d = store.get_dispatch(did)
    if d is None:
        raise KeyError("unknown dispatch")
    payload = d["payload"]
    nxt = next((h for h in payload["snapshot"]["hospitals"] if h["id"] not in payload["notified"]), None)
    if nxt is None:
        store.update_dispatch(did, status="NO_MORE_HOSPITALS")
        return None
    _add_hospital_message(did, payload["snapshot"], nxt, payload.get("ambulance"), backup=True)
    payload["notified"].append(nxt["id"])
    store.update_dispatch(did, payload=payload)
    logger.info("dispatch %s escalated to %s (%s)", did, nxt["id"], reason)
    history.event("dispatch_escalated", case_id=payload["snapshot"]["case_id"], hospital_id=nxt["id"], dispatch_id=did, reason=reason, hospital=nxt["name"])
    return nxt


def _refresh_status(did: str) -> str:
    d = store.get_dispatch(did)
    hosp = [m for m in d["messages"] if m["kind"] == "HOSPITAL"]
    latest = [(m["ack"] or {}).get("response") for m in hosp]
    if "ACCEPTED" in latest:
        st = "CONFIRMED"
    elif d["status"] == "NO_MORE_HOSPITALS":
        st = "NO_MORE_HOSPITALS"
    elif hosp and all(x == "DECLINED" for x in latest):
        st = "AWAITING_NEXT" if d["payload"]["snapshot"]["hospitals"] and len(d["payload"]["notified"]) < len(d["payload"]["snapshot"]["hospitals"]) else "NO_MORE_HOSPITALS"
    else:
        st = "AWAITING_CONFIRMATION"
    if st != d["status"]:
        store.update_dispatch(did, status=st)
    return st


def record_ack(token: str, response: str, note: str = "", eta_min: Optional[float] = None, source: str = "LINK") -> dict:
    msg = store.message_by_token(token)
    if msg is None or msg["kind"] != "HOSPITAL":
        raise KeyError("unknown confirmation link")
    resp = RESPONSES.get((response or "").strip().lower())
    if resp is None:
        raise ValueError("response must be accept, decline or eta")
    d = store.get_dispatch(msg["dispatch_id"])
    prior = next((m for m in d["messages"] if m["id"] == msg["id"]), {}).get("acks", [])
    bed: dict = {}
    if resp == "ACCEPTED" and not any(a["response"] == "ACCEPTED" for a in prior):  # reserve once, however often the link is clicked
        key = _bed_key(d["payload"]["snapshot"])
        try:
            from fallback import rule_based

            bed = rule_based.reserve_bed(msg["hospital_id"], key, token)
            bed["bed_key"] = key
        except Exception as exc:
            bed = {"ok": False, "reason": f"bed reservation unavailable: {type(exc).__name__}", "bed_key": key}
    store.add_ack(msg["id"], resp, (note or "")[:500], eta_min, bed, source)
    st = _refresh_status(msg["dispatch_id"])
    if resp == "DECLINED" and store.get_settings()["auto_escalate"]:
        escalate(msg["dispatch_id"], reason="declined")
        st = _refresh_status(msg["dispatch_id"])
    history.event("hospital_response", case_id=d["payload"]["snapshot"]["case_id"], hospital_id=msg["hospital_id"], dispatch_id=msg["dispatch_id"],
                  response=resp, source=source, eta_min=eta_min, note=(note or "")[:200], bed=bed, dispatch_status=st, hospital=msg["to_name"])
    return {"dispatch_id": msg["dispatch_id"], "response": resp, "bed": bed, "dispatch_status": st, "hospital": msg["to_name"]}


def manual_ack(did: str, message_id: str, response: str, note: str = "") -> dict:
    """A person records what a hospital said by phone (or any channel outside e-mail). Same effects as a link confirmation."""
    d = store.get_dispatch(did)
    msg = next((m for m in (d or {}).get("messages", []) if m["id"] == message_id and m["kind"] == "HOSPITAL"), None)
    if msg is None:
        raise KeyError("unknown hospital message")
    return record_ack(msg["token"], response, note=note or "(recorded by an operator)", source="OPERATOR")


def status(did: str, tick: bool = True) -> Optional[dict]:
    """Full dispatch view. With tick=True, a hospital that has not answered within the timeout triggers escalation (once)."""
    d = store.get_dispatch(did)
    if d is None:
        return None
    timeout = store.get_settings()["ack_timeout_s"]
    now = time.time()
    overdue = []
    for m in d["messages"]:
        if m["kind"] == "HOSPITAL" and m["status"] in ("SENT", "LOGGED") and not m["acks"] and m["sent_at"] and now - m["sent_at"] > timeout:
            overdue.append(m["id"])
    if tick and overdue and store.get_settings()["auto_escalate"]:
        payload = d["payload"]
        fresh = [mid for mid in overdue if mid not in payload["escalated_from"]]
        if fresh:
            payload["escalated_from"].extend(fresh)
            store.update_dispatch(did, payload=payload)
            escalate(did, reason="no answer in time")
    _refresh_status(did)
    d = store.get_dispatch(did)
    d["overdue"] = overdue
    d["ack_timeout_s"] = timeout
    d["mail"] = mailer.describe()
    for m in d["messages"]:
        m["ack_urls"] = ack_urls(m["token"]) if m["kind"] == "HOSPITAL" else None
        m["seconds_waiting"] = round(now - m["sent_at"]) if m["kind"] == "HOSPITAL" and m["sent_at"] and not m["acks"] else None
    return d


# ---------------------------------------------------------------- optional: read hospital replies from a mailbox
_TOKEN_RE = re.compile(r"\[CEDCS-ACK ([0-9a-f]{8})\]", re.I)


def imap_configured() -> bool:
    return all(os.environ.get(k) for k in ("IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD"))


def poll_imap() -> int:
    """Turn unread replies whose subject contains [CEDCS-ACK <id>] into acks. First word ACCEPT / DECLINE in the body decides."""
    if not imap_configured():
        return 0
    handled = 0
    box = imaplib.IMAP4_SSL(os.environ["IMAP_HOST"], int(os.environ.get("IMAP_PORT", "993")))
    try:
        box.login(os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"])
        box.select("INBOX")
        _, data = box.search(None, "UNSEEN", "SUBJECT", '"CEDCS-ACK"')
        for num in (data[0] or b"").split():
            _, parts = box.fetch(num, "(RFC822)")
            em = email.message_from_bytes(parts[0][1])
            m = _TOKEN_RE.search(em.get("Subject", ""))
            if not m:
                continue
            body = ""
            for part in em.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="replace")
                    break
            first = re.search(r"\b(accept|decline)\w*\b", body, re.I)
            if not first:
                continue
            prefix = m.group(1).lower()
            with store._lock:
                row = store.db().execute("SELECT token FROM messages WHERE kind='HOSPITAL' AND token LIKE ?", (prefix + "%",)).fetchone()
            if row is None:
                continue
            record_ack(row["token"], first.group(1), note=body.strip()[:300], source="EMAIL_REPLY")
            box.store(num, "+FLAGS", "\\Seen")
            handled += 1
    finally:
        try:
            box.logout()
        except Exception:
            pass
    return handled
