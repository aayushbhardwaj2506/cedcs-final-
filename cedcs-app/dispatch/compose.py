"""Compose the emergency emails from a finished case. Deterministic text only: no LLM writes what goes to a family or a
hospital, and nothing here diagnoses; it states what was reported and what the system assessed.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Optional

DISCLAIMER = ("This alert comes from the CEDCS emergency-routing PROTOTYPE running on synthetic data. "
              "It is not a certified medical or emergency service.")


def map_links(lat: float, lng: float) -> dict:
    return {
        "osm": f"https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lng:.5f}#map=17/{lat:.5f}/{lng:.5f}",
        "google": f"https://www.google.com/maps?q={lat:.5f},{lng:.5f}",
    }


def snapshot(case_id: str, request: dict, result: dict) -> dict:
    """Everything the emails need, taken from a finished /cases result (recommendation + trace)."""
    rec, trace = result["recommendation"], result["trace"]
    intake, triage = trace.get("intake", {}), trace.get("triage", {})
    flags = trace.get("red_flags", {})
    elig = {c["id"]: c for c in trace.get("eligibility", {}).get("candidates", [])}

    def hosp(h):
        c = elig.get(h["hospital_id"], {})
        return {"id": h["hospital_id"], "name": h["name"], "eta_min": h.get("eta_min"), "lat": c.get("lat"), "lng": c.get("lng"),
                "rank": h["rank"], "phone": c.get("phone"), "source": c.get("source", "registry"), "address": c.get("address")}

    return {
        "case_id": case_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "priority": flags.get("final_priority") or triage.get("priority", "UNKNOWN"),
        "location": {"lat": request["location_lat"], "lng": request["location_lng"], "address": request.get("location_address") or "",
                     "source": request.get("location_source", "MANUAL"), "accuracy_m": request.get("location_accuracy_m")},
        "patient": {"age": intake.get("age"), "gender": intake.get("gender"), "consciousness": intake.get("consciousness"),
                    "breathing": intake.get("breathing"), "bleeding": intake.get("bleeding"), "symptoms": intake.get("symptoms") or [],
                    "history": intake.get("history") or []},
        "needs": triage.get("required") or [], "red_flags": flags.get("rules") or [],
        "hospitals": [hosp(rec["primary"])] + [hosp(a) for a in rec.get("alternatives", [])],
        "escalated": bool(rec.get("escalated")), "confidence": rec.get("confidence_level"),
    }


def condition_text(snap: dict) -> str:
    p = snap["patient"]
    bits = []
    if p.get("consciousness") not in (None, "unknown"):
        bits.append({"alert": "awake", "confused": "confused", "unresponsive": "unresponsive"}.get(p["consciousness"], p["consciousness"]))
    if p.get("breathing") not in (None, "unknown"):
        bits.append({"normal": "breathing normally", "laboured": "breathing with difficulty", "slow": "breathing slowly",
                     "absent": "NOT breathing"}.get(p["breathing"], p["breathing"]))
    if p.get("bleeding") not in (None, "unknown", "none"):
        bits.append(f"{p['bleeding']} bleeding")
    who = ", ".join(x for x in [f"{p['age']} years old" if p.get("age") else "", p.get("gender") or ""] if x)
    parts = []
    if who:
        parts.append(who.capitalize())
    if bits:
        parts.append("Reported: " + ", ".join(bits))
    if p.get("symptoms"):
        parts.append("Symptoms: " + ", ".join(p["symptoms"]))
    if p.get("history"):
        parts.append("Known history: " + ", ".join(p["history"]))
    parts.append(f"Assessed urgency: {snap['priority']}")
    return ". ".join(parts) + "."


def _loc_lines(snap: dict) -> list:
    loc = snap["location"]
    links = map_links(loc["lat"], loc["lng"])
    acc = f" (accurate to about {int(loc['accuracy_m'])} m)" if loc.get("accuracy_m") else ""
    src = {"GPS": "device GPS", "GEOCODED": "searched address", "MANUAL": "manually chosen point"}.get(loc.get("source"), "")
    lines = [f"Location: {loc['lat']:.5f}, {loc['lng']:.5f}{acc}" + (f" [{src}]" if src else "")]
    if loc.get("address"):
        lines.append(f"Address: {loc['address']}")
    lines += [f"Map: {links['osm']}", f"Google Maps: {links['google']}"]
    return lines


def _html(text: str, buttons: Optional[list] = None) -> str:
    esc = html.escape(text).replace("\n", "<br>")
    for label, url, color in buttons or []:
        esc += f'<p><a href="{html.escape(url)}" style="display:inline-block;padding:10px 16px;background:{color};color:#fff;border-radius:6px;text-decoration:none;font-weight:600">{html.escape(label)}</a></p>'
    return f'<div style="font:15px/1.5 system-ui,sans-serif;color:#111">{esc}<hr><p style="color:#666;font-size:12px">{html.escape(DISCLAIMER)}</p></div>'


def family_email(snap: dict, contact: dict, ambulance: Optional[dict], is_update: bool = False, note: str = "") -> dict:
    h = snap["hospitals"][0] if snap["hospitals"] else None
    lines = [f"Hello {contact['name']},", ""]
    if is_update:
        lines += [f"UPDATE on the emergency case {snap['case_id'][:8]}:", note or "The situation has changed.", ""]
    else:
        lines += ["You are listed as an emergency contact. An emergency has been reported and help is being arranged.", ""]
    lines += ["CURRENT CONDITION", condition_text(snap), ""]
    lines += ["WHERE THE PERSON IS"] + _loc_lines(snap) + [""]
    if h:
        eta = f", about {h['eta_min']:.0f} min away" if h.get("eta_min") is not None else ""
        lines += ["WHERE THEY ARE BEING TAKEN", f"{h['name']}{eta}", ""]
    if ambulance:
        lines += ["AMBULANCE", f"{ambulance['name']} ({ambulance['type']}), about {ambulance['eta_min']:.0f} min from the patient", ""]
    lines += [f"Case reference: {snap['case_id'][:8]}", "You will receive another message if the situation changes."]
    text = "\n".join(lines) + f"\n\n{DISCLAIMER}"
    prefix = "UPDATE" if is_update else "EMERGENCY"
    return {"kind": "FAMILY", "to_name": contact["name"], "to_email": contact["email"],
            "subject": f"[{prefix}] Emergency case {snap['case_id'][:8]}: {snap['priority']} urgency", "body_text": text,
            "body_html": _html("\n".join(lines))}


def hospital_email(snap: dict, hospital: dict, ambulance: Optional[dict], ack_urls: dict, backup: bool = False) -> dict:
    h = hospital
    lines = [f"Emergency desk, {h['name']},", ""]
    lines += ["BACKUP REQUEST: the first hospital contacted did not accept in time. Please respond quickly." if backup else
              "INCOMING EMERGENCY PATIENT: please confirm you can receive them.", ""]
    lines += [f"Urgency: {snap['priority']}" + ("  (escalated: red-flag rules fired)" if snap["red_flags"] else ""), condition_text(snap), ""]
    if snap["needs"]:
        lines += ["Capabilities needed: " + ", ".join(n.replace("_", " ").lower() for n in snap["needs"]), ""]
    if h.get("eta_min") is not None:
        lines += [f"Estimated arrival: about {h['eta_min']:.0f} min"]
    if ambulance:
        lines += [f"Ambulance: {ambulance['name']} ({ambulance['type']}), {ambulance['phone']}"]
    lines += [""] + ["PATIENT LOCATION"] + _loc_lines(snap) + [""]
    lines += ["PLEASE CONFIRM using one of these links (a bed will be reserved when you accept):",
              f"Accept: {ack_urls['accept']}", f"Decline (no capacity): {ack_urls['decline']}", "",
              f"Case reference: {snap['case_id'][:8]}"]
    text = "\n".join(lines) + f"\n\n{DISCLAIMER}"
    buttons = [("Accept: we can receive the patient", ack_urls["accept"], "#0f9d6b"), ("Decline: no capacity", ack_urls["decline"], "#e11d48")]
    body = "\n".join(l for l in lines if not l.startswith(("Accept:", "Decline (")))
    return {"kind": "HOSPITAL", "hospital_id": h["id"], "to_name": h["name"], "to_email": None,  # address filled by the service
            "subject": f"[{'BACKUP ' if backup else ''}EMERGENCY] Incoming {snap['priority']} patient, ETA {int(h['eta_min']) if h.get('eta_min') is not None else '?'} min "
                       f"[CEDCS-ACK {ack_urls['token'][:8]}]",
            "body_text": text, "body_html": _html(body, buttons)}


def ambulance_email(snap: dict, ambulance: dict, hospital: Optional[dict]) -> dict:
    lines = [f"Dispatch desk, {ambulance['name']},", "", "EMERGENCY PICK-UP REQUEST", "",
             f"Urgency: {snap['priority']}", condition_text(snap), "", "PICK-UP LOCATION"] + _loc_lines(snap) + [""]
    if hospital:
        lines += [f"Destination: {hospital['name']}" + (f" (about {hospital['eta_min']:.0f} min from the patient)" if hospital.get("eta_min") is not None else ""), ""]
    lines += [f"Your estimated time to reach the patient: about {ambulance['eta_min']:.0f} min ({ambulance['distance_km']} km)", "",
              f"Case reference: {snap['case_id'][:8]}"]
    text = "\n".join(lines) + f"\n\n{DISCLAIMER}"
    return {"kind": "AMBULANCE", "to_name": ambulance["name"], "to_email": ambulance["email"],
            "subject": f"[EMERGENCY] Pick-up needed: {snap['priority']} urgency, case {snap['case_id'][:8]}", "body_text": text,
            "body_html": _html("\n".join(lines))}
