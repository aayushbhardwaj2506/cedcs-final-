"""Rule-based stand-ins for the LLM-facing stages, so the system works with no
API key or when the LLM is unavailable (design principle P8: degrade, don't fail).

  parse_intake   free text + toggles -> StructuredCase   (replaces intake agent)
  assess_triage  StructuredCase      -> TriageResult     (replaces triage agent)
  discover       location            -> candidates + resource snapshots via resource_service HTTP
  explain        ValidatedRecommendation -> plain-language text (replaces narrator)

Same contracts and same guardrails as the agent path: outputs are typed schemas,
triage never emits diagnosis terms, capability keys come only from the closed
vocabulary in deterministic_core.capability_map. Red flags are still applied by
the orchestrator afterwards, escalate-only.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import requests

from deterministic_core.capability_map import CAPABILITY_RESOURCE_MAP
from schemas.case import Incident, Location, Patient, StructuredCase
from schemas.hospital import CandidateHospital
from schemas.recommendation import ValidatedRecommendation
from schemas.resource import ResourceSnapshot
from schemas.triage import TriageResult

# ---------------------------------------------------------------- intake

# Severity scales; a higher rank is more alarming. Used to reconcile the free text with the caller's toggles.
_BREATHING = ["unknown", "normal", "laboured", "slow", "absent"]
_CONSCIOUS = ["unknown", "alert", "confused", "unresponsive"]
_BLEEDING = ["unknown", "none", "mild", "severe"]
_TEXT_SIGNALS = {
    "breathing": {
        "absent": ["not breathing", "no breathing", "stopped breathing", "isn't breathing", "isnt breathing", "no pulse", "pulseless"],
        "laboured": ["gasping", "struggling to breathe", "can't breathe", "cannot breathe", "short of breath", "difficulty breathing"],
    },
    "consciousness": {
        "unresponsive": ["unresponsive", "unconscious", "not responding", "not waking", "won't wake", "wont wake", "no response"],
        "confused": ["confused", "disoriented", "incoherent"],
    },
    "bleeding": {
        "severe": ["bleeding heavily", "heavy bleeding", "severe bleeding", "profuse bleeding", "spurting", "blood everywhere", "soaked in blood"],
    },
}
_SCALES = {"breathing": _BREATHING, "consciousness": _CONSCIOUS, "bleeding": _BLEEDING}


def reconcile_vitals(text: str, patient: Patient) -> tuple:
    """Escalate-only merge of the free text with the toggles: if the text describes a MORE alarming state than the
    toggle, the worse state wins and the conflict is reported. A stale or mistaken toggle can therefore never
    hide a life-threatening report from the hard red-flag rules. Returns (patient, conflicts)."""
    low = f" {text.lower()} "
    updates, conflicts = {}, []
    for field, signals in _TEXT_SIGNALS.items():
        scale, current = _SCALES[field], getattr(patient, field)
        for value, phrases in signals.items():  # ordered most-severe first within each field
            hit = next((ph for ph in phrases if _mentioned(low, ph)), None)
            if hit and scale.index(value) > scale.index(current):
                updates[field] = value
                conflicts.append({"field": field, "toggle": current, "text": value, "phrase": hit})
                break
    return (patient.model_copy(update=updates) if updates else patient), conflicts

_SYMPTOMS = {
    "chest pain": ["chest pain", "chest tightness", "chest pressure"],
    "collapse": ["collapse", "collapsed", "fainted", "passed out", "syncope"],
    "confusion": ["confused", "confusion", "disoriented", "drowsy"],
    "breathlessness": ["breathless", "short of breath", "shortness of breath", "gasping", "can't breathe", "cannot breathe"],
    "weakness one side": ["weakness", "one side", "facial droop", "face drooping", "slurred", "speech difficulty"],
    "seizure": ["seizure", "convulsion", "fits"],
    "severe bleeding": ["bleeding heavily", "heavy bleeding", "severe bleeding", "profuse bleeding"],
    "injury": ["accident", "crash", "fall ", "fell", "fracture", "broken", "injury", "stab", "gunshot"],
    "pregnancy": ["pregnant", "labour", "labor", "contractions", "delivery"],
    "burn": ["burn", "scald"],
    "poisoning": ["poison", "overdose", "swallowed", "ingested"],
    "fever": ["fever", "high temperature"],
    "vomiting": ["vomit"],
}
_HISTORY = {
    "diabetes": ["diabetic", "diabetes"],
    "hypertension": ["hypertension", "high blood pressure", "bp patient"],
    "heart disease": ["heart disease", "cardiac history", "bypass", "stent", "heart patient"],
    "asthma": ["asthma"],
    "kidney disease": ["dialysis", "kidney disease", "renal failure"],
}


_NEGATORS = ("no ", "not ", "denies ", "denied ", "without ", "never ", "free of ", "negative for ")


def _mentioned(low: str, word: str) -> bool:
    """True if `word` occurs and is not negated by a cue just before it ("no chest pain")."""
    start = 0
    while (i := low.find(word, start)) != -1:
        window = low[max(0, i - 30): i]
        window = re.split(r"[,.;]|\bbut\b|\band\b", window)[-1]  # a clause break resets negation scope
        if not any(n in window for n in _NEGATORS):
            return True
        start = i + len(word)
    return False


def _find(text: str, table: dict) -> list:
    low = f" {text.lower()} "
    return [name for name, words in table.items() if any(_mentioned(low, w) for w in words)]


def parse_intake(raw: dict[str, Any]) -> StructuredCase:
    text = str(raw.get("emergency_report", ""))
    low = text.lower()

    age = None
    m = re.search(r"(\d{1,3})\s*[- ]?\s*(?:years?|yrs?|y/?o|yr-old)", low) or re.search(r"\bage[d:]?\s*(\d{1,3})", low)
    if m and 0 < int(m.group(1)) < 120:
        age = int(m.group(1))
    elif re.search(r"\b(infant|baby|newborn)\b", low):
        age = 1
    elif re.search(r"\b(child|kid|toddler)\b", low):
        age = 6

    gender = None
    if re.search(r"\b(male|man|boy|he|his)\b", low):
        gender = "male"
    elif re.search(r"\b(female|woman|girl|she|her)\b", low):
        gender = "female"

    onset = None
    m = re.search(r"(\d+\s*(?:min|minutes|hours?|hrs?)\s*ago|sudden(?:ly)?)", low)
    if m:
        onset = m.group(1)

    patient = Patient(
        age=age, age_confidence=0.9 if age and m else (0.4 if age else 0.0), gender=gender,
        consciousness=raw.get("consciousness", "unknown"), breathing=raw.get("breathing", "unknown"),
        bleeding=raw.get("bleeding", "unknown"), symptoms=_find(text, _SYMPTOMS),
        onset_time=onset, medical_history=_find(text, _HISTORY),
    )
    tracked = {
        "age": age is not None, "gender": gender is not None, "symptoms": bool(patient.symptoms),
        "consciousness": patient.consciousness != "unknown", "breathing": patient.breathing != "unknown",
        "bleeding": patient.bleeding != "unknown", "onset_time": onset is not None,
        "medical_history": bool(patient.medical_history),
    }
    # ranked by decision impact, highest first
    impact_order = ["consciousness", "breathing", "bleeding", "symptoms", "age", "medical_history", "onset_time", "gender"]
    return StructuredCase(
        patient=patient,
        incident=Incident(mechanism="injury" if "injury" in patient.symptoms else None),
        location=Location(
            lat=raw.get("location_lat"), lng=raw.get("location_lng"),
            address=raw.get("location_address"), source=raw.get("location_source", "MANUAL"),
            accuracy_m=raw.get("location_accuracy_m"),
        ),
        missing_fields=[f for f in impact_order if not tracked[f]],
        field_confidence={k: (0.9 if v else 0.0) for k, v in tracked.items()},
        intake_completeness=round(sum(tracked.values()) / len(tracked), 2),
    )


# ---------------------------------------------------------------- triage

# symptom -> (category, required, preferred)
_RULES = {
    "chest pain": ("CARDIAC", ["EMERGENCY_DEPARTMENT", "CARDIAC_MONITORING"], ["CARDIOLOGY", "CATH_LAB", "BLOOD_BANK"]),
    "collapse": ("CARDIAC", ["EMERGENCY_DEPARTMENT", "CARDIAC_MONITORING"], ["CARDIOLOGY", "CT_SCAN"]),
    "confusion": ("NEUROLOGICAL", ["EMERGENCY_DEPARTMENT"], ["NEUROLOGY", "CT_SCAN"]),
    "weakness one side": ("NEUROLOGICAL", ["EMERGENCY_DEPARTMENT", "CT_SCAN"], ["NEUROLOGY", "ICU"]),
    "seizure": ("NEUROLOGICAL", ["EMERGENCY_DEPARTMENT"], ["NEUROLOGY", "CT_SCAN"]),
    "breathlessness": ("RESPIRATORY", ["EMERGENCY_DEPARTMENT"], ["VENTILATOR", "X_RAY"]),
    "severe bleeding": ("TRAUMA", ["EMERGENCY_DEPARTMENT", "BLOOD_BANK"], ["OT", "TRAUMA"]),
    "injury": ("TRAUMA", ["EMERGENCY_DEPARTMENT"], ["TRAUMA", "X_RAY", "CT_SCAN", "ORTHOPEDICS", "OT"]),
    "pregnancy": ("OBSTETRIC", ["EMERGENCY_DEPARTMENT", "OBSTETRICS"], ["BLOOD_BANK", "OT"]),
    "burn": ("BURNS", ["EMERGENCY_DEPARTMENT"], ["ICU", "OT"]),
    "poisoning": ("TOXICOLOGY", ["EMERGENCY_DEPARTMENT"], ["ICU", "LAB"]),
}
_HIGH_SYMPTOMS = {"chest pain", "weakness one side", "seizure", "severe bleeding", "poisoning", "breathlessness", "pregnancy", "burn"}


def assess_triage(case: StructuredCase) -> TriageResult:
    p = case.patient
    cats: list = []
    req: list = ["EMERGENCY_DEPARTMENT"]
    pref: list = []

    def add(lst, items):
        for i in items:
            if i in CAPABILITY_RESOURCE_MAP and i not in lst:
                lst.append(i)

    symptoms = set(p.symptoms)
    if p.breathing in ("laboured", "slow", "absent"):
        symptoms.add("breathlessness")
    if p.consciousness in ("confused", "unresponsive"):
        symptoms.add("confusion")
    if p.bleeding == "severe":
        symptoms.add("severe bleeding")

    for s in sorted(symptoms):
        if s in _RULES:
            cat, r, pr = _RULES[s]
            if cat not in cats:
                cats.append(cat)
            add(req, r)
            add(pref, pr)
    if p.age is not None and p.age < 12:
        cats.append("PAEDIATRIC")
        add(pref, ["PEDIATRICS", "PEDIATRIC_BED"])
    if any(h in p.medical_history for h in ("heart disease", "diabetes", "hypertension")) and "CARDIAC" in cats:
        add(pref, ["CARDIOLOGY", "CARDIAC_MONITORING"])
    if "kidney disease" in p.medical_history:
        cats.append("RENAL")
        add(pref, ["NEPHROLOGY", "DIALYSIS"])

    # Priority: escalate on combinations; red-flag rules in the orchestrator can only raise it further.
    score = 0
    if p.breathing == "absent" or p.consciousness == "unresponsive":
        score = 3
    elif ({"chest pain", "collapse"} & symptoms) and (
        p.consciousness == "confused" or p.breathing in ("laboured", "slow") or {"chest pain", "collapse"} <= symptoms
    ):
        score = 3
    elif symptoms & _HIGH_SYMPTOMS or p.breathing in ("laboured", "slow") or p.consciousness == "confused":
        score = 2
    elif symptoms:
        score = 1
    priority = ["LOW", "MODERATE", "HIGH", "CRITICAL"][score]

    if score >= 2:
        add(req, ["ICU"] if score == 3 else [])
        if score == 2:
            add(pref, ["ICU"])
    if p.breathing in ("absent", "slow"):
        add(req, ["VENTILATOR"])
    pref = [c for c in pref if c not in req]

    confidence = round(min(0.9, 0.45 + 0.45 * case.intake_completeness), 2)
    parts = []
    if p.consciousness in ("confused", "unresponsive"):
        parts.append(f"{p.consciousness} state")
    if p.breathing in ("laboured", "slow", "absent"):
        parts.append(f"{p.breathing} breathing")
    if symptoms:
        parts.append("reported " + ", ".join(sorted(symptoms)))
    rationale = (
        (", ".join(parts) or "limited information available")
        + f" indicates {priority.lower()} urgency and requires "
        + ", ".join(c.replace("_", " ").lower() for c in req)
        + " capability."
    )
    return TriageResult(
        priority=priority, category_set=cats, required_capabilities=req, preferred_capabilities=pref,
        triage_confidence=confidence, rationale=rationale,
    )


# ---------------------------------------------------------------- discovery

_session = requests.Session()  # keep-alive: avoids a fresh TCP handshake per call


def _base() -> str:
    url = os.environ.get("RESOURCE_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
    # "localhost" can cost ~200 ms per new connection on Windows (tries IPv6 first); use the IPv4 literal.
    return url.replace("://localhost", "://127.0.0.1")


def _merge_reported_status(facilities: list) -> list:
    """A real hospital that has reported its own status (console) is no longer 'assumed open'."""
    from dispatch import store

    reports = store.reports_for([f["hospital_id"] for f in facilities])
    for f in facilities:
        for rep in reports.get(f["hospital_id"], []):
            if rep["resource_key"] == "operating_status":
                f["operating_status"] = rep["value"].get("status", f["operating_status"])
                f["ipd_accepting"] = bool(rep["value"].get("ipd_accepting", f["ipd_accepting"]))
                f["status_verified"] = True
        c = store.contact_for(f["hospital_id"])
        if c and c.get("phone"):
            f["phone"] = c["phone"]
    return facilities


def _enrich_from_tags(facilities: list, lat: float, lng: float, radius: float) -> list:
    """Attach what each real hospital's map entry declares (phone, hours, website, facilities): unverified, labelled inferred."""
    from services import osm_tags

    tags = osm_tags.fetch(lat, lng, radius)
    for f in facilities:
        d = osm_tags.details(tags.get(f["hospital_id"], {}))
        f["phone"] = f.get("phone") or d["phone"]
        f.update(hours=d["hours"], website=d["website"], inferred=d["inferred"], beds_total=d["beds_total"], operator_type=d["operator_type"])
    return facilities


def nearby_raw(lat: float, lng: float, start_radius: int = 8) -> tuple:
    """(facility dicts, radius_km_used). 8 km first, 25 km if fewer than 3 are found (or start at 25)."""
    from services import hospitals_osm, registry

    if registry.source() == "real":
        found: Optional[list] = []
        radius = 8
        for radius in ((8, 25) if start_radius <= 8 else (25,)):
            found = hospitals_osm.nearby(lat, lng, radius)
            if found is None:
                raise RuntimeError("the map-data service (OpenStreetMap via LocationIQ) could not be reached")
            if len(found) >= 3:
                break
        return _merge_reported_status(_enrich_from_tags(found, lat, lng, radius)), radius
    facilities: list = []
    radius = 8
    for radius in ((8, 25) if start_radius <= 8 else (25,)):
        facilities = _session.get(
            f"{_base()}/facilities/nearby",
            params={"lat": lat, "lng": lng, "radius_km": radius, "limit": 50},
            timeout=(5, 30),
        ).json()
        if len(facilities) >= 3:
            break
    return facilities, radius


def discover_facilities(case: StructuredCase, start_radius: int = 8) -> tuple:
    """(candidates, radius_km_used). 8 km urban first, 25 km rural if fewer than 3 found.
    start_radius=25 skips straight to the wide search (used when too few hospitals are ELIGIBLE nearby)."""
    if case.location.lat is None or case.location.lng is None:
        raise RuntimeError("case has no resolved location")
    facilities, radius = nearby_raw(case.location.lat, case.location.lng, start_radius)
    return [
        CandidateHospital(
            hospital_id=f["hospital_id"], name=f["name"], operating_status=f["operating_status"],
            ipd_accepting=f["ipd_accepting"], distance_km=f["distance_km"], lat=f["lat"], lng=f["lng"],
            source=f.get("source", "registry"), status_verified=f.get("status_verified", True), phone=f.get("phone"),
            address=f.get("address"), osm_url=f.get("osm_url"), hours=f.get("hours"), website=f.get("website"),
            inferred=f.get("inferred") or [],
            eta_min=round(f["distance_km"] / 30 * 60 * 1.4, 1), eta_source_tier=4, eta_confidence=0.40,
        )
        for f in facilities
    ], radius


def _reported_snapshot(hospital_id: str, reports: list) -> ResourceSnapshot:
    from datetime import datetime, timezone

    from schemas.resource import ResourceRecord

    return ResourceSnapshot(hospital_id=hospital_id, records=[
        ResourceRecord(resource_key=r["resource_key"], value=r["value"], updated_at=datetime.fromtimestamp(r["updated_at"], timezone.utc),
                       source="HOSPITAL_CONSOLE", reporter_id=r.get("reporter") or None)
        for r in reports if r["resource_key"] != "operating_status"])


def fetch_snapshots(candidates: list) -> list:
    """Real hospitals: only what the hospital itself reported (often nothing). Seeded network: the resource service."""
    if not candidates:
        return []
    from services import registry

    real = [c for c in candidates if registry.is_real_id(c.hospital_id)]
    if real:
        from dispatch import store

        from services import osm_tags

        rep = store.reports_for([c.hospital_id for c in real])
        out = []
        for c in real:
            snap = _reported_snapshot(c.hospital_id, rep.get(c.hospital_id, []))
            # what the map entry declares goes in as INFERRED records; the hospital's own (fresher, fully trusted) reports win
            snap.records = osm_tags.records(c.hospital_id, osm_tags.tags_for(c.hospital_id, c.lat, c.lng)) + snap.records
            out.append(snap)
        rest = [c for c in candidates if not registry.is_real_id(c.hospital_id)]
        return out + (fetch_snapshots(rest) if rest else [])
    snaps = _session.post(f"{_base()}/snapshots", json={"hospital_ids": [c.hospital_id for c in candidates]}, timeout=(5, 30)).json()
    return [ResourceSnapshot.model_validate(x) for x in snaps]


def scan_beds(hospital_ids: list) -> dict:
    """Live bed scan. Real hospitals: their own latest reports (empty if they never reported). Seeded network: the resource
    service, bypassing its cache. Returns {snapshots: [ResourceSnapshot], scanned_at, took_ms}."""
    from services import registry

    if hospital_ids and all(registry.is_real_id(h) for h in hospital_ids):
        import time as _t
        from datetime import datetime, timezone

        from dispatch import store

        t0 = _t.time()
        rep = store.reports_for(hospital_ids)
        return {"snapshots": [_reported_snapshot(h, [r for r in rep.get(h, []) if r["resource_key"] in store.REPORT_KEYS_BEDS]) for h in hospital_ids],
                "scanned_at": datetime.now(timezone.utc).isoformat(), "took_ms": round((_t.time() - t0) * 1000)}
    r = _session.post(f"{_base()}/beds/scan", json={"hospital_ids": hospital_ids}, timeout=(5, 60))
    r.raise_for_status()
    j = r.json()
    return {"snapshots": [ResourceSnapshot.model_validate(x) for x in j["snapshots"]], "scanned_at": j["scanned_at"], "took_ms": j["took_ms"]}


def _write_headers() -> dict:
    key = os.environ.get("RESOURCE_WRITE_KEY", "")
    return {"X-Write-Key": key} if key else {}


def reserve_bed(hospital_id: str, bed_key: str, token: str = "") -> dict:
    from services import registry

    if registry.is_real_id(hospital_id):
        from dispatch import store

        return store.reserve_reported_bed(hospital_id, bed_key, reporter=f"cedcs-reserve:{token[:8]}")
    r = _session.post(f"{_base()}/beds/reserve", json={"hospital_id": hospital_id, "bed_key": bed_key, "token": token},
                      headers=_write_headers(), timeout=(5, 30))
    r.raise_for_status()
    return r.json()


def report_beds(hospital_id: str, bed_key: str, available: int, total=None, reporter_id: str = "hospital-report") -> dict:
    from services import registry

    if registry.is_real_id(hospital_id):
        from dispatch import store

        value = {"available": max(0, int(available)), "total": int(total) if total is not None else max(0, int(available))}
        store.upsert_report(hospital_id, bed_key, value, reporter_id)
        return {"ok": True, "hospital_id": hospital_id, "bed_key": bed_key, "value": value}
    r = _session.post(f"{_base()}/beds/report", json={"hospital_id": hospital_id, "bed_key": bed_key, "available": available,
                                                     "total": total, "reporter_id": reporter_id}, headers=_write_headers(), timeout=(5, 30))
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- explanation

def _cap_phrase(caps: list) -> str:
    return ", ".join(c.replace("_", " ").lower() for c in caps) or "no specific capability"


def explain(rec: ValidatedRecommendation, include_ai_notes: bool = True) -> str:
    p = rec.primary
    eta = f"about {p.eta_min:.0f} minutes away" if p.eta_min is not None else "at an unknown distance"
    lines = [
        f"**Recommended Destination:** {p.name}",
        (f"This is the best-ranked hospital that meets every required capability. It is {eta} and confirms: "
         f"{_cap_phrase(p.capability_match)}. Its resource data confidence is {p.resource_confidence:.0%}.") if p.capability_match else
        (f"This is the best-ranked hospital available. It is {eta}, but none of the needed capabilities could be confirmed from current "
         f"data (resource data confidence {p.resource_confidence:.0%}): please confirm by phone before sending the patient."),
    ]
    if rec.escalated:
        lines.append(f"**Escalated:** {rec.escalation_reason}. A human operator should review this case.")
    lines += ["", "**Alternatives:**"]
    lines += [f"- **{a.name}** (rank {a.rank}): {a.ranking_reason}" for a in rec.alternatives] or ["- None available."]
    lines += ["", "**Why Nearby Hospitals Were Not Recommended:**"]
    lines += [f"- **{r.name}** was excluded: {r.rejection_reason}." for r in rec.rejected] or ["- None."]
    notes = [c for c in rec.caveats if include_ai_notes or not c.caveat_type.startswith("AI_")]
    lines += ["", "**Important Limitations:**"] + [f"- {c.detail}" for c in notes]
    return "\n".join(lines)


# ---------------------------------------------------------------- clarification (rule-based)

_QUESTIONS = {
    "consciousness": ("Is the person awake and answering you?", "Determines whether neurological and airway support are required."),
    "breathing": ("Is the person breathing normally, with effort, or not at all?", "Decides whether ventilator and ICU capability are required."),
    "bleeding": ("Is there any heavy bleeding right now?", "Heavy bleeding requires a blood bank and surgical capability."),
    "symptoms": ("What is the main problem, in a few words?", "Symptoms determine which specialty capabilities are needed."),
    "age": ("About how old is the person?", "Children and older adults need different capabilities."),
    "medical_history": ("Do they have any known heart, breathing or kidney condition, or diabetes?", "Known conditions change the capabilities to prefer."),
    "onset_time": ("When did this start?", "Time since onset affects urgency."),
}


def clarify(case: StructuredCase):
    from schemas.case import ClarificationBatch, ClarificationQuestion

    qs = [
        ClarificationQuestion(field=f, question=_QUESTIONS[f][0], impact=_QUESTIONS[f][1])
        for f in case.missing_fields if f in _QUESTIONS
    ][:3]
    return ClarificationBatch(
        questions=qs,
        round_rationale="Asked first about the missing facts that would most change the capability requirements." if qs
        else "No impactful missing fields remain.",
    )
