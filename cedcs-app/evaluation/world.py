"""A synthetic emergency-care world with a hidden TRUE state, for evaluating destination-recommendation policies.

What is simulated, and what is not (see METHODS.md for the full list of assumptions):

  * Hospitals of several archetypes with a capability matrix (departments, equipment, blood bank, specialists) and bed counts.
  * At decision time T every resource has a TRUE value. What a policy can see is what the hospital REPORTED some time ago: the
    report was correct when made, but the resource kept changing since (beds fill and free up, equipment fails, ...) at a
    resource-specific rate, so an old report is wrong more often than a fresh one. Some fields are never reported (UNKNOWN).
  * Cases come from clinical templates that carry a GROUND-TRUTH requirement profile (REQUIRED / PREFERRED capabilities) defined
    independently of any hospital, so a recommendation can be judged against the true state of the hospital it picked.

Everything is seeded, so a run is exactly reproducible. All rates and probabilities below are plausibility choices, not fitted to
data; the weight/rate sensitivity experiments exist to show how much the conclusions depend on them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from deterministic_core.capability_map import evaluate_capability
from deterministic_core.freshness import score_records
from schemas.resource import ResourceRecord

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # the decision time for every case
CENTER = (13.05, 80.20)  # Chennai-like city centre
SPAN_KM = 30.0  # hospitals are placed in a square this wide
KM_PER_DEG_LAT = 111.0
BED_KEYS = ("icu_beds", "emergency_beds", "general_beds", "hdu_beds", "pediatric_beds", "ventilators")

# ---------------------------------------------------------------- how fast each resource changes (events per hour)
CHANGE_RATE_PER_H = {"icu_beds": 0.25, "emergency_beds": 1.5, "general_beds": 0.5, "hdu_beds": 0.3, "pediatric_beds": 0.3,
                     "ventilators": 0.2, "equipment": 0.01, "department": 0.0005, "blood_bank": 0.03, "specialist": 0.2}
P_MISSING = 0.10  # a field the hospital never reported (UNKNOWN to every policy)
SOURCES = (("HOSPITAL_CONSOLE", 0.55), ("GOVT", 0.25), ("FHIR", 0.20))

DEPARTMENTS = ["EMERGENCY_DEPARTMENT", "CARDIOLOGY", "NEUROLOGY", "TRAUMA", "ORTHOPEDICS", "PEDIATRICS", "OBSTETRICS", "GENERAL_MEDICINE"]
EQUIPMENT = ["CT_SCAN", "MRI", "X_RAY", "CATH_LAB", "OT", "CARDIAC_MONITOR"]
SPECIALISTS = ["CARDIOLOGY", "NEUROLOGY", "ORTHOPEDICS", "PEDIATRICS"]

# archetype -> (share of hospitals, P(department), P(equipment), ICU total range, P(has ICU), blood bank P)
ARCHETYPES = {
    "tertiary": (0.15, dict.fromkeys(DEPARTMENTS, 0.97), dict(CT_SCAN=1, MRI=.9, X_RAY=1, CATH_LAB=.75, OT=1, CARDIAC_MONITOR=1), (8, 20), 1.0, 1.0),
    "trauma": (0.10, dict(EMERGENCY_DEPARTMENT=1, TRAUMA=1, ORTHOPEDICS=.95, NEUROLOGY=.6, GENERAL_MEDICINE=.9, CARDIOLOGY=.4, PEDIATRICS=.3, OBSTETRICS=.2),
               dict(CT_SCAN=.95, MRI=.5, X_RAY=1, CATH_LAB=.15, OT=1, CARDIAC_MONITOR=.9), (6, 12), 1.0, .95),
    "district": (0.30, dict(EMERGENCY_DEPARTMENT=.92, GENERAL_MEDICINE=1, ORTHOPEDICS=.5, PEDIATRICS=.6, OBSTETRICS=.55, CARDIOLOGY=.25, NEUROLOGY=.1, TRAUMA=.3),
                 dict(CT_SCAN=.4, MRI=.05, X_RAY=.95, CATH_LAB=0, OT=.65, CARDIAC_MONITOR=.7), (2, 6), .6, .6),
    "private_mid": (0.25, dict(EMERGENCY_DEPARTMENT=.9, GENERAL_MEDICINE=1, CARDIOLOGY=.55, ORTHOPEDICS=.6, PEDIATRICS=.5, OBSTETRICS=.55, NEUROLOGY=.3, TRAUMA=.35),
                    dict(CT_SCAN=.7, MRI=.4, X_RAY=1, CATH_LAB=.2, OT=.85, CARDIAC_MONITOR=.9), (3, 8), .85, .75),
    "clinic": (0.20, dict(EMERGENCY_DEPARTMENT=.35, GENERAL_MEDICINE=.9, PEDIATRICS=.3, OBSTETRICS=.3, ORTHOPEDICS=.15),
               dict(CT_SCAN=0, MRI=0, X_RAY=.7, CATH_LAB=0, OT=.2, CARDIAC_MONITOR=.2), (0, 0), 0.0, .15),
}


# ---------------------------------------------------------------- hospitals
@dataclass
class Hospital:
    hid: str
    name: str
    archetype: str
    lat: float
    lng: float
    true: dict = field(default_factory=dict)  # resource_key -> value, the state at T0
    reported: dict = field(default_factory=dict)  # resource_key -> ResourceRecord (only fields that were reported)


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def eta_minutes(distance_km: float) -> float:
    """Same estimate the application uses when no routing service is available (30 km/h, x1.4 for roads and traffic)."""
    return round(distance_km / 30 * 60 * 1.4, 1)


LOAD_FREE_SCALE = {"relaxed": 1.8, "normal": 1.0, "crowded": 0.4}  # multiplies the chance that a bed is free: how busy the city is


def _true_state(rng: random.Random, arch: str, free_scale: float = 1.0) -> dict:
    _, pdept, peq, icu_rng, p_icu, p_blood = ARCHETYPES[arch]
    st: dict = {}
    for d in DEPARTMENTS:
        st[f"department_{d}"] = {"active": rng.random() < pdept.get(d, 0.0)}
    for e in EQUIPMENT:
        st[f"equipment_{e}"] = {"operational": rng.random() < peq.get(e, 0.0)}
    st["blood_bank"] = {"available": rng.random() < p_blood}
    for s in SPECIALISTS:
        has = st[f"department_{s}"]["active"]
        st[f"specialist_{s}"] = {"status": (rng.choice(["on_site", "on_site", "on_call"]) if has else "none")}
    has_icu = icu_rng[1] > 0 and rng.random() < p_icu
    icu_total = rng.randint(*icu_rng) if has_icu else 0
    scale = {"tertiary": 3.0, "trauma": 2.0, "private_mid": 1.4, "district": 1.0, "clinic": 0.0}[arch]

    def beds(total: int, p_free: float) -> dict:
        free = sum(rng.random() < min(0.95, p_free * free_scale) for _ in range(total))
        return {"total": total, "available": free}

    st["icu_beds"] = beds(icu_total, 0.25)  # about 75 % occupied: a crunch is the normal case
    st["hdu_beds"] = beds(round(icu_total * 0.8), 0.3)
    st["emergency_beds"] = beds(max(0, round(rng.randint(4, 12) * min(scale, 1.5))), 0.35)
    st["general_beds"] = beds(round(rng.randint(30, 90) * max(scale, 0.0)), 0.3)
    st["pediatric_beds"] = beds(round(rng.randint(4, 14) * (1 if st["department_PEDIATRICS"]["active"] else 0)), 0.35)
    st["ventilators"] = beds(max(0, round(icu_total * 0.7)), 0.3)
    st["ipd_admission_delay_est_min"] = {"minutes": rng.randint(5, 90)}
    return st


def _perturb(key: str, value: dict, age_h: float, rng: random.Random) -> dict:
    """The value the hospital would have reported `age_h` hours before T0: the true value passed back through random changes."""
    v = dict(value)
    if key in BED_KEYS:
        n = _poisson(rng, CHANGE_RATE_PER_H[key] * age_h)
        a = v["available"]
        for _ in range(n):
            a = min(v["total"], max(0, a + rng.choice((-1, 1))))
        v["available"] = a
    elif key.startswith("equipment_"):
        if _poisson(rng, CHANGE_RATE_PER_H["equipment"] * age_h) % 2:
            v["operational"] = not v["operational"]
    elif key.startswith("department_"):
        if _poisson(rng, CHANGE_RATE_PER_H["department"] * age_h) % 2:
            v["active"] = not v["active"]
    elif key == "blood_bank":
        if _poisson(rng, CHANGE_RATE_PER_H["blood_bank"] * age_h) % 2:
            v["available"] = not v["available"]
    elif key.startswith("specialist_"):
        if _poisson(rng, CHANGE_RATE_PER_H["specialist"] * age_h) > 0:
            v["status"] = rng.choice(["on_site", "on_call", "none"])
    return v


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def sample_age_h(rng: random.Random, fixed_h: Optional[float] = None) -> float:
    """Age of one reported value in hours. Default: log-normal, median 45 min, a long tail (about 8 % older than 6 h)."""
    if fixed_h is not None:
        return max(0.0, fixed_h * rng.uniform(0.8, 1.2))
    return math.exp(rng.gauss(math.log(0.75), 1.1))


def build_world(seed: int, n_hospitals: int = 60, fixed_age_h: Optional[float] = None, p_missing: float = P_MISSING, load: str = "normal") -> list:
    """Hospitals with a true state at T0 and what each has reported (ages from `sample_age_h`). `load`: relaxed | normal | crowded."""
    rng = random.Random(seed)
    names = list(ARCHETYPES)
    weights = [ARCHETYPES[a][0] for a in names]
    out = []
    for i in range(n_hospitals):
        arch = rng.choices(names, weights)[0]
        lat = CENTER[0] + rng.uniform(-SPAN_KM / 2, SPAN_KM / 2) / KM_PER_DEG_LAT
        lng = CENTER[1] + rng.uniform(-SPAN_KM / 2, SPAN_KM / 2) / (KM_PER_DEG_LAT * math.cos(math.radians(CENTER[0])))
        h = Hospital(hid=f"W{seed}-H{i:03d}", name=f"{arch.title()} Hospital {i:03d}", archetype=arch, lat=lat, lng=lng, true=_true_state(rng, arch, LOAD_FREE_SCALE[load]))
        for key, val in h.true.items():
            if rng.random() < p_missing:
                continue  # never reported: UNKNOWN to every policy
            age = sample_age_h(rng, fixed_age_h)
            src = rng.choices([s for s, _ in SOURCES], [w for _, w in SOURCES])[0]
            h.reported[key] = ResourceRecord(resource_key=key, value=_perturb(key, val, age, rng), updated_at=T0 - timedelta(hours=age), source=src,
                                             reporter_id=None)
        out.append(h)
    return out


# ---------------------------------------------------------------- truth, as seen by the evaluator
def _truth_records(h: Hospital) -> list:
    recs = [ResourceRecord(resource_key=k, value=v, updated_at=T0, source="HOSPITAL_CONSOLE") for k, v in h.true.items()]
    return score_records(recs, now=T0)


def true_status(h: Hospital, capability: str) -> str:
    """PRESENT / ABSENT for a capability in the hospital's TRUE state at T0 (never UNKNOWN: the evaluator sees everything)."""
    cache = getattr(h, "_truth", None)
    if cache is None:
        cache = h._truth = _truth_records(h)
    return evaluate_capability(capability, cache).status


# ---------------------------------------------------------------- cases
@dataclass(frozen=True)
class Template:
    name: str
    priority: str
    categories: tuple
    required: tuple
    preferred: tuple
    specialty: Optional[str]  # the department a "specialty filter" baseline would look for
    texts: tuple


TEMPLATES = [
    Template("cardiac", "CRITICAL", ("CARDIAC",), ("EMERGENCY_DEPARTMENT", "CARDIAC_MONITORING", "ICU", "CARDIOLOGY"), ("CATH_LAB", "CT_SCAN", "BLOOD_BANK"), "CARDIOLOGY",
             ("60 year old man, crushing chest pain spreading to left arm, sweating, short of breath",
              "elderly woman collapsed at home, clutching her chest, pale and sweating",
              "man in his fifties, severe chest pressure for 30 minutes, nauseous, known diabetic")),
    Template("neuro", "CRITICAL", ("NEUROLOGICAL",), ("EMERGENCY_DEPARTMENT", "CT_SCAN", "NEUROLOGY", "ICU"), ("MRI", "BLOOD_BANK"), "NEUROLOGY",
             ("72 year old woman, sudden weakness on the left side, slurred speech, face drooping",
              "man suddenly confused, cannot speak properly, one arm limp, started 20 minutes ago",
              "woman collapsed, severe sudden headache, vomiting, drowsy")),
    Template("trauma", "CRITICAL", ("TRAUMA",), ("EMERGENCY_DEPARTMENT", "TRAUMA", "OT", "BLOOD_BANK", "ICU"), ("CT_SCAN", "ORTHOPEDICS"), "TRAUMA",
             ("road accident, young man, heavy bleeding from the thigh, pale, drowsy",
              "fell from scaffolding, unconscious, bleeding from the head",
              "motorbike collision, leg deformed, bleeding heavily, conscious but in severe pain")),
    Template("paediatric", "HIGH", ("PAEDIATRIC",), ("EMERGENCY_DEPARTMENT", "PEDIATRICS", "PEDIATRIC_BED"), ("CT_SCAN", "ICU"), "PEDIATRICS",
             ("5 year old child, high fever, seizure, now drowsy",
              "toddler not breathing well, fast breathing, blue lips",
              "child of 8 with severe vomiting and diarrhoea, very weak")),
    Template("obstetric", "HIGH", ("OBSTETRIC",), ("EMERGENCY_DEPARTMENT", "OBSTETRICS", "OT", "BLOOD_BANK"), ("ICU",), "OBSTETRICS",
             ("28 year old pregnant woman, 34 weeks, severe abdominal pain and bleeding",
              "woman in labour, bleeding heavily, baby not moving",
              "pregnant woman with fits and severe headache, 8 months")),
    Template("respiratory", "HIGH", ("RESPIRATORY",), ("EMERGENCY_DEPARTMENT", "ICU", "VENTILATOR"), ("X_RAY", "CT_SCAN"), "GENERAL_MEDICINE",
             ("45 year old man, severe difficulty breathing, wheezing, known asthma, lips bluish",
              "elderly man, breathless at rest, cough and fever for a week, getting worse",
              "woman gasping for breath after a bee sting, face swollen")),
    Template("fracture", "MODERATE", (), ("EMERGENCY_DEPARTMENT", "X_RAY", "ORTHOPEDICS"), ("OT",), "ORTHOPEDICS",
             ("19 year old fell from a bike, wrist swollen and deformed, can walk, in pain",
              "elderly woman slipped, cannot put weight on her hip, in a lot of pain",
              "boy fell from a tree, arm bent at an odd angle, crying")),
    Template("minor", "LOW", (), ("EMERGENCY_DEPARTMENT",), ("X_RAY",), None,
             ("35 year old woman, deep cut on the hand, bleeding controlled with pressure",
              "man with a high fever since two days, otherwise talking normally",
              "child with a minor burn on the arm, not in distress")),
]
MIX = {"CRITICAL": 0.30, "HIGH": 0.30, "MODERATE": 0.25, "LOW": 0.15}


@dataclass(frozen=True)
class Case:
    cid: str
    template: Template
    text: str
    lat: float
    lng: float


def build_cases(seed: int, n: int) -> list:
    rng = random.Random(seed * 7919 + 1)
    by_pri = {p: [t for t in TEMPLATES if t.priority == p] for p in MIX}
    out = []
    for i in range(n):
        pri = rng.choices(list(MIX), list(MIX.values()))[0]
        t = rng.choice(by_pri[pri])
        lat = CENTER[0] + rng.uniform(-10, 10) / KM_PER_DEG_LAT
        lng = CENTER[1] + rng.uniform(-10, 10) / (KM_PER_DEG_LAT * math.cos(math.radians(CENTER[0])))
        out.append(Case(f"S{seed}-C{i:04d}", t, rng.choice(t.texts), lat, lng))
    return out


def distance_km(case: Case, h: Hospital) -> float:
    return haversine_km(case.lat, case.lng, h.lat, h.lng)
