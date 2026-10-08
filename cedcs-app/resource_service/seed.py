"""
CEDCS seeded hospital dataset generator.

Populates ~40 synthetic hospitals (F4.4 in the design doc: "realistic seeded
hospitals for prototype and evaluation") into the facility registry, with
resource records spread across a range of ages/sources so the freshness
model (FRESH/RECENT/STALE/UNKNOWN) has something real to decay.

All facility names are fictional/generic (e.g. "Adyar General Hospital") --
this is explicitly labelled SEED/SEEDED data, matching the design doc's own
requirement (§12) to never present simulated availability as real. Do not
rename these to real institutions.

Usage:
    pip install psycopg2-binary
    export DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
    python seed.py
"""

import os
import random
import sys
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
from psycopg2.extras import Json

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    sys.exit("Set DATABASE_URL to your Postgres connection string before running seed.py")

random.seed(42)

# Roughly the Chennai metro area, matching the worked example in the design
# doc (Tambaram). Swap this center/radius for your own city if needed.
CENTER_LAT, CENTER_LNG = 12.9716, 80.2000
SPREAD_DEG = 0.28  # ~30 km

AREA_NAMES = [
    "Adyar", "Tambaram", "Velachery", "Anna Nagar", "Mylapore", "T Nagar",
    "Guindy", "Porur", "Perambur", "Kilpauk", "Nungambakkam", "Chromepet",
    "Ambattur", "Vadapalani", "Egmore", "Saidapet", "Sholinganallur",
    "Pallavaram", "Alwarpet", "Royapettah", "Medavakkam", "Thoraipakkam",
    "Nanganallur", "Kodambakkam", "Ashok Nagar", "Poonamallee", "Tondiarpet",
    "Manapakkam", "West Mambalam", "Mogappair",
]

HOSPITAL_TYPES = [
    ("General Hospital", "govt"),
    ("Multispecialty Hospital", "private"),
    ("Medical Centre", "private"),
    ("Trust Hospital", "trust"),
    ("Institute of Medical Sciences", "private"),
]

# Capability taxonomy used by requirements.py — keep these codes in sync
# with the deterministic core.
DEPARTMENTS = [
    "EMERGENCY_DEPARTMENT", "CARDIOLOGY", "NEUROLOGY", "TRAUMA", "ORTHOPEDICS",
    "PEDIATRICS", "OBSTETRICS", "ONCOLOGY", "NEPHROLOGY", "GENERAL_MEDICINE",
]
EQUIPMENT = ["CT_SCAN", "MRI", "X_RAY", "USG", "CATH_LAB", "OT", "LAB", "DIALYSIS", "CARDIAC_MONITOR"]
SPECIALTIES = ["CARDIOLOGY", "NEUROLOGY", "ORTHOPEDICS", "PEDIATRICS", "GENERAL_SURGERY"]
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
SOURCE_WEIGHTS = [("HOSPITAL_CONSOLE", 0.55), ("GOVT", 0.20), ("SEED", 0.25)]


def weighted_choice(pairs):
    r = random.random()
    upto = 0.0
    for value, weight in pairs:
        upto += weight
        if r <= upto:
            return value
    return pairs[-1][0]


def random_age_minutes():
    # Deliberately spread across FRESH / RECENT / STALE / UNKNOWN bands so the
    # freshness model has a realistic mix to classify, not all-fresh data.
    band = random.choices(
        ["very_fresh", "recent", "stale", "very_stale"],
        weights=[0.50, 0.28, 0.14, 0.08],
    )[0]
    return {
        "very_fresh": random.uniform(1, 20),
        "recent": random.uniform(20, 120),
        "stale": random.uniform(120, 600),
        "very_stale": random.uniform(600, 2000),
    }[band]


def record(hospital_id, key, value, records):
    updated_at = datetime.now(timezone.utc) - timedelta(minutes=random_age_minutes())
    records.append((hospital_id, key, Json(value), updated_at, weighted_choice(SOURCE_WEIGHTS), "seed-script"))


def build_hospital(i):
    area = AREA_NAMES[i % len(AREA_NAMES)]
    suffix, htype = HOSPITAL_TYPES[i % len(HOSPITAL_TYPES)]
    hospital_id = f"CEDCS-SEED-{i:03d}"
    name = f"{area} {suffix}" if i < len(AREA_NAMES) else f"New {area} {suffix}"  # keep names unique
    lat = CENTER_LAT + random.uniform(-SPREAD_DEG, SPREAD_DEG)
    lng = CENTER_LNG + random.uniform(-SPREAD_DEG, SPREAD_DEG)
    operating_status = random.choices(
        ["OPERATIONAL", "DIVERTING", "CLOSED"], weights=[0.88, 0.09, 0.03]
    )[0]
    trauma_level = random.choices(["I", "II", "III", "none"], weights=[0.1, 0.15, 0.2, 0.55])[0]
    ipd_accepting = operating_status == "OPERATIONAL" and random.random() > 0.08
    return {
        "hospital_id": hospital_id,
        "name": name,
        "type": htype,
        "lat": lat,
        "lng": lng,
        "address": f"{name}, {area}, Chennai, Tamil Nadu",
        "phone": f"+91{random.randint(7000000000, 9999999999)}",
        "emergency_line": f"+91{random.randint(7000000000, 9999999999)}",
        "operating_status": operating_status,
        "trauma_level": trauma_level,
        "ipd_accepting": ipd_accepting,
        "data_source": "SEEDED",
    }


def build_records(h):
    records = []
    hid = h["hospital_id"]

    # Every hospital has an emergency department; the rest are a random subset
    # weighted by facility size (bigger "type" strings get more departments).
    active_depts = {"EMERGENCY_DEPARTMENT"}
    n_extra = random.randint(1, len(DEPARTMENTS) - 1)
    active_depts |= set(random.sample(DEPARTMENTS[1:], n_extra))
    for dept in DEPARTMENTS:
        record(hid, f"department_{dept}", {"active": dept in active_depts}, records)

    for bed_type, (lo, hi) in {
        "general_beds": (20, 150), "emergency_beds": (4, 20),
        "icu_beds": (2, 20), "hdu_beds": (0, 10),
        "pediatric_beds": (0, 20), "ventilators": (1, 12),
    }.items():
        total = random.randint(lo, hi)
        available = random.randint(0, total) if total else 0
        # ~12% chance a required-bed-type figure is simply not known yet —
        # exercises the eligibility engine's PROVISIONAL path (rule R5).
        if random.random() < 0.12:
            record(hid, bed_type, {"total": total, "available": None}, records)
        else:
            record(hid, bed_type, {"total": total, "available": available}, records)

    for eq in EQUIPMENT:
        operational = random.random() > 0.12
        record(hid, f"equipment_{eq}", {"operational": operational}, records)

    for spec in SPECIALTIES:
        if spec == "CARDIOLOGY" and "CARDIOLOGY" not in active_depts:
            continue
        if spec == "NEUROLOGY" and "NEUROLOGY" not in active_depts:
            continue
        status = random.choices(["on_site", "on_call", "unavailable"], weights=[0.5, 0.35, 0.15])[0]
        record(hid, f"specialist_{spec}", {"status": status}, records)

    blood_groups = {g: random.randint(0, 40) for g in BLOOD_GROUPS}
    record(hid, "blood_bank", {"available": True, "groups": blood_groups}, records)

    record(hid, "opd", {"queue_estimate_min": random.randint(5, 90)}, records)
    record(hid, "ipd_admission_delay_est_min", {"minutes": random.randint(0, 120)}, records)

    return records


def main(n_hospitals=40):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    with open(os.path.join(os.path.dirname(__file__), "schema.sql")) as f:
        cur.execute(f.read())

    cur.execute("TRUNCATE resource_records, hospitals RESTART IDENTITY CASCADE")

    hospitals = [build_hospital(i) for i in range(n_hospitals)]
    cur.executemany(
        """
        INSERT INTO hospitals
            (hospital_id, name, type, lat, lng, address, phone, emergency_line,
             operating_status, trauma_level, ipd_accepting, data_source)
        VALUES (%(hospital_id)s, %(name)s, %(type)s, %(lat)s, %(lng)s, %(address)s,
                %(phone)s, %(emergency_line)s, %(operating_status)s, %(trauma_level)s,
                %(ipd_accepting)s, %(data_source)s)
        """,
        hospitals,
    )

    all_records = []
    for h in hospitals:
        all_records.extend(build_records(h))
    # one batched statement instead of a network round trip per row (matters for remote DBs like Neon)
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO resource_records (hospital_id, resource_key, value, updated_at, source, reporter_id) VALUES %s",
        all_records,
        page_size=500,
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Seeded {len(hospitals)} hospitals and {len(all_records)} resource records.")


if __name__ == "__main__":
    main()
