-- CEDCS Facility Registry schema
-- Matches the canonical HOSPITAL / RESOURCE_RECORD data model in the
-- CEDCS Final Design Document, §3.5.
--
-- Run this once against a fresh Postgres database (e.g. a free Neon project)
-- before running seed.py.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS hospitals (
    hospital_id      TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    type             TEXT NOT NULL CHECK (type IN ('govt', 'private', 'trust')),
    lat              DOUBLE PRECISION NOT NULL,
    lng              DOUBLE PRECISION NOT NULL,
    address          TEXT,
    phone            TEXT,
    emergency_line   TEXT,
    operating_status TEXT NOT NULL DEFAULT 'OPERATIONAL'
                         CHECK (operating_status IN ('OPERATIONAL', 'DIVERTING', 'CLOSED')),
    trauma_level     TEXT NOT NULL DEFAULT 'none'
                         CHECK (trauma_level IN ('I', 'II', 'III', 'none')),
    ipd_accepting    BOOLEAN NOT NULL DEFAULT TRUE,
    data_source      TEXT NOT NULL DEFAULT 'SEEDED'
                         CHECK (data_source IN ('MANUAL', 'HIS_FHIR', 'GOVT_FEED', 'SEEDED')),
    geog             GEOGRAPHY(POINT, 4326)
                         GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(lng, lat), 4326)::geography) STORED
);

CREATE INDEX IF NOT EXISTS idx_hospitals_geog ON hospitals USING GIST (geog);

-- The atomic unit of P3 ("every fact carries provenance and time"):
-- one row per (hospital, resource_key) fact, never a bare value.
-- resource_key examples used by seed.py:
--   department_<CODE>            {"active": true}
--   icu_beds                     {"total": 10, "available": 3}
--   emergency_beds                {"total": 8,  "available": 2}
--   hdu_beds / pediatric_beds / general_beds   (same shape)
--   ventilators                  {"total": 6, "available": 1}
--   equipment_<CODE>             {"operational": true}
--   specialist_<SPECIALTY>       {"status": "on_site" | "on_call" | "unavailable"}
--   blood_bank                   {"available": true, "groups": {"A+": 12, "O-": 3, ...}}
--   opd                          {"queue_estimate_min": 25}
--   ipd_admission_delay_est_min  {"minutes": 40}
CREATE TABLE IF NOT EXISTS resource_records (
    id            BIGSERIAL PRIMARY KEY,
    hospital_id   TEXT NOT NULL REFERENCES hospitals(hospital_id) ON DELETE CASCADE,
    resource_key  TEXT NOT NULL,
    value         JSONB NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL,
    source        TEXT NOT NULL CHECK (source IN ('HOSPITAL_CONSOLE', 'FHIR', 'GOVT', 'INFERRED', 'SEED')),
    reporter_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_resource_records_hospital ON resource_records(hospital_id);
CREATE INDEX IF NOT EXISTS idx_resource_records_key ON resource_records(hospital_id, resource_key);
