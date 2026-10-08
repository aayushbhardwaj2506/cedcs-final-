# CEDCS Resource Retrieval stub — get your two missing env vars

This folder gives you both remaining values your CrewAI builder is asking for:

- `DATABASE_URL` (server-side only) — a real Postgres database, seeded with ~40 synthetic hospitals
- `RESOURCE_SERVICE_URL` — a small deployed service that reads from that database

Total cost: **$0**, no credit card, on Neon (database) + Render (service).

## Step 1 — Create the free database (Neon)

1. Go to **neon.com** → sign up free → **New Project**. Name it `cedcs`. Pick a region close to you if offered.
2. Once it's created, open the **SQL Editor** in the Neon console, paste in the contents of `schema.sql`, and run it. This creates the `hospitals` and `resource_records` tables and enables PostGIS.
3. In the Neon dashboard, go to **Connection Details** and copy the connection string. It looks like:
   ```
   postgresql://<user>:<password>@<host>/<db>?sslmode=require
   ```
4. Set it as `DATABASE_URL` on this service. The CrewAI tools do NOT get the Postgres string — they get `RESOURCE_SERVICE_URL`, the HTTP base URL of this service (e.g. `http://localhost:8000`).

## Step 2 — Seed it with hospitals

On your own machine (or anywhere with Python):

```bash
pip install psycopg2-binary
export DATABASE_URL="<the same Neon connection string from step 1>"
python seed.py
```

This inserts ~40 fictional Chennai-area hospitals with randomized departments, bed capacity, equipment, specialists, blood bank stock, and — importantly — **timestamps spread across fresh/recent/stale/very-stale bands**, so your freshness model (`freshness.py`) actually has something realistic to decay when you test it, rather than everything looking equally fresh.

Re-run `seed.py` any time to reset to a fresh random dataset (it truncates and reseeds).

## Step 3 — Deploy the resource service (Render, free tier)

1. Push this folder to a GitHub repo (or a subfolder of your CEDCS repo).
2. Go to **render.com** → sign up free → **New +** → **Web Service** → connect that repo.
3. Render will detect the `Dockerfile` automatically. Leave build/start commands as default.
4. Under **Environment**, add one variable: `DATABASE_URL` = the same Neon connection string from Step 1.
5. Deploy. Render gives you a public URL like `https://cedcs-resource-stub.onrender.com`.
6. Check it's alive: `curl https://cedcs-resource-stub.onrender.com/health` → `{"status":"ok"}`.
7. **Paste that URL into `RESOURCE_SERVICE_URL`** in your CrewAI builder. Both fields are now done.

> Free Render web services spin down after ~15 minutes idle and take a few seconds to wake back up on the next request — fine for a prototype/demo, not something to rely on for the ≤6s/≤10s latency targets in your design doc once you're timing things for real.

## What the service actually does

- `POST /snapshots` `{"hospital_ids": ["CEDCS-SEED-001", ...]}` → returns each hospital's resource records (departments, beds, equipment, specialists, blood bank, OPD/IPD), each one carrying `updated_at`, `source`, and `reporter_id` — matching the `ResourceSnapshot` contract in the build prompt. This is what `ResourceSnapshotTool` should call.
- `POST /facilities/batch` `{"hospital_ids": [...]}` → same shape as `/facilities/nearby`, looked up by id.
- `GET /facilities/nearby?lat=..&lng=..&radius_km=8` → a PostGIS radius query over the same hospitals — a ready-made HTTP fallback for `FacilityRegistryLookupTool` if you'd rather call this service than embed DB credentials directly in that tool. (If you kept the original design — that tool connecting straight to `FACILITY_REGISTRY_DB_URL` — you don't need this endpoint at all; it's just here in case it's more convenient.)

## What this does *not* do

It does not compute freshness classes, run eligibility, or rank anything — those stay in `freshness.py` / `eligibility.py` / `ranking.py` exactly as the build prompt specifies. This service's only job is to hand back raw, provenance-tagged facts.
