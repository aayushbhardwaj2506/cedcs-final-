# CEDCS — Emergency hospital routing (V1 backend + command view)

```
free text ─► Intake ─► Triage ─► Red flags ─► Discovery ─► Resources      (AI layer: interprets)
                                                   │
        Requirements ─► Freshness ─► Eligibility ─► Escalation ─► Ranking ─► Engine   (deterministic core: decides)
                                                   │
                                   Validation gate ─► Explanation ─► recommendation
```
Principle P1: the AI layer interprets, the deterministic core decides, guardrails can only escalate or block.

## Run
`python scripts/run_all.py` → UI at http://127.0.0.1:8001 (needs Docker for Postgres/PostGIS).
The UI shows the live workflow, a per-step latency waterfall (with handoff gaps), the reasoning at every step, a map,
and p50/p95 latency across runs (also `GET /metrics`).

## Configuration (`.env`, see `.env.example`)
`GROQ_API_KEY` (LLM agents) · `LOCATIONIQ_KEY` (address search + real road travel times) · `DATABASE_URL` (Neon/Postgres+PostGIS) · `RESOURCE_SERVICE_URL`.

## Modes (`CEDCS_MODE`)
- `auto` (default): Groq if `GROQ_API_KEY` is set; else CrewAI if installed and configured; else rule-based.
- `groq`: intake, triage and explanation run on Groq (`llm/groq_agents.py`, prompts read from the CrewAI YAML config), each validated against the
  same schemas and retried once. Any LLM failure falls back to the rule-based stage. The rule engine is an escalate-only priority floor over the LLM.
  Three different models are used because free-tier rate limits are per model (`CEDCS_LLM_MODEL_<STAGE>` overrides).
- `offline`: rule-based only. `crewai`: the exported CrewAI crew (untested here).

## Tests
`python -m pytest` (the 2 skipped integration tests need `crewai` + an LLM key + `CEDCS_CREW_SRC` / `CEDCS_DETERMINISTIC_CORE_SRC`).

## Layout
`deterministic_core/` decision logic · `orchestrator/` pipeline + audit/timing · `guardrails/` · `schemas/` · `fallback/` rule-based stand-ins ·
`cedcs_ai_multi_agent_layer/` CrewAI export · `resource_service/` hospital data API · `api/` HTTP + UI.

## Agents and orchestration (Groq mode)
The orchestrator plans, dispatches and joins. Every command and reply is a message on a bus that the UI draws live (Orchestration card; `▶ Replay` re-plays a finished run).

| Agent | Job | Authority |
|---|---|---|
| Intake | free text → structured case | toggles and location stay authoritative |
| Triage | urgency + capabilities | rule engine is an escalate-only floor |
| Clarifier | what to ask the caller next | advisory, runs in the background |
| Critic | challenges the triage, finds contradictions | can only raise priority / add PREFERRED capabilities / request review |
| Advisor | second opinion on the shortlist | consulted only on near-ties or shaky top picks; score nudge capped at ±0.05, eligible hospitals only |
| Narrator | plain-language explanation | cannot change the recommendation; rejected if it fails the integrity check |

The hard gates never involve an LLM: eligibility, ranking arithmetic, engine re-verification, confidence, red-flag rules and the validation gate.
Every agent has a rule-based fallback, so an LLM outage or rate limit degrades the case instead of failing it.
Adaptive orchestration: independent work runs in parallel (triage ∥ clarifier; critic ∥ hospital search); the advisor is skipped when there is a clear winner.

## Live map
Leaflet (vendored in `api/static/vendor/leaflet`, no CDN) over LocationIQ tiles. Tiles and road routes are fetched through the server
(`/tiles/{style}/{z}/{x}/{y}.png`, `/route`), so the API key never reaches the browser and tiles are cached in memory.
Markers appear live as eligibility and ranking complete; road routes to the recommended hospital and alternatives are drawn at the end.
Dark/streets tiles follow the theme. Click the map to move the patient. Without `LOCATIONIQ_KEY` it falls back to a simple schematic map.

## Current location and live hospital search
`📍 Use my current location` asks the browser for GPS (works on localhost/HTTPS), shows the accuracy circle, reverse-geocodes the address
and sends `location_source=GPS` and `location_accuracy_m` with the case (a poor fix, over 300 m, adds a caveat).
Whenever the location changes, `/nearby` searches for hospitals and the map shows a radar sweep with hospitals appearing by distance.
During a run the map narrates each phase (searching 8 km, widening to 25 km, checking capabilities, ranking...) and draws the routes at the end.

## Bed Checker (manual) and emergency dispatch
**Bed Checker** — off by default and never turns itself on. A user can switch it on for their own cases (the 🛏 toggle) unless the admin
disallows that; the admin can enable it for everyone. When on, a live scan reads nearby hospitals' bed records straight from the database
(bypassing caches) and bed availability becomes the largest single ranking weight (35%; the other weights are scaled to 65%). Unknown is
never treated as available, and older reports count for less.

**Emergency dispatch** — after a recommendation, *Send emergency alerts* e-mails (with the patient's GPS location, map links and condition):
family contacts, the best-ranked hospital, and the nearest ambulance. It needs explicit confirmation and happens once per case.
* Default is a **dry run**: nothing is sent; every message is logged in the outbox. Set `DISPATCH_MODE=smtp` plus SMTP_* for real delivery.
  `DISPATCH_TEST_INBOX` redirects *every* message to one inbox (recommended first). Synthetic hospital/ambulance addresses (example.org) are
  refused in live mode without it, and credentials never appear in logs, errors or API responses.
* **Confirmation loop:** each hospital e-mail has Accept/Decline links to a confirmation page (opening a link changes nothing; only the
  form POST records it, since mail scanners prefetch links). Optional IMAP polling turns "ACCEPT"/"DECLINE" replies into acknowledgements.
  Accepting reserves a bed in the hospital's records; declining, or no answer within the timeout, notifies the next-ranked hospital.
* Family can be sent status updates. Auto-dispatch for CRITICAL/HIGH cases exists but is an admin opt-in (off by default).
* Admin settings need `ADMIN_TOKEN` (in `.env`). State (contacts, outbox, acks, settings) is in `data/cedcs_app.db`.

## Two LLM providers: Groq + NVIDIA (`llm/providers.py`, `llm/groq_agents.py`)
Set `GROQ_API_KEY` and/or `NVIDIA_API_KEY`. Each agent has a *plan*, an ordered list of `provider:model`, and the call goes to the first
entry that is healthy and has headroom. Default split: **Groq** for triage, advisor (gpt-oss-120b) and the explanation (gpt-oss-20b);
**NVIDIA** for intake, clarification and critic (nemotron-3-super, thinking off, ~2-3 s). Every plan continues onto the other provider.
* **Failover** on rate limits, timeouts, 5xx, empty answers; a transient NVIDIA 503 is retried once first.
* **Groq** limits are per model per minute *and per day* (200k tokens/day/model on the free tier); the budget comes from response headers and
  `retry-after`. **NVIDIA** is limited by request rate (soft cap 35/min) and its free endpoints cold-start (25-40 s), so it gets a short
  timeout, a startup warm-up and a keep-warm ping, and a model that 404s for the account is retired for 10 minutes.
* **Circuit breaker** per provider (3 consecutive failures opens it for 25 s); if nothing looks available the plan is probed anyway.
* Optional agents (clarifier, critic, advisor) are skipped when the shared token budget is tight; essential ones (intake, triage, explanation)
  never are, and fall back to the rule-based stage only if every model fails.
* `CEDCS_LLM_PROVIDERS=groq` (or `nvidia`) restricts to one provider; `CEDCS_LLM_MODEL_<STAGE>=provider:model` overrides one agent.
* The UI's **AI providers** card (`GET /llm/status`) shows each provider's health, latency, calls/min, budgets and every agent's plan.

## Real hospitals (OpenStreetMap) vs the synthetic network

Setting `hospital_source` (admin, default `real`; effective only when `LOCATIONIQ_KEY` is set) chooses the network.

- **Real mode** discovers actual hospitals near the patient via LocationIQ Nearby (OpenStreetMap data). Their beds, equipment and
  departments are **not reported** anywhere, so every real hospital is *provisional* (ranked with the unknown-capability penalty) and
  the recommendation says to confirm by phone. Only the hospital's *name* is used to rule out eye/dental/homeopathic clinics or
  out-of-field specialty hospitals, and that is labelled as an inference.
- **Hospital console**: an admin issues a private link (`POST /admin/consoles`, or the map popup) so a hospital's staff can report
  beds, facilities and status. Reports are timestamped, decay with the usual freshness model, and feed ranking and bed reservation.
- **Dispatch** never invents a hospital email: without a registered console contact it falls back to phone / manual acknowledgement.
- Ambulance units are always simulated. The synthetic network (Neon) remains available for demos and tests.

### Facilities declared on the map (OpenStreetMap tags)

For real hospitals, `services/osm_tags.py` queries Overpass once per search area (cached 24 h; disable with `CEDCS_OSM_TAGS=0`) and
reads what the entry declares: `emergency=yes/no`, `healthcare:speciality`, phone, website, opening hours, operator, beds.
These become **INFERRED** resource records (trust 0.5, aged from `check_date`, else 30 days). They lift a hospital in ranking
(share of required/preferred capabilities evidenced), a declared `emergency=no` excludes it, but a capability backed only by a
map tag stays **provisional**: nobody at the hospital has confirmed it. Coverage is sparse (in a sample of 152 Chennai-area entries,
8 had `emergency`, 4 a speciality, 5 a phone), so most hospitals still show "not reported". Hospital console reports always win.

## Hospital portal

A second, parallel screen for the receiving hospital: `/portal/<token>`. The operator clicks **Open this hospital's portal** on a
hospital in the dispatch card (admin token; creates or reuses that hospital's private link). The portal shows the hospital's
incoming patients live (priority, condition, needs, ambulance ETA, pick-up location; never the family's contact details) with
**Accept** / **Decline**, optional free beds, prep ETA and a note. An answer updates the operator's dispatch immediately (accepting
reserves a bed if the hospital has reported its beds) and declining triggers the next-hospital escalation. The portal links to the
beds and facilities form.

## Saved history (latency reports and everything that happened)

Every run is saved, with its full latency report, reasoning trace, AI usage and outcome, plus an event log of what happens around
it: dispatch sent, hospital accepted/declined, escalation, family update, beds reported or reserved, settings changed.

- **Where:** Neon Postgres when `HISTORY_DATABASE_URL` (or `DATABASE_URL`) is set, in its own schema `cedcs_history` (tables
  `case_runs`, `stage_timings`, `events`; JSON columns are JSONB, so they can be queried in Neon's SQL editor). Otherwise, or if Neon
  cannot be reached, a local file `data/cedcs_history.db` (set `CEDCS_HISTORY_BACKEND=sqlite` to force it).
- **Never slows a case:** writes go through a background worker. `CEDCS_HISTORY=off` disables it; `HISTORY_MAX_RUNS` (default 2000)
  keeps the newest runs.
- **See it:** the *Saved history* card in the operator screen (admin token), `/history/runs`, `/history/runs/{id}`,
  `/history/events`, `/history/latency` (admin token) and `/history/status`. The live latency panel is rebuilt from saved runs after a restart.
- **Privacy:** a saved run contains the emergency description and coordinates. Family e-mail addresses are never saved here.

## Analytics dashboard

`http://localhost:8001/analytics` (admin token) turns the saved history into charts and research data:

- **Latency:** median / p95 / max, the time of every run (coloured by urgency), a distribution, and per-phase and per-step timings.
- **AI agents:** calls, success and fallback rate, latency and tokens per provider (Groq vs NVIDIA), and which provider answered each agent.
- **Decisions:** urgency mix, hospitals found / eligible / provisional / rejected per run, confidence levels, most-recommended hospitals, ETA.
- **Dispatch:** hospital responses and the time to the first reply, from the event log.
- **Research data:** `runs.csv`, `stage-timings.csv`, `events.csv`, the dashboard as JSON, a summary table that copies as Markdown, and
  per-run JSON. Percentiles use linear interpolation and are over completed runs; aim for 30+ runs before quoting them.

The same figures are available as JSON at `/history/analytics?days=N` and as CSV at `/history/export/{runs|stages|events}`.
