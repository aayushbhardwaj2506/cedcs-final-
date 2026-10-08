"""HTTP API + web UI for CEDCS.

  GET  /                UI
  POST /cases           run the pipeline, return recommendation + timeline + decision trace
  POST /cases/stream    same, but as Server-Sent Events so the UI can animate live
  GET  /metrics         per-stage latency statistics across recent runs
"""

from __future__ import annotations

import html as _html
import os
import json
import queue
import secrets
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


try:  # local dev: read secrets from cedcs-app/.env (never overrides real environment variables)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from dispatch import ambulances, compose, mailer, service, store
from services import history
from orchestrator.audit import AuditLog
from orchestrator.orchestrator import CedcsOrchestrator



@asynccontextmanager
async def lifespan(_app):
    # warm imports/session so the first request is not slower than the rest
    import deterministic_core.engine  # noqa: F401
    import fallback.rule_based  # noqa: F401
    from llm import groq_agents

    def _load_history():  # the latency panel survives a restart: reload the saved runs
        try:
            runs = history.stage_runs(500)
            with _RUNS_LOCK:
                _RUNS.extendleft(reversed(runs))
        except Exception as exc:
            print(f"[cedcs] could not load run history: {type(exc).__name__}")

    if history.enabled():
        threading.Thread(target=_load_history, daemon=True).start()
    threading.Thread(target=groq_agents.prime_budget, daemon=True).start()  # no-op without a key
    threading.Thread(target=groq_agents.keep_warm_forever, daemon=True).start()  # keeps NVIDIA's free endpoints from going cold
    if service.imap_configured():  # optional: hospital replies by e-mail become acknowledgements
        def _imap_loop():
            import time as _t

            while True:
                try:
                    service.poll_imap()
                except Exception as exc:
                    print(f"[cedcs] imap poll failed: {type(exc).__name__}")
                _t.sleep(20)

        threading.Thread(target=_imap_loop, daemon=True).start()

    yield


app = FastAPI(title="CEDCS API", version="0.2.0", lifespan=lifespan)

# ---------------------------------------------------------------- optional access gate for a public deployment
OPEN_PREFIXES = ("/portal/", "/console/", "/ack/", "/health", "/favicon.ico")  # hospital staff and mail recipients arrive through secret links


@app.middleware("http")
async def access_gate(request: Request, call_next):
    """With ACCESS_PASSWORD set, every operator page and API needs HTTP Basic auth; the token-link pages stay open."""
    password = os.environ.get("ACCESS_PASSWORD", "")
    if password and not request.url.path.startswith(OPEN_PREFIXES):
        import base64

        given = ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("basic "):
            try:
                given = base64.b64decode(auth[6:]).decode("utf-8", "replace").partition(":")[2]
            except Exception:
                given = ""
        if not secrets.compare_digest(given.encode(), password.encode()):
            from fastapi.responses import Response

            return Response("Sign in to use CEDCS.", status_code=401, headers={"WWW-Authenticate": 'Basic realm="CEDCS"'})
    return await call_next(request)


_CASES: "OrderedDict[str, dict]" = OrderedDict()  # finished cases, so a dispatch can be built from one (last 100)
_CASES_LOCK = threading.Lock()
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN") or secrets.token_urlsafe(12)
if not os.environ.get("ADMIN_TOKEN"):
    print(f"[cedcs] admin token for this session (set ADMIN_TOKEN to choose your own): {ADMIN_TOKEN}")


def require_admin(token: Optional[str]) -> None:
    if not token or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="admin token required")


_RUNS: deque = deque(maxlen=500)  # {"total_ms": float, "stages": {name: ms}} per run
_RUNS_LOCK = threading.Lock()


class CaseRequest(BaseModel):
    emergency_report: str
    consciousness: str = "unknown"
    breathing: str = "unknown"
    bleeding: str = "unknown"
    location_lat: float
    location_lng: float
    location_address: str = ""
    location_source: str = "MANUAL"  # GPS | GEOCODED | MANUAL
    location_accuracy_m: Optional[float] = None
    bed_check: bool = False  # user switches the Bed Checker on for this case (only honoured if the admin allows it)
    case_id: Optional[str] = None


class CaseResponse(BaseModel):
    case_id: str
    halted: bool
    halt_reason: Optional[str] = None
    handoff_required: bool = False
    mode: str = ""
    total_ms: float = 0.0
    recommendation: Optional[dict[str, Any]] = None
    explanation_text: Optional[str] = None
    timeline: list[dict[str, Any]] = []
    trace: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []


def _percentile(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def _remember_case(case_id: str, request: dict, result) -> None:
    if result.halted or result.recommendation is None:
        return
    with _CASES_LOCK:
        _CASES[case_id] = {"request": request, "result": {"recommendation": result.recommendation.model_dump(), "trace": result.trace}}
        while len(_CASES) > 100:
            _CASES.popitem(last=False)


def _maybe_auto_dispatch(orch, audit, req: CaseRequest, result) -> Optional[dict]:
    """If an admin enabled auto-dispatch, CRITICAL/HIGH cases alert family, hospital and ambulance without a click."""
    _remember_case(result.case_id, req.model_dump(exclude_none=True), result)
    st = store.get_settings()
    if not st["auto_dispatch"] or result.halted or result.recommendation is None:
        return None
    if (result.trace.get("red_flags", {}).get("final_priority") or "") not in ("CRITICAL", "HIGH"):
        return None
    audit.message("ORCH", "DISPATCH", "command", "notify family, hospital and ambulance (auto-dispatch)")
    try:
        d = service.send_dispatch(compose.snapshot(result.case_id, req.model_dump(exclude_none=True), _CASES[result.case_id]["result"]), confirmed=True)
        n = len(d["messages"])
        audit.message("DISPATCH", "ORCH", "result", f"{n} message(s) {'logged' if d['mode'] == 'outbox' else 'sent'}", dispatch_id=d["id"])
        return {"dispatch_id": d["id"], "messages": n}
    except Exception as exc:
        audit.message("DISPATCH", "ORCH", "error", f"auto-dispatch failed: {str(exc)[:60]}")
        return {"error": str(exc)[:120]}


def _execute(req: CaseRequest, on_event=None) -> CaseResponse:
    audit = AuditLog(on_event=on_event)
    orch = CedcsOrchestrator(audit=audit)
    result = orch.run_case(req.model_dump(exclude_none=True))
    timeline = audit.timeline()
    total = next((s["duration_ms"] for s in timeline if s["stage"] == "PIPELINE"), 0.0)
    with _RUNS_LOCK:
        _RUNS.append({"total_ms": total, "stages": {s["stage"]: s["duration_ms"] for s in timeline}})
    auto = _maybe_auto_dispatch(orch, audit, req, result)
    if auto:
        result.trace["auto_dispatch"] = auto
    response = CaseResponse(
        case_id=result.case_id,
        halted=result.halted,
        halt_reason=result.halt_reason,
        handoff_required=result.halted,  # no safe automated recommendation -> human operator
        mode=str(result.trace.get("mode", "")),
        total_ms=total,
        recommendation=result.recommendation.model_dump() if result.recommendation else None,
        explanation_text=result.explanation_text,
        timeline=timeline,
        messages=audit.messages,
        trace=result.trace,
        audit=result.audit.as_list(),
    )
    from services import registry

    history.record_case(req.model_dump(exclude_none=True), response.model_dump(), hospital_source=registry.source())
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ---------------------------------------------------------------- settings (admin) and contacts
@app.get("/settings")
def get_settings():
    from services import hospitals_osm, registry

    return {**store.get_settings(), "hospital_source_effective": registry.source(), "real_hospitals_available": hospitals_osm.enabled(),
            "mail": mailer.describe(), "imap": service.imap_configured(),
            "public_base_url": service.public_base(), "ambulances": len(ambulances.AMBULANCES)}


class SettingsUpdate(BaseModel):
    hospital_source: Optional[str] = None
    bed_checker_enabled: Optional[bool] = None
    allow_user_toggle: Optional[bool] = None
    auto_dispatch: Optional[bool] = None
    auto_escalate: Optional[bool] = None
    ack_timeout_s: Optional[int] = None


@app.put("/settings")
def put_settings(body: SettingsUpdate, x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    out = store.put_settings(changes)
    history.event("settings_changed", **changes)
    return out


class ContactIn(BaseModel):
    name: str
    email: str
    relation: str = ""


@app.get("/contacts")
def list_contacts():
    return {"contacts": store.list_contacts()}


@app.post("/contacts")
def add_contact(c: ContactIn):
    import re

    if not c.name.strip() or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", c.email.strip()):
        raise HTTPException(status_code=422, detail="a name and a valid email address are required")
    if len(store.list_contacts()) >= 10:
        raise HTTPException(status_code=422, detail="at most 10 emergency contacts")
    return store.add_contact(c.name, c.email, c.relation)


@app.delete("/contacts/{contact_id}")
def delete_contact(contact_id: str):
    return {"ok": store.delete_contact(contact_id)}


# ---------------------------------------------------------------- hospital console (real hospitals report their own capacity)
class ConsoleIn(BaseModel):
    hospital_id: str
    name: str = ""
    email: str = ""
    phone: str = ""


@app.post("/admin/consoles")
def admin_create_console(body: ConsoleIn, x_admin_token: Optional[str] = Header(None)):
    """Issue a private link a hospital's staff can use to report beds, facilities and status. Admin only."""
    import re

    require_admin(x_admin_token)
    if not body.hospital_id.strip():
        raise HTTPException(status_code=422, detail="hospital_id is required")
    if body.email.strip() and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", body.email.strip()):
        raise HTTPException(status_code=422, detail="that email address is not valid")
    c = store.create_console(body.hospital_id.strip(), body.name.strip(), body.email, body.phone)
    return {**c, "url": f"{service.public_base()}/console/{c['token']}"}


@app.get("/admin/consoles")
def admin_list_consoles(x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    return {"consoles": store.list_consoles()}


_CONSOLE_CAPS = [("department_EMERGENCY_DEPARTMENT", "Emergency department", "active"), ("department_CARDIOLOGY", "Cardiology", "active"),
                 ("department_NEUROLOGY", "Neurology", "active"), ("department_TRAUMA", "Trauma care", "active"),
                 ("department_ORTHOPEDICS", "Orthopaedics", "active"), ("department_PEDIATRICS", "Paediatrics", "active"),
                 ("department_OBSTETRICS", "Obstetrics", "active"), ("department_GENERAL_MEDICINE", "General medicine", "active"),
                 ("equipment_CT_SCAN", "CT scan", "operational"), ("equipment_MRI", "MRI", "operational"), ("equipment_X_RAY", "X-ray", "operational"),
                 ("equipment_CATH_LAB", "Cath lab", "operational"), ("equipment_OT", "Operating theatre", "operational"),
                 ("equipment_DIALYSIS", "Dialysis", "operational"), ("equipment_CARDIAC_MONITOR", "Cardiac monitoring", "operational"),
                 ("blood_bank", "Blood bank stock", "available")]
_CONSOLE_BEDS = [("icu_beds", "ICU beds"), ("emergency_beds", "Emergency beds"), ("general_beds", "General beds"), ("hdu_beds", "HDU beds"),
                 ("pediatric_beds", "Paediatric beds"), ("ventilators", "Ventilators")]


def _console_page(console: dict, saved: bool = False) -> str:
    e = _html.escape
    rep = {r["resource_key"]: r for r in store.reports_for([console["hospital_id"]]).get(console["hospital_id"], [])}
    import datetime as _dt

    def when(k):
        return _dt.datetime.fromtimestamp(rep[k]["updated_at"]).strftime("%d %b %H:%M") if k in rep else ""

    beds = "".join(
        f'<tr><td>{e(label)}</td><td><input name="{k}_avail" type="number" min="0" max="2000" value="{rep.get(k, {}).get("value", {}).get("available", "")}" placeholder="free now"></td>'
        f'<td><input name="{k}_total" type="number" min="0" max="2000" value="{rep.get(k, {}).get("value", {}).get("total", "")}" placeholder="total"></td><td class="m">{e(when(k))}</td></tr>'
        for k, label in _CONSOLE_BEDS)

    def sel(k, field):
        v = rep.get(k, {}).get("value", {}).get(field)
        opts = [("", "not reporting"), ("yes", "available"), ("no", "not available")]
        cur = "" if v is None else ("yes" if v else "no")
        return "".join(f'<option value="{o}" {"selected" if o == cur else ""}>{t}</option>' for o, t in opts)

    caps = "".join(f'<tr><td>{e(label)}</td><td><select name="{k}">{sel(k, field)}</select></td><td class="m">{e(when(k))}</td></tr>' for k, label, field in _CONSOLE_CAPS)
    st = rep.get("operating_status", {}).get("value", {})
    status_opts = "".join(f'<option value="{o}" {"selected" if st.get("status") == o else ""}>{t}</option>' for o, t in
                          [("", "not reporting"), ("OPERATIONAL", "Open: receiving emergencies"), ("DIVERTING", "Diverting: send patients elsewhere"), ("CLOSED", "Closed")])
    body = f"""<h1>{e(console.get('name') or console['hospital_id'])}</h1><p class="m">Hospital console &middot; what you report here is shown to emergency teams choosing where to send patients.
Report only what is true right now; leave anything you are unsure of as &ldquo;not reporting&rdquo;. Each value is stamped with the time you save it.</p>
{'<p class="ok">Saved. Thank you.</p>' if saved else ''}
<form method="post"><h2>Status</h2><p><select name="operating_status">{status_opts}</select> <label><input type="checkbox" name="ipd_accepting" {"checked" if st.get("ipd_accepting", True) else ""}> accepting admissions</label></p>
<h2>Beds free right now</h2><table><tr><th></th><th>Free</th><th>Total</th><th>Last reported</th></tr>{beds}</table>
<h2>Facilities</h2><table><tr><th></th><th>Status</th><th>Last reported</th></tr>{caps}</table><p><button>Save report</button></p></form>"""
    return _ack_shell(body).replace("</style>", "table{border-collapse:collapse;width:100%}td,th{padding:5px 8px;border-bottom:1px solid #ddd;text-align:left}.m{color:#666;font-size:13px}.ok{color:#0a7d3c;font-weight:600}select,input{padding:6px}</style>", 1)


@app.get("/console/{token}", response_class=HTMLResponse)
def console_page(token: str):
    c = store.console_by_token(token)
    if c is None:
        raise HTTPException(status_code=404, detail="unknown console link")
    return _console_page(c)


@app.post("/console/{token}", response_class=HTMLResponse)
async def console_submit(token: str, request: Request):
    c = store.console_by_token(token)
    if c is None:
        raise HTTPException(status_code=404, detail="unknown console link")
    form = {k: str(v).strip() for k, v in (await request.form()).items()}
    hid, who = c["hospital_id"], f"console:{token[:6]}"

    def num(name):
        try:
            v = int(form.get(name, ""))
            return v if 0 <= v <= 2000 else None
        except ValueError:
            return None

    for key, _ in _CONSOLE_BEDS:
        avail, total = num(f"{key}_avail"), num(f"{key}_total")
        if avail is not None:
            store.upsert_report(hid, key, {"available": avail, "total": total if total is not None and total >= avail else avail}, who)
    for key, _, field in _CONSOLE_CAPS:
        v = form.get(key, "")
        if v in ("yes", "no"):
            store.upsert_report(hid, key, {field: v == "yes"}, who)
    if form.get("operating_status") in ("OPERATIONAL", "DIVERTING", "CLOSED"):
        store.upsert_report(hid, "operating_status", {"status": form["operating_status"], "ipd_accepting": "ipd_accepting" in form}, who)
    return _console_page(c, saved=True)


# ---------------------------------------------------------------- hospital portal (runs in parallel with the operator's Command View)
def _portal_ctx(token: str) -> dict:
    c = store.console_by_token(token)
    if c is None:
        raise HTTPException(status_code=404, detail="unknown portal link")
    return c


@app.get("/portal/{token}", response_class=HTMLResponse)
def portal_page(token: str):
    _portal_ctx(token)
    return (Path(__file__).parent / "static" / "portal.html").read_text(encoding="utf-8")


@app.get("/portal/{token}/data")
def portal_data(token: str):
    """Everything the hospital's staff see: their incoming patients (newest first), each patient's case summary and the state of
    their own answer, plus what they have reported. Contacts of the family are never included."""
    c = _portal_ctx(token)
    hid = c["hospital_id"]
    incoming = []
    for item in store.hospital_inbox(hid):
        m, d = item["message"], item["dispatch"]
        snap = d["payload"]["snapshot"]
        me = next((h for h in snap["hospitals"] if h["id"] == hid), {})
        amb = d["payload"].get("ambulance") or {}
        others_accepted = any(x["kind"] == "HOSPITAL" and x["id"] != m["id"] and x.get("ack") and x["ack"]["response"] == "ACCEPTED" for x in d["messages"])
        incoming.append({
            "message_id": m["id"], "received_at": m["created_at"], "dispatch_status": d["status"], "case_id": snap["case_id"],
            "priority": snap["priority"], "condition": compose.condition_text(snap), "needs": snap.get("needs", []),
            "symptoms": snap["patient"].get("symptoms", []), "history": snap["patient"].get("history", []), "red_flags": snap.get("red_flags", []),
            "patient": {k: snap["patient"].get(k) for k in ("age", "gender")}, "eta_min": me.get("eta_min"), "rank": me.get("rank"),
            "location": snap["location"], "ambulance": {"name": amb.get("name"), "eta_min": amb.get("eta_min"), "simulated": amb.get("simulated")} if amb else None,
            "backup": bool(m["subject"] and "BACKUP" in m["subject"].upper()), "answer": m["ack"], "another_hospital_accepted": others_accepted,
        })
    rep = store.reports_for([hid]).get(hid, [])
    return {"hospital": {"id": hid, "name": c.get("name") or hid}, "incoming": incoming, "console_url": f"/console/{token}",
            "reported": {r["resource_key"]: r["value"] for r in rep if r["resource_key"] in store.REPORT_KEYS_BEDS},
            "server_time": time.time()}


class PortalRespond(BaseModel):
    message_id: str
    response: str  # accept | decline
    note: str = ""
    eta_min: Optional[float] = None
    beds_free: Optional[int] = None


@app.post("/portal/{token}/respond")
def portal_respond(token: str, body: PortalRespond):
    c = _portal_ctx(token)
    item = next((i for i in store.hospital_inbox(c["hospital_id"], 100) if i["message"]["id"] == body.message_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="that alert is not addressed to this hospital")
    if body.beds_free is not None and body.response.lower().startswith("accept"):
        from fallback import rule_based

        try:
            rule_based.report_beds(c["hospital_id"], service._bed_key(item["dispatch"]["payload"]["snapshot"]), max(0, body.beds_free), reporter_id=f"portal:{token[:6]}")
        except Exception:
            pass  # a failed bed report must not lose the answer itself
    try:
        r = service.record_ack(item["message"]["token"], body.response, note=body.note, eta_min=body.eta_min, source="PORTAL")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True, "response": r["response"], "bed": r["bed"], "dispatch_status": r["dispatch_status"]}


@app.post("/dispatch/{dispatch_id}/portal")
def dispatch_portal_link(dispatch_id: str, hospital_id: str, x_admin_token: Optional[str] = Header(None)):
    """Admin: open (creating if needed) the portal link of a hospital in this dispatch, so it can be shown in a second window."""
    require_admin(x_admin_token)
    d = store.get_dispatch(dispatch_id)
    if d is None or not any(m.get("hospital_id") == hospital_id for m in d["messages"]):
        raise HTTPException(status_code=404, detail="that hospital is not part of this dispatch")
    c = store.console_for_hospital(hospital_id)
    if c is None:
        name = next((m["to_name"] for m in d["messages"] if m.get("hospital_id") == hospital_id), hospital_id)
        c = store.create_console(hospital_id, name)
    return {"url": f"{service.public_base()}/portal/{c['token']}", "hospital_id": hospital_id}


# ---------------------------------------------------------------- hospital portal (runs in parallel with the operator's Command View)
def _portal_ctx(token: str) -> dict:
    c = store.console_by_token(token)
    if c is None:
        raise HTTPException(status_code=404, detail="unknown portal link")
    return c


@app.get("/portal/{token}", response_class=HTMLResponse)
def portal_page(token: str):
    _portal_ctx(token)
    return (Path(__file__).parent / "static" / "portal.html").read_text(encoding="utf-8")


@app.get("/portal/{token}/data")
def portal_data(token: str):
    """What a hospital's staff see: their incoming patients (newest first), each patient's case summary and the state of their own
    answer, plus what they have reported. The family's contact details are never included."""
    c = _portal_ctx(token)
    hid = c["hospital_id"]
    incoming = []
    for item in store.hospital_inbox(hid):
        m, d = item["message"], item["dispatch"]
        snap = d["payload"]["snapshot"]
        me = next((h for h in snap["hospitals"] if h["id"] == hid), {})
        amb = d["payload"].get("ambulance") or {}
        others_accepted = any(x["kind"] == "HOSPITAL" and x["id"] != m["id"] and x.get("ack") and x["ack"]["response"] == "ACCEPTED"
                              for x in d["messages"])
        incoming.append({
            "message_id": m["id"], "received_at": m["created_at"], "dispatch_status": d["status"], "case_id": snap["case_id"],
            "priority": snap["priority"], "condition": compose.condition_text(snap), "needs": snap.get("needs", []),
            "symptoms": snap["patient"].get("symptoms", []), "history": snap["patient"].get("history", []), "red_flags": snap.get("red_flags", []),
            "patient": {k: snap["patient"].get(k) for k in ("age", "gender")}, "eta_min": me.get("eta_min"), "rank": me.get("rank"),
            "location": snap["location"],
            "ambulance": {"name": amb.get("name"), "eta_min": amb.get("eta_min"), "simulated": amb.get("simulated")} if amb else None,
            "backup": bool(m["subject"] and "BACKUP" in m["subject"].upper()), "answer": m["ack"], "another_hospital_accepted": others_accepted,
        })
    rep = store.reports_for([hid]).get(hid, [])
    return {"hospital": {"id": hid, "name": c.get("name") or hid}, "incoming": incoming, "console_url": f"/console/{token}",
            "reported": {r["resource_key"]: r["value"] for r in rep if r["resource_key"] in store.REPORT_KEYS_BEDS},
            "server_time": time.time()}


class PortalRespond(BaseModel):
    message_id: str
    response: str  # accept | decline
    note: str = ""
    eta_min: Optional[float] = None
    beds_free: Optional[int] = None


@app.post("/portal/{token}/respond")
def portal_respond(token: str, body: PortalRespond):
    c = _portal_ctx(token)
    item = next((i for i in store.hospital_inbox(c["hospital_id"], 100) if i["message"]["id"] == body.message_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="that alert is not addressed to this hospital")
    if body.beds_free is not None and body.response.lower().startswith("accept"):
        from fallback import rule_based

        try:
            rule_based.report_beds(c["hospital_id"], service._bed_key(item["dispatch"]["payload"]["snapshot"]), max(0, body.beds_free),
                                   reporter_id=f"portal:{token[:6]}")
        except Exception:
            pass  # a failed bed report must not lose the answer itself
    try:
        r = service.record_ack(item["message"]["token"], body.response, note=body.note, eta_min=body.eta_min, source="PORTAL")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True, "response": r["response"], "bed": r["bed"], "dispatch_status": r["dispatch_status"]}


@app.post("/dispatch/{dispatch_id}/portal")
def dispatch_portal_link(dispatch_id: str, hospital_id: str, x_admin_token: Optional[str] = Header(None)):
    """Admin: open (creating if needed) the portal link of a hospital in this dispatch, to show it in a second window."""
    require_admin(x_admin_token)
    d = store.get_dispatch(dispatch_id)
    if d is None or not any(m.get("hospital_id") == hospital_id for m in d["messages"]):
        raise HTTPException(status_code=404, detail="that hospital is not part of this dispatch")
    c = store.console_for_hospital(hospital_id)
    if c is None:
        name = next((m["to_name"] for m in d["messages"] if m.get("hospital_id") == hospital_id), hospital_id)
        c = store.create_console(hospital_id, name)
    return {"url": f"{service.public_base()}/portal/{c['token']}", "hospital_id": hospital_id}


# ---------------------------------------------------------------- emergency dispatch
class DispatchIn(BaseModel):
    case_id: str
    confirm: bool = False
    include_family: bool = True


def _snapshot_for(case_id: str) -> dict:
    with _CASES_LOCK:
        c = _CASES.get(case_id)
    if c is None:
        raise HTTPException(status_code=404, detail="unknown or expired case: run the case again")
    return compose.snapshot(case_id, c["request"], c["result"])


@app.post("/dispatch/preview")
def dispatch_preview(body: DispatchIn):
    return service.preview(_snapshot_for(body.case_id))


@app.post("/dispatch")
def dispatch_send(body: DispatchIn):
    if not body.confirm:
        raise HTTPException(status_code=400, detail="confirmation required: set confirm=true to send emergency alerts")
    try:
        return service.send_dispatch(_snapshot_for(body.case_id), confirmed=True, include_family=body.include_family)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/dispatch/{dispatch_id}")
def dispatch_status(dispatch_id: str):
    d = service.status(dispatch_id)
    if d is None:
        raise HTTPException(status_code=404, detail="unknown dispatch")
    return d


class UpdateIn(BaseModel):
    note: str


@app.post("/dispatch/{dispatch_id}/update")
def dispatch_update(dispatch_id: str, body: UpdateIn):
    try:
        return service.send_update(dispatch_id, body.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown dispatch")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/dispatch/{dispatch_id}/escalate")
def dispatch_escalate(dispatch_id: str):
    try:
        nxt = service.escalate(dispatch_id, reason="manual")
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown dispatch")
    return {"notified": nxt, "dispatch": service.status(dispatch_id, tick=False)}


class SimAck(BaseModel):
    message_id: str
    response: str


@app.post("/dispatch/{dispatch_id}/simulate-ack")
def dispatch_simulate_ack(dispatch_id: str, body: SimAck):
    """Dry-run only: play the hospital's part so the confirmation loop can be tried without a real mailbox."""
    if mailer.mode() != "outbox":
        raise HTTPException(status_code=403, detail="simulated replies are only available in dry-run mode")
    d = store.get_dispatch(dispatch_id)
    msg = next((m for m in (d or {}).get("messages", []) if m["id"] == body.message_id and m["kind"] == "HOSPITAL"), None)
    if msg is None:
        raise HTTPException(status_code=404, detail="unknown hospital message")
    try:
        r = service.record_ack(msg["token"], body.response, note="(simulated reply)", source="SIMULATED")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {**r, "dispatch": service.status(dispatch_id, tick=False)}


class ManualAck(BaseModel):
    message_id: str
    response: str
    note: str = ""


@app.post("/dispatch/{dispatch_id}/manual-ack")
def dispatch_manual_ack(dispatch_id: str, body: ManualAck):
    """A person records a hospital's answer given by phone. Works in every mode (unlike simulated replies)."""
    try:
        r = service.manual_ack(dispatch_id, body.message_id, body.response, body.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown hospital message")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {**r, "dispatch": service.status(dispatch_id, tick=False)}


@app.get("/outbox")
def outbox():
    return {"mail": mailer.describe(), "messages": store.list_outbox()}


# ---------------------------------------------------------------- hospital confirmation page (link in the email)
def _ack_page(msg: dict, preset: str = "", done: Optional[dict] = None) -> str:
    d = store.get_dispatch(msg["dispatch_id"])
    snap = d["payload"]["snapshot"]
    e = _html.escape
    body = f"""<h1>Incoming emergency patient</h1><p><b>{e(msg['to_name'])}</b> &middot; case {e(snap['case_id'][:8])}</p>
<p class="pill">{e(snap['priority'])} urgency</p><p>{e(compose.condition_text(snap))}</p>
<p>Needs: {e(', '.join(n.replace('_', ' ').lower() for n in snap['needs']) or 'general emergency care')}</p>"""
    if done is not None:
        bed = done.get("bed") or {}
        extra = (f"<p>Bed reserved: {e(bed.get('bed_key', ''))} {bed.get('before')} &rarr; {bed.get('after')}.</p>" if bed.get("ok")
                 else (f"<p>No bed was reserved ({e(str(bed.get('reason', '')))}). Please prepare manually.</p>" if bed else ""))
        return _ack_shell(body + f"<h2>Thank you: recorded as {e(done['response'])}</h2>{extra}")
    body += f"""<form method="post"><label><input type="radio" name="response" value="accept" {'checked' if preset == 'accept' else ''} required> <b>Accept</b>: we can receive this patient (a bed will be reserved)</label><br>
<label><input type="radio" name="response" value="decline" {'checked' if preset == 'decline' else ''}> <b>Decline</b>: no capacity</label>
<p><label>Free beds of the needed type right now (optional) <input name="beds_free" type="number" min="0" max="500"></label></p>
<p><label>Your ETA to prepare, minutes (optional) <input name="eta_min" type="number" min="0" max="240"></label></p>
<p><label>Note <input name="note" maxlength="300" style="width:100%"></label></p><button>Send confirmation</button></form>"""
    return _ack_shell(body)


def _ack_shell(inner: str) -> str:
    return ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>CEDCS confirmation</title>"
            "<style>body{font:16px/1.5 system-ui;max-width:560px;margin:24px auto;padding:0 16px;color:#111}.pill{display:inline-block;padding:2px 12px;background:#fee;color:#b00020;border-radius:99px;font-weight:700}"
            "button{padding:10px 18px;border:0;border-radius:6px;background:#2563eb;color:#fff;font-weight:600}input{padding:6px}</style>"
            f"{inner}<hr><small>{_html.escape(compose.DISCLAIMER)}</small>")


@app.get("/ack/{token}", response_class=HTMLResponse)
def ack_page(token: str, response: str = ""):
    msg = store.message_by_token(token)
    if msg is None or msg["kind"] != "HOSPITAL":
        raise HTTPException(status_code=404, detail="unknown confirmation link")
    return _ack_page(msg, preset=response.lower())  # GET never changes anything: mail scanners prefetch links


@app.post("/ack/{token}", response_class=HTMLResponse)
async def ack_submit(token: str, request: Request):
    msg = store.message_by_token(token)
    if msg is None or msg["kind"] != "HOSPITAL":
        raise HTTPException(status_code=404, detail="unknown confirmation link")
    form = {k: str(v) for k, v in (await request.form()).items()}
    try:
        beds_free = int(form["beds_free"]) if form.get("beds_free", "").strip() else None
        eta = float(form["eta_min"]) if form.get("eta_min", "").strip() else None
        d = store.get_dispatch(msg["dispatch_id"])
        if beds_free is not None and form.get("response", "").lower().startswith("accept"):
            from fallback import rule_based

            try:
                rule_based.report_beds(msg["hospital_id"], service._bed_key(d["payload"]["snapshot"]), beds_free, reporter_id=f"ack:{token[:8]}")
            except Exception:
                pass  # a failed bed report must not lose the confirmation itself
        done = service.record_ack(token, form.get("response", ""), note=form.get("note", ""), eta_min=eta)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _ack_page(msg, done=done)


@app.get("/llm/status")
def llm_status():
    """Provider health, Groq token budgets and each agent's plan: no secrets."""
    from llm import groq_agents

    return groq_agents.status()


@app.get("/config")
def config():
    from services import locationiq

    return {"tiles": locationiq.enabled(), "geocoding": locationiq.enabled()}


@app.get("/tiles/{style}/{z}/{x}/{y}.png")
def map_tile(style: str, z: int, x: int, y: int):
    """Map tile proxy: keeps the LocationIQ key server-side and lets the server cache tiles."""
    from fastapi import Response

    from services import locationiq

    data = locationiq.tile(style, z, x, y)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/route")
def road_route(from_lat: float, from_lng: float, to_lat: float, to_lng: float):
    """Road geometry between two points, for drawing the route on the map."""
    from services import locationiq

    r = locationiq.route_geometry((from_lat, from_lng), (to_lat, to_lng))
    return {"ok": r is not None, **(r or {})}


@app.get("/nearby")
def nearby(lat: float, lng: float):
    """Live hospital search around a point (before any case is run): 8 km first, widening to 25 km if fewer than 3."""
    from fallback import rule_based

    try:
        facilities, radius = rule_based.nearby_raw(lat, lng)
    except Exception as exc:  # resource service down: the UI shows a message instead of failing the page
        return {"ok": False, "error": str(exc)[:120], "radius_km": 8, "hospitals": []}
    return {
        "ok": True, "radius_km": radius,
        "hospitals": [
            {"id": f["hospital_id"], "name": f["name"], "lat": f["lat"], "lng": f["lng"], "distance_km": f["distance_km"],
             "operating_status": f["operating_status"], "ipd_accepting": f["ipd_accepting"], "trauma_level": f["trauma_level"],
             "source": f.get("source", "registry"), "status_verified": f.get("status_verified", True), "phone": f.get("phone"),
             "address": f.get("address"), "osm_url": f.get("osm_url"), "hours": f.get("hours"), "website": f.get("website"),
             "inferred": f.get("inferred") or [], "beds_total": f.get("beds_total")}
            for f in facilities
        ],
    }


@app.get("/reverse")
def reverse_geocode(lat: float, lng: float):
    from services import locationiq

    return {"address": locationiq.reverse(lat, lng)}


@app.get("/geocode")
def geocode(q: str):
    """Address search for the UI (proxied so the LocationIQ key never reaches the browser)."""
    from services import locationiq

    return {"enabled": locationiq.enabled(), "results": locationiq.search(q)}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi import Response

    return Response(status_code=204)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/cases", response_model=CaseResponse)
def create_case(req: CaseRequest):
    return _execute(req)


@app.post("/cases/stream")
def create_case_stream(req: CaseRequest):
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            resp = _execute(req, on_event=q.put)
            q.put({"type": "result", "data": resp.model_dump()})
        except Exception as exc:  # surface unexpected failures to the UI instead of hanging the stream
            q.put({"type": "error", "message": str(exc)})
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def events():
        while (item := q.get()) is not None:
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/metrics")
def metrics():
    with _RUNS_LOCK:
        runs = list(_RUNS)
    return _summarise(runs)


def _summarise(runs: list) -> dict:
    per_stage: dict = {}
    for r in runs:
        for stage, ms in r["stages"].items():
            per_stage.setdefault(stage, []).append(ms)
    stages = {}
    for stage, vals in per_stage.items():
        v = sorted(vals)
        stages[stage] = {
            "count": len(v), "mean_ms": round(sum(v) / len(v), 3), "p50_ms": round(_percentile(v, 0.5), 3),
            "p95_ms": round(_percentile(v, 0.95), 3), "max_ms": round(v[-1], 3),
        }
    totals = sorted(r["total_ms"] for r in runs)
    return {
        "runs": len(runs),
        "total": {
            "mean_ms": round(sum(totals) / len(totals), 3) if totals else 0.0,
            "p50_ms": round(_percentile(totals, 0.5), 3), "p95_ms": round(_percentile(totals, 0.95), 3),
            "max_ms": round(totals[-1], 3) if totals else 0.0,
        },
        "stages": stages,
    }


# ---------------------------------------------------------------- saved history (admin: it contains patient descriptions)
@app.get("/history/status")
def history_status():
    return history.status()


@app.get("/history/runs")
def history_runs(limit: int = 50, x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    return {"runs": history.recent_runs(min(max(limit, 1), 500)), "status": history.status()}


@app.get("/history/runs/{case_id}")
def history_run(case_id: str, x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    r = history.get_run(case_id)
    if r is None:
        raise HTTPException(status_code=404, detail="no saved run with that id")
    return r


@app.get("/history/events")
def history_events(limit: int = 100, case_id: Optional[str] = None, x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    return {"events": history.recent_events(min(max(limit, 1), 1000), case_id=case_id)}


@app.get("/history/analytics")
def history_analytics(days: float = 0, x_admin_token: Optional[str] = Header(None)):
    """Everything the Analytics dashboard shows, computed from the saved runs (days=0: all of them)."""
    require_admin(x_admin_token)
    from services import analytics

    return analytics.analytics(time.time() - days * 86400 if days else 0.0)


@app.get("/history/export/{kind}")
def history_export(kind: str, days: float = 0, x_admin_token: Optional[str] = Header(None)):
    """Research export as CSV: runs (one row per run), stages (one row per stage per run) or events."""
    require_admin(x_admin_token)
    from fastapi.responses import Response
    from services import analytics

    text = analytics.export_csv(kind, time.time() - days * 86400 if days else 0.0)
    if text is None:
        raise HTTPException(status_code=404, detail="export kinds: runs, stages, events")
    return Response(text, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="cedcs-{kind}.csv"'})


@app.get("/analytics", include_in_schema=False)
def analytics_page():
    return FileResponse(Path(__file__).parent / "static" / "analytics.html")


@app.get("/history/latency")
def history_latency(x_admin_token: Optional[str] = Header(None)):
    """Latency report over every saved run (survives restarts), in the same shape as /metrics."""
    require_admin(x_admin_token)
    return _summarise(history.stage_runs(2000))


@app.delete("/metrics")
def reset_metrics():
    with _RUNS_LOCK:
        _RUNS.clear()
    return {"ok": True}


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")  # vendored Leaflet
