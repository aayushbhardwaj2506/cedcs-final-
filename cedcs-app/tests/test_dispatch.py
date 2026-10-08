"""Emergency dispatch: mailer safety, email content, confirmation loop, escalation, admin gating, auto-dispatch."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from dispatch import ambulances, compose, mailer, service, store
from fallback import rule_based
from schemas.hospital import CandidateHospital
from schemas.resource import ResourceRecord, ResourceSnapshot
from tests.test_groq_agents import RAW

NOW = datetime.now(timezone.utc)


def _snap_records(hid):
    r = lambda k, v: ResourceRecord(resource_key=k, value=v, updated_at=NOW - timedelta(minutes=2), source="HOSPITAL_CONSOLE")
    return ResourceSnapshot(hospital_id=hid, records=[
        r("department_EMERGENCY_DEPARTMENT", {"active": True}), r("icu_beds", {"total": 8, "available": 3}),
        r("emergency_beds", {"total": 8, "available": 4}), r("equipment_CARDIAC_MONITOR", {"operational": True}),
        r("ipd_admission_delay_est_min", {"minutes": 20})])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    monkeypatch.delenv("DISPATCH_MODE", raising=False)
    monkeypatch.delenv("DISPATCH_TEST_INBOX", raising=False)
    cands = [CandidateHospital(hospital_id=f"H{i}", name=f"Hospital {i}", eta_min=8 + i * 4, eta_source_tier=4, eta_confidence=0.4,
                               lat=12.9 + i * .01, lng=80.1, distance_km=3 + i) for i in range(3)]
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: (cands, 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cs: [_snap_records(c.hospital_id) for c in cs])
    monkeypatch.setattr(service, "_road_minutes", lambda o, d: None)  # estimates only: no network
    reserved = []
    monkeypatch.setattr(rule_based, "reserve_bed", lambda hid, key, token="": (reserved.append((hid, key)), {"ok": True, "before": 3, "after": 2})[1])
    c = TestClient(api_main.app)
    c.reserved = reserved
    return c


BODY = {"emergency_report": "60 year old male, chest pain, confused", "consciousness": "confused", "breathing": "laboured", "bleeding": "none",
        "location_lat": 12.9249, "location_lng": 80.1, "location_source": "GPS", "location_accuracy_m": 18, "location_address": "Tambaram, Chennai"}


def run_case(client, **over):
    d = client.post("/cases", json={**BODY, **over}).json()
    assert not d["halted"], d["halt_reason"]
    return d


def add_family(client, n=2):
    for i in range(n):
        assert client.post("/contacts", json={"name": f"Family {i}", "email": f"fam{i}@mail.test", "relation": "sister"}).status_code == 200


# ---------------------------------------------------------------- mailer: safety rules
def test_default_mode_sends_nothing(monkeypatch):
    monkeypatch.delenv("DISPATCH_MODE", raising=False)
    monkeypatch.setattr(mailer.smtplib, "SMTP", lambda *a, **k: pytest.fail("SMTP must not be touched in dry-run"))
    r = mailer.send({"to_email": "a@b.test", "subject": "s", "body_text": "t", "body_html": ""})
    assert r["status"] == "LOGGED" and r["actual_to"] == "a@b.test" and mailer.describe()["mode"] == "outbox"


def _smtp_env(monkeypatch, sandbox=None):
    monkeypatch.setenv("DISPATCH_MODE", "smtp")
    for k, v in {"SMTP_HOST": "smtp.test", "SMTP_USER": "u", "SMTP_PASSWORD": "hunter2-secret", "SMTP_FROM": "cedcs@test.example"}.items():
        monkeypatch.setenv(k, v)
    if sandbox:
        monkeypatch.setenv("DISPATCH_TEST_INBOX", sandbox)
    else:
        monkeypatch.delenv("DISPATCH_TEST_INBOX", raising=False)


class FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout=None, context=None):
        pass

    def starttls(self, context=None):
        pass

    def login(self, u, p):
        pass

    def send_message(self, em):
        FakeSMTP.sent.append(em)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_live_mode_without_smtp_config_fails_cleanly(monkeypatch):
    monkeypatch.setenv("DISPATCH_MODE", "smtp")
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"):
        monkeypatch.delenv(k, raising=False)
    r = mailer.send({"to_email": "a@b.test", "subject": "s", "body_text": "t", "body_html": ""})
    assert r["status"] == "FAILED" and "not configured" in r["error"]


def test_synthetic_addresses_are_refused_in_live_mode_without_a_sandbox(monkeypatch):
    _smtp_env(monkeypatch)
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    FakeSMTP.sent.clear()
    r = mailer.send({"to_email": ambulances.hospital_email("H1"), "subject": "s", "body_text": "t", "body_html": ""})
    assert r["status"] == "FAILED" and "synthetic" in r["error"] and not FakeSMTP.sent
    real = mailer.send({"to_email": "sister@gmail.com", "subject": "s", "body_text": "t", "body_html": "<b>t</b>"})
    assert real["status"] == "SENT" and FakeSMTP.sent[-1]["To"] == "sister@gmail.com"


def test_sandbox_redirects_everything_and_says_who_it_was_for(monkeypatch):
    _smtp_env(monkeypatch, sandbox="me@mytest.dev")
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    FakeSMTP.sent.clear()
    for to in (ambulances.hospital_email("H1"), "sister@gmail.com"):
        r = mailer.send({"to_email": to, "subject": "Alert", "body_text": "body", "body_html": ""})
        assert r["status"] == "SENT" and r["actual_to"] == "me@mytest.dev"
        assert f"[SANDBOX for {to}]" in FakeSMTP.sent[-1]["Subject"] and to in FakeSMTP.sent[-1].get_content()
    assert all(m["To"] == "me@mytest.dev" for m in FakeSMTP.sent)


def test_smtp_errors_never_leak_the_password(monkeypatch):
    _smtp_env(monkeypatch, sandbox="me@mytest.dev")

    class Boom(FakeSMTP):
        def login(self, u, p):
            raise RuntimeError(f"auth failed for password {p}")

    monkeypatch.setattr(mailer.smtplib, "SMTP", Boom)
    r = mailer.send({"to_email": "x@y.test", "subject": "s", "body_text": "t", "body_html": ""})
    assert r["status"] == "FAILED" and "hunter2-secret" not in str(r) and "hunter2-secret" not in str(mailer.describe())


# ---------------------------------------------------------------- content
def test_family_email_has_gps_map_link_condition_and_no_diagnosis(client):
    add_family(client, 1)
    run = run_case(client)
    snap = api_main._snapshot_for(run["case_id"])
    m = compose.family_email(snap, store.list_contacts()[0], {"name": "Adyar Ambulance Station", "type": "ALS", "eta_min": 9})
    t = m["body_text"]
    assert "12.92490, 80.10000" in t and "accurate to about 18 m" in t and "device GPS" in t and "Tambaram, Chennai" in t
    assert "openstreetmap.org" in t and "google.com/maps?q=12.92490,80.10000" in t
    assert "CURRENT CONDITION" in t and "confused" in t and "60 years old" in t and "Adyar Ambulance Station" in t
    assert "PROTOTYPE" in t and m["subject"].startswith("[EMERGENCY]")
    for word in ("heart attack", "stroke", "infarction", "sepsis"):
        assert word not in t.lower()


def test_hospital_email_has_confirmation_links_and_needs(client):
    snap = api_main._snapshot_for(run_case(client)["case_id"])
    urls = service.ack_urls("a" * 32)
    m = compose.hospital_email(snap, snap["hospitals"][0], None, urls)
    assert urls["accept"] in m["body_text"] and urls["decline"] in m["body_text"] and "[CEDCS-ACK aaaaaaaa]" in m["subject"]
    assert "Capabilities needed" in m["body_text"] and "PATIENT LOCATION" in m["body_text"]
    assert urls["accept"].split("?")[0].endswith("/ack/" + "a" * 32)


def test_html_escapes_user_supplied_text(client):
    snap = api_main._snapshot_for(run_case(client)["case_id"])
    snap["location"]["address"] = "<script>alert(1)</script>"
    m = compose.family_email(snap, {"name": "<b>x</b>", "email": "a@b.test"}, None)
    assert "<script>" not in m["body_html"] and "&lt;script&gt;" in m["body_html"] and "<b>x</b>" not in m["body_html"]


# ---------------------------------------------------------------- ambulances
def test_nearest_available_ambulance_and_road_refinement():
    lat, lng = 13.0, 80.2
    unit = ambulances.nearest(lat, lng)
    assert unit["available"] and unit["method"] == "estimate" and unit["eta_min"] > 0
    closest = min((a for a in ambulances.AMBULANCES if a["available"]), key=lambda a: ambulances.haversine_km(lat, lng, a["lat"], a["lng"]))
    assert unit["id"] == closest["id"]
    trio = sorted((a for a in ambulances.AMBULANCES if a["available"]), key=lambda a: ambulances.haversine_km(lat, lng, a["lat"], a["lng"]))[:3]
    routed = ambulances.nearest(lat, lng, road_minutes=lambda o, d: [30.0, 4.0, 20.0])  # the 2nd-closest is fastest by road
    assert routed["id"] == trio[1]["id"] and routed["method"] == "road" and routed["eta_min"] == 4.0
    assert all(a["email"].endswith("example.org") for a in ambulances.AMBULANCES)  # cannot reach a real mailbox


# ---------------------------------------------------------------- dispatch flow (dry run)
def test_dispatch_requires_explicit_confirmation(client):
    run = run_case(client)
    r = client.post("/dispatch", json={"case_id": run["case_id"], "confirm": False})
    assert r.status_code == 400 and "confirm" in r.json()["detail"] and store.list_outbox() == []
    with pytest.raises(PermissionError):
        service.send_dispatch(api_main._snapshot_for(run["case_id"]), confirmed=False)


def test_dispatch_notifies_family_hospital_and_ambulance_and_logs_in_the_outbox(client):
    add_family(client, 2)
    run = run_case(client)
    prev = client.post("/dispatch/preview", json={"case_id": run["case_id"]}).json()
    assert [m["kind"] for m in prev["messages"]] == ["FAMILY", "FAMILY", "HOSPITAL", "AMBULANCE"] and store.list_outbox() == []  # preview sends nothing
    d = client.post("/dispatch", json={"case_id": run["case_id"], "confirm": True}).json()
    kinds = [m["kind"] for m in d["messages"]]
    assert kinds.count("FAMILY") == 2 and kinds.count("HOSPITAL") == 1 and kinds.count("AMBULANCE") == 1
    assert all(m["status"] == "LOGGED" for m in d["messages"]) and d["mode"] == "outbox" and d["status"] == "AWAITING_CONFIRMATION"
    hosp = next(m for m in d["messages"] if m["kind"] == "HOSPITAL")
    assert hosp["hospital_id"] == run["recommendation"]["primary"]["hospital_id"]  # the RECOMMENDED hospital is the one told
    assert hosp["ack_urls"]["accept"].endswith("?response=accept")
    assert len(client.get("/outbox").json()["messages"]) == 4


def test_a_case_is_dispatched_only_once(client):
    add_family(client, 1)
    run = run_case(client)
    a = client.post("/dispatch", json={"case_id": run["case_id"], "confirm": True}).json()
    b = client.post("/dispatch", json={"case_id": run["case_id"], "confirm": True}).json()
    assert a["id"] == b["id"] and len(store.list_outbox()) == 3  # no duplicate emails to anyone


def test_dispatch_for_an_unknown_case_is_404(client):
    assert client.post("/dispatch", json={"case_id": "nope", "confirm": True}).status_code == 404


def test_family_update_email_carries_the_note(client):
    add_family(client, 2)
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    d2 = client.post(f"/dispatch/{d['id']}/update", json={"note": "Now conscious and talking."}).json()
    updates = [m for m in d2["messages"] if m["subject"].startswith("[UPDATE]")]
    assert len(updates) == 2 and all("Now conscious and talking." in store.db().execute("SELECT body_text FROM messages WHERE id=?", (m["id"],)).fetchone()[0] for m in updates)
    assert client.post(f"/dispatch/{d['id']}/update", json={"note": "  "}).status_code == 422


# ---------------------------------------------------------------- confirmation loop
def _hosp_msg(d):
    return next(m for m in d["messages"] if m["kind"] == "HOSPITAL" and not m["acks"])


def test_ack_page_get_changes_nothing_but_post_records_and_reserves_a_bed(client):
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    tok = _hosp_msg(d)["token"]
    page = client.get(f"/ack/{tok}?response=accept")
    assert page.status_code == 200 and "Incoming emergency patient" in page.text and "checked" in page.text
    assert client.reserved == [] and client.get(f"/dispatch/{d['id']}").json()["status"] == "AWAITING_CONFIRMATION"  # a link prefetch is harmless
    r = client.post(f"/ack/{tok}", data={"response": "accept", "eta_min": "7", "note": "ICU ready"})
    assert r.status_code == 200 and "recorded as ACCEPTED" in r.text and "Bed reserved" in r.text
    assert client.reserved == [(_hosp_msg(d)["hospital_id"], "icu_beds")]  # ICU needed -> an ICU bed is reserved
    after = client.get(f"/dispatch/{d['id']}").json()
    ack = next(m for m in after["messages"] if m["kind"] == "HOSPITAL")["ack"]
    assert after["status"] == "CONFIRMED" and ack["response"] == "ACCEPTED" and ack["eta_min"] == 7 and ack["note"] == "ICU ready" and ack["source"] == "LINK"


def test_clicking_accept_twice_reserves_only_one_bed(client):
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    tok = _hosp_msg(d)["token"]
    client.post(f"/ack/{tok}", data={"response": "accept"})
    client.post(f"/ack/{tok}", data={"response": "accept"})
    assert len(client.reserved) == 1


def test_decline_escalates_to_the_next_ranked_hospital(client):
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    first = _hosp_msg(d)
    client.post(f"/ack/{first['token']}", data={"response": "decline", "note": "ICU full"})
    after = client.get(f"/dispatch/{d['id']}").json()
    hosp = [m for m in after["messages"] if m["kind"] == "HOSPITAL"]
    assert len(hosp) == 2 and hosp[0]["ack"]["response"] == "DECLINED" and hosp[1]["hospital_id"] != hosp[0]["hospital_id"]
    assert "BACKUP" in hosp[1]["subject"] and after["status"] == "AWAITING_CONFIRMATION"
    client.post(f"/ack/{hosp[1]['token']}", data={"response": "accept"})
    assert client.get(f"/dispatch/{d['id']}").json()["status"] == "CONFIRMED"


def test_every_hospital_declining_ends_in_no_more_hospitals(client):
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    for _ in range(6):
        cur = client.get(f"/dispatch/{d['id']}").json()
        pending = [m for m in cur["messages"] if m["kind"] == "HOSPITAL" and not m["acks"]]
        if not pending:
            break
        client.post(f"/ack/{pending[0]['token']}", data={"response": "decline"})
    final = client.get(f"/dispatch/{d['id']}").json()
    assert final["status"] == "NO_MORE_HOSPITALS" and len([m for m in final["messages"] if m["kind"] == "HOSPITAL"]) == 3


def test_auto_escalation_can_be_turned_off(client):
    store.put_settings({"auto_escalate": False})
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    client.post(f"/ack/{_hosp_msg(d)['token']}", data={"response": "decline"})
    after = client.get(f"/dispatch/{d['id']}").json()
    assert len([m for m in after["messages"] if m["kind"] == "HOSPITAL"]) == 1 and after["status"] == "AWAITING_NEXT"
    assert client.post(f"/dispatch/{d['id']}/escalate").json()["notified"]["id"]  # a person can still escalate by hand


def test_silence_past_the_timeout_escalates_exactly_once(client, monkeypatch):
    store.put_settings({"ack_timeout_s": 60})
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    early = client.get(f"/dispatch/{d['id']}").json()
    assert early["overdue"] == [] and len([m for m in early["messages"] if m["kind"] == "HOSPITAL"]) == 1
    real = time.time
    monkeypatch.setattr(service.time, "time", lambda: real() + 90)  # 90 s later, nobody answered
    late = client.get(f"/dispatch/{d['id']}").json()
    assert len(late["overdue"]) == 1 and len([m for m in late["messages"] if m["kind"] == "HOSPITAL"]) == 2
    again = client.get(f"/dispatch/{d['id']}").json()
    assert len([m for m in again["messages"] if m["kind"] == "HOSPITAL"]) == 2  # not re-escalated on every poll


def test_simulated_replies_only_exist_in_dry_run(client, monkeypatch):
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    msg = _hosp_msg(d)
    ok = client.post(f"/dispatch/{d['id']}/simulate-ack", json={"message_id": msg["id"], "response": "accept"})
    assert ok.status_code == 200 and ok.json()["dispatch"]["status"] == "CONFIRMED"
    assert next(m for m in ok.json()["dispatch"]["messages"] if m["kind"] == "HOSPITAL")["ack"]["source"] == "SIMULATED"
    monkeypatch.setenv("DISPATCH_MODE", "smtp")
    assert client.post(f"/dispatch/{d['id']}/simulate-ack", json={"message_id": msg["id"], "response": "accept"}).status_code == 403


def test_unknown_or_kind_wrong_tokens_are_rejected(client):
    add_family(client, 1)
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    fam = next(m for m in d["messages"] if m["kind"] == "FAMILY")
    assert client.get("/ack/doesnotexist").status_code == 404
    assert client.post(f"/ack/{fam['token']}", data={"response": "accept"}).status_code == 404  # family links cannot confirm a bed
    assert client.post(f"/ack/{_hosp_msg(d)['token']}", data={"response": "maybe"}).status_code == 422


# ---------------------------------------------------------------- e-mail reply loop (IMAP)
def test_imap_reply_becomes_an_acknowledgement(client, monkeypatch):
    d = client.post("/dispatch", json={"case_id": run_case(client)["case_id"], "confirm": True}).json()
    msg = _hosp_msg(d)
    for k, v in {"IMAP_HOST": "imap.test", "IMAP_USER": "u", "IMAP_PASSWORD": "p"}.items():
        monkeypatch.setenv(k, v)
    raw = (f"Subject: Re: [CEDCS-ACK {msg['token'][:8]}] Incoming\r\nFrom: er@hospital.test\r\nContent-Type: text/plain\r\n\r\n"
           "ACCEPT - ICU bed ready, send them straight in.\r\n").encode()

    class FakeIMAP:
        marked = []

        def __init__(self, *a, **k): pass
        def login(self, u, p): pass
        def select(self, box): return "OK", [b"1"]
        def search(self, *a): return "OK", [b"1"]
        def fetch(self, num, spec): return "OK", [(b"1", raw)]
        def store(self, num, op, flag): FakeIMAP.marked.append(flag)
        def logout(self): pass

    monkeypatch.setattr(service.imaplib, "IMAP4_SSL", FakeIMAP)
    assert service.poll_imap() == 1 and FakeIMAP.marked == ["\\Seen"]
    ack = next(m for m in client.get(f"/dispatch/{d['id']}").json()["messages"] if m["kind"] == "HOSPITAL")["ack"]
    assert ack["response"] == "ACCEPTED" and ack["source"] == "EMAIL_REPLY" and "ICU bed ready" in ack["note"]


def test_imap_is_a_noop_without_configuration(monkeypatch):
    for k in ("IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    assert service.poll_imap() == 0


# ---------------------------------------------------------------- settings, contacts, auto-dispatch
def test_only_an_admin_can_change_settings(client):
    assert client.put("/settings", json={"bed_checker_enabled": True}).status_code == 403
    assert client.put("/settings", json={"bed_checker_enabled": True}, headers={"X-Admin-Token": "wrong"}).status_code == 403
    r = client.put("/settings", json={"bed_checker_enabled": True, "auto_dispatch": True}, headers={"X-Admin-Token": api_main.ADMIN_TOKEN})
    assert r.status_code == 200 and r.json()["bed_checker_enabled"] is True
    pub = client.get("/settings").json()
    assert pub["bed_checker_enabled"] and "token" not in str(pub).lower() and pub["mail"]["mode"] == "outbox"


def test_contact_validation(client):
    assert client.post("/contacts", json={"name": "", "email": "a@b.co"}).status_code == 422
    assert client.post("/contacts", json={"name": "A", "email": "not-an-email"}).status_code == 422
    c = client.post("/contacts", json={"name": "A", "email": "a@b.co"}).json()
    assert client.get("/contacts").json()["contacts"][0]["id"] == c["id"]
    assert client.delete(f"/contacts/{c['id']}").json()["ok"] and client.get("/contacts").json()["contacts"] == []


def test_auto_dispatch_is_off_by_default_and_only_fires_for_urgent_cases_when_enabled(client):
    add_family(client, 1)
    quiet = run_case(client)
    assert "auto_dispatch" not in quiet["trace"] and store.list_outbox() == []  # default: nothing goes out by itself
    client.put("/settings", json={"auto_dispatch": True}, headers={"X-Admin-Token": api_main.ADMIN_TOKEN})
    urgent = run_case(client, emergency_report="man not breathing after collapse", breathing="absent")
    assert urgent["trace"]["auto_dispatch"]["messages"] == 3 and len(store.list_outbox()) == 3
    assert any(m["to"] == "DISPATCH" and m["kind"] == "command" for m in urgent["messages"])
    assert any(m["from"] == "DISPATCH" and m["kind"] == "result" for m in urgent["messages"])
    before = len(store.list_outbox())
    mild = run_case(client, emergency_report="mild fever", consciousness="alert", breathing="normal")
    assert mild["trace"]["red_flags"]["final_priority"] in ("LOW", "MODERATE") and "auto_dispatch" not in mild["trace"] and len(store.list_outbox()) == before


def test_bed_check_flag_flows_from_the_request_into_the_pipeline(client, monkeypatch):
    calls = []
    monkeypatch.setattr(rule_based, "scan_beds", lambda ids: (calls.append(ids), {"snapshots": [_snap_records(i) for i in ids], "scanned_at": NOW.isoformat(), "took_ms": 5})[1])
    off = run_case(client)
    assert calls == [] and off["trace"]["bed_check"]["active"] is False
    on = run_case(client, bed_check=True)
    assert len(calls) == 1 and on["trace"]["bed_check"] == {"active": True, "by": "user"} and "BEDS" in on["trace"]["ranking"]["weights"]


# ------------------------------------------------------------------ hospital portal
ADMIN_H = {"X-Admin-Token": api_main.ADMIN_TOKEN}


def _dispatch(client):
    add_family(client, 1)
    case = run_case(client)
    d = client.post("/dispatch", json={"case_id": case["case_id"], "confirm": True}).json()
    hid = next(m["hospital_id"] for m in d["messages"] if m["kind"] == "HOSPITAL")
    return d, hid


def test_portal_link_needs_admin_and_only_for_hospitals_in_the_dispatch(client):
    d, hid = _dispatch(client)
    assert client.post(f"/dispatch/{d['id']}/portal", params={"hospital_id": hid}).status_code == 403
    assert client.post(f"/dispatch/{d['id']}/portal", params={"hospital_id": "H999"}, headers=ADMIN_H).status_code == 404
    a = client.post(f"/dispatch/{d['id']}/portal", params={"hospital_id": hid}, headers=ADMIN_H).json()
    b = client.post(f"/dispatch/{d['id']}/portal", params={"hospital_id": hid}, headers=ADMIN_H).json()
    assert a["url"] == b["url"] and "/portal/" in a["url"]  # reused, not duplicated


def test_portal_shows_incoming_patient_and_answer_reaches_the_operator(client):
    d, hid = _dispatch(client)
    token = client.post(f"/dispatch/{d['id']}/portal", params={"hospital_id": hid}, headers=ADMIN_H).json()["url"].rsplit("/", 1)[1]
    assert client.get(f"/portal/{token}").status_code == 200 and client.get("/portal/nope/data").status_code == 404
    data = client.get(f"/portal/{token}/data").json()
    assert data["hospital"]["id"] == hid and len(data["incoming"]) == 1
    p = data["incoming"][0]
    assert p["priority"] and p["condition"] and p["answer"] is None
    assert "fam0@mail.test" not in str(data)  # the family's contact details never reach a hospital
    r = client.post(f"/portal/{token}/respond", json={"message_id": p["message_id"], "response": "accept", "note": "ready", "eta_min": 5})
    assert r.status_code == 200 and r.json()["dispatch_status"] == "CONFIRMED" and client.reserved
    assert client.get(f"/portal/{token}/data").json()["incoming"][0]["answer"]["source"] == "PORTAL"
    assert client.get(f"/dispatch/{d['id']}").json()["status"] == "CONFIRMED"  # the operator's view updates


def test_portal_cannot_answer_another_hospitals_alert(client):
    d, hid = _dispatch(client)
    other = store.create_console("H2" if hid != "H2" else "H1", "Other")["token"]
    msg = next(m for m in d["messages"] if m["kind"] == "HOSPITAL")
    assert client.post(f"/portal/{other}/respond", json={"message_id": msg["id"], "response": "accept"}).status_code == 404


# ------------------------------------------------------------------ deployment: access gate and Postgres translation
def test_access_gate_protects_operator_pages_but_not_token_links(client, monkeypatch):
    import base64

    monkeypatch.setenv("ACCESS_PASSWORD", "s3cret")
    assert client.get("/").status_code == 401 and client.get("/analytics").status_code == 401
    assert client.get("/health").status_code == 200  # health check and hospital/mail links stay open
    assert client.get("/portal/nope/data").status_code == 404 and client.get("/ack/nope").status_code == 404
    bad = {"Authorization": "Basic " + base64.b64encode(b"u:wrong").decode()}
    good = {"Authorization": "Basic " + base64.b64encode(b"anyone:s3cret").decode()}
    assert client.get("/", headers=bad).status_code == 401 and client.get("/", headers=good).status_code == 200
    monkeypatch.delenv("ACCESS_PASSWORD")
    assert client.get("/").status_code == 200  # no password configured: open (local use)


def test_pg_translate_makes_sqlite_sql_valid_for_postgres():
    q = store.pg_translate("SELECT * FROM messages WHERE dispatch_id=? ORDER BY created_at, rowid", "cedcs_app")
    assert q == "SELECT * FROM cedcs_app.messages WHERE dispatch_id=%s ORDER BY created_at"
    assert "DOUBLE PRECISION" in store.pg_translate("CREATE TABLE IF NOT EXISTS acks(eta_min REAL)", "s")
    assert "s.hospital_reports" in store.pg_translate("INSERT INTO hospital_reports VALUES(?,?)", "s")
    assert store.pg_translate("SELECT '100%' AS x FROM settings", "s") == "SELECT '100%%' AS x FROM s.settings"
