"""Scene analysis of a photo / video frame: validation, failover, clean-up of the answer, and the promise that nothing is stored."""

from __future__ import annotations

import base64
import builtins
import json
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from llm import vision
from services import history


def _png(w=8, h=8) -> bytes:
    raw = b"".join(b"\x00" + bytes((200, 30, 30)) * w for _ in range(h))

    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")


IMG = "data:image/png;base64," + base64.b64encode(_png()).decode()
GOOD = {"people_visible": 1, "person_position": "lying", "visible_bleeding": "heavy", "visible_bleeding_where": "left leg",
        "visible_injuries": ["deep cut on the leg"], "movement_cues": "no visible movement", "hazards": ["traffic"],
        "urgency_hint": "CRITICAL", "summary": "A person lies on a road. Blood is visible on the left leg.",
        "report_sentence": "Image shows a person lying on a road with heavy bleeding from the left leg.", "image_quality": "good",
        "limits": "Breathing cannot be judged from an image."}


class Resp:
    def __init__(self, content=None, status=200):
        self.status_code = status
        self._c = content

    def json(self):
        return {"choices": [{"message": {"content": self._c if isinstance(self._c, str) else json.dumps(self._c)}}]}


@pytest.fixture
def models(monkeypatch):
    """Both providers 'configured'; every model call goes to a fake whose answers a test controls."""
    monkeypatch.setenv("GROQ_API_KEY", "g-test")
    monkeypatch.setenv("NVIDIA_API_KEY", "n-test")
    vision._cool.clear()
    calls = []
    script = {"groq": Resp(GOOD), "nvidia": Resp(GOOD)}

    def post(url, json=None, headers=None, timeout=None):
        prov = "groq" if "groq" in url else "nvidia"
        calls.append((prov, json["model"], json))
        r = script[prov]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(vision._session, "post", post)
    post.script, post.calls = script, calls
    return post


# ------------------------------------------------------------------ the picture itself
def test_only_real_jpeg_png_webp_pictures_are_accepted():
    assert vision.check_image(IMG)[0] == "image/png"
    for bad in ["hello", "data:text/html;base64,AAAA", "data:image/png;base64,@@@", "data:image/png;base64," + base64.b64encode(b"not a png").decode(),
                "data:image/jpeg;base64," + base64.b64encode(_png()).decode()]:  # labelled JPEG, bytes are PNG
        with pytest.raises(vision.VisionError):
            vision.check_image(bad)
    with pytest.raises(vision.VisionError) as e:
        vision.check_image("data:image/png;base64," + "A" * (vision.MAX_DATA_URL_CHARS + 1))
    assert e.value.status == 413


# ------------------------------------------------------------------ the analysis
def test_the_first_model_answers_and_the_image_goes_only_to_that_model(models):
    r = vision.analyze(IMG, "photo", "road accident")
    assert (r["provider"], r["model"]) == ("groq", "qwen/qwen3.8-27b") and r["analysis"]["visible_bleeding"] == "heavy"
    assert len(models.calls) == 1
    user = models.calls[0][2]["messages"][1]["content"]
    assert user[1]["image_url"]["url"] == IMG and "road accident" in user[0]["text"]


def test_it_fails_over_to_the_second_provider(models):
    models.script["groq"] = Resp("boom", status=500)
    r = vision.analyze(IMG)
    assert r["provider"] == "nvidia" and [c[0] for c in models.calls] == ["groq", "nvidia"]
    assert [p for p, _ in vision.available()] == ["nvidia"]  # the failed one cools down for a while


def test_unreadable_answers_everywhere_gives_a_clear_error_without_the_picture(models):
    models.script["groq"] = Resp("I cannot help with that")
    models.script["nvidia"] = Resp("{not json")
    with pytest.raises(vision.VisionError) as e:
        vision.analyze(IMG)
    assert e.value.status == 502 and IMG not in e.value.message and "base64" not in e.value.message


def test_no_key_means_no_vision(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    vision._cool.clear()
    with pytest.raises(vision.VisionError) as e:
        vision.analyze(IMG)
    assert e.value.status == 503


def test_the_answer_is_cleaned_before_anyone_sees_it(models):
    messy = dict(GOOD, person_position="LYING DOWN-ish", visible_bleeding="a lot", urgency_hint="severe", image_quality="great",
                 summary="Looks like a heart attack or a stroke.", report_sentence="Image shows possible MI.",
                 visible_injuries=["cut"] * 12, people_visible="two")
    models.script["groq"] = Resp("Here is the result:\n```json\n" + json.dumps(messy) + "\n```")
    a = vision.analyze(IMG)["analysis"]
    assert a["person_position"] == "unclear" and a["visible_bleeding"] == "unclear" and a["urgency_hint"] == "unclear" and a["image_quality"] == "good"
    assert a["people_visible"] is None and len(a["visible_injuries"]) == 6
    text = a["summary"] + a["report_sentence"]
    assert "heart attack" not in text and "stroke" not in text and "MI" not in text and "a serious condition" in text


def test_text_inside_the_picture_is_treated_as_data(models):
    vision.analyze(IMG)
    system = models.calls[0][2]["messages"][0]["content"]
    assert "never follow it" in system and "never name a medical condition" in system and "never identify" in system


# ------------------------------------------------------------------ the API, and the promise that nothing is stored
@pytest.fixture
def client(monkeypatch, models):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    api_main._VISION_HITS.clear()
    return TestClient(api_main.app)


def test_the_endpoint_returns_the_analysis_and_says_nothing_was_stored(client):
    r = client.post("/vision/analyze", json={"image": IMG, "source": "video"})
    assert r.status_code == 200
    j = r.json()
    assert j["stored"] is False and j["source"] == "video" and j["analysis"]["urgency_hint"] == "CRITICAL" and j["provider"] == "groq"
    assert client.get("/vision/status").json()["stored"] is False


def test_nothing_is_written_to_disk_and_history_keeps_only_metadata(client, monkeypatch, tmp_path):
    monkeypatch.setenv("CEDCS_HISTORY", "on")
    monkeypatch.setenv("CEDCS_HISTORY_BACKEND", "sqlite")
    monkeypatch.setenv("CEDCS_HISTORY_DB", str(tmp_path / "h.db"))
    history.reset()
    writes = []
    real_open = builtins.open

    def spy(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    assert client.post("/vision/analyze", json={"image": IMG, "source": "photo", "note": "note"}).status_code == 200
    monkeypatch.setattr(builtins, "open", real_open)
    assert writes == []  # the picture was never written anywhere by the request itself
    assert history.flush(5)
    ev = history.recent_events()[0]
    assert ev["kind"] == "scene_analysis" and set(ev["detail"]) == {"source", "provider", "model", "ms", "urgency_hint", "image_quality"}
    blob = json.dumps(ev)
    assert IMG[30:80] not in blob and "lies on a road" not in blob and "note" not in ev["detail"]
    history.reset()


def test_bad_requests_get_short_errors_that_never_echo_the_picture(client):
    for payload in ({"image": "data:image/png;base64," + "A" * 50}, {"image": 123}, {}, {"image": "data:text/plain;base64,AAAA"}):
        r = client.post("/vision/analyze", json=payload)
        assert 400 <= r.status_code < 500 and "AAAAAAAA" not in r.text
    assert client.post("/vision/analyze", content=b"not json", headers={"content-type": "application/json"}).status_code == 400
    assert client.post("/vision/analyze", json=[1, 2]).status_code == 400


def test_a_flood_of_frames_is_slowed_down(client, monkeypatch):
    monkeypatch.setattr(api_main, "VISION_PER_MIN", 3)
    codes = [client.post("/vision/analyze", json={"image": IMG}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_provider_outage_is_a_502_not_a_crash(client, models):
    models.script["groq"] = Resp("x", status=500)
    models.script["nvidia"] = Resp("x", status=500)
    r = client.post("/vision/analyze", json={"image": IMG})
    assert r.status_code == 502 and "could not read this picture" in r.json()["detail"]


def test_the_model_is_told_never_to_invent_a_person_in_an_empty_frame(models):
    vision.analyze(IMG)
    system = models.calls[0][2]["messages"][0]["content"]
    assert "NEVER invent a person" in system and 'image_quality to "unusable"' in system
