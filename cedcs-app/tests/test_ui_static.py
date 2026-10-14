"""Guards for mistakes that only show up in the browser."""

from __future__ import annotations

import re
from pathlib import Path

HTML = (Path(__file__).resolve().parent.parent / "api" / "static" / "index.html").read_text(encoding="utf-8")


def test_every_seg_toggle_group_has_a_data_key():
    """The page's start-up code (drawSeg) rebuilds EVERY element with class "seg" from its data-key. One without it throws, and
    everything after it, including the live map, never starts. Other tab bars must use their own class."""
    tags = re.findall(r'<[^>]*class="seg"[^>]*>', HTML)
    assert tags, "expected the case form's toggle groups"
    assert all("data-key=" in t for t in tags), [t for t in tags if "data-key=" not in t]


def test_the_scene_analysis_card_does_not_use_the_form_toggle_class():
    card = re.search(r'<div class="card" id="visCard">.*?id="visOut"', HTML, re.S).group(0)
    assert 'class="seg"' not in card and 'id="visTabs"' in card


def test_pages_and_scripts_are_revalidated_so_browsers_never_show_an_old_copy():
    from fastapi.testclient import TestClient

    import api.main as api_main

    c = TestClient(api_main.app)
    for path in ("/", "/analytics", "/static/vision.js"):
        r = c.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", path
    assert "no-cache" not in c.get("/static/vendor/leaflet/leaflet.js").headers.get("cache-control", "")
