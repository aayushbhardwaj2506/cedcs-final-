"""Tile proxy, route geometry and config endpoints, with the LocationIQ HTTP calls mocked."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from services import locationiq


class Resp:
    def __init__(self, content=b"PNG", status=200, ctype="image/png", js=None):
        self.content, self.status_code, self.headers, self._js = content, status, {"content-type": ctype}, js

    def json(self):
        return self._js


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LOCATIONIQ_KEY", "test-key")
    monkeypatch.setattr(locationiq.time, "sleep", lambda s: None)
    locationiq._TILE_CACHE.clear()
    locationiq._ROUTE_GEOM_CACHE.clear()
    return TestClient(api_main.app)


def test_config_reports_whether_tiles_are_available(client, monkeypatch):
    assert client.get("/config").json()["tiles"] is True
    monkeypatch.setenv("LOCATIONIQ_KEY", "")
    assert client.get("/config").json()["tiles"] is False


def test_tile_proxy_serves_and_caches_and_never_leaks_the_key(client, monkeypatch):
    calls = []
    monkeypatch.setattr(locationiq._session, "get", lambda url, params=None, timeout=None: (calls.append((url, params)), Resp())[1])
    r = client.get("/tiles/dark/12/2926/1898.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and "max-age" in r.headers["cache-control"]
    assert "test-key" not in r.text and "test-key" not in str(r.headers)
    client.get("/tiles/dark/12/2926/1898.png")
    assert len(calls) == 1  # second request served from the cache
    assert calls[0][0].startswith("https://") and "/v3/dark/r/12/2926/1898.png" in calls[0][0]


@pytest.mark.parametrize("path", ["/tiles/satellite/1/1/1.png", "/tiles/streets/25/1/1.png"])
def test_tile_proxy_rejects_unknown_style_or_zoom(client, path):
    assert client.get(path).status_code == 404


def test_tile_proxy_404s_when_provider_fails(client, monkeypatch):
    monkeypatch.setattr(locationiq._session, "get", lambda *a, **k: Resp(status=500))
    assert client.get("/tiles/streets/10/1/1.png").status_code == 404


def test_route_geometry_converts_lnglat_to_latlng_and_caches(client, monkeypatch):
    calls = []
    js = {"code": "Ok", "routes": [{"distance": 29040.0, "duration": 1482.0, "geometry": {"coordinates": [[80.1, 12.9], [80.2, 13.0]]}}]}
    monkeypatch.setattr(locationiq._session, "get", lambda url, params=None, timeout=None: (calls.append(url), Resp(js=js, ctype="application/json"))[1])
    d = client.get("/route", params={"from_lat": 12.9, "from_lng": 80.1, "to_lat": 13.0, "to_lng": 80.2}).json()
    assert d == {"ok": True, "coords": [[12.9, 80.1], [13.0, 80.2]], "km": 29.0, "minutes": 24.7}
    assert "80.1,12.9;80.2,13.0" in calls[0]  # provider wants lng,lat
    client.get("/route", params={"from_lat": 12.9, "from_lng": 80.1, "to_lat": 13.0, "to_lng": 80.2})
    assert len(calls) == 1


def test_route_endpoint_degrades_gracefully(client, monkeypatch):
    monkeypatch.setattr(locationiq._session, "get", lambda *a, **k: Resp(js={"code": "NoRoute"}, ctype="application/json"))
    assert client.get("/route", params={"from_lat": 1, "from_lng": 2, "to_lat": 3, "to_lng": 4}).json() == {"ok": False}


def test_leaflet_is_vendored_not_loaded_from_a_cdn(client):
    assert client.get("/static/vendor/leaflet/leaflet.js").status_code == 200
    assert client.get("/static/vendor/leaflet/leaflet.css").status_code == 200
