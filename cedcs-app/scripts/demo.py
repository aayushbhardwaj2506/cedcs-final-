"""POST /cases through the real API, no test doubles. Needs resource_service on :8000.
Uses the rule-based path automatically when crewai/LLM key are absent (CEDCS_MODE=offline forces it)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from api.main import app

def run(body):
    d = TestClient(app).post("/cases", json=body).json()
    r = d["recommendation"]
    print("halted:", d["halted"], d["halt_reason"] or "")
    if r:
        print(f"primary: {r['primary']['name']} (ETA {r['primary']['eta_min']} min)  confidence: {r['confidence_level']} {r['confidence']}  escalated: {r['escalated']} {r['escalation_reason'] or ''}")
        print("alternatives:", [a["name"] for a in r["alternatives"]])
    return d

if __name__ == "__main__":
    base = {"location_lat": 12.9249, "location_lng": 80.1, "location_address": "Tambaram, Chennai"}
    print("--- worked example"); d = run({**base, "emergency_report": "60 year old male, sudden collapse, confused, chest pain, diabetic", "consciousness": "confused", "breathing": "laboured", "bleeding": "none"})
    print(d["explanation_text"])
    print("\n--- absent breathing (red flag)"); run({**base, "emergency_report": "man not breathing after collapse", "breathing": "absent", "consciousness": "unresponsive"})
    print("\n--- road accident"); run({**base, "emergency_report": "road accident, 30 year old woman, fracture, bleeding heavily", "bleeding": "severe", "consciousness": "alert", "breathing": "normal"})
    print("\n--- no location coverage"); run({"location_lat": 28.6, "location_lng": 77.2, "emergency_report": "chest pain"})
