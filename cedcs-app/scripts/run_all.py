"""One command to run everything locally:  python scripts/run_all.py

Reads cedcs-app/.env. If DATABASE_URL is set (e.g. Neon) it is used as-is; otherwise a local
Postgres+PostGIS container is started with Docker and seeded. Then starts the resource service
(:8000) and the API + UI (:8001). Ctrl+C stops both servers.
"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
env = {**os.environ, "RESOURCE_SERVICE_URL": os.environ.get("RESOURCE_SERVICE_URL", "http://127.0.0.1:8000")}


def sh(*a, **kw):
    return subprocess.run(a, cwd=ROOT, env=env, capture_output=True, text=True, **kw)


if not os.environ.get("DATABASE_URL"):
    env["DATABASE_URL"] = "postgresql://postgres:cedcs@localhost:5432/postgres"
    print("DATABASE_URL not set: using a local Docker Postgres")
    if "cedcs-db" not in sh("docker", "ps", "-a", "--format", "{{.Names}}").stdout:
        sh("docker", "run", "-d", "--name", "cedcs-db", "-e", "POSTGRES_PASSWORD=cedcs", "-p", "5432:5432", "postgis/postgis:16-3.4")
    else:
        sh("docker", "start", "cedcs-db")
    for _ in range(30):
        if sh("docker", "exec", "cedcs-db", "pg_isready", "-U", "postgres").returncode == 0:
            break
        time.sleep(2)
    time.sleep(2)
    have = sh("docker", "exec", "cedcs-db", "psql", "-U", "postgres", "-tAc", "select count(*) from hospitals").stdout.strip()
    if not have.isdigit() or int(have) == 0:
        subprocess.run(["docker", "exec", "-i", "cedcs-db", "psql", "-U", "postgres", "-q"],
                       input=(ROOT / "resource_service" / "schema.sql").read_text(), text=True, cwd=ROOT)
        subprocess.run([sys.executable, "resource_service/seed.py"], cwd=ROOT, env=env)
else:
    print("Using DATABASE_URL from .env (remote database; first cache load can take ~30 s)")

print("LLM:", " + ".join(n for n, k in (("Groq", "GROQ_API_KEY"), ("NVIDIA", "NVIDIA_API_KEY")) if env.get(k)) or "off (rule-based)", "| routing:", "LocationIQ" if env.get("LOCATIONIQ_KEY") else "off (estimates)")
procs = [
    subprocess.Popen([sys.executable, "-m", "uvicorn", "resource_service.main:app", "--port", "8000"], cwd=ROOT, env=env),
    subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8001"], cwd=ROOT, env=env),
]
print("\nUI:  http://127.0.0.1:8001    API docs: http://127.0.0.1:8001/docs    (Ctrl+C to stop)\n")
try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    for p in procs:
        p.terminate()
