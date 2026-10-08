# CEDCS: emergency hospital routing

AI-assisted, safety-bounded routing of emergency patients to the right hospital. The AI agents only advise; a deterministic core
decides. See [`cedcs-app/README.md`](cedcs-app/README.md) for the architecture, setup, the live map, dispatch, the hospital portal and
the analytics dashboard.

Hospitals are discovered live from OpenStreetMap. Their beds and equipment are unknown unless a hospital reports them, and the
ambulance directory and e-mail alerts are simulated. This is a research prototype and decision support, not a medical device.

Deploy: `render.yaml` is a Render Blueprint (free web service). Secrets are never in the repo; see `cedcs-app/.env.example`.
