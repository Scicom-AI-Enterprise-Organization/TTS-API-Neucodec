# Vendored

`bettersentryio.py` — the heartbeat + error-capture client, copied verbatim from
`Scicom-AI-Enterprise-Organization/bettersentryio` at `clients/python/bettersentryio.py`.

Vendored rather than installed because it is a single stdlib-only file with nothing to
`pip install`, and the GPU nodes should not need network access at start to fetch it.

To update, re-copy it from the engine you report to:

    curl -o app/_vendor/bettersentryio.py \
      https://bsio-ingest.aies.scicom.dev/clients/python/bettersentryio.py
