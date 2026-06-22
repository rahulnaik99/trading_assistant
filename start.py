#!/usr/bin/env python3
"""Start Trade Assistant — all services."""

import subprocess, sys, time, os, signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV  = {**os.environ, "PYTHONPATH": str(ROOT)}

SERVICES = {
    "analysis":  {
        "cmd":   [sys.executable, "-m", "services.analysis_agent_service", "--port", "8101"],
        "label": "Analysis Agent   A2A :8101",
    },
    "execution": {
        "cmd":   [sys.executable, "-m", "services.execution_agent_service", "--port", "8102"],
        "label": "Execution Agent  A2A :8102",
    },
    "backend": {
        "cmd":   [sys.executable, "-m", "uvicorn", "backend.main:app",
                  "--host", "0.0.0.0", "--port", "8100", "--reload"],
        "label": "FastAPI Gateway      :8100",
    },
    "frontend": {
        "cmd":   [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
                  "--server.port", "8501", "--server.headless", "true"],
        "label": "Streamlit UI         :8501",
    },
}

procs = []
for name, svc in SERVICES.items():
    print(f"  Starting {svc['label']} ...")
    procs.append(subprocess.Popen(svc["cmd"], cwd=str(ROOT), env=ENV))
    time.sleep(1.5)

print("\n  Trade Assistant running:")
print("  Dashboard  → http://localhost:8501")
print("  API docs   → http://localhost:8100/docs")
print("  Analysis   → http://localhost:8101/.well-known/agent.json")
print("  Execution  → http://localhost:8102/.well-known/agent.json")
print("  Ctrl+C to stop all\n")


def _stop(sig, frame):
    for p in procs:
        try:
            p.terminate()
        except OSError:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT,  _stop)
signal.signal(signal.SIGTERM, _stop)

while True:
    for p in procs:
        if p.poll() is not None:
            print(f"Process {p.pid} exited — shutting down")
            _stop(None, None)
    time.sleep(2)
