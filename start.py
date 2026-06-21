#!/usr/bin/env python3
"""Start Trade Assistant — backend + frontend."""

import subprocess, sys, time, os, signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV  = {**os.environ, "PYTHONPATH": str(ROOT)}

SERVICES = {
    "backend":  [sys.executable, "-m", "uvicorn", "backend.main:app",
                 "--host", "0.0.0.0", "--port", "8100", "--reload"],
    "frontend": [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
                 "--server.port", "8501", "--server.headless", "true"],
}

procs = []
for name, cmd in SERVICES.items():
    print(f"  Starting {name}...")
    procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=ENV))
    time.sleep(1.5)

print("\n  Trade Assistant running:")
print("  Dashboard → http://localhost:8501")
print("  API docs  → http://localhost:8100/docs")
print("  Press Ctrl+C to stop\n")

def _stop(sig, frame):
    for p in procs:
        try: p.terminate()
        except: pass
    sys.exit(0)

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

while True:
    for p in procs:
        if p.poll() is not None:
            print(f"Process {p.pid} exited — shutting down")
            _stop(None, None)
    time.sleep(2)
