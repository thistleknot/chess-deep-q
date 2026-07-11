"""Single point of entry for the chess Q-learning control panel.

Run ONE command:  python app.py
It boots the FastAPI server and opens the dashboard in your browser. From there the UI drives everything —
start/stop training, run Optuna, calibrate Elo — no other script to run. Training runs as detached
subprocesses the server spawns for you, so they survive a browser refresh (the page auto-reconnects) and
even an app restart (relaunch and it re-attaches to the latest run).

Ctrl+C stops the server (and the browser panel); any detached training run keeps going and can be
re-attached by relaunching app.py.
"""
import os
import threading
import time
import webbrowser

import uvicorn

HOST = os.environ.get("QLEARN_HOST", "127.0.0.1")
PORT = int(os.environ.get("QLEARN_PORT", "8000"))


def _open_browser():
    time.sleep(1.5)                       # give uvicorn a moment to bind
    try:
        webbrowser.open(f"http://{HOST}:{PORT}/")
    except Exception:
        pass


if __name__ == "__main__":
    print(f"chess Q-learning control panel -> http://{HOST}:{PORT}/  (Ctrl+C to stop)")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("server:app", host=HOST, port=PORT)
