"""Lightweight HTTP + SSE server for the pipeline viz dashboard.

Runs in a background daemon thread while the pipeline executes.
StatusTracker calls broadcast() on every update, pushing the full
state to all connected EventSource clients.
"""
from __future__ import annotations

import json
import mimetypes
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8766
DASHBOARD_DIR = Path("dashboard")

# One queue per connected SSE client
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def broadcast(state: dict) -> None:
    """Push the current state snapshot to every connected SSE client."""
    msg = ("data: " + json.dumps(state, ensure_ascii=False, default=str) + "\n\n").encode()
    with _sse_lock:
        dead: list[queue.Queue] = []
        for q in _sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


class VizHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:  # silence access log
        pass

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path in ("/", "/index.html"):
            self._serve_file(DASHBOARD_DIR / "index.html")
        elif path == "/api/status/stream":
            self._sse_stream()
        elif path.startswith(("/css/", "/js/", "/ui/", "/pages/")):
            self._serve_file(DASHBOARD_DIR / path.lstrip("/"))
        else:
            self.send_response(404)
            self.end_headers()

    def _sse_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        client_q: queue.Queue = queue.Queue(maxsize=64)
        with _sse_lock:
            _sse_clients.append(client_q)

        try:
            while True:
                try:
                    msg = client_q.get(timeout=20)
                    self.wfile.write(msg)
                    self.wfile.flush()
                except queue.Empty:
                    # keepalive comment so the connection stays open
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _sse_lock:
                if client_q in _sse_clients:
                    _sse_clients.remove(client_q)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        mime, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_viz_server(port: int = PORT) -> None:
    """Start the viz server in a background daemon thread and open the browser."""
    try:
        server = HTTPServer(("localhost", port), VizHandler)
    except OSError:
        # Port already in use — skip (dashboard will fall back to polling)
        return

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}"
    print(f"\nPipeline Dashboard: {url}\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
