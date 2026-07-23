#!/usr/bin/env python3
"""Tiny static-file + /api reverse-proxy for the shareable local demo.

Serves the built Angular app (DIST_DIR) and forwards every /api/* request to
the local backend (BACKEND), stripping the /api prefix — the same shape as
the CloudFront setup, but on one local port so a single tunnel URL serves the
UI and proxies the API on the same origin (no CORS). Streams responses so the
DQC generation SSE still arrives progressively.

Env: DIST_DIR (required), BACKEND (default http://127.0.0.1:8000), PORT (4200).
Stdlib only — no extra dependencies.
"""
from __future__ import annotations

import mimetypes
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIST = os.environ.get("DIST_DIR", "")
BACKEND = os.environ.get("BACKEND", "http://127.0.0.1:8000").rstrip("/")
PORT = int(os.environ.get("PORT", "4200"))

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("font/woff2", ".woff2")

_HOP = {"host", "content-length", "connection", "keep-alive",
        "proxy-authenticate", "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade", "accept-encoding"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ── /api/* → backend (prefix stripped), streamed back ──────────────────
    def _proxy(self) -> None:
        backend_path = self.path[4:] or "/"          # drop leading "/api"
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(BACKEND + backend_path, data=body,
                                     method=self.command)
        for k, v in self.headers.items():
            if k.lower() not in _HOP:
                req.add_header(k, v)
        try:
            resp = urllib.request.urlopen(req, timeout=180)
        except urllib.error.HTTPError as e:
            resp = e                                  # forward 4xx/5xx bodies
        except urllib.error.URLError as e:
            self.send_error(502, f"backend unreachable: {e.reason}")
            return
        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in _HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    # ── everything else → static file, SPA-fallback to index.html ──────────
    def _static(self) -> None:
        rel = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        full = os.path.normpath(os.path.join(DIST, rel))
        if not (full.startswith(DIST) and os.path.isfile(full)):
            full = os.path.join(DIST, "index.html")   # SPA route → index
        try:
            with open(full, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _route(self) -> None:
        (self._proxy if self.path.startswith("/api") else self._static)()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _route

    def log_message(self, *args) -> None:          # keep the console quiet
        pass


if __name__ == "__main__":
    if not DIST or not os.path.isdir(DIST):
        sys.exit(f"✗ DIST_DIR not set or missing: {DIST!r}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
