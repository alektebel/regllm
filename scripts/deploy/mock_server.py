#!/usr/bin/env python3
"""
Mock server for verifying Docker networking, port forwarding,
and Cloudflare tunnel before loading the real model.

Zero dependencies — uses Python's built-in http.server.

Usage:
    python scripts/mock_server.py          # port 7860
    python scripts/mock_server.py --port 8080
"""

import argparse
import http.server
import socketserver

HTML = b"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RegLLM \u2014 Mock</title>
  <style>
    body { font-family: sans-serif; max-width: 600px; margin: 80px auto; padding: 0 20px; }
    h1   { color: #2d6a4f; }
    .ok  { background: #d8f3dc; border: 1px solid #74c69d;
           border-radius: 8px; padding: 16px 24px; }
    code { background: #eee; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>RegLLM</h1>
  <div class="ok">
    <strong>\u2705 Mock server is reachable.</strong><br><br>
    Docker networking, port forwarding and Cloudflare tunnel are working.<br>
    Replace with the real app when ready:
    <pre><code>docker compose up -d</code></pre>
  </div>
</body>
</html>
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(HTML)))
        self.end_headers()
        self.wfile.write(HTML)

    def log_message(self, fmt, *args):
        print(f"[mock] {self.address_string()} — {fmt % args}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RegLLM mock server")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    with socketserver.TCPServer(("0.0.0.0", args.port), _Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"Mock server listening on http://0.0.0.0:{args.port}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
