#!/usr/bin/env python3
"""
Monte Carlo Dashboard Server
Serves the simulation results + interactive HTML dashboard.
"""

import http.server
import json
import os
import sys

PORT = 8501
DIR = os.path.dirname(os.path.abspath(__file__))


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
        elif self.path == "/api/results":
            self.send_json("results.json")
            return
        super().do_GET()

    def send_json(self, filename):
        fpath = os.path.join(DIR, filename)
        if not os.path.exists(fpath):
            self.send_error(404, "Results not found — run simulation.py first")
            return
        with open(fpath) as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, format, *args):
        pass  # quiet


if __name__ == "__main__":
    with http.server.HTTPServer(("0.0.0.0", PORT), DashboardHandler) as srv:
        print(f"Monte Carlo Dashboard: http://0.0.0.0:{PORT}")
        sys.stdout.flush()
        srv.serve_forever()
