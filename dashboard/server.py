#!/usr/bin/env python3
"""Simple HTTP server for the Team Eval dashboard.

Usage:
    python dashboard/server.py
    python dashboard/server.py --logs logs_pigdice --port 8081
"""

import csv
import json
import re
import os
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

STATIC_DIR = Path(__file__).parent


class Handler(BaseHTTPRequestHandler):
    log_dir: Path = None

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/api/results":
            self._serve_results()
        elif path == "/api/ratings":
            self._serve_ratings()
        elif path.startswith("/api/ratings/"):
            self._serve_env_ratings(path[len("/api/ratings/"):])
        elif path == "/api/envs":
            self._serve_envs()
        elif path == "/api/games":
            self._serve_games_list()
        elif path.startswith("/api/games/"):
            self._serve_game(path[len("/api/games/"):])
        else:
            self.send_error(404)

    def _serve_file(self, path, content_type):
        try:
            data = Path(path).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_results(self):
        p = self.__class__.log_dir / "results.csv"
        if not p.exists():
            return self._json([])
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self._json(rows)

    def _serve_ratings(self):
        p = self.__class__.log_dir / "state.json"
        if not p.exists():
            return self._json({})
        state = json.loads(p.read_text())
        self._json({"ratings": state.get("ratings", {}), "ci": state.get("ci", {})})

    def _serve_envs(self):
        p = self.__class__.log_dir / "results.csv"
        if not p.exists():
            return self._json([])
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
        seen = {}
        for r in rows:
            name = r.get("env", "")
            if name and name not in seen:
                key = re.sub(r'[^a-zA-Z0-9_-]', '_', name).lower()
                seen[name] = key
        self._json([{"name": name, "key": key} for name, key in seen.items()])

    def _serve_env_ratings(self, env_key):
        safe = Path(env_key).name
        p = self.__class__.log_dir / f"{safe}.json"
        if not p.exists():
            return self._json({})
        state = json.loads(p.read_text())
        self._json({"ratings": state.get("ratings", {}), "ci": state.get("ci", {})})

    def _serve_games_list(self):
        d = self.__class__.log_dir / "games"
        if not d.exists():
            return self._json([])
        self._json(sorted([p.stem for p in d.glob("*.json")], reverse=True))

    def _serve_game(self, game_id):
        # Sanitize: strip anything that looks like a path traversal
        safe_id = Path(game_id).name
        p = self.__class__.log_dir / "games" / f"{safe_id}.json"
        if not p.exists():
            return self.send_error(404)
        self._json(json.loads(p.read_text()))

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description="Team Eval dashboard server")
    parser.add_argument("--logs", default="logs", help="Log directory relative to project root (default: logs)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    log_dir = (Path(__file__).parent.parent / args.logs).resolve()
    Handler.log_dir = log_dir

    print(f"Logs:      {log_dir}")
    print(f"Dashboard: http://localhost:{args.port}")
    HTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
