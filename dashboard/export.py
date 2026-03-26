#!/usr/bin/env python3
"""Export log data to static JSON files for GitHub Pages.

Usage:
    python dashboard/export.py
    python dashboard/export.py --logs logs_pigdice
    python dashboard/export.py --logs logs --out dashboard/data
"""

import csv
import json
import re
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Export log data to static JSON")
    parser.add_argument("--logs", default="logs", help="Log directory relative to project root (default: logs)")
    parser.add_argument("--out", default="dashboard/data", help="Output directory relative to project root (default: dashboard/data)")
    args = parser.parse_args()

    log_dir = (PROJECT_ROOT / args.logs).resolve()
    out_dir = (PROJECT_ROOT / args.out).resolve()

    print(f"Logs:   {log_dir}")
    print(f"Output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    results_csv = log_dir / "results.csv"
    if results_csv.exists():
        with results_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        write_json(out_dir / "results.json", rows)
        print(f"  results.json ({len(rows)} rows)")
    else:
        write_json(out_dir / "results.json", [])
        print("  results.json (empty — results.csv not found)")

    # envs.json
    envs = []
    if results_csv.exists():
        with results_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        seen = {}
        for r in rows:
            name = r.get("env", "")
            if name and name not in seen:
                key = re.sub(r'[^a-zA-Z0-9_-]', '_', name).lower()
                seen[name] = key
        envs = [{"name": name, "key": key} for name, key in seen.items()]
    write_json(out_dir / "envs.json", envs)
    print(f"  envs.json ({len(envs)} envs)")

    # ratings.json (global state)
    state_file = log_dir / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        write_json(out_dir / "ratings.json", {"ratings": state.get("ratings", {}), "ci": state.get("ci", {})})
        print("  ratings.json")
    else:
        write_json(out_dir / "ratings.json", {})
        print("  ratings.json (empty — state.json not found)")

    # per-env ratings: data/ratings/{key}.json
    ratings_dir = out_dir / "ratings"
    ratings_dir.mkdir(exist_ok=True)
    for env in envs:
        key = env["key"]
        p = log_dir / f"{key}.json"
        if p.exists():
            state = json.loads(p.read_text())
            write_json(ratings_dir / f"{key}.json", {"ratings": state.get("ratings", {}), "ci": state.get("ci", {})})
            print(f"  ratings/{key}.json")
        else:
            write_json(ratings_dir / f"{key}.json", {})

    # games
    games_src = log_dir / "games"
    games_out = out_dir / "games"
    games_out.mkdir(exist_ok=True)

    if games_src.exists():
        game_ids = sorted([p.stem for p in games_src.glob("*.json")], reverse=True)
        write_json(games_out / "index.json", game_ids)
        print(f"  games/index.json ({len(game_ids)} games)")

        for gid in game_ids:
            src = games_src / f"{gid}.json"
            dst = games_out / f"{gid}.json"
            shutil.copy2(src, dst)
        print(f"  games/*.json (copied {len(game_ids)} files)")
    else:
        write_json(games_out / "index.json", [])
        print("  games/index.json (empty — no games directory found)")

    print("Done.")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, separators=(',', ':')))


if __name__ == "__main__":
    main()
