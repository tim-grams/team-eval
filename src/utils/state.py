import json
import re
from pathlib import Path


_EXCLUDED_CFG_KEYS = {"games", "concurrency", "dry_run", "backend", "api_key", "temperature", "top_p", "timeout", "env", "envs", "error_allowance", "team_sampler"}


def env_state_filename(env_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', env_name).lower() + ".json"


def save_state(path: str | Path, sampler, cfg: dict = None) -> None:
    ci = getattr(sampler, "_ci", {})
    state = {
        "ratings": sampler._ratings,
        "ci": {k: list(v) for k, v in ci.items()},
        "results": sampler._results,
    }
    if cfg is not None:
        state["config"] = {k: v for k, v in cfg.items() if k not in _EXCLUDED_CFG_KEYS}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def load_state(path: str) -> tuple[dict, dict[str, float], list]:
    data = json.loads(Path(path).read_text())
    config = {k: v for k, v in data.get("config", {}).items() if k not in _EXCLUDED_CFG_KEYS}
    ratings = {k: float(v) for k, v in data.get("ratings", {}).items()}
    results = [tuple(r) for r in data.get("results", [])]
    return config, ratings, results
