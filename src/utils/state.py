import json
from pathlib import Path


def save_state(path: str, sampler, cfg: dict) -> None:
    state = {
        "config": cfg,
        "k_factor": sampler.k_factor,
        "initial_rating": sampler.initial_rating,
        "ratings": sampler._ratings,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def load_state(path: str) -> tuple[dict, dict[str, float]]:
    data = json.loads(Path(path).read_text())
    ratings = {k: float(v) for k, v in data["ratings"].items()}
    return data.get("config", {}), ratings
