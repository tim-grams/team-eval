import asyncio
import argparse
from pathlib import Path

import yaml

from src.teams.team import build_teams_from_cfg
from src.rating.sampler import build_team_sampler
from src.runner import MatchRunner
from src.utils.state import save_state, load_state


def load_config(name: str) -> dict:
    path = Path("configs") / f"{name}.yaml"
    return yaml.safe_load(path.read_text())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test", help="Config name under configs/")
    parser.add_argument("--state", default=None, help="Path to state file for save/load")
    args = parser.parse_args()

    if args.state and Path(args.state).exists():
        cfg, saved_ratings = load_state(args.state)
        print(f"Loaded config and ratings from {args.state}")
    else:
        cfg = load_config(args.config)
        saved_ratings = {}

    teams = build_teams_from_cfg(cfg)
    sampler = build_team_sampler(cfg, teams)

    for name, rating in saved_ratings.items():
        if name in sampler._ratings:
            sampler._ratings[name] = rating

    concurrency = cfg.get("concurrency", 1)
    semaphore = asyncio.Semaphore(concurrency)

    async def run_match(match_num: int):
        async with semaphore:
            team0, team1 = sampler.sample_match()
            runner = MatchRunner(cfg["env"], [team0, team1], sampler)
            log = await runner.run()
            winner = log["winner"] or "Draw"
            print(f"Match {match_num} | {team0.name} vs {team1.name} | Winner: {winner} | "
                  f"Ratings: {team0.name}={log['final_ratings'][team0.name]:.1f}  "
                  f"{team1.name}={log['final_ratings'][team1.name]:.1f}")
            if args.state:
                save_state(args.state, sampler, cfg)

    await asyncio.gather(*[run_match(i) for i in range(1, cfg["games"] + 1)])

    print("\n=== Final Leaderboard ===")
    for rank, (team_name, rating) in enumerate(sampler.leaderboard(), 1):
        print(f"  {rank}. {team_name:<20} {rating:.1f}")

    if cfg.get("backend") == "vllm_actor":
        import ray
        pools = {id(a._pool): a._pool for t in teams for a in t.agents if hasattr(a, "_pool") and a._pool}
        for pool in pools.values():
            loop = asyncio.get_event_loop()
            await asyncio.gather(*[
                loop.run_in_executor(None, ray.get, slot.actor.unload_model.remote())
                for slot in pool._slots if slot.model_name is not None
            ])
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
