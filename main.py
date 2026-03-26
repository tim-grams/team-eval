import asyncio
import argparse
import shutil
from pathlib import Path

import yaml

from src.teams.team import build_teams_from_cfg
from src.samplers.sampler import build_team_sampler, BaseTeamSampler
from src.runner import MatchRunner
from src.utils.state import save_state, load_state, env_state_filename


def load_config(name: str) -> dict:
    path = Path("configs") / f"{name}.yaml"
    return yaml.safe_load(path.read_text())


def _restore_sampler(sampler: BaseTeamSampler, saved_ratings: dict, saved_results: list) -> None:
    for name, rating in saved_ratings.items():
        if name in sampler._ratings: sampler._ratings[name] = rating
    sampler._results.extend(saved_results)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test", help="Config name under configs/")
    parser.add_argument("--log-dir", default="logs", help="Directory for logs (default: logs)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    state_path = log_dir / "state.json"

    cfg = load_config(args.config)
    if state_path.exists():
        saved_cfg, _, _ = load_state(state_path)
        cfg.update(saved_cfg)
        print(f"Resuming from {state_path}")

    # Support both `env` (single) and `envs` (multi)
    envs: list[str] = cfg.get("envs") or [cfg["env"]]

    outputs_dir = log_dir / "outputs"
    if outputs_dir.exists(): shutil.rmtree(outputs_dir)

    teams = build_teams_from_cfg(cfg, log_dir=args.log_dir)

    # Build per-env samplers, restoring from saved state if present
    per_env_samplers: dict[str, BaseTeamSampler] = {}
    for env_name in envs:
        sampler = build_team_sampler(cfg, teams, log_dir=args.log_dir, env_name=env_name)
        env_path = log_dir / env_state_filename(env_name)
        if env_path.exists():
            _, saved_ratings, saved_results = load_state(env_path)
            _restore_sampler(sampler, saved_ratings, saved_results)
            print(f"  {env_name}: resuming ({len(saved_results)} previous games)")
        per_env_samplers[env_name] = sampler

    # Global sampler aggregates results from all envs
    global_sampler = build_team_sampler(cfg, teams, log_dir=args.log_dir)
    for sampler in per_env_samplers.values():
        global_sampler._results.extend(sampler._results)

    concurrency = cfg.get("concurrency", 1)
    semaphore = asyncio.Semaphore(concurrency)
    counter_lock = asyncio.Lock()
    completed = [0]

    async def run_match(env_name: str, match_num: int):
        async with semaphore:
            env_sampler = per_env_samplers[env_name]
            team0, team1 = env_sampler.sample_match()
            runner = MatchRunner(env_name, [team0, team1], env_sampler, log_dir=args.log_dir,
                                 error_allowance=cfg.get("error_allowance", 0))
            log = await runner.run()
            if log.get("discarded"):
                print(f"[discarded] {env_name} | {team0.name} vs {team1.name} | network error")
                return
            # Mirror outcome to global sampler
            winner = log["winner"]
            outcome = 0.5 if winner is None else (1.0 if winner == team0.name else 0.0)
            await global_sampler.update(team0.name, team1.name, outcome)
            async with counter_lock:
                completed[0] += 1
                n = completed[0]
            print(f"[{n}] {env_name} | {team0.name} vs {team1.name} | Winner: {winner or 'Draw'} | "
                  f"Ratings: {team0.name}={log['final_ratings'][team0.name]:.1f}  "
                  f"{team1.name}={log['final_ratings'][team1.name]:.1f}")

    matches = [(env_name, i) for env_name in envs for i in range(1, cfg["games"] + 1)]
    await asyncio.gather(*[run_match(env, i) for env, i in matches])

    # Compute and save per-env states
    for env_name, sampler in per_env_samplers.items():
        sampler.compute_ratings()
        env_path = log_dir / env_state_filename(env_name)
        save_state(env_path, sampler)

    # Compute and save global state
    global_sampler.compute_ratings()
    save_state(state_path, global_sampler, cfg)
    print(f"\nState saved to {log_dir}")

    print("\n=== Final Leaderboard ===")
    for rank, (team_name, rating) in enumerate(global_sampler.leaderboard(), 1):
        ci = global_sampler.get_ci(team_name) if hasattr(global_sampler, "get_ci") else None
        ci_str = f"  [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else ""
        print(f"  {rank}. {team_name:<20} {rating:.1f}{ci_str}")

    pools = {id(a._pool): a._pool for t in teams for a in t.agents if hasattr(a, "_pool") and a._pool}
    if pools:
        import ray
        for pool in pools.values():
            loop = asyncio.get_event_loop()
            await asyncio.gather(*[
                loop.run_in_executor(None, ray.get, actor.unload_model.remote())
                for actor in pool.all_actors()
            ])
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
