import csv
import json
import asyncio
from datetime import datetime
from pathlib import Path

import textarena as ta

from src.teams.team import BaseTeam
from src.rating.sampler import BaseTeamSampler

_RESULTS_CSV = Path("logs") / "results.csv"
_CSV_FIELDS = ["timestamp", "env", "team0", "team1", "elo0_before", "elo1_before",
               "elo0_after", "elo1_after", "winner", "num_turns"]


def _append_csv(row: dict) -> None:
    _RESULTS_CSV.parent.mkdir(exist_ok=True)
    write_header = not _RESULTS_CSV.exists()
    with _RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class MatchRunner:
    def __init__(self, env_name: str, teams: list[BaseTeam], sampler: BaseTeamSampler):
        self.env_name = env_name
        self.teams = teams
        self._team_by_player = {i: t for i, t in enumerate(teams)}
        self.sampler = sampler

    async def run(self) -> dict:
        env = ta.make(self.env_name)
        env.reset(num_players=len(self.teams))
        turns = []
        terminated = False
        while not terminated:
            player_id, obs = env.get_observation()
            team = self._team_by_player[player_id]
            action, transcript = await team(obs)
            turns.append({"player_id": player_id, "team": team.name, "action": action, "transcript": transcript})
            terminated, _ = env.step(action)
        rewards, _ = env.close()

        team0, team1 = self.teams[0], self.teams[1]

        elo0_before = self.sampler.get_rating(team0.name)
        elo1_before = self.sampler.get_rating(team1.name)

        if rewards:
            max_reward = max(rewards.values())
            winners = [pid for pid, r in rewards.items() if r == max_reward]
            winner_pid = winners[0] if len(winners) == 1 else None
        else:
            winner_pid = None

        winner_name = self._team_by_player[winner_pid].name if winner_pid is not None else None
        outcome = 0.5 if winner_pid is None else (1.0 if winner_pid == 0 else 0.0)
        await self.sampler.update(team0.name, team1.name, outcome)

        elo0_after = self.sampler.get_rating(team0.name)
        elo1_after = self.sampler.get_rating(team1.name)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        match_log = {
            "env_name": self.env_name,
            "teams": {
                t.name: [{"agent_id": a.agent_id, "model": a.model_name} for a in t.agents]
                for t in self.teams
            },
            "winner": winner_name,
            "turns": turns,
            "final_ratings": {team0.name: elo0_after, team1.name: elo1_after},
        }

        game_path = Path("logs") / "games" / f"{ts}.json"
        game_path.parent.mkdir(parents=True, exist_ok=True)
        game_path.write_text(json.dumps(match_log, indent=2))

        _append_csv({
            "timestamp": ts,
            "env": self.env_name,
            "team0": team0.name,
            "team1": team1.name,
            "elo0_before": f"{elo0_before:.1f}",
            "elo1_before": f"{elo1_before:.1f}",
            "elo0_after": f"{elo0_after:.1f}",
            "elo1_after": f"{elo1_after:.1f}",
            "winner": winner_name or "draw",
            "num_turns": len(turns),
        })

        return match_log
