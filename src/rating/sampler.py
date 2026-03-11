import math
import random
import asyncio
from abc import ABC, abstractmethod


def build_team_sampler(cfg: dict, teams: list) -> "BaseTeamSampler":
    classes = {"random": RandomTeamSampler, "elo": EloTeamSampler}
    sampler_type = cfg.get("team_sampler", {}).get("sample", "random")
    cls = classes.get(sampler_type)
    if cls is None: raise ValueError(f"Unknown team_sampler type '{sampler_type}'. Use: {list(classes)}")
    return cls(teams)


class BaseTeamSampler(ABC):
    def __init__(self, teams: list, k_factor: int = 32, initial_rating: float = 1000.0):
        self.teams = teams
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self._ratings: dict[str, float] = {t.name: initial_rating for t in teams}
        self._lock = asyncio.Lock()

    def get_rating(self, team_name: str) -> float: return self._ratings.get(team_name, self.initial_rating)

    async def update(self, team0_name: str, team1_name: str, outcome: float) -> None:
        async with self._lock:
            r0 = self.get_rating(team0_name)
            r1 = self.get_rating(team1_name)
            exp0 = 1 / (1 + 10 ** ((r1 - r0) / 400))
            exp1 = 1 - exp0
            self._ratings[team0_name] = r0 + self.k_factor * (outcome - exp0)
            self._ratings[team1_name] = r1 + self.k_factor * ((1 - outcome) - exp1)

    def leaderboard(self) -> list[tuple[str, float]]: return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    @abstractmethod
    def sample_match(self) -> tuple:
        """Return (team0, team1) for the next match."""


class RandomTeamSampler(BaseTeamSampler):
    def sample_match(self) -> tuple:
        return tuple(random.sample(self.teams, 2))


class EloTeamSampler(BaseTeamSampler):
    def __init__(self, teams: list, k_factor: int = 32, initial_rating: float = 1000.0, temperature: float = 100.0):
        super().__init__(teams, k_factor, initial_rating)
        self.temperature = temperature

    def sample_match(self) -> tuple:
        team0 = random.choice(self.teams)
        opponents = [t for t in self.teams if t is not team0]
        r0 = self.get_rating(team0.name)
        weights = [math.exp(-abs(self.get_rating(t.name) - r0) / self.temperature) for t in opponents]
        team1 = random.choices(opponents, weights=weights, k=1)[0]
        return (team0, team1)