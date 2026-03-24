import math
import random
import asyncio
import numpy as np
from abc import ABC, abstractmethod


def build_team_sampler(cfg: dict, teams: list) -> "BaseTeamSampler":
    classes = {"random": RandomTeamSampler, "elo": EloTeamSampler, "bradley_terry": BradleyTerryTeamSampler}
    sampler_type = cfg.get("team_sampler", {}).get("sample", "random")
    cls = classes.get(sampler_type)
    if cls is None: raise ValueError(f"Unknown team_sampler type '{sampler_type}'. Use: {list(classes)}")
    return cls(teams)


class BaseTeamSampler(ABC):
    def __init__(self, teams: list, initial_rating: float = 1000.0):
        self.teams = teams
        self.initial_rating = initial_rating
        self._ratings: dict[str, float] = {t.name: initial_rating for t in teams}
        self._results: list[tuple[str, str, float]] = []
        self._lock = asyncio.Lock()

    def get_rating(self, team_name: str) -> float: return self._ratings.get(team_name, self.initial_rating)

    async def update(self, team0_name: str, team1_name: str, outcome: float) -> None:
        async with self._lock: self._results.append((team0_name, team1_name, outcome))

    def compute_ratings(self) -> None: pass

    def leaderboard(self) -> list[tuple[str, float]]: return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    @abstractmethod
    def sample_match(self) -> tuple: """Return (team0, team1) for the next match."""


class RandomTeamSampler(BaseTeamSampler):
    def sample_match(self) -> tuple: return tuple(random.sample(self.teams, 2))


class EloTeamSampler(BaseTeamSampler):
    def __init__(self, teams: list, k_factor: int = 16, initial_rating: float = 1000.0,
                 temperature: float = 100.0, n_bootstrap: int = 200):
        super().__init__(teams, initial_rating)
        self.k_factor = k_factor
        self.temperature = temperature
        self.n_bootstrap = n_bootstrap
        self._ci: dict[str, tuple[float, float]] = {}

    def sample_match(self) -> tuple:
        team0 = random.choice(self.teams)
        opponents = [t for t in self.teams if t is not team0]
        r0 = self.get_rating(team0.name)
        weights = [math.exp(-abs(self.get_rating(t.name) - r0) / self.temperature) for t in opponents]
        team1 = random.choices(opponents, weights=weights, k=1)[0]
        return (team0, team1)

    def compute_ratings(self) -> None:
        names = [t.name for t in self.teams]
        samples: dict[str, list[float]] = {n: [] for n in names}
        for _ in range(self.n_bootstrap):
            ratings = {n: self.initial_rating for n in names}
            for t0, t1, outcome in random.sample(self._results, len(self._results)):
                r0, r1 = ratings[t0], ratings[t1]
                exp0 = 1.0 / (1.0 + 10 ** ((r1 - r0) / 400))
                ratings[t0] = r0 + self.k_factor * (outcome - exp0)
                ratings[t1] = r1 + self.k_factor * ((1.0 - outcome) - (1.0 - exp0))
            for n in names: samples[n].append(ratings[n])
        for n in names:
            s = sorted(samples[n])
            self._ratings[n] = sum(s) / len(s)
            lo = s[int(0.025 * len(s))]
            hi = s[min(int(0.975 * len(s)), len(s) - 1)]
            self._ci[n] = (lo, hi)

    def get_ci(self, team_name: str) -> tuple[float, float] | None: return self._ci.get(team_name)


class BradleyTerryTeamSampler(BaseTeamSampler):
    def __init__(self, teams: list, initial_rating: float = 1000.0):
        super().__init__(teams, initial_rating)
        self._ci: dict[str, tuple[float, float]] = {}

    def sample_match(self) -> tuple: return tuple(random.sample(self.teams, 2))

    def _fit_bt(self, results: list) -> dict[str, float]:
        """Fit Bradley-Terry via MM algorithm. Returns raw strengths β_i (geometric mean = 1)."""
        names = [t.name for t in self.teams]
        strengths = {n: 1.0 for n in names}
        wins: dict[str, float] = {n: 0.5 for n in names}
        games: dict[tuple[str, str], int] = {}
        for t0, t1, outcome in results:
            wins[t0] += outcome
            wins[t1] += 1.0 - outcome
            pair = (min(t0, t1), max(t0, t1))
            games[pair] = games.get(pair, 0) + 1
        for _ in range(1000):
            new_strengths: dict[str, float] = {}
            for i in names:
                denom = sum(
                    games.get((min(i, j), max(i, j)), 0) / (strengths[i] + strengths[j])
                    for j in names if j != i
                    if games.get((min(i, j), max(i, j)), 0) > 0
                )
                new_strengths[i] = wins[i] / denom if denom > 0 else strengths[i]
            log_mean = sum(math.log(max(s, 1e-300)) for s in new_strengths.values()) / len(names)
            scale = math.exp(log_mean)
            new_strengths = {n: s / scale for n, s in new_strengths.items()}
            if max(abs(new_strengths[n] - strengths[n]) for n in names) < 1e-8:
                strengths = new_strengths
                break
            strengths = new_strengths
        return strengths

    def compute_ratings(self) -> None:
        names = [t.name for t in self.teams]
        idx = {n: i for i, n in enumerate(names)}
        strengths = self._fit_bt(self._results)

        # Elo-like scale: 1000 + 400 * log10(β_i)
        self._ratings = {n: 1000.0 + 400.0 * math.log10(max(s, 1e-300)) for n, s in strengths.items()}

        # Fisher information matrix of log-likelhood w.r.t. α_i = log(β_i)
        # I[i,i] = Σ_{j≠i} n_ij * p_ij * (1-p_ij)
        # I[i,j] = -n_ij * p_ij * (1-p_ij)
        n = len(names)
        I = np.zeros((n, n))
        for t0, t1, _ in self._results:
            i, j = idx[t0], idx[t1]
            if i == j:
                continue
            b0, b1 = strengths[t0], strengths[t1]
            p = b0 / (b0 + b1)
            v = p * (1 - p)
            I[i, i] += v
            I[j, j] += v
            I[i, j] -= v
            I[j, i] -= v

        # Pseudoinverse handles the rank-n-1 singularity (model is identified up to additive const)
        cov = np.linalg.pinv(I)

        # SE on rating scale: r_i = 1000 + (400/ln10) * α_i  →  SE(r_i) = (400/ln10) * sqrt(cov[i,i])
        rating_scale = 400.0 / math.log(10)
        for i, name in enumerate(names):
            se = rating_scale * math.sqrt(max(float(cov[i, i]), 0.0))
            margin = 1.96 * se
            self._ci[name] = (self._ratings[name] - margin, self._ratings[name] + margin)

    def get_ci(self, team_name: str) -> tuple[float, float] | None:
        return self._ci.get(team_name)