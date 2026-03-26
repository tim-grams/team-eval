import asyncio
import re
from abc import ABC, abstractmethod
from src.agents.agent import BaseAgent
from src.utils.logging import setup_logger


_TEAM_CLASSES = {"voting": lambda: VotingTeam, "reflection": lambda: ReflectionTeam}


def build_teams_from_cfg(cfg: dict, log_dir: str = "logs") -> list:
    def _make_team(team_cls, agents, team_info, team_id, team_name):
        return team_cls(agents=agents, debate_rounds=team_info.get("rounds", 1), team_id=team_id, name=team_name, log_dir=log_dir)

    team_cfg = cfg["team_sampler"]["teams"]
    global_kwargs = {k: cfg[k] for k in ("temperature", "top_p", "timeout") if k in cfg}

    # Determine if any agent uses vllm backend (needs actor pool)
    def _agent_backend(agent_cfg) -> str:
        if isinstance(agent_cfg, dict): return agent_cfg.get("backend", "vllm")
        return "vllm"

    has_vllm = any(
        _agent_backend(agent_cfg) == "vllm"
        for team_info in team_cfg.values()
        for agent_cfg in team_info["agents"].values()
    )

    pool = None
    if has_vllm and "actors" in cfg:
        from src.agents.vllm_actor import VLLMActorPool
        import ray
        ray.init(ignore_reinit_error=True)
        pool = VLLMActorPool(cfg["actors"], log_dir=log_dir)

    openrouter_key = None
    def _get_openrouter_key() -> str:
        nonlocal openrouter_key
        if openrouter_key is None:
            import os
            raw_key = cfg.get("openrouter_api_key", "${OPENROUTER_API_KEY}")
            openrouter_key = os.path.expandvars(raw_key)
        return openrouter_key

    def _build_agent(agent_name: str, agent_cfg) -> BaseAgent:
        if isinstance(agent_cfg, str):
            model_name, backend, template, reasoning = agent_cfg, "vllm", cfg.get("template", "default"), None
        else:
            model_name = agent_cfg["model"]
            backend = agent_cfg.get("backend", "vllm")
            template = agent_cfg.get("template", cfg.get("template", "default"))
            reasoning = agent_cfg.get("reasoning")

        if backend == "vllm":
            from src.agents.vllm_actor import VLLMActorAgent
            return VLLMActorAgent(model_name=model_name, pool=pool, template=template, name=agent_name)
        if backend == "openrouter":
            from src.agents.agent import OpenRouterAgent
            kwargs = dict(global_kwargs)
            if reasoning is not None: kwargs["reasoning"] = reasoning
            return OpenRouterAgent(model_name=model_name, api_key=_get_openrouter_key(), name=agent_name, **kwargs)
        # Legacy plain vllm/ollama via server_url
        from src.agents.agent import VLLMAgent, OllamaAgent
        agent_cls = {"plain_vllm": VLLMAgent, "ollama": OllamaAgent}[backend]
        return agent_cls(model_name=model_name, server_url=cfg["server_url"], name=agent_name, **global_kwargs)

    return [
        _make_team(
            _TEAM_CLASSES[team_info.get("type", "voting")](),
            [_build_agent(agent_name, agent_cfg) for agent_name, agent_cfg in team_info["agents"].items()],
            team_info, team_id, team_name,
        )
        for team_id, (team_name, team_info) in enumerate(team_cfg.items())
    ]


class BaseTeam(ABC):
    def __init__(self, agents: list[BaseAgent], debate_rounds: int = 3, team_id: int = -1, name: str = "", log_dir: str = "logs"):
        self.agents = agents
        self.debate_rounds = debate_rounds
        self.team_id = team_id
        self.name = name or str(team_id)
        self._log = setup_logger(self.name, f"{log_dir}/outputs")

    _TOKEN_LIMIT = 2000

    async def _maybe_summarize(self, agent: BaseAgent, justification: str) -> str:
        if len(justification) // 4 <= self._TOKEN_LIMIT: return justification
        self._log.info("Justification from agent %s exceeds %d tokens, requesting summary", agent.name, self._TOKEN_LIMIT)
        _, summary = await agent(f"The following justification is too long. Summarize it in fewer than {self._TOKEN_LIMIT} tokens:\n\n{justification}")
        return summary

    async def _run_debate(self, prompt: str) -> tuple[list[dict], list[dict]]: raise NotImplementedError

    @abstractmethod
    async def __call__(self, prompt: str) -> tuple[str, list[dict]]: """Run the debate and return (final_action, transcript)."""


class ReflectionTeam(BaseTeam):
    _RESPONSE_FORMAT = (
        "Respond in exactly this format:\n"
        "ACTION: <your action>\n"
        "JUSTIFICATION: <short justification>"
    )

    def _build_prompt(self, prompt: str, prev: dict | None, round_num: int) -> str:
        if prev:
            return (
                f"{prompt}\n\n"
                f"Your previous proposal (round {round_num - 1}/{self.debate_rounds}):\n"
                f"ACTION: {prev['proposal']}\nJUSTIFICATION: {prev['justification']}\n\n"
                f"You may revise your action.\n{self._RESPONSE_FORMAT}"
            )
        return f"{prompt}\n\n{self._RESPONSE_FORMAT}"

    @staticmethod
    def _parse_response(text: str) -> tuple[str, str]:
        action_matches = list(re.finditer(r"ACTION:\s*(.+)", text))
        action_match = action_matches[-1] if action_matches else None
        if action_match:
            action_text = action_match.group(1).strip()
            brackets = re.findall(r"\[([^\]]+)\]", action_text)
            action = f"[{brackets[-1]}]" if brackets else action_text
            just_match = re.search(r"JUSTIFICATION:\s*(.+)", text[action_match.start():], re.DOTALL)
            justification = just_match.group(1).strip() if just_match else ""
            return action, justification
        return text.strip(), ""

    async def __call__(self, prompt: str) -> tuple[str, list[dict]]:
        agent = self.agents[0]
        transcript: list[dict] = []
        prev: dict | None = None
        for round_num in range(1, self.debate_rounds + 1):
            self._log.info("Querying agent %s in round %d", agent.name, round_num)
            sent_prompt, completion = await agent(self._build_prompt(prompt, prev, round_num))
            self._log.info("Received response from agent %s in round %d", agent.name, round_num)
            action, justification = self._parse_response(completion)
            justification = await self._maybe_summarize(agent, justification)
            transcript.append({
                "round": round_num,
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "prompt": sent_prompt,
                "completion": completion,
                "proposal": action,
                "justification": justification,
            })
            if prev and prev["proposal"] == action:
                break
            prev = {"proposal": action, "justification": justification}
        return prev["proposal"], transcript


class VotingTeam(BaseTeam):
    _RESPONSE_FORMAT = (
        "Respond in exactly this format:\n"
        "ACTION: <your action>\n"
        "JUSTIFICATION: <short justification>"
    )

    @staticmethod
    def _format_prior(prev_proposals: list[dict]) -> str:
        return "\n".join(f"{p.get('agent_name', p['agent_id'])}:\nACTION: {p['proposal']}\nJUSTIFICATION: {p['justification']}" for p in prev_proposals)

    def _build_prompt(self, prompt: str, prev_proposals: list[dict] | None, agent_name: str) -> str:
        if prev_proposals:
            prior = self._format_prior(prev_proposals)
            return (
                f"{prompt}\n\nYou are agent {agent_name} debating with your teammates about the optimal next action. These are the current action proposals:\n{prior}\n\n"
                f"Based on the opinion of the other agents, you may revise your action proposal.\n{self._RESPONSE_FORMAT}"
            )
        return f"{prompt}\n\n{self._RESPONSE_FORMAT}"

    @staticmethod
    def _parse_response(text: str) -> tuple[str, str]:
        action_matches = list(re.finditer(r"ACTION:\s*(.+)", text))
        action_match = action_matches[-1] if action_matches else None
        if action_match:
            action_text = action_match.group(1).strip()
            brackets = re.findall(r"\[([^\]]+)\]", action_text)
            action = f"[{brackets[-1]}]" if brackets else action_text
            just_match = re.search(r"JUSTIFICATION:\s*(.+)", text[action_match.start():], re.DOTALL)
            justification = just_match.group(1).strip() if just_match else ""
            return action, justification
        return text.strip(), ""

    async def _run_debate(self, prompt: str) -> tuple[list[dict], list[dict]]:
        transcript: list[dict] = []
        prev_proposals: list[dict] = []
        for round_num in range(1, self.debate_rounds + 1):
            prior = prev_proposals if round_num > 1 else None
            for agent in self.agents: self._log.info("Querying agent %s in round %d", agent.name, round_num)
            responses = await asyncio.gather(*[
                agent(self._build_prompt(prompt, prior, agent_name=agent.name))
                for agent in self.agents
            ])
            prev_proposals = []
            for agent, (sent_prompt, completion) in zip(self.agents, responses):
                self._log.info("Received response from agent %s in round %d", agent.name, round_num)
                action, justification = self._parse_response(completion)
                justification = await self._maybe_summarize(agent, justification)
                entry = {
                    "round": round_num,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "prompt": sent_prompt,
                    "completion": completion,
                    "proposal": action,
                    "justification": justification,
                }
                transcript.append(entry)
                prev_proposals.append(entry)
            if len(set(e["proposal"] for e in prev_proposals)) == 1:
                break
        return transcript, prev_proposals

    _JUDGE_FORMAT = (
        "Respond in exactly this format:\n"
        "ACTION: <your action>\n"
        "JUSTIFICATION: <short justification>"
    )

    def _build_judge_prompt(self, prompt: str, final_proposals: list[dict]) -> str:
        proposals_text = self._format_prior(final_proposals)
        return (
            f"{prompt}\n\n"
            f"After the debate, here are the team's final action proposals:\n{proposals_text}\n\n"
            f"As the judge, select the single best action for the team.\n{self._JUDGE_FORMAT}"
        )

    async def __call__(self, prompt: str) -> tuple[str, list[dict]]:
        transcript, final_proposals = await self._run_debate(prompt)
        judge_agent = self.agents[0]
        sent_prompt, completion = await judge_agent(self._build_judge_prompt(prompt, final_proposals))
        final_action, justification = self._parse_response(completion)
        transcript.append({
            "round": self.debate_rounds + 1,
            "role": "judge",
            "agent_id": judge_agent.agent_id,
            "agent_name": judge_agent.name,
            "prompt": sent_prompt,
            "completion": completion,
            "proposal": final_action,
            "justification": justification,
        })
        return final_action, transcript