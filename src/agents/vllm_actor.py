import os
import time
import uuid
import asyncio
import concurrent.futures
from collections import deque
from typing import Optional

import ray
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.inputs import TextPrompt
from vllm.sampling_params import SamplingParams

from src.agents.agent import BaseAgent
from src.utils.templates import OBSERVATION_FORMATTING, get_hf_formatter
from src.utils.logging import setup_logger
from src.utils.exceptions import ContextLimitExceeded

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=512)


@ray.remote
class VLLMActor:
    def __init__(self, num_gpus: int, slot_id: int, log_dir: str = "logs"):
        self.num_gpus = num_gpus
        self.slot_id = slot_id
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.engine: Optional[AsyncLLMEngine] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.model_name: Optional[str] = None
        self._log = setup_logger(f"vllm_actor_{slot_id + 1}", f"{log_dir}/outputs")
        self._active: int = 0
        self._tok_hist: deque = deque()
        self._report_task: Optional[asyncio.Task] = None

    async def load_model(
        self,
        model_name: str,
        max_model_len: int = 4096,
        max_num_seqs: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 1024,
        gpu_memory_utilization: float = 0.85,
    ) -> None:
        if self.engine is not None: await self.unload_model()
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=self.num_gpus,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            disable_log_stats=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )
        self.model_name = model_name
        self._report_task = asyncio.create_task(self._report_loop())
        self._log.info("Model loaded: %s", model_name)

    async def unload_model(self) -> None:
        if self._report_task is not None:
            self._report_task.cancel()
            try: await self._report_task
            except asyncio.CancelledError: pass
            self._report_task = None
        if self.engine is not None:
            self.engine.shutdown()
            del self.engine
            self.engine = None
            torch.cuda.empty_cache()

    async def submit_prompt(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self.engine is None: raise RuntimeError("No model loaded. Call load_model() first.")
        params = SamplingParams(
            temperature=temperature if temperature is not None else self.sampling_params.temperature,
            top_p=top_p if top_p is not None else self.sampling_params.top_p,
            top_k=top_k if top_k is not None else self.sampling_params.top_k,
            max_tokens=max_tokens if max_tokens is not None else self.sampling_params.max_tokens,
            skip_special_tokens=True,
        )
        request_id = str(uuid.uuid4())
        self._active += 1
        try:
            final_output = None
            prev_len = 0
            async for output in self.engine.generate(TextPrompt(prompt=prompt), params, request_id):
                final_output = output
                if output.outputs:
                    cur_len = len(output.outputs[-1].token_ids)
                    new_toks = cur_len - prev_len
                    prev_len = cur_len
                    if new_toks > 0:
                        now = time.monotonic()
                        for _ in range(new_toks): self._tok_hist.append(now)
            return final_output.outputs[0].text
        finally: self._active -= 1

    async def _report_loop(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            self._log.info("processing=%d  tok/s=%.1f", self._active, self._tok_rate())

    def _tok_rate(self, window: float = 5.0) -> float:
        now = time.monotonic()
        while self._tok_hist and now - self._tok_hist[0] > window: self._tok_hist.popleft()
        return len(self._tok_hist) / window

    def ready(self) -> bool: return True


_MODEL_KWARGS = ("max_model_len", "max_num_seqs", "max_tokens", "gpu_memory_utilization",
                 "temperature", "top_p", "top_k")


class VLLMActorPool:
    def __init__(self, actors_cfg: dict, log_dir: str = "logs"):
        self._by_model: dict[str, list] = {}
        self._all_actors: list = []
        self._rr: dict[str, int] = {}
        entries = []
        for slot_id, (actor_name, cfg) in enumerate(actors_cfg.items()):
            num_gpus = cfg["num_gpus"]
            model_name = cfg["model"]
            actor = VLLMActor.options(num_gpus=num_gpus).remote(num_gpus, slot_id, log_dir)
            kwargs = {k: cfg[k] for k in _MODEL_KWARGS if k in cfg}
            entries.append((actor, model_name, kwargs))
            self._all_actors.append(actor)
        ray.get([actor.load_model.remote(model_name, **kwargs) for actor, model_name, kwargs in entries])
        for actor, model_name, _ in entries: self._by_model.setdefault(model_name, []).append(actor)

    async def get_actor(self, model_name: str) -> object:
        actors = self._by_model.get(model_name)
        if not actors:
            raise RuntimeError(f"No actor loaded for model '{model_name}'")
        idx = self._rr.get(model_name, 0) % len(actors)
        self._rr[model_name] = idx + 1
        return actors[idx]

    def all_actors(self) -> list: return self._all_actors


class VLLMActorAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        actor=None,
        pool: Optional[VLLMActorPool] = None,
        system_prompt: str = "",
        template: str = "default",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        name: str = "",
    ):
        if actor is None and pool is None: raise ValueError("Provide either actor or pool.")
        self._actor = actor
        self._pool = pool
        if template == "hf":
            self._fmt = get_hf_formatter(model_name, system_prompt or None)
            system_prompt = ""  # system prompt already baked into the formatter
        elif template in OBSERVATION_FORMATTING: self._fmt = OBSERVATION_FORMATTING[template]
        else: raise ValueError(f"Unknown template '{template}'. Available: hf, {list(OBSERVATION_FORMATTING)}")
        self.template = template
        self.model_name = model_name
        self.server_url = ""
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.timeout = None
        self.agent_id = str(uuid.uuid4())
        self.name = name or self.agent_id[:8]

    async def _resolve_actor(self):
        if self._pool is not None: return await self._pool.get_actor(self.model_name)
        return self._actor

    async def __call__(self, prompt: str) -> tuple[str, str]:
        full_prompt = self._fmt(f"{self.system_prompt}\n\n{prompt}" if self.system_prompt else prompt)
        actor = await self._resolve_actor()
        loop = asyncio.get_event_loop()
        try:
            completion = await loop.run_in_executor(
                _executor,
                ray.get,
                actor.submit_prompt.remote(
                    full_prompt,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_tokens=self.max_tokens,
                ),
            )
        except Exception as e:
            if "VLLMValidationError" in type(e).__name__ or "input tokens" in str(e): raise ContextLimitExceeded(str(e)) from e
            raise
        return full_prompt, completion
