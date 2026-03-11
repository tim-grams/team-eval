import os
import time
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Optional

import concurrent.futures

import ray
import torch

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=512)
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.inputs import TextPrompt
from vllm.sampling_params import SamplingParams

from src.agents.agent import BaseAgent
from src.utils.templates import OBSERVATION_FORMATTING, get_hf_formatter


@ray.remote
class VLLMActor:
    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.engine: Optional[AsyncLLMEngine] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.model_name: Optional[str] = None

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
        if self.engine is not None:
            await self.unload_model()
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

    async def unload_model(self) -> None:
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
        final_output = None
        async for output in self.engine.generate(TextPrompt(prompt=prompt), params, request_id): final_output = output
        return final_output.outputs[0].text

    def ready(self) -> bool:
        return True


@dataclass
class _Slot:
    actor: object
    model_name: Optional[str] = None
    last_used: float = field(default_factory=time.monotonic)


class VLLMActorPool:
    def __init__(self, num_slots: int, num_gpus_per_slot: int, **model_kwargs):
        self._model_kwargs = model_kwargs
        self._slots: list[_Slot] = [
            _Slot(actor=VLLMActor.options(num_gpus=num_gpus_per_slot).remote(num_gpus_per_slot))
            for _ in range(num_slots)
        ]
        self._loading: dict[str, asyncio.Event] = {}
        self._rr: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def get_actor(self, model_name: str) -> object:
        async with self._lock:
            model_slots = [s for s in self._slots if s.model_name == model_name]
            empty_slots = [s for s in self._slots if s.model_name is None]
            if empty_slots and model_name not in self._loading:
                target = empty_slots[0]
                event = asyncio.Event()
                self._loading[model_name] = event
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, self._load_slot, target, model_name, event, loop)
            if model_slots:
                idx = self._rr.get(model_name, 0) % len(model_slots)
                self._rr[model_name] = idx + 1
                model_slots[idx].last_used = time.monotonic()
                return model_slots[idx].actor
            event = self._loading[model_name]
        await event.wait()
        async with self._lock:
            model_slots = [s for s in self._slots if s.model_name == model_name]
            idx = self._rr.get(model_name, 0) % len(model_slots)
            self._rr[model_name] = idx + 1
            model_slots[idx].last_used = time.monotonic()
            return model_slots[idx].actor

    def _load_slot(
        self,
        slot: _Slot,
        model_name: str,
        event: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        try:
            ray.get(slot.actor.load_model.remote(model_name, **self._model_kwargs))
            slot.model_name = model_name
            slot.last_used = time.monotonic()
        finally:
            self._loading.pop(model_name, None)
            loop.call_soon_threadsafe(event.set)

    def status(self) -> list[dict]: return [{"slot": i, "model": s.model_name, "last_used": s.last_used} for i, s in enumerate(self._slots)]


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
        return full_prompt, completion
