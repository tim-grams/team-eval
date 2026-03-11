import uuid
import random
import httpx
from abc import ABC, abstractmethod



class BaseAgent(ABC):
    def __init__(self, model_name: str, server_url: str, system_prompt: str = "",
                 temperature: float | None = None, top_p: float | None = None, top_k: int | None = None,
                 timeout: float = 300.0, name: str = ""):
        self.model_name = model_name
        self.server_url = server_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.agent_id = str(uuid.uuid4())
        self.name = name or self.agent_id[:8]

    @abstractmethod
    async def __call__(self, prompt: str) -> tuple[str, str]:
        raise NotImplementedError("Subclasses must implement this method.")


class VLLMAgent(BaseAgent):
    async def __call__(self, prompt: str) -> tuple[str, str]:
        messages = []
        if self.system_prompt: messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model_name, "messages": messages}
        if self.temperature is not None: payload["temperature"] = self.temperature
        if self.top_p is not None: payload["top_p"] = self.top_p
        if self.top_k is not None: payload["top_k"] = self.top_k
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.server_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            completion = response.json()["choices"][0]["message"]["content"]
        return prompt, completion


class OllamaAgent(BaseAgent):
    async def __call__(self, prompt: str) -> tuple[str, str]:
        messages = []
        if self.system_prompt: messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        options = {}
        if self.temperature is not None: options["temperature"] = self.temperature
        if self.top_p is not None: options["top_p"] = self.top_p
        if self.top_k is not None: options["top_k"] = self.top_k
        payload = {"model": self.model_name, "messages": messages, "stream": False}
        if options: payload["options"] = options
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.server_url}/api/chat", json=payload)
            response.raise_for_status()
            completion = response.json()["message"]["content"]
        return prompt, completion


class RandomAgent(BaseAgent):
    def __init__(self, action_space: list[str]):
        self.action_space = action_space
        self.model_name = "random"
        self.server_url = ""
        self.system_prompt = ""
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.agent_id = str(uuid.uuid4())

    async def __call__(self, prompt: str) -> tuple[str, str]:
        completion = random.choice(self.action_space)
        return prompt, completion
