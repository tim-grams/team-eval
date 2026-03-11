import re
from typing import Callable, Optional


TEMPLATE_PARTS: dict[str, dict] = {
    "default": {
        "user": lambda obs: f"<|im_start|>user\n{obs}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n",
    },
    "llama": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>",
        "user": lambda obs: f"<|start_header_id|>user<|end_header_id|>\n\n{obs}<|eot_id|>\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>",
    },
    "gemma": {
        "user": lambda obs: f"<bos><start_of_turn>user\n{obs}<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n",
    },
}


def apply_template(template_name: str, observation: str) -> str:
    parts = TEMPLATE_PARTS[template_name]
    system = parts.get("system", "")
    user = parts["user"](observation)
    assistant = parts.get("assistant", "")
    return f"{system}{user}{assistant}"


def extract_boxed_action(raw: str) -> tuple[str, dict]:
    """Extract action from \\boxed{} and report whether format was correct."""
    matches = re.findall(r"\\boxed\{(.*?)\}", raw)
    if matches:
        last = matches[-1].strip()
        if last:
            action = last if "[" in last else f"[{last}]"
            return action, {"correct_answer_format": True}
    return raw, {"correct_answer_format": False}


OBSERVATION_FORMATTING: dict[str, Callable[[str], str]] = {
    name: (lambda n=name: lambda obs: apply_template(n, obs))()
    for name in TEMPLATE_PARTS
}


def get_hf_formatter(model_name: str, system_prompt: Optional[str] = None) -> Callable[[str], str]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def fmt(observation: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": observation})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return fmt