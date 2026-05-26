from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptCase:
    name: str
    scenario: str
    messages: list[dict[str, str]]


def load_prompt_suite(path: Path) -> list[PromptCase]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} must define a non-empty prompts list")

    cases: list[PromptCase] = []
    for index, item in enumerate(prompts):
        if not isinstance(item, dict):
            raise ValueError(f"prompt #{index} must be a mapping")
        name = str(item.get("name", "")).strip()
        scenario = str(item.get("scenario", name)).strip()
        messages = _normalize_messages(item.get("messages"), name=name or f"#{index}")
        if not name:
            raise ValueError(f"prompt #{index} must define name")
        if not scenario:
            raise ValueError(f"prompt {name} must define scenario")
        cases.append(PromptCase(name=name, scenario=scenario, messages=messages))
    return cases


def _normalize_messages(raw_messages: Any, *, name: str) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"prompt {name} must define at least one message")

    messages: list[dict[str, str]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise ValueError(f"prompt {name} message #{index} must be a mapping")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"prompt {name} message #{index} has invalid role {role!r}")
        if not content:
            raise ValueError(f"prompt {name} message #{index} must define content")
        messages.append({"role": role, "content": content})
    return messages
