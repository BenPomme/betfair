from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config


_VALID_MODES = {"full", "cost_saver", "hard_stop"}


def normalize_agentic_factory_mode(value: Any) -> str:
    mode = str(value or "full").strip().lower()
    if mode in _VALID_MODES:
        return mode
    return "full"


@dataclass(frozen=True)
class AgenticFactoryRuntimeMode:
    value: str = "full"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", normalize_agentic_factory_mode(self.value))

    @property
    def is_full(self) -> bool:
        return self.value == "full"

    @property
    def is_cost_saver(self) -> bool:
        return self.value == "cost_saver"

    @property
    def is_hard_stop(self) -> bool:
        return self.value == "hard_stop"

    @property
    def agentic_tokens_allowed(self) -> bool:
        return self.is_full

    @property
    def factory_influence_allowed(self) -> bool:
        return not self.is_hard_stop

    @property
    def deterministic_research_allowed(self) -> bool:
        return not self.is_hard_stop

    @property
    def process_start_allowed(self) -> bool:
        return not self.is_hard_stop

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentic_factory_mode": self.value,
            "agentic_tokens_allowed": self.agentic_tokens_allowed,
            "factory_influence_allowed": self.factory_influence_allowed,
        }


def current_agentic_factory_runtime_mode() -> AgenticFactoryRuntimeMode:
    return AgenticFactoryRuntimeMode(getattr(config, "AGENTIC_FACTORY_MODE", "full"))


def research_factory_start_blocker(portfolio_id: str) -> str | None:
    research_factory_id = str(getattr(config, "RESEARCH_FACTORY_PORTFOLIO_ID", "research_factory"))
    if portfolio_id != research_factory_id:
        return None
    if current_agentic_factory_runtime_mode().is_hard_stop:
        return "agentic_factory_hard_stopped"
    return None
