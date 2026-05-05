from dataclasses import dataclass, field
import gymnasium as gym
from typing import Any

@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    agent_ids: list[str]
    observation_spaces: dict[str, gym.Space]
    action_spaces: dict[str, gym.Space]
    is_multi_agent: bool
    max_episode_steps: int | None = None

class EnvSpecValidationError(ValueError):
    """Raised when EnvSpec validation fails."""
    pass

@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_params: dict[str, Any] = field(default_factory=dict)
    reward_config: dict[str, float] = field(default_factory=dict)
    episode_limit: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)