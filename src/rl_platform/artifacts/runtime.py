"""
RunConfig — Pydantic schema for a full training run.
ConfigLoader — loads and validates a YAML config file into a RunConfig.

Example YAML layout (TCP/UE5 worker — used by `rlp train`)
-----------------------------------------------------------
run_name: ue5_ppo
total_steps: 500000
data_dir: runs

env:
  # env_id: omit (or null) for TCP/UE5 workers.
  # For `rlp eval` against a gym env, set env_id: CartPole-v1
  is_multi_agent: false      # M0: must be false (multi-agent is M1+)

worker:
  host: 127.0.0.1
  port: 7777
  num_workers: 1             # M0: must be 1 (multi-worker is M3+)

learner:
  type: sb3                  # key into LEARNER_REGISTRY
  params:
    algorithm: PPO
    learning_rate: 3e-4
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    verbose: 0

checkpoint:
  save_freq: 10000           # save a checkpoint every N steps
  keep_best: true

eval:
  eval_freq: 0               # M0: must be 0 — use `rlp eval` after training
  num_episodes: 5            # episodes per `rlp eval` pass
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

class EnvConfig(BaseModel):
    env_id: str | None = None
    is_multi_agent: bool = False

    @field_validator("is_multi_agent")
    @classmethod
    def _m0_single_agent_only(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "M0 only supports single-agent environments — "
                "set is_multi_agent: false (multi-agent is M1+)"
            )
        return v


class WorkerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7777
    num_workers: int = 1

    @field_validator("num_workers")
    @classmethod
    def _m0_single_worker_only(cls, v: int) -> int:
        if v != 1:
            raise ValueError(
                f"M0 only supports num_workers=1, got {v} "
                "(multi-worker is M3+)"
            )
        return v


class LearnerConfig(BaseModel):
    """
    `type` is the registry key (e.g. 'sb3') used to instantiate the plugin.
    `params` is an opaque dict passed verbatim to LearnerPlugin.configure().
    For SB3 that means {"algorithm": "PPO", "learning_rate": ..., ...}.
    """
    type: str = "sb3"
    params: dict[str, Any] = Field(default_factory=dict)


class CheckpointConfig(BaseModel):
    save_freq: int = 10_000
    keep_best: bool = True


class EvalConfig(BaseModel):
    eval_freq: int = 0
    num_episodes: int = 5

    @field_validator("eval_freq")
    @classmethod
    def _m0_no_periodic_eval(cls, v: int) -> int:
        if v != 0:
            raise ValueError(
                "Periodic eval during training (eval_freq > 0) is not supported in M0 — "
                "set eval_freq: 0 and use `rlp eval` after training (periodic eval is M3+)"
            )
        return v


class RunConfig(BaseModel):
    run_name: str
    total_steps: int
    data_dir: Path = Path("runs")

    env: EnvConfig = Field(default_factory=EnvConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    learner: LearnerConfig = Field(default_factory=LearnerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


class ConfigLoader:
    """Loads a YAML file and validates it into a RunConfig."""

    @staticmethod
    def load(path: Path | str) -> RunConfig:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(
                f"Config file must be a YAML mapping, got {type(raw).__name__}: {path}"
            )

        return RunConfig.model_validate(raw)
