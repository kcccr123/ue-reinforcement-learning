from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class WorkerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7777


class LearnerConfig(BaseModel):
    type: str = "sb3"
    params: dict[str, Any] = Field(default_factory=dict)


class CheckpointConfig(BaseModel):
    save_freq: int = 10_000


class RunConfig(BaseModel):
    run_name: str
    total_steps: int
    data_dir: Path = Path("runs")

    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    learner: LearnerConfig = Field(default_factory=LearnerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


class ConfigLoader:
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
