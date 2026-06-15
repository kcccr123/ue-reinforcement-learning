from pathlib import Path
from typing import Any

from rl_platform.artifacts.database import Database
from rl_platform.core.protocols import LearnerPlugin


class CheckpointManager:

    def __init__(self, data_dir: Path, database: Database) -> None:
        self.data_dir = Path(data_dir)
        self.database = database

    def save(
        self,
        run_id: str,
        step: int,
        learner: LearnerPlugin,
        metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
    ) -> str:

        ckpt_dir = self._checkpoint_dir(run_id, step)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_path = ckpt_dir / "model"
        learner.save(model_path)

        zip_path = Path(str(model_path) + ".zip")
        path_to_record = str(zip_path) if zip_path.exists() else str(model_path)

        ckpt_id = self.database.add_checkpoint(
            run_id=run_id,
            step=step,
            path=path_to_record,
            metrics=metrics,
            tags=tags,
        )
        return ckpt_id

    def load(self, run_id: str, checkpoint_id: str) -> Path:
        all_checkpoints = self.database.get_checkpoints(run_id)
        if all_checkpoints is None:
            raise KeyError(f"Checkpoint '{checkpoint_id}' not found in run '{run_id}'")

        for checkpoint in all_checkpoints:
            if checkpoint["id"] == checkpoint_id:
                return Path(checkpoint["path"])
        raise KeyError(f"Checkpoint '{checkpoint_id}' not found in run '{run_id}'")

    def get_latest(self, run_id: str) -> Path | None:
        """Return the path of the most recent checkpoint for a run, or None."""
        record = self.database.get_latest_checkpoint(run_id)
        if record is None:
            return None
        return Path(record["path"])

    def get_best(
        self, run_id: str, metric: str, higher_is_better: bool = True
    ) -> Path | None:
        """Return the path of the checkpoint with the best value for a metric."""
        record = self.database.get_best_checkpoint(run_id, metric, higher_is_better)
        if record is None:
            return None
        return Path(record["path"])

    def _checkpoint_dir(self, run_id: str, step: int) -> Path:
        return self.data_dir / run_id / f"step_{step}"
