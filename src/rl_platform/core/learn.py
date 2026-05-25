from typing import Callable

import gymnasium as gym
import structlog

from rl_platform.core.protocols import LearnerPlugin, TrainingCallback
from rl_platform.core.specifications import TaskSpec
from rl_platform.artifacts.checkpointing import CheckpointManager

log = structlog.get_logger()

class _PlatformCallback:

    def __init__(
        self,
        run_id: str,
        learner: LearnerPlugin,
        checkpoint_mgr: CheckpointManager,
        checkpoint_freq: int,
    ) -> None:
        self._run_id = run_id
        self._learner = learner
        self._ckpt_mgr = checkpoint_mgr
        self._ckpt_freq = checkpoint_freq
        self._last_ckpt_step = 0

    def on_step(self, step: int, metrics: dict[str, float]) -> bool:
        if self._ckpt_freq > 0 and (step - self._last_ckpt_step) >= self._ckpt_freq:
            self._save_checkpoint(step, metrics)
            self._last_ckpt_step = step
        return True

    def on_batch_end(self, step: int, metrics: dict[str, float]) -> None:
        pass

    def on_update_end(self, step: int, metrics: dict[str, float]) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def _save_checkpoint(self, step: int, metrics: dict[str, float]) -> None:
        ckpt_id = self._ckpt_mgr.save(
            run_id=self._run_id,
            step=step,
            learner=self._learner,
            metrics=metrics if metrics else None,
        )
        log.info("checkpoint_saved", run_id=self._run_id, step=step, ckpt_id=ckpt_id)


class LearnerDriver:

    def __init__(
        self,
        learner: LearnerPlugin,
        checkpoint_mgr: CheckpointManager,
        run_id: str,
        checkpoint_freq: int = 10_000,
    ) -> None:
        self._learner = learner
        self._ckpt_mgr = checkpoint_mgr
        self._run_id = run_id
        self._ckpt_freq = checkpoint_freq

    def train_stage(
        self,
        worker_addrs: list[tuple[str, int]] | None,
        env_fns: list[Callable[[], gym.Env]] | None = None,
        task: TaskSpec | None = None,
        total_steps: int = 0,
        extra_callbacks: list[TrainingCallback] | None = None,
    ) -> dict[str, float]:

        platform_cb = _PlatformCallback(
            run_id=self._run_id,
            learner=self._learner,
            checkpoint_mgr=self._ckpt_mgr,
            checkpoint_freq=self._ckpt_freq,
        )

        callbacks: list[TrainingCallback] = [platform_cb]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        log.info(
            "train_stage_start",
            run_id=self._run_id,
            total_steps=total_steps,
            has_worker_addrs=worker_addrs is not None,
            has_env_fns=env_fns is not None,
        )

        results = self._learner.train(
            worker_addrs=worker_addrs,
            env_fns=env_fns,
            task=task,
            total_steps=total_steps,
            callbacks=callbacks,
        )

        log.info("train_stage_complete", run_id=self._run_id, results=results)
        return results
