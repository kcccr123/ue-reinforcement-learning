from pathlib import Path

import structlog

from rl_platform.artifacts.runtime import RunConfig, ConfigLoader
from rl_platform.artifacts.database import Database
from rl_platform.artifacts.checkpointing import CheckpointManager
from rl_platform.core.specifications import EnvSpec, TaskSpec
from rl_platform.core.protocols import LearnerPlugin, CurriculumPlugin
from rl_platform.core.learn import LearnerDriver
from rl_platform.plugins import LEARNER_REGISTRY
from rl_platform.core.curriculum import SingleTaskCurriculum
from rl_platform.infra.workers import WorkerManager, WorkerSpec, DebugWorkerBackend
from rl_platform.infra.gym_envs import TCPGymEnv
from rl_platform.utils.logging import setup_logging

log = structlog.get_logger()


class Coordinator:

    def __init__(self, config: RunConfig) -> None:
        self._config = config

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "Coordinator":
        config = ConfigLoader.load(config_path)
        return cls(config)

    def run(self) -> dict:
        cfg = self._config
        setup_logging()
        structlog.contextvars.bind_contextvars(run_name=cfg.run_name)

        log.info("coordinator_start", run_name=cfg.run_name, total_steps=cfg.total_steps)

        db = Database(cfg.data_dir / "registry.db")
        run_id = db.create_run(cfg.run_name, cfg.model_dump(mode="json"))
        db.update_run_status(run_id, "running")

        ckpt_mgr = CheckpointManager(data_dir=cfg.data_dir, database=db)

        learner = self._create_learner(cfg)
        curriculum = self._create_curriculum(cfg)

        is_tcp = cfg.env.env_id is None
        worker_mgr: WorkerManager | None = None
        worker_addrs: list[tuple[str, int]] | None = None

        try:
            if is_tcp:
                worker_mgr = self._setup_workers(cfg)
                worker_addrs = worker_mgr.get_all_addresses()
                env_spec = self._probe_tcp_env_spec(worker_addrs[0])
            else:
                raise ValueError(
                    "Coordinator only supports TCP workers in M0. "
                    "Set env.env_id to null and provide worker.host/port in your config. "
                    "For gym env evaluation use `rlp eval` instead."
                )

            log.info(
                "env_spec_discovered",
                env_id=env_spec.env_id,
                agent_ids=env_spec.agent_ids,
                is_multi_agent=env_spec.is_multi_agent,
            )

            learner.configure(env_spec, cfg.learner.params)

            driver = LearnerDriver(
                learner=learner,
                checkpoint_mgr=ckpt_mgr,
                run_id=run_id,
                checkpoint_freq=cfg.checkpoint.save_freq,
            )

            stage_results = {}
            while True:
                task = curriculum.get_task()
                log.info("curriculum_stage", stage=curriculum.current_stage, task=task.name)

                results = driver.train_stage(
                    worker_addrs=worker_addrs,
                    env_fns=None,
                    task=task,
                    total_steps=cfg.total_steps,
                )
                stage_results = results

                curriculum.report(results)
                if not curriculum.should_advance():
                    break
                curriculum.advance()

            final_ckpt_id = ckpt_mgr.save(
                run_id=run_id,
                step=int(stage_results.get("num_timesteps", 0)),
                learner=learner,
                metrics=stage_results,
                tags=["final"],
            )
            log.info("final_checkpoint_saved", ckpt_id=final_ckpt_id)

            db.update_run_status(run_id, "completed")
            log.info("coordinator_complete", run_id=run_id, results=stage_results)

            return {
                "run_id": run_id,
                "status": "completed",
                "results": stage_results,
                "final_checkpoint_id": final_ckpt_id,
            }

        except Exception:
            db.update_run_status(run_id, "failed")
            log.exception("coordinator_failed", run_id=run_id)
            raise

        finally:
            if worker_mgr is not None:
                worker_mgr.teardown_all()
            db.close()

    def _create_learner(self, cfg: RunConfig) -> LearnerPlugin:
        learner_type = cfg.learner.type
        if learner_type not in LEARNER_REGISTRY:
            raise ValueError(
                f"Unknown learner type {learner_type!r}. "
                f"Available: {list(LEARNER_REGISTRY)}"
            )
        cls = LEARNER_REGISTRY[learner_type]
        return cls()

    def _create_curriculum(self, cfg: RunConfig) -> CurriculumPlugin:
        task = TaskSpec(name=cfg.run_name)
        return SingleTaskCurriculum(task)

    def _setup_workers(self, cfg: RunConfig) -> WorkerManager:
        backend = DebugWorkerBackend()
        manager = WorkerManager(backend)
        spec = WorkerSpec(
            runtime_type="external",
            host=cfg.worker.host,
            port=cfg.worker.port,
        )
        manager.launch_workers([spec])
        log.info("workers_registered", addresses=manager.get_all_addresses())
        return manager

    def _probe_tcp_env_spec(self, addr: tuple[str, int]) -> EnvSpec:
        log.info("probing_tcp_env", host=addr[0], port=addr[1])
        probe_env = TCPGymEnv(addr[0], addr[1])
        env_spec = probe_env.env_spec
        probe_env.close()
        return env_spec
