from pathlib import Path

import structlog
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_platform.config import ConfigLoader
from rl_platform.env import TCPGymEnv
from rl_platform.logging_setup import setup_logging
from rl_platform.sb3 import SB3Learner

log = structlog.get_logger()


def train(config_path: Path | str) -> dict[str, float | int]:
    cfg = ConfigLoader.load(config_path)
    setup_logging()
    structlog.contextvars.bind_contextvars(run_name=cfg.run_name)

    if cfg.learner.type != "sb3":
        raise ValueError(
            f"Unknown learner type {cfg.learner.type!r}; only 'sb3' is supported"
        )

    run_dir = cfg.data_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "train_start",
        run_name=cfg.run_name,
        total_steps=cfg.total_steps,
        host=cfg.worker.host,
        port=cfg.worker.port,
    )

    env = TCPGymEnv(cfg.worker.host, cfg.worker.port)
    if env.env_spec.is_multi_agent:
        env.close()
        raise NotImplementedError("multi-agent training is M1+")

    log.info(
        "env_spec_discovered",
        env_id=env.env_spec.env_id,
        agent_ids=env.env_spec.agent_ids,
    )

    vec_env = DummyVecEnv([lambda: Monitor(env)])
    learner = SB3Learner(cfg.learner.params)
    results = learner.train(
        vec_env, cfg.total_steps, run_dir, cfg.checkpoint.save_freq
    )

    learner.save(run_dir / "model")
    log.info("train_complete", run_name=cfg.run_name, results=results)
    return results

def multi_train(config_path: Path | str) -> dict[str, float | int]:
    cfg = ConfigLoader.load(config_path)
    setup_logging()
    structlog.contextvars.bind_contextvars(run_name=cfg.run_name)

    if cfg.learner.type != "sb3":
        raise ValueError(
            f"Unknown learner type {cfg.learner.type!r}; only 'sb3' is supported"
        )

    run_dir = cfg.data_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)