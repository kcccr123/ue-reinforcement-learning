"""Integration tests for LearnerDriver.

Tests that LearnerDriver:
- runs train_stage end-to-end with CartPole (gym path)
- fires checkpoints at the configured frequency
- passes extra_callbacks through to the learner
"""
import gymnasium as gym
import pytest
from pathlib import Path

from rl_platform.artifacts.database import Database
from rl_platform.artifacts.checkpointing import CheckpointManager
from rl_platform.core.learn import LearnerDriver
from rl_platform.core.specifications import EnvSpec
from rl_platform.plugins.sb3 import SB3Plugin


SHORT_STEPS = 256


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "registry.db")
    yield database
    database.close()


@pytest.fixture
def run_id(db):
    return db.create_run("driver_test_run", {})


@pytest.fixture
def ckpt_mgr(db, tmp_path):
    return CheckpointManager(data_dir=tmp_path / "data", database=db)


@pytest.fixture
def cartpole_env_spec():
    env = gym.make("CartPole-v1")
    spec = EnvSpec(
        env_id="CartPole-v1",
        agent_ids=["agent_0"],
        observation_spaces={"agent_0": env.observation_space},
        action_spaces={"agent_0": env.action_space},
        is_multi_agent=False,
    )
    env.close()
    return spec


@pytest.fixture
def ppo_config():
    return {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "verbose": 0,
    }


@pytest.fixture
def configured_learner(cartpole_env_spec, ppo_config):
    learner = SB3Plugin()
    learner.configure(cartpole_env_spec, ppo_config)
    return learner


def cartpole_env_fn():
    return gym.make("CartPole-v1")


# ---------------------------------------------------------------------------
# Basic train_stage
# ---------------------------------------------------------------------------

class TestLearnerDriverTrainStage:
    def test_train_stage_returns_dict(
        self, configured_learner, ckpt_mgr, run_id
    ):
        driver = LearnerDriver(
            learner=configured_learner,
            checkpoint_mgr=ckpt_mgr,
            run_id=run_id,
            checkpoint_freq=0,
        )
        results = driver.train_stage(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
        )
        assert isinstance(results, dict)

    def test_train_stage_results_contain_num_timesteps(
        self, configured_learner, ckpt_mgr, run_id
    ):
        driver = LearnerDriver(
            learner=configured_learner,
            checkpoint_mgr=ckpt_mgr,
            run_id=run_id,
            checkpoint_freq=0,
        )
        results = driver.train_stage(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
        )
        assert "num_timesteps" in results
        assert results["num_timesteps"] > 0


# ---------------------------------------------------------------------------
# Checkpointing via LearnerDriver
# ---------------------------------------------------------------------------

class TestLearnerDriverCheckpointing:
    def test_checkpoint_saved_when_freq_reached(
        self, configured_learner, ckpt_mgr, db, run_id
    ):
        """With checkpoint_freq=64, a 256-step run should produce checkpoints."""
        driver = LearnerDriver(
            learner=configured_learner,
            checkpoint_mgr=ckpt_mgr,
            run_id=run_id,
            checkpoint_freq=64,
        )
        driver.train_stage(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
        )
        ckpts = db.get_checkpoints(run_id)
        assert len(ckpts) >= 1

    def test_no_checkpoint_when_freq_is_zero(
        self, configured_learner, ckpt_mgr, db, run_id
    ):
        """checkpoint_freq=0 means disabled; no periodic checkpoints saved."""
        driver = LearnerDriver(
            learner=configured_learner,
            checkpoint_mgr=ckpt_mgr,
            run_id=run_id,
            checkpoint_freq=0,
        )
        driver.train_stage(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
        )
        ckpts = db.get_checkpoints(run_id)
        assert len(ckpts) == 0

    def test_checkpoint_files_exist_on_disk(
        self, configured_learner, ckpt_mgr, db, run_id
    ):
        driver = LearnerDriver(
            learner=configured_learner,
            checkpoint_mgr=ckpt_mgr,
            run_id=run_id,
            checkpoint_freq=64,
        )
        driver.train_stage(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
        )
        ckpts = db.get_checkpoints(run_id)
        for ckpt in ckpts:
            model_path = Path(ckpt["path"])
            assert model_path.parent.exists(), (
                f"Checkpoint directory missing: {model_path.parent}"
            )


# ---------------------------------------------------------------------------
# Periodic eval guard (M0: eval_freq > 0 rejected at config level)
# ---------------------------------------------------------------------------

class TestEvalConfigGuard:
    def test_eval_freq_nonzero_rejected_by_config(self):
        """Periodic eval during training is deferred to M3+; config must reject it."""
        from pydantic import ValidationError
        from rl_platform.artifacts.runtime import EvalConfig
        with pytest.raises(ValidationError, match="eval_freq"):
            EvalConfig(eval_freq=100)


# ---------------------------------------------------------------------------
# Extra callbacks are forwarded
# ---------------------------------------------------------------------------

class _RecordingCallback:
    def __init__(self):
        self.steps = []
        self.training_ended = False

    def on_step(self, step, metrics):
        self.steps.append(step)
        return True

    def on_batch_end(self, step, metrics):
        pass

    def on_update_end(self, step, metrics):
        pass

    def on_training_end(self):
        self.training_ended = True


class TestLearnerDriverExtraCallbacks:
    def test_extra_callback_receives_steps(
        self, configured_learner, ckpt_mgr, run_id
    ):
        driver = LearnerDriver(
            learner=configured_learner,
            checkpoint_mgr=ckpt_mgr,
            run_id=run_id,
            checkpoint_freq=0,
        )
        cb = _RecordingCallback()
        driver.train_stage(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            extra_callbacks=[cb],
        )
        assert len(cb.steps) > 0

    def test_extra_callback_training_ended_fires(
        self, configured_learner, ckpt_mgr, run_id
    ):
        driver = LearnerDriver(
            learner=configured_learner,
            checkpoint_mgr=ckpt_mgr,
            run_id=run_id,
            checkpoint_freq=0,
        )
        cb = _RecordingCallback()
        driver.train_stage(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            extra_callbacks=[cb],
        )
        assert cb.training_ended
