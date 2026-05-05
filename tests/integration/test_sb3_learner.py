"""Integration tests for SB3Plugin: CartPole-v1 training, save/load, callbacks.

Most tests use a very short training run (128-512 steps) to stay fast in CI.
The acceptance-criteria test (50 k steps, mean return > 200) is marked
@pytest.mark.slow and skipped by default.

Run the slow suite with:
    pytest -m slow tests/integration/test_sb3_learner.py
"""
import numpy as np
import pytest
import gymnasium as gym
from pathlib import Path

from rl_platform.plugins.sb3 import SB3Plugin
from rl_platform.core.protocols import InferencePolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHORT_STEPS = 256  # fast enough for any CI box


class RecordingCallback:
    """Concrete TrainingCallback that records every hook invocation."""

    def __init__(self):
        self.steps_seen: list[int] = []
        self.metrics_seen: list[dict] = []
        self.batch_ends: list[int] = []
        self.training_ended: bool = False
        self._stop_after: int | None = None

    def stop_after(self, n: int) -> "RecordingCallback":
        """Tell the callback to request a stop after n on_step calls."""
        self._stop_after = n
        return self

    def on_step(self, step: int, metrics: dict[str, float]) -> bool:
        self.steps_seen.append(step)
        self.metrics_seen.append(metrics)
        if self._stop_after is not None and len(self.steps_seen) >= self._stop_after:
            return False
        return True

    def on_batch_end(self, step: int, metrics: dict[str, float]) -> None:
        self.batch_ends.append(step)

    def on_update_end(self, step: int, metrics: dict[str, float]) -> None:
        pass  # SB3 doesn't surface this; no-op is fine

    def on_training_end(self) -> None:
        self.training_ended = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestCartPoleTraining:
    def test_train_returns_expected_keys(self, configured_plugin, cartpole_env_fn):
        results = configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            callbacks=[],
        )
        assert "num_timesteps" in results
        assert results["num_timesteps"] > 0

    def test_train_with_multiple_envs(self, cartpole_env_spec, ppo_config, cartpole_env_fn):
        """Multiple env factories should be wrapped in DummyVecEnv without error."""
        plugin = SB3Plugin()
        plugin.configure(cartpole_env_spec, ppo_config)
        results = plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn, cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            callbacks=[],
        )
        assert results["num_timesteps"] > 0

    def test_episode_metrics_present_after_episodes_complete(
        self, configured_plugin, cartpole_env_fn
    ):
        """After enough steps to finish at least one episode, reward metrics are returned."""
        # CartPole episodes end quickly; 512 steps always produces episodes.
        results = configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=512,
            callbacks=[],
        )
        assert "mean_reward" in results, (
            f"No episodes completed in 512 steps; got keys: {list(results)}"
        )
        assert "mean_length" in results
        assert "last_100_mean_reward" in results
        assert "total_episodes" in results
        assert results["total_episodes"] >= 1

    def test_episode_metrics_are_floats(self, configured_plugin, cartpole_env_fn):
        results = configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=512,
            callbacks=[],
        )
        for key in ("mean_reward", "mean_length", "last_100_mean_reward"):
            if key in results:
                assert isinstance(results[key], float), (
                    f"{key} should be float, got {type(results[key])}"
                )

    def test_reconfigure_and_retrain(self, cartpole_env_spec, ppo_config, cartpole_env_fn):
        """Calling configure() again must allow a fresh train() with a clean slate."""
        plugin = SB3Plugin()
        plugin.configure(cartpole_env_spec, ppo_config)
        plugin.train(
            worker_addrs=None, env_fns=[cartpole_env_fn],
            task=None, total_steps=SHORT_STEPS, callbacks=[],
        )
        plugin.configure(cartpole_env_spec, ppo_config)  # re-configure
        assert plugin._trained_model is None
        results = plugin.train(
            worker_addrs=None, env_fns=[cartpole_env_fn],
            task=None, total_steps=SHORT_STEPS, callbacks=[],
        )
        assert results["num_timesteps"] > 0


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_on_step_fires_during_training(self, configured_plugin, cartpole_env_fn):
        cb = RecordingCallback()
        configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            callbacks=[cb],
        )
        assert len(cb.steps_seen) > 0, "on_step was never called"

    def test_on_batch_end_fires(self, configured_plugin, cartpole_env_fn):
        cb = RecordingCallback()
        configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            callbacks=[cb],
        )
        assert len(cb.batch_ends) > 0, "on_batch_end was never called"

    def test_on_training_end_fires(self, configured_plugin, cartpole_env_fn):
        cb = RecordingCallback()
        configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            callbacks=[cb],
        )
        assert cb.training_ended, "on_training_end was never called"

    def test_on_step_step_values_are_ints(self, configured_plugin, cartpole_env_fn):
        cb = RecordingCallback()
        configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            callbacks=[cb],
        )
        for step in cb.steps_seen:
            assert isinstance(step, int), f"step should be int, got {type(step)}"

    def test_on_step_metrics_values_are_float(self, configured_plugin, cartpole_env_fn):
        """When an episode ends inside a step, emitted metrics must be float."""
        cb = RecordingCallback()
        configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=512,
            callbacks=[cb],
        )
        for metrics in cb.metrics_seen:
            for key, value in metrics.items():
                assert isinstance(value, float), (
                    f"metrics[{key!r}] should be float, got {type(value)}"
                )

    def test_callback_can_stop_training_early(self, configured_plugin, cartpole_env_fn):
        """Returning False from on_step should halt training before total_steps."""
        stop_at = 3
        cb = RecordingCallback().stop_after(stop_at)
        configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=10_000,  # would take a while if not stopped
            callbacks=[cb],
        )
        # SB3 stops at the next rollout boundary, so allow a small margin
        assert configured_plugin._trained_model.num_timesteps < 10_000, (
            "Training did not stop early despite callback returning False"
        )

    def test_multiple_callbacks_all_called(self, configured_plugin, cartpole_env_fn):
        cb1 = RecordingCallback()
        cb2 = RecordingCallback()
        configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            callbacks=[cb1, cb2],
        )
        assert cb1.training_ended
        assert cb2.training_ended
        assert len(cb1.steps_seen) > 0
        assert len(cb2.steps_seen) > 0


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def _train_short(self, plugin, env_fn):
        plugin.train(
            worker_addrs=None,
            env_fns=[env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            callbacks=[],
        )

    def test_save_returns_metadata(self, configured_plugin, cartpole_env_fn, tmp_path):
        self._train_short(configured_plugin, cartpole_env_fn)
        meta = configured_plugin.save(tmp_path / "model")
        assert "algorithm" in meta
        assert "step" in meta
        assert meta["algorithm"] == "PPO"
        assert meta["step"] > 0

    def test_save_creates_file(self, configured_plugin, cartpole_env_fn, tmp_path):
        self._train_short(configured_plugin, cartpole_env_fn)
        configured_plugin.save(tmp_path / "model")
        # SB3 saves as <name>.zip
        assert (tmp_path / "model.zip").exists()

    def test_load_restores_model(
        self, cartpole_env_spec, ppo_config, cartpole_env_fn, tmp_path
    ):
        plugin_a = SB3Plugin()
        plugin_a.configure(cartpole_env_spec, ppo_config)
        self._train_short(plugin_a, cartpole_env_fn)
        plugin_a.save(tmp_path / "model")

        plugin_b = SB3Plugin()
        plugin_b.configure(cartpole_env_spec, ppo_config)
        plugin_b.load(tmp_path / "model.zip")
        # After load, get_policy should work without raising
        policy = plugin_b.get_policy()
        assert policy is not None

    def test_loaded_policy_predicts_valid_actions(
        self, cartpole_env_spec, ppo_config, cartpole_env_fn, tmp_path
    ):
        plugin = SB3Plugin()
        plugin.configure(cartpole_env_spec, ppo_config)
        self._train_short(plugin, cartpole_env_fn)
        plugin.save(tmp_path / "model")

        plugin2 = SB3Plugin()
        plugin2.configure(cartpole_env_spec, ppo_config)
        plugin2.load(tmp_path / "model.zip")
        policy = plugin2.get_policy()

        env = gym.make("CartPole-v1")
        obs, _ = env.reset()
        actions = policy.get_actions(obs[np.newaxis, :])  # batch of 1
        env.close()

        assert actions.shape == (1,)
        assert actions[0] in (0, 1)

    def test_save_before_train_raises(self, configured_plugin, tmp_path):
        with pytest.raises(RuntimeError):
            configured_plugin.save(tmp_path / "model")


# ---------------------------------------------------------------------------
# InferencePolicy
# ---------------------------------------------------------------------------

class TestInferencePolicy:
    def _trained_plugin(self, cartpole_env_spec, ppo_config, cartpole_env_fn):
        plugin = SB3Plugin()
        plugin.configure(cartpole_env_spec, ppo_config)
        plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=SHORT_STEPS,
            callbacks=[],
        )
        return plugin

    def test_get_policy_returns_inference_policy(
        self, cartpole_env_spec, ppo_config, cartpole_env_fn
    ):
        plugin = self._trained_plugin(cartpole_env_spec, ppo_config, cartpole_env_fn)
        policy = plugin.get_policy()
        assert callable(getattr(policy, "get_actions", None)), (
            "get_policy() result must have a callable get_actions()"
        )

    def test_get_actions_correct_shape_single(
        self, cartpole_env_spec, ppo_config, cartpole_env_fn
    ):
        plugin = self._trained_plugin(cartpole_env_spec, ppo_config, cartpole_env_fn)
        policy = plugin.get_policy()

        env = gym.make("CartPole-v1")
        obs, _ = env.reset()
        env.close()

        actions = policy.get_actions(obs[np.newaxis, :])  # (1, 4) -> (1,)
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (1,)

    def test_get_actions_correct_shape_batch(
        self, cartpole_env_spec, ppo_config, cartpole_env_fn
    ):
        plugin = self._trained_plugin(cartpole_env_spec, ppo_config, cartpole_env_fn)
        policy = plugin.get_policy()

        batch_obs = np.random.randn(8, 4).astype(np.float32)
        actions = policy.get_actions(batch_obs)
        assert actions.shape == (8,)

    def test_get_actions_valid_discrete_values(
        self, cartpole_env_spec, ppo_config, cartpole_env_fn
    ):
        plugin = self._trained_plugin(cartpole_env_spec, ppo_config, cartpole_env_fn)
        policy = plugin.get_policy()
        obs = np.random.randn(16, 4).astype(np.float32)
        actions = policy.get_actions(obs)
        assert set(actions.tolist()).issubset({0, 1}), (
            f"Actions outside CartPole action space: {set(actions.tolist())}"
        )

    def test_multiple_get_policy_calls_work(
        self, cartpole_env_spec, ppo_config, cartpole_env_fn
    ):
        """EvalDriver re-fetches policy before each eval pass; this must not break."""
        plugin = self._trained_plugin(cartpole_env_spec, ppo_config, cartpole_env_fn)
        p1 = plugin.get_policy()
        p2 = plugin.get_policy()
        obs = np.random.randn(4, 4).astype(np.float32)
        np.testing.assert_array_equal(p1.get_actions(obs), p2.get_actions(obs))


# ---------------------------------------------------------------------------
# M0.1 acceptance criterion (slow — skipped in regular CI)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cartpole_50k_mean_return_above_200(cartpole_env_fn):
    """M0.1 acceptance: PPO on CartPole-v1 for 50k steps, mean return > 200."""
    env = gym.make("CartPole-v1")
    spec = __import__(
        "rl_platform.core.specifications", fromlist=["EnvSpec"]
    ).EnvSpec(
        env_id="CartPole-v1",
        agent_ids=["agent_0"],
        observation_spaces={"agent_0": env.observation_space},
        action_spaces={"agent_0": env.action_space},
        is_multi_agent=False,
    )
    env.close()

    config = {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "verbose": 0,
    }

    plugin = SB3Plugin()
    plugin.configure(spec, config)
    results = plugin.train(
        worker_addrs=None,
        env_fns=[cartpole_env_fn],
        task=None,
        total_steps=50_000,
        callbacks=[],
    )

    assert "mean_reward" in results, "No episodes recorded during 50k-step run"
    assert results["mean_reward"] > 200, (
        f"M0.1 acceptance failed: mean_reward={results['mean_reward']:.1f}, expected > 200"
    )
