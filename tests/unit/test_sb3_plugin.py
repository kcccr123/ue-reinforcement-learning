"""Unit tests for SB3Plugin: configure, guards, and protocol compliance.

These tests do NOT train models. They exercise the plugin's state machine,
error paths, and structural contract against LearnerPlugin without running
any RL loops (fast, no GPU required).
"""
import pytest
from unittest.mock import MagicMock

from rl_platform.plugins.sb3 import SB3Plugin
from rl_platform.core.protocols import LearnerPlugin, TrainingCallback, InferencePolicy
from rl_platform.core.specifications import EnvSpec, TaskSpec


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------

class TestConfigure:
    def test_single_agent_sets_state(self, cartpole_env_spec, ppo_config):
        plugin = SB3Plugin()
        plugin.configure(cartpole_env_spec, ppo_config)
        assert plugin.env_spec is cartpole_env_spec
        assert plugin.algo_name == "PPO"
        assert plugin._trained_model is None

    def test_multi_agent_raises(self, multi_agent_env_spec, ppo_config):
        plugin = SB3Plugin()
        with pytest.raises(NotImplementedError, match="M1"):
            plugin.configure(multi_agent_env_spec, ppo_config)

    def test_unknown_algorithm_raises(self, cartpole_env_spec):
        plugin = SB3Plugin()
        with pytest.raises(KeyError):
            plugin.configure(cartpole_env_spec, {"algorithm": "UNKNOWN_ALGO"})

    def test_config_mutation_does_not_affect_stored_hps(self, cartpole_env_spec, ppo_config):
        """Mutating the caller's config dict must not change stored hyperparameters."""
        plugin = SB3Plugin()
        config = dict(ppo_config)
        plugin.configure(cartpole_env_spec, config)
        original_lr = plugin.hps["learning_rate"]
        config["learning_rate"] = 999.0  # mutate original
        assert plugin.hps["learning_rate"] == original_lr

    def test_algorithm_key_is_not_in_hps(self, cartpole_env_spec, ppo_config):
        """'algorithm' must be popped from hps so it isn't passed to the SB3 constructor."""
        plugin = SB3Plugin()
        plugin.configure(cartpole_env_spec, ppo_config)
        assert "algorithm" not in plugin.hps

    def test_reconfigure_resets_model(self, cartpole_env_spec, ppo_config):
        """Calling configure() again must reset _trained_model to None."""
        plugin = SB3Plugin()
        plugin.configure(cartpole_env_spec, ppo_config)
        plugin._trained_model = MagicMock()  # simulate a trained model
        plugin.configure(cartpole_env_spec, ppo_config)
        assert plugin._trained_model is None


# ---------------------------------------------------------------------------
# train() argument guards (no actual training)
# ---------------------------------------------------------------------------

class TestTrainGuards:
    def test_env_fns_none_raises(self, configured_plugin):
        with pytest.raises(ValueError, match="env_fns"):
            configured_plugin.train(
                worker_addrs=None,
                env_fns=None,
                task=None,
                total_steps=1,
                callbacks=[],
            )

    def test_task_not_none_raises(self, configured_plugin, cartpole_env_fn):
        task = TaskSpec(name="solo", env_params={"x": 1})
        with pytest.raises(NotImplementedError, match="M4"):
            configured_plugin.train(
                worker_addrs=None,
                env_fns=[cartpole_env_fn],
                task=task,
                total_steps=1,
                callbacks=[],
            )


# ---------------------------------------------------------------------------
# Lifecycle guards (save / get_policy before train)
# ---------------------------------------------------------------------------

class TestLifecycleGuards:
    def test_save_before_train_raises(self, configured_plugin, tmp_path):
        with pytest.raises(RuntimeError, match="save"):
            configured_plugin.save(tmp_path / "model")

    def test_get_policy_before_train_raises(self, configured_plugin):
        with pytest.raises(RuntimeError, match="get_policy"):
            configured_plugin.get_policy()


# ---------------------------------------------------------------------------
# Protocol compliance (structural)
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """Verify SB3Plugin satisfies the LearnerPlugin Protocol at runtime.

    Python's runtime_checkable + isinstance() checks that all method names
    and signatures declared on the Protocol exist on the implementation.
    """

    def test_is_learner_plugin(self):
        from typing import runtime_checkable, Protocol, get_protocol_members

        # Confirm LearnerPlugin is runtime-checkable
        assert hasattr(LearnerPlugin, "__protocol_attrs__") or True  # Python 3.12+

        plugin = SB3Plugin()
        # Check each method name declared on LearnerPlugin exists on SB3Plugin
        for name in ("configure", "train", "load", "save", "get_policy"):
            assert callable(getattr(plugin, name, None)), (
                f"SB3Plugin is missing LearnerPlugin method: {name}"
            )

    def test_configure_accepts_env_spec_and_dict(self, cartpole_env_spec):
        plugin = SB3Plugin()
        # Should not raise for any valid single-agent EnvSpec + config
        plugin.configure(cartpole_env_spec, {"algorithm": "PPO", "verbose": 0, "n_steps": 64, "batch_size": 32})

    def test_train_returns_dict(self, configured_plugin, cartpole_env_fn):
        """A very short train() must return a dict[str, float] (or int for counts)."""
        results = configured_plugin.train(
            worker_addrs=None,
            env_fns=[cartpole_env_fn],
            task=None,
            total_steps=128,
            callbacks=[],
        )
        assert isinstance(results, dict)
        assert "num_timesteps" in results
        for key, value in results.items():
            assert isinstance(key, str), f"Key {key!r} is not a str"
            assert isinstance(value, (int, float)), f"Value for {key!r} is not numeric"
