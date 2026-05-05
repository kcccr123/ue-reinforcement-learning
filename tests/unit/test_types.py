"""Unit tests for core dataclasses: EnvSpec, TaskSpec, EnvSpecValidationError."""
import pytest
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

from rl_platform.core.specifications import EnvSpec, TaskSpec, EnvSpecValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box() -> Box:
    return Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


def _discrete() -> Discrete:
    return Discrete(2)


# ---------------------------------------------------------------------------
# EnvSpec
# ---------------------------------------------------------------------------

class TestEnvSpec:
    def test_single_agent_construction(self):
        spec = EnvSpec(
            env_id="CartPole-v1",
            agent_ids=["agent_0"],
            observation_spaces={"agent_0": _box()},
            action_spaces={"agent_0": _discrete()},
            is_multi_agent=False,
        )
        assert spec.env_id == "CartPole-v1"
        assert spec.agent_ids == ["agent_0"]
        assert isinstance(spec.observation_spaces["agent_0"], Box)
        assert isinstance(spec.action_spaces["agent_0"], Discrete)
        assert spec.is_multi_agent is False
        assert spec.max_episode_steps is None

    def test_max_episode_steps_optional(self):
        spec = EnvSpec(
            env_id="env",
            agent_ids=["a"],
            observation_spaces={"a": _box()},
            action_spaces={"a": _discrete()},
            is_multi_agent=False,
            max_episode_steps=500,
        )
        assert spec.max_episode_steps == 500

    def test_is_multi_agent_required(self):
        """Omitting is_multi_agent must raise TypeError (no default)."""
        with pytest.raises(TypeError):
            EnvSpec(  # type: ignore[call-arg]
                env_id="env",
                agent_ids=["a"],
                observation_spaces={"a": _box()},
                action_spaces={"a": _discrete()},
            )

    def test_frozen_immutability(self):
        """EnvSpec is frozen — attribute assignment must raise."""
        spec = EnvSpec(
            env_id="env",
            agent_ids=["a"],
            observation_spaces={"a": _box()},
            action_spaces={"a": _discrete()},
            is_multi_agent=False,
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.env_id = "other"  # type: ignore[misc]

    def test_multi_agent_construction(self):
        spec = EnvSpec(
            env_id="combat",
            agent_ids=["red_0", "blue_0"],
            observation_spaces={"red_0": _box(), "blue_0": _box()},
            action_spaces={"red_0": _discrete(), "blue_0": _discrete()},
            is_multi_agent=True,
        )
        assert spec.is_multi_agent is True
        assert len(spec.agent_ids) == 2

    def test_cartpole_spaces_from_gym(self):
        """EnvSpec can hold spaces derived from a live gymnasium env."""
        env = gym.make("CartPole-v1")
        spec = EnvSpec(
            env_id="CartPole-v1",
            agent_ids=["agent_0"],
            observation_spaces={"agent_0": env.observation_space},
            action_spaces={"agent_0": env.action_space},
            is_multi_agent=False,
        )
        env.close()
        assert isinstance(spec.observation_spaces["agent_0"], Box)
        assert spec.observation_spaces["agent_0"].shape == (4,)
        assert isinstance(spec.action_spaces["agent_0"], Discrete)
        assert spec.action_spaces["agent_0"].n == 2


# ---------------------------------------------------------------------------
# EnvSpecValidationError
# ---------------------------------------------------------------------------

class TestEnvSpecValidationError:
    def test_is_value_error(self):
        err = EnvSpecValidationError("bad spec")
        assert isinstance(err, ValueError)

    def test_can_be_raised_and_caught_as_value_error(self):
        with pytest.raises(ValueError, match="mismatch"):
            raise EnvSpecValidationError("mismatch")


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------

class TestTaskSpec:
    def test_minimal_construction(self):
        """Only name is required; all other fields have defaults."""
        spec = TaskSpec(name="solo")
        assert spec.name == "solo"
        assert spec.env_params == {}
        assert spec.reward_config == {}
        assert spec.episode_limit is None
        assert spec.metadata == {}

    def test_env_params_positional(self):
        """Second positional arg should bind to env_params, not reward_config."""
        spec = TaskSpec("solo", {"num_agents": 1})
        assert spec.env_params == {"num_agents": 1}
        assert spec.reward_config == {}

    def test_all_fields(self):
        spec = TaskSpec(
            name="1v1",
            env_params={"num_agents": 2},
            reward_config={"win": 1.0, "loss": -1.0},
            episode_limit=500,
            metadata={"stage": "beginner"},
        )
        assert spec.episode_limit == 500
        assert spec.reward_config["win"] == 1.0
        assert spec.metadata["stage"] == "beginner"

    def test_frozen_immutability(self):
        spec = TaskSpec(name="solo")
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "other"  # type: ignore[misc]

    def test_env_params_defaults_are_independent(self):
        """Default factories must not share the same dict across instances."""
        a = TaskSpec(name="a")
        b = TaskSpec(name="b")
        assert a.env_params is not b.env_params
        assert a.metadata is not b.metadata
