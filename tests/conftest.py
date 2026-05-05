"""Shared fixtures for M0.1 tests."""
import gymnasium as gym
import pytest

from rl_platform.core.specifications import EnvSpec, TaskSpec


# ---------------------------------------------------------------------------
# Env / spec helpers
# ---------------------------------------------------------------------------

def _make_cartpole_env() -> gym.Env:
    """Factory producing a fresh CartPole-v1 env. Used as an env_fn callable."""
    return gym.make("CartPole-v1")


@pytest.fixture
def cartpole_env_fn():
    """Single env-factory callable for CartPole-v1."""
    return _make_cartpole_env


@pytest.fixture
def cartpole_env_spec() -> EnvSpec:
    """EnvSpec derived from CartPole-v1 spaces (single-agent)."""
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    act_space = env.action_space
    env.close()
    return EnvSpec(
        env_id="CartPole-v1",
        agent_ids=["agent_0"],
        observation_spaces={"agent_0": obs_space},
        action_spaces={"agent_0": act_space},
        is_multi_agent=False,
    )


@pytest.fixture
def multi_agent_env_spec() -> EnvSpec:
    """Minimal multi-agent EnvSpec for guard testing."""
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    act_space = env.action_space
    env.close()
    return EnvSpec(
        env_id="CartPole-v1",
        agent_ids=["agent_0", "agent_1"],
        observation_spaces={"agent_0": obs_space, "agent_1": obs_space},
        action_spaces={"agent_0": act_space, "agent_1": act_space},
        is_multi_agent=True,
    )


# ---------------------------------------------------------------------------
# Plugin config helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def ppo_config() -> dict:
    """Minimal PPO config that runs quickly (used with DummyVecEnv)."""
    return {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "verbose": 0,
    }


@pytest.fixture
def configured_plugin(cartpole_env_spec, ppo_config):
    """SB3Plugin already past configure(); ready for train()."""
    from rl_platform.plugins.sb3 import SB3Plugin
    plugin = SB3Plugin()
    plugin.configure(cartpole_env_spec, ppo_config)
    return plugin
