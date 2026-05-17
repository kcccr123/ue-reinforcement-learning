"""Integration tests for TCPGymEnv against MockTcpServer.

Tests the full wire protocol path: handshake, reset, step, close.
"""
import threading
import time
import pytest
import numpy as np
import gymnasium as gym

from rl_platform.infra.gym_envs import TCPGymEnv
from tests.helpers.mock_tcp_server import MockTcpServer


SERVER_IP = "127.0.0.1"
BASE_PORT = 19100


def _get_port(request):
    """Derive a unique port from the test's node id to avoid collisions."""
    return BASE_PORT + hash(request.node.nodeid) % 500


@pytest.fixture
def server_and_env(request):
    """Start a MockTcpServer in a background thread and yield a connected TCPGymEnv."""
    port = _get_port(request)
    server = MockTcpServer(SERVER_IP, port)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.1)

    env = TCPGymEnv(SERVER_IP, port)
    yield server, env

    env.close()
    thread.join(timeout=2.0)


@pytest.fixture
def tcp_env(server_and_env):
    """Just the env, for tests that don't need direct server access."""
    _, env = server_and_env
    return env


# ---------------------------------------------------------------------------
# Handshake / Construction
# ---------------------------------------------------------------------------

class TestHandshake:
    def test_env_spec_populated(self, tcp_env):
        assert tcp_env.env_spec is not None
        assert tcp_env.env_spec.env_id == "test_env"
        assert tcp_env.agent_id == "agent_1"

    def test_observation_space_is_box(self, tcp_env):
        assert isinstance(tcp_env.observation_space, gym.spaces.Box)
        assert tcp_env.observation_space.shape == (4,)

    def test_action_space_is_box(self, tcp_env):
        assert isinstance(tcp_env.action_space, gym.spaces.Box)
        assert tcp_env.action_space.shape == (1,)

    def test_single_agent(self, tcp_env):
        assert tcp_env.env_spec.is_multi_agent is False


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_and_info(self, tcp_env):
        obs, info = tcp_env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_obs_shape(self, tcp_env):
        obs, _ = tcp_env.reset()
        assert obs.shape == (4,)

    def test_reset_obs_dtype(self, tcp_env):
        obs, _ = tcp_env.reset()
        assert obs.dtype == np.float32

    def test_multiple_resets(self, tcp_env):
        for _ in range(3):
            obs, info = tcp_env.reset()
            assert obs.shape == (4,)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_5_tuple(self, tcp_env):
        tcp_env.reset()
        action = tcp_env.action_space.sample()
        result = tcp_env.step(action)
        assert len(result) == 5

    def test_step_obs_shape(self, tcp_env):
        tcp_env.reset()
        action = tcp_env.action_space.sample()
        obs, reward, terminated, truncated, info = tcp_env.step(action)
        assert obs.shape == (4,)

    def test_step_reward_is_float(self, tcp_env):
        tcp_env.reset()
        action = tcp_env.action_space.sample()
        _, reward, _, _, _ = tcp_env.step(action)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self, tcp_env):
        tcp_env.reset()
        action = tcp_env.action_space.sample()
        _, _, terminated, _, _ = tcp_env.step(action)
        assert isinstance(terminated, (bool, np.bool_))

    def test_step_truncated_is_bool(self, tcp_env):
        tcp_env.reset()
        action = tcp_env.action_space.sample()
        _, _, _, truncated, _ = tcp_env.step(action)
        assert isinstance(truncated, (bool, np.bool_))

    def test_step_info_is_dict(self, tcp_env):
        tcp_env.reset()
        action = tcp_env.action_space.sample()
        _, _, _, _, info = tcp_env.step(action)
        assert isinstance(info, dict)

    def test_multiple_steps(self, tcp_env):
        tcp_env.reset()
        for _ in range(10):
            action = tcp_env.action_space.sample()
            obs, reward, terminated, truncated, info = tcp_env.step(action)
            assert obs.shape == (4,)
            if terminated or truncated:
                tcp_env.reset()


# ---------------------------------------------------------------------------
# Episode lifecycle (reset on done)
# ---------------------------------------------------------------------------

class TestEpisodeLifecycle:
    def test_episode_terminates_eventually(self, tcp_env):
        """CartPole episodes end within 500 steps (or sooner with random actions)."""
        tcp_env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            action = tcp_env.action_space.sample()
            _, _, terminated, truncated, _ = tcp_env.step(action)
            done = terminated or truncated
            steps += 1
        assert done, f"Episode did not terminate within 500 steps"

    def test_reset_after_done_works(self, tcp_env):
        tcp_env.reset()
        done = False
        while not done:
            action = tcp_env.action_space.sample()
            _, _, terminated, truncated, _ = tcp_env.step(action)
            done = terminated or truncated
        obs, info = tcp_env.reset()
        assert obs.shape == (4,)

    def test_multiple_episodes(self, tcp_env):
        for episode in range(3):
            tcp_env.reset()
            done = False
            steps = 0
            while not done and steps < 500:
                action = tcp_env.action_space.sample()
                _, _, terminated, truncated, _ = tcp_env.step(action)
                done = terminated or truncated
                steps += 1
            assert done, f"Episode {episode} did not terminate"
