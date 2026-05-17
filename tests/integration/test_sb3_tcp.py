"""Integration tests for SB3Plugin training via worker_addrs (TCP path).

Proves that the same SB3 training loop works through TCPGymEnv + MockTcpServer
as it does with direct env_fns.
"""
import threading
import time
import pytest
import numpy as np
import gymnasium as gym

from rl_platform.plugins.sb3 import SB3Plugin
from rl_platform.core.specifications import EnvSpec
from tests.helpers.mock_tcp_server import MockTcpServer


SERVER_IP = "127.0.0.1"
BASE_PORT = 19700


def _get_port(request):
    return BASE_PORT + hash(request.node.nodeid) % 500


@pytest.fixture
def tcp_server(request):
    """Start a MockTcpServer in a daemon thread, return (server, port)."""
    port = _get_port(request)
    server = MockTcpServer(SERVER_IP, port)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield server, port
    thread.join(timeout=2.0)


@pytest.fixture
def tcp_env_spec():
    """EnvSpec matching MockTcpServer's handshake (agent_1, obs=4, act=1)."""
    return EnvSpec(
        env_id="test_env",
        agent_ids=["agent_1"],
        observation_spaces={"agent_1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))},
        action_spaces={"agent_1": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))},
        is_multi_agent=False,
    )


@pytest.fixture
def tcp_ppo_config():
    return {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "verbose": 0,
    }


# ---------------------------------------------------------------------------
# Training via worker_addrs
# ---------------------------------------------------------------------------

class TestSB3TcpTraining:
    def test_train_with_worker_addrs_returns_results(
        self, tcp_server, tcp_env_spec, tcp_ppo_config
    ):
        _, port = tcp_server
        plugin = SB3Plugin()
        plugin.configure(tcp_env_spec, tcp_ppo_config)
        results = plugin.train(
            worker_addrs=[(SERVER_IP, port)],
            env_fns=None,
            task=None,
            total_steps=256,
            callbacks=[],
        )
        assert "num_timesteps" in results
        assert results["num_timesteps"] > 0

    def test_train_completes_episodes(
        self, tcp_server, tcp_env_spec, tcp_ppo_config
    ):
        _, port = tcp_server
        plugin = SB3Plugin()
        plugin.configure(tcp_env_spec, tcp_ppo_config)
        results = plugin.train(
            worker_addrs=[(SERVER_IP, port)],
            env_fns=None,
            task=None,
            total_steps=512,
            callbacks=[],
        )
        assert "total_episodes" in results, "No episodes completed in 512 steps"
        assert results["total_episodes"] >= 1

    def test_train_episode_metrics_are_numeric(
        self, tcp_server, tcp_env_spec, tcp_ppo_config
    ):
        _, port = tcp_server
        plugin = SB3Plugin()
        plugin.configure(tcp_env_spec, tcp_ppo_config)
        results = plugin.train(
            worker_addrs=[(SERVER_IP, port)],
            env_fns=None,
            task=None,
            total_steps=512,
            callbacks=[],
        )
        for key in ("mean_reward", "mean_length", "last_100_mean_reward"):
            if key in results:
                assert isinstance(results[key], float), f"{key} should be float"


# ---------------------------------------------------------------------------
# Save / Load through TCP-trained model
# ---------------------------------------------------------------------------

class TestSB3TcpSaveLoad:
    def test_save_after_tcp_training(
        self, tcp_server, tcp_env_spec, tcp_ppo_config, tmp_path
    ):
        _, port = tcp_server
        plugin = SB3Plugin()
        plugin.configure(tcp_env_spec, tcp_ppo_config)
        plugin.train(
            worker_addrs=[(SERVER_IP, port)],
            env_fns=None,
            task=None,
            total_steps=256,
            callbacks=[],
        )
        meta = plugin.save(tmp_path / "tcp_model")
        assert (tmp_path / "tcp_model.zip").exists()
        assert meta["algorithm"] == "PPO"
        assert meta["step"] > 0

    def test_get_policy_after_tcp_training(
        self, tcp_server, tcp_env_spec, tcp_ppo_config
    ):
        _, port = tcp_server
        plugin = SB3Plugin()
        plugin.configure(tcp_env_spec, tcp_ppo_config)
        plugin.train(
            worker_addrs=[(SERVER_IP, port)],
            env_fns=None,
            task=None,
            total_steps=256,
            callbacks=[],
        )
        policy = plugin.get_policy()
        obs = np.random.randn(4, 4).astype(np.float32)
        actions = policy.get_actions(obs)
        assert actions.shape == (4, 1)


# ---------------------------------------------------------------------------
# Callback bridge works through TCP path
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


class TestSB3TcpCallbacks:
    def test_callbacks_fire_during_tcp_training(
        self, tcp_server, tcp_env_spec, tcp_ppo_config
    ):
        _, port = tcp_server
        plugin = SB3Plugin()
        plugin.configure(tcp_env_spec, tcp_ppo_config)
        cb = _RecordingCallback()
        plugin.train(
            worker_addrs=[(SERVER_IP, port)],
            env_fns=None,
            task=None,
            total_steps=256,
            callbacks=[cb],
        )
        assert len(cb.steps) > 0
        assert cb.training_ended
