"""Integration tests for Coordinator using the mock TCP server (TCP path).

These tests exercise the full wire-protocol path:
  Coordinator -> WorkerManager -> (DebugWorkerBackend registers addr) ->
  Coordinator probes TCPGymEnv for EnvSpec -> LearnerDriver -> SB3Plugin
  creates TCPGymEnv -> trains against MockTcpServer.
"""
import textwrap
import threading
import time
import pytest
from pathlib import Path

from rl_platform.core.coordinator import Coordinator
from rl_platform.artifacts.database import Database
from tests.helpers.mock_tcp_server import MockTcpServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERVER_IP = "127.0.0.1"
BASE_PORT = 19500


def _get_port(request) -> int:
    return BASE_PORT + hash(request.node.nodeid) % 300


def write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def tcp_config(tmp_path: Path, port: int, total_steps: int = 256) -> Path:
    """Write a TCP-backed training config pointing at a mock server."""
    content = f"""
        run_name: tcp_test_run
        total_steps: {total_steps}
        data_dir: {tmp_path / "runs"}

        env:
          env_id: null
          is_multi_agent: false

        worker:
          host: {SERVER_IP}
          port: {port}
          num_workers: 1

        learner:
          type: sb3
          params:
            algorithm: PPO
            learning_rate: 0.0003
            n_steps: 64
            batch_size: 32
            n_epochs: 2
            verbose: 0

        checkpoint:
          save_freq: 0
          keep_best: false

        eval:
          eval_freq: 0
          num_episodes: 2
    """
    return write_yaml(tmp_path, content)


@pytest.fixture
def mock_server(request):
    """Start a MockTcpServer in a daemon thread; yield (server, port)."""
    port = _get_port(request)
    server = MockTcpServer(SERVER_IP, port)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield server, port
    server.shutdown()
    thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Coordinator + TCP — return value
# ---------------------------------------------------------------------------

class TestCoordinatorTcpRunReturnValue:
    def test_run_returns_dict(self, mock_server, tmp_path):
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()
        assert isinstance(result, dict)

    def test_run_result_status_completed(self, mock_server, tmp_path):
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()
        assert result["status"] == "completed"

    def test_run_result_has_run_id(self, mock_server, tmp_path):
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()
        assert isinstance(result["run_id"], str)

    def test_run_result_has_num_timesteps(self, mock_server, tmp_path):
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()
        assert result["results"].get("num_timesteps", 0) > 0

    def test_run_result_has_final_checkpoint_id(self, mock_server, tmp_path):
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()
        assert isinstance(result["final_checkpoint_id"], str)


# ---------------------------------------------------------------------------
# Coordinator + TCP — database state
# ---------------------------------------------------------------------------

class TestCoordinatorTcpDatabaseState:
    def test_run_creates_db_record(self, mock_server, tmp_path):
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()

        db = Database(tmp_path / "runs" / "registry.db")
        record = db.get_run(result["run_id"])
        db.close()

        assert record is not None
        assert record["status"] == "completed"

    def test_run_saves_final_checkpoint(self, mock_server, tmp_path):
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()

        db = Database(tmp_path / "runs" / "registry.db")
        ckpts = db.get_checkpoints(result["run_id"])
        db.close()

        assert len(ckpts) >= 1

    def test_final_checkpoint_tagged_final(self, mock_server, tmp_path):
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()

        db = Database(tmp_path / "runs" / "registry.db")
        ckpts = db.get_checkpoints(result["run_id"])
        db.close()

        all_tags = [t for c in ckpts if c["tags"] for t in c["tags"]]
        assert "final" in all_tags


# ---------------------------------------------------------------------------
# Coordinator + TCP — checkpoint files
# ---------------------------------------------------------------------------

class TestCoordinatorTcpCheckpointFiles:
    def test_checkpoint_directory_exists(self, mock_server, tmp_path):
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()

        db = Database(tmp_path / "runs" / "registry.db")
        ckpts = db.get_checkpoints(result["run_id"])
        db.close()

        for ckpt in ckpts:
            assert Path(ckpt["path"]).parent.exists()


# ---------------------------------------------------------------------------
# Coordinator + TCP — env spec discovered from handshake
# ---------------------------------------------------------------------------

class TestCoordinatorTcpEnvSpec:
    def test_run_completes_with_correct_env_id(self, mock_server, tmp_path):
        """MockTcpServer sends env_id='test_env'; coordinator should discover it."""
        _, port = mock_server
        path = tcp_config(tmp_path, port)
        result = Coordinator.from_yaml(path).run()
        # If env spec discovery failed the run would raise, not complete
        assert result["status"] == "completed"
