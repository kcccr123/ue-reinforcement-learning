"""Unit tests for ConfigLoader and RunConfig (Pydantic schema + YAML loading)."""
import textwrap
import pytest
from pathlib import Path
from pydantic import ValidationError

from rl_platform.artifacts.runtime import (
    ConfigLoader,
    RunConfig,
    EnvConfig,
    WorkerConfig,
    LearnerConfig,
    CheckpointConfig,
    EvalConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(tmp_path: Path, content: str) -> Path:
    """Write an indented YAML string to a temp file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


MINIMAL_VALID_YAML = """
    run_name: test_run
    total_steps: 1000
    env:
      env_id: CartPole-v1
      is_multi_agent: false
    learner:
      type: sb3
      params:
        algorithm: PPO
        learning_rate: 0.0003
        n_steps: 64
        batch_size: 32
        n_epochs: 2
        verbose: 0
"""


# ---------------------------------------------------------------------------
# ConfigLoader — file handling
# ---------------------------------------------------------------------------

class TestConfigLoaderFileHandling:
    def test_load_valid_yaml_returns_run_config(self, tmp_path):
        path = write_yaml(tmp_path, MINIMAL_VALID_YAML)
        cfg = ConfigLoader.load(path)
        assert isinstance(cfg, RunConfig)

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load(tmp_path / "nonexistent.yaml")

    def test_load_accepts_string_path(self, tmp_path):
        path = write_yaml(tmp_path, MINIMAL_VALID_YAML)
        cfg = ConfigLoader.load(str(path))
        assert isinstance(cfg, RunConfig)

    def test_load_non_mapping_yaml_raises_value_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError):
            ConfigLoader.load(p)


# ---------------------------------------------------------------------------
# RunConfig — required fields
# ---------------------------------------------------------------------------

class TestRunConfigRequiredFields:
    def test_run_name_is_set(self, tmp_path):
        cfg = ConfigLoader.load(write_yaml(tmp_path, MINIMAL_VALID_YAML))
        assert cfg.run_name == "test_run"

    def test_total_steps_is_set(self, tmp_path):
        cfg = ConfigLoader.load(write_yaml(tmp_path, MINIMAL_VALID_YAML))
        assert cfg.total_steps == 1000

    def test_missing_run_name_raises(self, tmp_path):
        yaml = """
            total_steps: 1000
            env:
              env_id: CartPole-v1
        """
        with pytest.raises(ValidationError):
            ConfigLoader.load(write_yaml(tmp_path, yaml))

    def test_missing_total_steps_raises(self, tmp_path):
        yaml = """
            run_name: test
            env:
              env_id: CartPole-v1
        """
        with pytest.raises(ValidationError):
            ConfigLoader.load(write_yaml(tmp_path, yaml))


# ---------------------------------------------------------------------------
# EnvConfig — M0 guard
# ---------------------------------------------------------------------------

class TestEnvConfig:
    def test_is_multi_agent_false_is_valid(self):
        cfg = EnvConfig(env_id="CartPole-v1", is_multi_agent=False)
        assert cfg.is_multi_agent is False

    def test_is_multi_agent_true_raises(self):
        with pytest.raises(ValidationError):
            EnvConfig(env_id="CartPole-v1", is_multi_agent=True)

    def test_env_id_none_is_valid(self):
        cfg = EnvConfig(env_id=None)
        assert cfg.env_id is None

    def test_is_multi_agent_true_in_yaml_raises(self, tmp_path):
        yaml = """
            run_name: bad
            total_steps: 100
            env:
              env_id: CartPole-v1
              is_multi_agent: true
        """
        with pytest.raises(ValidationError):
            ConfigLoader.load(write_yaml(tmp_path, yaml))


# ---------------------------------------------------------------------------
# WorkerConfig — M0 guard
# ---------------------------------------------------------------------------

class TestWorkerConfig:
    def test_num_workers_1_is_valid(self):
        cfg = WorkerConfig(num_workers=1)
        assert cfg.num_workers == 1

    def test_num_workers_2_raises(self):
        with pytest.raises(ValidationError):
            WorkerConfig(num_workers=2)

    def test_num_workers_0_raises(self):
        with pytest.raises(ValidationError):
            WorkerConfig(num_workers=0)

    def test_num_workers_2_in_yaml_raises(self, tmp_path):
        yaml = """
            run_name: bad
            total_steps: 100
            env:
              env_id: CartPole-v1
            worker:
              num_workers: 2
        """
        with pytest.raises(ValidationError):
            ConfigLoader.load(write_yaml(tmp_path, yaml))

    def test_default_host_and_port(self):
        cfg = WorkerConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 7777


# ---------------------------------------------------------------------------
# LearnerConfig
# ---------------------------------------------------------------------------

class TestLearnerConfig:
    def test_learner_type_is_set(self, tmp_path):
        cfg = ConfigLoader.load(write_yaml(tmp_path, MINIMAL_VALID_YAML))
        assert cfg.learner.type == "sb3"

    def test_learner_params_passed_through(self, tmp_path):
        cfg = ConfigLoader.load(write_yaml(tmp_path, MINIMAL_VALID_YAML))
        assert cfg.learner.params["algorithm"] == "PPO"
        assert cfg.learner.params["learning_rate"] == pytest.approx(3e-4)

    def test_learner_params_default_empty(self):
        cfg = LearnerConfig(type="sb3")
        assert cfg.params == {}


# ---------------------------------------------------------------------------
# EvalConfig — M0 guard
# ---------------------------------------------------------------------------

class TestEvalConfig:
    def test_eval_freq_zero_is_valid(self):
        cfg = EvalConfig(eval_freq=0)
        assert cfg.eval_freq == 0

    def test_eval_freq_nonzero_raises(self):
        with pytest.raises(ValidationError, match="eval_freq"):
            EvalConfig(eval_freq=1000)

    def test_eval_freq_nonzero_in_yaml_raises(self, tmp_path):
        yaml = """
            run_name: bad
            total_steps: 100
            eval:
              eval_freq: 500
        """
        with pytest.raises(ValidationError):
            ConfigLoader.load(write_yaml(tmp_path, yaml))


# ---------------------------------------------------------------------------
# CheckpointConfig + EvalConfig defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_checkpoint_save_freq_default(self, tmp_path):
        cfg = ConfigLoader.load(write_yaml(tmp_path, MINIMAL_VALID_YAML))
        assert cfg.checkpoint.save_freq == 10_000

    def test_eval_freq_default_is_zero(self, tmp_path):
        cfg = ConfigLoader.load(write_yaml(tmp_path, MINIMAL_VALID_YAML))
        assert cfg.eval.eval_freq == 0

    def test_eval_num_episodes_default(self, tmp_path):
        cfg = ConfigLoader.load(write_yaml(tmp_path, MINIMAL_VALID_YAML))
        assert cfg.eval.num_episodes == 5

    def test_data_dir_default(self, tmp_path):
        cfg = ConfigLoader.load(write_yaml(tmp_path, MINIMAL_VALID_YAML))
        assert cfg.data_dir == Path("runs")

    def test_data_dir_can_be_overridden(self, tmp_path):
        custom_dir = tmp_path / "custom_runs"
        content = f"""
            run_name: test_run
            total_steps: 1000
            data_dir: {custom_dir}
            env:
              env_id: CartPole-v1
              is_multi_agent: false
            learner:
              type: sb3
              params:
                algorithm: PPO
                verbose: 0
        """
        cfg = ConfigLoader.load(write_yaml(tmp_path, content))
        assert cfg.data_dir == custom_dir
