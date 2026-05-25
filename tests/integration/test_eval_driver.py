"""Integration tests for EvalDriver.

Uses CartPole-v1 with a random policy so tests stay fast (no training).
"""
import gymnasium as gym
import numpy as np
import pytest

from rl_platform.artifacts.database import Database
from rl_platform.core.eval import EvalDriver, EvalScenario, EvalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def eval_driver(db):
    return EvalDriver(database=db)


@pytest.fixture
def run_id(db):
    return db.create_run("eval_test_run", {})


def cartpole_env_fn():
    return gym.make("CartPole-v1")


class _RandomPolicy:
    """Picks a random discrete action regardless of observation."""
    def __init__(self, n_actions: int = 2):
        self._n = n_actions

    def get_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        batch_size = obs_batch.shape[0]
        return np.random.randint(0, self._n, size=(batch_size,))


@pytest.fixture
def random_policy():
    return _RandomPolicy(n_actions=2)


@pytest.fixture
def scenario():
    return EvalScenario(name="smoke_eval", num_episodes=3)


# ---------------------------------------------------------------------------
# EvalScenario dataclass
# ---------------------------------------------------------------------------

class TestEvalScenario:
    def test_name_stored(self):
        s = EvalScenario(name="test", num_episodes=5)
        assert s.name == "test"

    def test_num_episodes_stored(self):
        s = EvalScenario(name="test", num_episodes=7)
        assert s.num_episodes == 7

    def test_env_params_default_empty(self):
        s = EvalScenario(name="test")
        assert s.env_params == {}


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_mean_length_property(self):
        result = EvalResult(
            scenario_name="s",
            mean_reward=5.0,
            episode_rewards=[5.0, 10.0],
            episode_lengths=[100, 200],
        )
        assert result.mean_length == pytest.approx(150.0)

    def test_mean_length_empty_is_zero(self):
        result = EvalResult(scenario_name="s", mean_reward=0.0)
        assert result.mean_length == 0.0

    def test_num_episodes_property(self):
        result = EvalResult(
            scenario_name="s",
            mean_reward=1.0,
            episode_rewards=[1.0, 2.0, 3.0],
        )
        assert result.num_episodes == 3


# ---------------------------------------------------------------------------
# EvalDriver.evaluate
# ---------------------------------------------------------------------------

class TestEvalDriverEvaluate:
    def test_evaluate_returns_eval_result(self, eval_driver, random_policy, scenario):
        result = eval_driver.evaluate(
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        assert isinstance(result, EvalResult)

    def test_evaluate_correct_episode_count(self, eval_driver, random_policy, scenario):
        result = eval_driver.evaluate(
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        assert result.num_episodes == scenario.num_episodes

    def test_evaluate_episode_rewards_are_floats(self, eval_driver, random_policy, scenario):
        result = eval_driver.evaluate(
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        for r in result.episode_rewards:
            assert isinstance(r, float)

    def test_evaluate_episode_lengths_are_ints(self, eval_driver, random_policy, scenario):
        result = eval_driver.evaluate(
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        for length in result.episode_lengths:
            assert isinstance(length, int)
            assert length > 0

    def test_evaluate_mean_reward_equals_average(self, eval_driver, random_policy, scenario):
        result = eval_driver.evaluate(
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        expected = sum(result.episode_rewards) / len(result.episode_rewards)
        assert result.mean_reward == pytest.approx(expected)

    def test_evaluate_scenario_name_in_result(self, eval_driver, random_policy, scenario):
        result = eval_driver.evaluate(
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        assert result.scenario_name == scenario.name

    def test_evaluate_rewards_are_positive_cartpole(self, eval_driver, random_policy):
        """CartPole gives +1 per step so reward must be >= 1."""
        scenario = EvalScenario(name="test", num_episodes=5)
        result = eval_driver.evaluate(
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        for r in result.episode_rewards:
            assert r >= 1.0

    def test_evaluate_closes_env(self, eval_driver, random_policy):
        """If the env is not closed, repeated calls would exhaust file descriptors.
        This just verifies multiple calls don't raise."""
        scenario = EvalScenario(name="test", num_episodes=2)
        for _ in range(3):
            eval_driver.evaluate(
                policy=random_policy,
                env_fn=cartpole_env_fn,
                scenario=scenario,
            )


# ---------------------------------------------------------------------------
# EvalDriver.evaluate_and_store
# ---------------------------------------------------------------------------

class TestEvalDriverEvaluateAndStore:
    def test_evaluate_and_store_returns_eval_result(
        self, eval_driver, db, run_id, random_policy, scenario
    ):
        result = eval_driver.evaluate_and_store(
            run_id=run_id,
            checkpoint_id=None,
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        assert isinstance(result, EvalResult)

    def test_evaluate_and_store_persists_to_db(
        self, eval_driver, db, run_id, random_policy, scenario
    ):
        eval_driver.evaluate_and_store(
            run_id=run_id,
            checkpoint_id=None,
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        evals = db.get_evaluations(run_id)
        assert len(evals) == 1

    def test_evaluate_and_store_records_scenario_name(
        self, eval_driver, db, run_id, random_policy, scenario
    ):
        eval_driver.evaluate_and_store(
            run_id=run_id,
            checkpoint_id=None,
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        ev = db.get_evaluations(run_id)[0]
        assert ev["scenario"] == scenario.name

    def test_evaluate_and_store_multiple_calls(
        self, eval_driver, db, run_id, random_policy
    ):
        for i in range(3):
            eval_driver.evaluate_and_store(
                run_id=run_id,
                checkpoint_id=None,
                policy=random_policy,
                env_fn=cartpole_env_fn,
                scenario=EvalScenario(name=f"eval_{i}", num_episodes=2),
            )
        assert len(db.get_evaluations(run_id)) == 3

    def test_evaluate_and_store_with_checkpoint_id(
        self, eval_driver, db, run_id, random_policy, scenario
    ):
        ckpt_id = db.add_checkpoint(run_id, step=100, path="/fake/model")
        eval_driver.evaluate_and_store(
            run_id=run_id,
            checkpoint_id=ckpt_id,
            policy=random_policy,
            env_fn=cartpole_env_fn,
            scenario=scenario,
        )
        ev = db.get_evaluations(run_id)[0]
        assert ev["checkpoint_id"] == ckpt_id
