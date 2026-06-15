"""Unit tests for Database (SQLite registry).

All tests use pytest's tmp_path fixture so no state leaks between tests.
"""
import pytest

from rl_platform.artifacts.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def run_id(db):
    return db.create_run("test_run", {"some": "config"})


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class TestRuns:
    def test_create_run_returns_string_id(self, db):
        run_id = db.create_run("my_run", {})
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_create_run_default_status_is_created(self, db):
        run_id = db.create_run("my_run", {})
        record = db.get_run(run_id)
        assert record["status"] == "created"

    def test_create_run_stores_name(self, db):
        run_id = db.create_run("named_run", {})
        record = db.get_run(run_id)
        assert record["name"] == "named_run"

    def test_create_run_stores_config(self, db):
        cfg = {"algo": "PPO", "lr": 3e-4}
        run_id = db.create_run("cfg_run", cfg)
        record = db.get_run(run_id)
        assert record["config"]["algo"] == "PPO"
        assert record["config"]["lr"] == pytest.approx(3e-4)

    def test_create_run_ids_are_unique(self, db):
        id1 = db.create_run("run_a", {})
        id2 = db.create_run("run_b", {})
        assert id1 != id2

    def test_get_run_returns_none_for_unknown_id(self, db):
        assert db.get_run("nonexistent") is None

    def test_update_run_status(self, db, run_id):
        db.update_run_status(run_id, "running")
        assert db.get_run(run_id)["status"] == "running"

    def test_update_run_status_completed(self, db, run_id):
        db.update_run_status(run_id, "completed")
        assert db.get_run(run_id)["status"] == "completed"

    def test_update_run_status_failed(self, db, run_id):
        db.update_run_status(run_id, "failed")
        assert db.get_run(run_id)["status"] == "failed"


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

class TestCheckpoints:
    def test_add_checkpoint_returns_string_id(self, db, run_id):
        ckpt_id = db.add_checkpoint(run_id, step=100, path="/some/path/model")
        assert isinstance(ckpt_id, str)
        assert len(ckpt_id) > 0

    def test_add_checkpoint_ids_are_unique(self, db, run_id):
        id1 = db.add_checkpoint(run_id, step=100, path="/path/a")
        id2 = db.add_checkpoint(run_id, step=200, path="/path/b")
        assert id1 != id2

    def test_get_checkpoints_empty_for_new_run(self, db, run_id):
        assert db.get_checkpoints(run_id) == []

    def test_get_checkpoints_returns_all(self, db, run_id):
        db.add_checkpoint(run_id, step=100, path="/path/a")
        db.add_checkpoint(run_id, step=200, path="/path/b")
        ckpts = db.get_checkpoints(run_id)
        assert len(ckpts) == 2

    def test_get_checkpoints_ordered_by_step_asc(self, db, run_id):
        db.add_checkpoint(run_id, step=300, path="/path/c")
        db.add_checkpoint(run_id, step=100, path="/path/a")
        db.add_checkpoint(run_id, step=200, path="/path/b")
        steps = [c["step"] for c in db.get_checkpoints(run_id)]
        assert steps == sorted(steps)

    def test_add_checkpoint_stores_metrics(self, db, run_id):
        db.add_checkpoint(run_id, step=100, path="/p", metrics={"mean_reward": 42.5})
        ckpt = db.get_checkpoints(run_id)[0]
        assert ckpt["metrics"]["mean_reward"] == pytest.approx(42.5)

    def test_add_checkpoint_stores_tags(self, db, run_id):
        db.add_checkpoint(run_id, step=100, path="/p", tags=["best", "final"])
        ckpt = db.get_checkpoints(run_id)[0]
        assert "best" in ckpt["tags"]
        assert "final" in ckpt["tags"]

    def test_add_checkpoint_no_metrics_is_none(self, db, run_id):
        db.add_checkpoint(run_id, step=100, path="/p")
        ckpt = db.get_checkpoints(run_id)[0]
        assert ckpt["metrics"] is None

    def test_get_checkpoints_does_not_return_other_runs(self, db):
        run_a = db.create_run("run_a", {})
        run_b = db.create_run("run_b", {})
        db.add_checkpoint(run_a, step=100, path="/a")
        db.add_checkpoint(run_b, step=100, path="/b")
        assert len(db.get_checkpoints(run_a)) == 1
        assert len(db.get_checkpoints(run_b)) == 1


class TestLatestCheckpoint:
    def test_get_latest_returns_none_when_no_checkpoints(self, db, run_id):
        assert db.get_latest_checkpoint(run_id) is None

    def test_get_latest_returns_highest_step(self, db, run_id):
        db.add_checkpoint(run_id, step=100, path="/a")
        db.add_checkpoint(run_id, step=500, path="/b")
        db.add_checkpoint(run_id, step=300, path="/c")
        latest = db.get_latest_checkpoint(run_id)
        assert latest["step"] == 500

    def test_get_latest_path_is_correct(self, db, run_id):
        db.add_checkpoint(run_id, step=100, path="/a")
        db.add_checkpoint(run_id, step=200, path="/the_latest")
        assert db.get_latest_checkpoint(run_id)["path"] == "/the_latest"


class TestBestCheckpoint:
    def test_get_best_returns_none_when_no_checkpoints(self, db, run_id):
        assert db.get_best_checkpoint(run_id, "mean_reward") is None

    def test_get_best_returns_none_when_metric_absent(self, db, run_id):
        db.add_checkpoint(run_id, step=100, path="/a", metrics={"other": 1.0})
        assert db.get_best_checkpoint(run_id, "mean_reward") is None

    def test_get_best_higher_is_better(self, db, run_id):
        db.add_checkpoint(run_id, step=100, path="/low", metrics={"mean_reward": 50.0})
        db.add_checkpoint(run_id, step=200, path="/high", metrics={"mean_reward": 200.0})
        best = db.get_best_checkpoint(run_id, "mean_reward", higher_is_better=True)
        assert best["path"] == "/high"

    def test_get_best_lower_is_better(self, db, run_id):
        db.add_checkpoint(run_id, step=100, path="/low_loss", metrics={"loss": 0.1})
        db.add_checkpoint(run_id, step=200, path="/high_loss", metrics={"loss": 5.0})
        best = db.get_best_checkpoint(run_id, "loss", higher_is_better=False)
        assert best["path"] == "/low_loss"

    def test_get_best_invalid_metric_name_raises(self, db, run_id):
        with pytest.raises(ValueError):
            db.get_best_checkpoint(run_id, "bad; DROP TABLE runs; --")


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------

class TestEvaluations:
    def test_add_evaluation_returns_string_id(self, db, run_id):
        eval_id = db.add_evaluation(run_id, None, "smoke", {"mean_reward": 1.0})
        assert isinstance(eval_id, str)
        assert len(eval_id) > 0

    def test_get_evaluations_empty_for_new_run(self, db, run_id):
        assert db.get_evaluations(run_id) == []

    def test_get_evaluations_returns_all(self, db, run_id):
        db.add_evaluation(run_id, None, "eval_a", {"mean_reward": 10.0})
        db.add_evaluation(run_id, None, "eval_b", {"mean_reward": 20.0})
        evals = db.get_evaluations(run_id)
        assert len(evals) == 2

    def test_get_evaluations_stores_results(self, db, run_id):
        db.add_evaluation(run_id, None, "scenario_x", {"mean_reward": 42.0, "episodes": 5})
        ev = db.get_evaluations(run_id)[0]
        assert ev["results"]["mean_reward"] == pytest.approx(42.0)
        assert ev["results"]["episodes"] == 5

    def test_get_evaluations_does_not_return_other_runs(self, db):
        run_a = db.create_run("run_a", {})
        run_b = db.create_run("run_b", {})
        db.add_evaluation(run_a, None, "s", {"r": 1})
        db.add_evaluation(run_b, None, "s", {"r": 2})
        assert len(db.get_evaluations(run_a)) == 1
        assert len(db.get_evaluations(run_b)) == 1

    def test_evaluation_with_checkpoint_id(self, db, run_id):
        ckpt_id = db.add_checkpoint(run_id, step=100, path="/p")
        eval_id = db.add_evaluation(run_id, ckpt_id, "s", {"r": 1})
        ev = db.get_evaluations(run_id)[0]
        assert ev["checkpoint_id"] == ckpt_id
