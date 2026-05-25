"""Unit tests for CheckpointManager.

Uses a stub LearnerPlugin so no SB3/training is needed — this tests
the save/load/query logic and the DB integration only.
"""
import pytest
from pathlib import Path
from typing import Any

from rl_platform.artifacts.database import Database
from rl_platform.artifacts.checkpointing import CheckpointManager
from rl_platform.core.specifications import EnvSpec, TaskSpec


# ---------------------------------------------------------------------------
# Stub learner — just writes a marker file, no actual model weights
# ---------------------------------------------------------------------------

class _StubLearner:
    def __init__(self):
        self.saved_paths: list[Path] = []
        self._step = 42

    def save(self, path: Path) -> dict[str, Any]:
        path = Path(str(path) + ".stub")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub_model")
        self.saved_paths.append(path)
        return {"algorithm": "stub", "step": self._step}

    def configure(self, env_spec, config): pass
    def train(self, **kwargs): return {}
    def load(self, path: Path): pass
    def get_policy(self): return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "registry.db")
    yield database
    database.close()


@pytest.fixture
def ckpt_mgr(db, tmp_path):
    return CheckpointManager(data_dir=tmp_path / "data", database=db)


@pytest.fixture
def run_id(db):
    return db.create_run("test_run", {})


@pytest.fixture
def learner():
    return _StubLearner()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_returns_string_checkpoint_id(self, ckpt_mgr, run_id, learner):
        ckpt_id = ckpt_mgr.save(run_id=run_id, step=100, learner=learner)
        assert isinstance(ckpt_id, str)
        assert len(ckpt_id) > 0

    def test_save_calls_learner_save(self, ckpt_mgr, run_id, learner):
        ckpt_mgr.save(run_id=run_id, step=100, learner=learner)
        assert len(learner.saved_paths) == 1

    def test_save_creates_directory(self, ckpt_mgr, run_id, learner, tmp_path):
        ckpt_mgr.save(run_id=run_id, step=200, learner=learner)
        ckpt_dir = tmp_path / "data" / run_id / "step_200"
        assert ckpt_dir.exists()

    def test_save_registers_checkpoint_in_db(self, ckpt_mgr, db, run_id, learner):
        ckpt_mgr.save(run_id=run_id, step=100, learner=learner)
        ckpts = db.get_checkpoints(run_id)
        assert len(ckpts) == 1
        assert ckpts[0]["step"] == 100

    def test_save_stores_metrics_in_db(self, ckpt_mgr, db, run_id, learner):
        ckpt_mgr.save(run_id=run_id, step=100, learner=learner, metrics={"mean_reward": 99.0})
        ckpts = db.get_checkpoints(run_id)
        assert ckpts[0]["metrics"]["mean_reward"] == pytest.approx(99.0)

    def test_save_stores_tags_in_db(self, ckpt_mgr, db, run_id, learner):
        ckpt_mgr.save(run_id=run_id, step=100, learner=learner, tags=["final"])
        ckpts = db.get_checkpoints(run_id)
        assert "final" in ckpts[0]["tags"]

    def test_multiple_saves_produce_multiple_records(self, ckpt_mgr, db, run_id, learner):
        ckpt_mgr.save(run_id=run_id, step=100, learner=learner)
        ckpt_mgr.save(run_id=run_id, step=200, learner=learner)
        assert len(db.get_checkpoints(run_id)) == 2


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_path(self, ckpt_mgr, run_id, learner):
        ckpt_id = ckpt_mgr.save(run_id=run_id, step=100, learner=learner)
        path = ckpt_mgr.load(run_id=run_id, checkpoint_id=ckpt_id)
        assert isinstance(path, Path)

    def test_load_unknown_checkpoint_raises(self, ckpt_mgr, run_id):
        with pytest.raises(KeyError):
            ckpt_mgr.load(run_id=run_id, checkpoint_id="does_not_exist")


# ---------------------------------------------------------------------------
# get_latest
# ---------------------------------------------------------------------------

class TestGetLatest:
    def test_get_latest_returns_none_with_no_checkpoints(self, ckpt_mgr, run_id):
        assert ckpt_mgr.get_latest(run_id) is None

    def test_get_latest_returns_path_of_highest_step(self, ckpt_mgr, run_id, learner):
        ckpt_mgr.save(run_id=run_id, step=100, learner=learner)
        ckpt_mgr.save(run_id=run_id, step=500, learner=learner)
        ckpt_mgr.save(run_id=run_id, step=300, learner=learner)
        latest_path = ckpt_mgr.get_latest(run_id)
        # path should contain "step_500"
        assert "step_500" in str(latest_path)

    def test_get_latest_returns_path_instance(self, ckpt_mgr, run_id, learner):
        ckpt_mgr.save(run_id=run_id, step=100, learner=learner)
        assert isinstance(ckpt_mgr.get_latest(run_id), Path)


# ---------------------------------------------------------------------------
# get_best
# ---------------------------------------------------------------------------

class TestGetBest:
    def test_get_best_returns_none_with_no_checkpoints(self, ckpt_mgr, run_id):
        assert ckpt_mgr.get_best(run_id, "mean_reward") is None

    def test_get_best_selects_highest_metric(self, ckpt_mgr, run_id, learner):
        ckpt_mgr.save(run_id=run_id, step=100, learner=learner, metrics={"mean_reward": 50.0})
        ckpt_mgr.save(run_id=run_id, step=200, learner=learner, metrics={"mean_reward": 250.0})
        best = ckpt_mgr.get_best(run_id, "mean_reward")
        assert "step_200" in str(best)

    def test_get_best_lower_is_better(self, ckpt_mgr, run_id, learner):
        ckpt_mgr.save(run_id=run_id, step=100, learner=learner, metrics={"loss": 0.5})
        ckpt_mgr.save(run_id=run_id, step=200, learner=learner, metrics={"loss": 0.1})
        best = ckpt_mgr.get_best(run_id, "loss", higher_is_better=False)
        assert "step_200" in str(best)
