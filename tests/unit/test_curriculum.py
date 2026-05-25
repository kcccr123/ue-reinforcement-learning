"""Unit tests for SingleTaskCurriculum."""
import pytest

from rl_platform.core.curriculum import SingleTaskCurriculum
from rl_platform.core.specifications import TaskSpec


@pytest.fixture
def task():
    return TaskSpec(name="solo_task", env_params={"difficulty": 0.5})


@pytest.fixture
def curriculum(task):
    return SingleTaskCurriculum(task)


class TestSingleTaskCurriculum:
    def test_get_task_returns_the_task(self, curriculum, task):
        assert curriculum.get_task() is task

    def test_get_task_always_returns_same_task(self, curriculum, task):
        for _ in range(5):
            assert curriculum.get_task() is task

    def test_should_advance_is_always_false(self, curriculum):
        assert curriculum.should_advance() is False

    def test_should_advance_after_report_is_still_false(self, curriculum):
        curriculum.report({"mean_reward": 500.0})
        assert curriculum.should_advance() is False

    def test_report_does_not_raise(self, curriculum):
        curriculum.report({})
        curriculum.report({"mean_reward": 100.0, "episodes": 10})

    def test_advance_returns_same_task(self, curriculum, task):
        result = curriculum.advance()
        assert result is task

    def test_current_stage_is_string(self, curriculum):
        assert isinstance(curriculum.current_stage, str)

    def test_current_stage_does_not_change_after_report(self, curriculum):
        stage_before = curriculum.current_stage
        curriculum.report({"mean_reward": 999.0})
        assert curriculum.current_stage == stage_before
