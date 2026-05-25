from rl_platform.core.specifications import TaskSpec


class SingleTaskCurriculum:
    """No-op curriculum that returns the same TaskSpec forever.

    Keeps the Coordinator's curriculum loop functional without
    real stage-advancement logic (that comes in M4).
    """

    def __init__(self, task: TaskSpec):
        self._task = task
        self._stage = "default"

    def get_task(self) -> TaskSpec:
        return self._task

    def report(self, metrics: dict[str, float]) -> None:
        pass

    def should_advance(self) -> bool:
        return False

    def advance(self) -> TaskSpec:
        return self._task

    @property
    def current_stage(self) -> str:
        return self._stage

