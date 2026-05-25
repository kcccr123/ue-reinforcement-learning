import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

@dataclass
class WorkerSpec:

    runtime_type: str               # e.g. "ue5", "mock"
    host: str = "localhost"
    port: int = 7777
    image: str | None = None        # Docker image (M3+)
    binary_path: Path | None = None # local binary (M3+)
    env_vars: dict[str, str] = field(default_factory=dict)
    gpu_required: bool = False


class OrchestratorBackend(Protocol):
    def launch(self, spec: WorkerSpec) -> str: ...
    def health_check(self, worker_id: str) -> bool: ...
    def teardown(self, worker_id: str) -> None: ...
    def teardown_all(self) -> None: ...

class DebugWorkerBackend:

    def __init__(self) -> None:
        # worker_id -> (host, port)
        self._workers: dict[str, tuple[str, int]] = {}

    def launch(self, spec: WorkerSpec) -> str:
        worker_id = uuid.uuid4().hex
        self._workers[worker_id] = (spec.host, spec.port)
        return worker_id

    def health_check(self, worker_id: str) -> bool:
        if worker_id not in self._workers:
            return False
        host, port = self._workers[worker_id]
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            return False

    def teardown(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)

    def teardown_all(self) -> None:
        self._workers.clear()

    def all_worker_ids(self) -> list[str]:
        return list(self._workers.keys())

class WorkerManager:

    def __init__(self, backend: OrchestratorBackend) -> None:
        self._backend = backend
        # ordered list of (worker_id, host, port) preserves launch order
        self._workers: list[tuple[str, str, int]] = []

    def launch_workers(self, specs: list[WorkerSpec]) -> list[str]:
        ids = []
        for spec in specs:
            wid = self._backend.launch(spec)
            self._workers.append((wid, spec.host, spec.port))
            ids.append(wid)
        return ids

    def get_worker_address(self, worker_id: str) -> tuple[str, int]:
        for wid, host, port in self._workers:
            if wid == worker_id:
                return host, port
        raise KeyError(f"Unknown worker_id: {worker_id!r}")

    def get_all_addresses(self) -> list[tuple[str, int]]:
        return [(host, port) for _, host, port in self._workers]

    def health_check_all(self) -> dict[str, bool]:
        return {
            wid: self._backend.health_check(wid)
            for wid, _, _ in self._workers
        }

    def teardown_all(self) -> None:
        self._backend.teardown_all()
        self._workers.clear()
