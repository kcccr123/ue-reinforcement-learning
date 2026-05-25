"""Unit tests for WorkerManager and DebugWorkerBackend.

No real sockets are needed for most tests — DebugWorkerBackend simply
registers host:port pairs. The health_check tests that do open a socket
are marked as needing a live server; they just verify the TCP probe
returns False when nothing is listening.
"""
import pytest

from rl_platform.infra.workers import WorkerSpec, WorkerManager, DebugWorkerBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return DebugWorkerBackend()


@pytest.fixture
def manager(backend):
    return WorkerManager(backend)


@pytest.fixture
def spec():
    return WorkerSpec(runtime_type="external", host="127.0.0.1", port=19999)


# ---------------------------------------------------------------------------
# WorkerSpec
# ---------------------------------------------------------------------------

class TestWorkerSpec:
    def test_defaults(self):
        spec = WorkerSpec(runtime_type="external")
        assert spec.host == "localhost"
        assert spec.port == 7777
        assert spec.image is None
        assert spec.gpu_required is False

    def test_custom_values(self):
        spec = WorkerSpec(runtime_type="ue5", host="192.168.1.1", port=8888)
        assert spec.host == "192.168.1.1"
        assert spec.port == 8888


# ---------------------------------------------------------------------------
# DebugWorkerBackend
# ---------------------------------------------------------------------------

class TestDebugWorkerBackend:
    def test_launch_returns_string_id(self, backend, spec):
        wid = backend.launch(spec)
        assert isinstance(wid, str)
        assert len(wid) > 0

    def test_launch_ids_are_unique(self, backend, spec):
        id1 = backend.launch(spec)
        id2 = backend.launch(spec)
        assert id1 != id2

    def test_all_worker_ids_lists_launched(self, backend, spec):
        id1 = backend.launch(spec)
        id2 = backend.launch(spec)
        ids = backend.all_worker_ids()
        assert id1 in ids
        assert id2 in ids

    def test_teardown_removes_worker(self, backend, spec):
        wid = backend.launch(spec)
        backend.teardown(wid)
        assert wid not in backend.all_worker_ids()

    def test_teardown_unknown_is_silent(self, backend):
        backend.teardown("ghost_worker")  # should not raise

    def test_teardown_all_clears_all_workers(self, backend, spec):
        backend.launch(spec)
        backend.launch(spec)
        backend.teardown_all()
        assert backend.all_worker_ids() == []

    def test_health_check_returns_false_when_nothing_listening(self, backend, spec):
        wid = backend.launch(spec)
        result = backend.health_check(wid)
        assert result is False

    def test_health_check_unknown_worker_returns_false(self, backend):
        assert backend.health_check("ghost") is False


# ---------------------------------------------------------------------------
# WorkerManager
# ---------------------------------------------------------------------------

class TestWorkerManager:
    def test_launch_workers_returns_ids(self, manager, spec):
        ids = manager.launch_workers([spec])
        assert len(ids) == 1
        assert isinstance(ids[0], str)

    def test_launch_multiple_workers(self, manager, spec):
        spec2 = WorkerSpec(runtime_type="external", host="127.0.0.1", port=20000)
        ids = manager.launch_workers([spec, spec2])
        assert len(ids) == 2

    def test_get_worker_address(self, manager, spec):
        ids = manager.launch_workers([spec])
        addr = manager.get_worker_address(ids[0])
        assert addr == ("127.0.0.1", 19999)

    def test_get_all_addresses_single(self, manager, spec):
        manager.launch_workers([spec])
        addrs = manager.get_all_addresses()
        assert len(addrs) == 1
        assert addrs[0] == ("127.0.0.1", 19999)

    def test_get_all_addresses_multiple(self, manager):
        specs = [
            WorkerSpec(runtime_type="external", host="127.0.0.1", port=20001),
            WorkerSpec(runtime_type="external", host="127.0.0.1", port=20002),
        ]
        manager.launch_workers(specs)
        addrs = manager.get_all_addresses()
        ports = [a[1] for a in addrs]
        assert 20001 in ports
        assert 20002 in ports

    def test_get_all_addresses_preserves_order(self, manager):
        specs = [
            WorkerSpec(runtime_type="external", host="127.0.0.1", port=20003),
            WorkerSpec(runtime_type="external", host="127.0.0.1", port=20004),
        ]
        manager.launch_workers(specs)
        ports = [a[1] for a in manager.get_all_addresses()]
        assert ports == [20003, 20004]

    def test_health_check_all_returns_dict(self, manager, spec):
        ids = manager.launch_workers([spec])
        results = manager.health_check_all()
        assert isinstance(results, dict)
        assert ids[0] in results

    def test_teardown_all_clears_workers(self, manager, spec):
        manager.launch_workers([spec])
        manager.teardown_all()
        assert manager.get_all_addresses() == []

    def test_teardown_all_idempotent(self, manager, spec):
        manager.launch_workers([spec])
        manager.teardown_all()
        manager.teardown_all()  # second call should not raise
        assert manager.get_all_addresses() == []
