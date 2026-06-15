import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_to_dict(
    row: sqlite3.Row,
    json_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    d = dict(row)
    for field in json_fields:
        val = d.get(field)
        if val is not None:
            d[field] = json.loads(val)
    return d


class Database:

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                config      TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'created',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id          TEXT PRIMARY KEY,
                run_id      TEXT NOT NULL REFERENCES runs(id),
                step        INTEGER NOT NULL,
                path        TEXT NOT NULL,
                metrics     TEXT,
                tags        TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_step ON checkpoints(run_id, step);

            CREATE TABLE IF NOT EXISTS evaluations (
                id              TEXT PRIMARY KEY,
                run_id          TEXT NOT NULL REFERENCES runs(id),
                checkpoint_id   TEXT REFERENCES checkpoints(id),
                scenario        TEXT NOT NULL,
                results         TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_evaluations_run_id ON evaluations(run_id);
            """
        )

    def create_run(self, name: str, config: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        now = now_iso()
        self._conn.execute(
            "INSERT INTO runs (id, name, config, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'created', ?, ?)",
            (run_id, name, json.dumps(config), now, now),
        )
        self._conn.commit()
        return run_id

    def update_run_status(self, run_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
            (status, now_iso(), run_id),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return row_to_dict(row, json_fields=("config",))

    def add_checkpoint(
        self,
        run_id: str,
        step: int,
        path: str,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        ckpt_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO checkpoints (id, run_id, step, path, metrics, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ckpt_id,
                run_id,
                step,
                path,
                json.dumps(metrics) if metrics else None,
                json.dumps(tags) if tags else None,
                now_iso(),
            ),
        )
        self._conn.commit()
        return ckpt_id

    def get_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step ASC",
            (run_id,),
        ).fetchall()
        return [row_to_dict(r, json_fields=("metrics", "tags")) for r in rows]

    def get_latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return row_to_dict(row, json_fields=("metrics", "tags"))

    def get_best_checkpoint(
        self, run_id: str, metric: str, higher_is_better: bool = True
    ) -> dict[str, Any] | None:
        if not metric.replace("_", "").isalnum():
            raise ValueError(f"invalid metric name: {metric!r}")
        order = "DESC" if higher_is_better else "ASC"
        json_path = f"$.{metric}"

        row = self._conn.execute(
            f"""
            SELECT *
            FROM checkpoints
            WHERE run_id = ?
            AND json_extract(metrics, ?) IS NOT NULL
            ORDER BY json_extract(metrics, ?) {order}
            LIMIT 1
            """,
            (run_id, json_path, json_path),
        ).fetchone()

        if row is None:
            return None
        return row_to_dict(row, json_fields=("metrics", "tags"))

    def add_evaluation(
        self,
        run_id: str,
        checkpoint_id: str | None,
        scenario: str,
        results: dict[str, Any],
    ) -> str:
        eval_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO evaluations (id, run_id, checkpoint_id, scenario, results, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (eval_id, run_id, checkpoint_id, scenario, json.dumps(results), now_iso()),
        )
        self._conn.commit()
        return eval_id

    def get_evaluations(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM evaluations WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        return [row_to_dict(r, json_fields=("results",)) for r in rows]

    def close(self) -> None:
        self._conn.close()