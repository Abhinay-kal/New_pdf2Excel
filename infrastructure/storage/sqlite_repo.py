from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from domain.interfaces import ReviewRepository


class SQLiteReviewRepo(ReviewRepository):
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self):
        # check_same_thread=False is required for Streamlit's multi-threading
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_items (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    payload TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_next_pending(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, payload
                FROM review_items
                WHERE status = 'pending'
                ORDER BY rowid
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            payload = {"payload": row["payload"]}

        if isinstance(payload, dict):
            payload.setdefault("id", row["id"])
            return payload

        return {"id": row["id"], "payload": payload}

    def get_pending_count(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS pending_count FROM review_items WHERE status = 'pending'"
            ).fetchone()
            return int(row["pending_count"] if row is not None else 0)

    def save_finalized(self, item_id: str, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("id", item_id)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO review_items (id, status, payload, updated_at)
                VALUES (?, 'finalized', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (item_id, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()
