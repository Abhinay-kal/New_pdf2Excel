from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config.settings import OUTPUT_DIR
from infrastructure.storage.sqlite_repo import SQLiteReviewRepo


def _load_records(queue_json: Path) -> List[Dict[str, Any]]:
    with queue_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("queue JSON must contain a list of review records")
    return [record for record in data if isinstance(record, dict)]


def load_queue(queue_json: Path, db_path: Path) -> int:
    repo = SQLiteReviewRepo(db_path)
    records = _load_records(queue_json)

    with repo._get_connection() as conn:
        for index, record in enumerate(records, start=1):
            item_id = str(record.get("id") or f"item_{index}")
            payload = dict(record)
            payload.setdefault("id", item_id)
            status = str(payload.get("status") or "pending")
            if status not in {"pending", "finalized", "skipped"}:
                status = "pending"
            conn.execute(
                """
                INSERT INTO review_items (id, status, payload, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (item_id, status, json.dumps(payload, ensure_ascii=False)),
            )
        conn.commit()

    return len(records)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load review queue JSON into SQLite.")
    parser.add_argument(
        "--queue-json",
        type=Path,
        default=OUTPUT_DIR / "human_review_queue.json",
        help="Path to human_review_queue.json",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=OUTPUT_DIR / "review_queue.sqlite3",
        help="Path to the SQLite database",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    inserted = load_queue(args.queue_json, args.db_path)
    print(f"Loaded {inserted} review records into {args.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())