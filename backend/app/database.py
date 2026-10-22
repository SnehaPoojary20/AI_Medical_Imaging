
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.config import DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    """Create and return a database connection."""
    database_path = Path(DATABASE_PATH)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(database_path),
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """Provide a database connection and manage transactions."""
    connection = get_connection()

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Create the reviews table if it does not exist."""
    with get_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id TEXT NOT NULL
                    CHECK(length(study_id) BETWEEN 1 AND 100),
                decision TEXT NOT NULL
                    CHECK(decision IN ('approve', 'reject')),
                notes TEXT
                    CHECK(notes IS NULL OR length(notes) <= 1000),
                created_at TEXT NOT NULL
            )
            """
        )


def create_review(
    study_id: str,
    decision: str,
    notes: str | None = None,
) -> dict:
    """Save a review and return the created record."""
    created_at = datetime.now(timezone.utc).isoformat()

    with get_db() as connection:
        cursor = connection.execute(
            """
            INSERT INTO reviews (
                study_id,
                decision,
                notes,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (study_id, decision, notes, created_at),
        )

        review_id = cursor.lastrowid

        row = connection.execute(
            "SELECT * FROM reviews WHERE id = ?",
            (review_id,),
        ).fetchone()

        return dict(row)


def get_review(review_id: int) -> dict | None:
    """Retrieve a review by its database ID."""
    with get_db() as connection:
        row = connection.execute(
            "SELECT * FROM reviews WHERE id = ?",
            (review_id,),
        ).fetchone()

        return dict(row) if row is not None else None


def get_reviews_for_study(study_id: str) -> list[dict]:
    """Retrieve all reviews associated with a study."""
    with get_db() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM reviews
            WHERE study_id = ?
            ORDER BY id DESC
            """,
            (study_id,),
        ).fetchall()

        return [dict(row) for row in rows]


