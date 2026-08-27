"""
Audit trail — first-class decision logging.

Every decision, model version, threshold version, and failure event
is logged to SQLite so "why did the system decide this on that day"
is always answerable.

This is NOT an afterthought. The audit trail is a core design element
of the system, treated as essential infrastructure for compliance and
debugging in a financial context.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import AUDIT_DB_PATH, AUDIT_TABLE_NAME


class AuditTrail:
    """
    SQLite-backed audit logger for return decisions.

    Thread-safe for single-writer scenarios (which covers a demo).
    For production, use PostgreSQL or a dedicated audit service.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or AUDIT_DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        """Create the decisions table if it doesn't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {AUDIT_TABLE_NAME} (
                    decision_id TEXT PRIMARY KEY,
                    return_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tabular_score REAL,
                    semantic_similarity REAL,
                    empty_box_flag INTEGER,
                    modality_confidence REAL,
                    trust_score REAL,
                    decision TEXT NOT NULL,
                    nudge_type TEXT,
                    nudge_message TEXT,
                    rephoto_count INTEGER DEFAULT 0,
                    prior_store_credit_count INTEGER DEFAULT 0,
                    failure_event TEXT,
                    failure_details TEXT,
                    effective_approve_threshold REAL,
                    effective_review_threshold REAL,
                    decision_reason TEXT,
                    forced_by TEXT,
                    model_version TEXT,
                    config_snapshot TEXT
                )
            """)
            conn.commit()

            # ── Schema migration: add columns that may be absent in old DBs ──
            # SQLite has no "ADD COLUMN IF NOT EXISTS"; catch OperationalError instead.
            _migrations = [
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN nudge_type TEXT",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN nudge_message TEXT",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN rephoto_count INTEGER DEFAULT 0",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN prior_store_credit_count INTEGER DEFAULT 0",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN failure_event TEXT",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN failure_details TEXT",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN effective_approve_threshold REAL",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN effective_review_threshold REAL",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN decision_reason TEXT",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN forced_by TEXT",
                f"ALTER TABLE {AUDIT_TABLE_NAME} ADD COLUMN config_snapshot TEXT",
            ]
            for stmt in _migrations:
                try:
                    conn.execute(stmt)
                    conn.commit()
                except Exception:
                    pass  # Column already exists — this is expected for fresh DBs

    def log_decision(
        self,
        return_id: str,
        tabular_score: Optional[float],
        semantic_similarity: Optional[float],
        empty_box_flag: Optional[int],
        modality_confidence: float,
        trust_score: float,
        decision: str,
        nudge_type: Optional[str] = None,
        nudge_message: Optional[str] = None,
        rephoto_count: int = 0,
        prior_store_credit_count: int = 0,
        failure_event: Optional[str] = None,
        failure_details: Optional[str] = None,
        effective_approve_threshold: Optional[float] = None,
        effective_review_threshold: Optional[float] = None,
        decision_reason: Optional[str] = None,
        forced_by: Optional[str] = None,
        model_version: Optional[str] = None,
        config_snapshot: Optional[dict] = None,
    ) -> str:
        """
        Log a single decision to the audit trail.

        Returns the generated decision_id for reference.
        """
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                f"""
                INSERT INTO {AUDIT_TABLE_NAME} (
                    decision_id, return_id, timestamp,
                    tabular_score, semantic_similarity, empty_box_flag,
                    modality_confidence, trust_score, decision,
                    nudge_type, nudge_message, rephoto_count,
                    prior_store_credit_count,
                    failure_event, failure_details,
                    effective_approve_threshold, effective_review_threshold,
                    decision_reason, forced_by,
                    model_version, config_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id, return_id, timestamp,
                    tabular_score, semantic_similarity, empty_box_flag,
                    modality_confidence, trust_score, decision,
                    nudge_type, nudge_message, rephoto_count,
                    prior_store_credit_count,
                    failure_event, failure_details,
                    effective_approve_threshold, effective_review_threshold,
                    decision_reason, forced_by,
                    model_version,
                    json.dumps(config_snapshot) if config_snapshot else None,
                ),
            )
            conn.commit()

        return decision_id

    def get_decisions_for_return(self, return_id: str) -> list[dict]:
        """Get all decisions made for a specific return request."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"SELECT * FROM {AUDIT_TABLE_NAME} WHERE return_id = ? ORDER BY timestamp",
                (return_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_decisions(self, limit: int = 50) -> list[dict]:
        """Get the most recent decisions."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"SELECT * FROM {AUDIT_TABLE_NAME} ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get aggregate statistics on decisions made."""
        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}"
            ).fetchone()[0]

            decision_counts = {}
            for row in conn.execute(
                f"SELECT decision, COUNT(*) FROM {AUDIT_TABLE_NAME} GROUP BY decision"
            ):
                decision_counts[row[0]] = row[1]

            failure_count = conn.execute(
                f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME} WHERE failure_event IS NOT NULL"
            ).fetchone()[0]

            forced_count = conn.execute(
                f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME} WHERE forced_by IS NOT NULL"
            ).fetchone()[0]

        return {
            "total_decisions": total,
            "decision_counts": decision_counts,
            "failure_events": failure_count,
            "forced_decisions": forced_count,
        }

    def get_failure_events(self) -> list[dict]:
        """Get all decisions that involved a failure event."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"SELECT * FROM {AUDIT_TABLE_NAME} WHERE failure_event IS NOT NULL ORDER BY timestamp DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
