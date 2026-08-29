"""
audit.py — asynchronous, append-only forensic audit trail.

Writes happen on a background thread pool so the Streamlit main thread is
never blocked by disk or network I/O. Always writes local JSONL; optionally
mirrors to an Azure Blob append-blob for a tamper-evident off-box copy.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("clinical_rag.audit")


class AsyncForensicAuditLogger:
    """
    Append-only, structured audit trail. Writes happen on a background
    thread pool so the caller's main thread is never blocked by disk or
    network I/O. Writes to local JSONL always; additionally mirrors to an
    Azure Blob append-blob when Blob Storage is configured, for a
    tamper-evident off-box copy suitable for compliance review.
    """

    def __init__(self, local_path: str = "audit_log.jsonl",
                 blob_container_client=None, blob_name: str = "audit_log.jsonl"):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="audit")
        self.local_path = local_path
        self.blob_container_client = blob_container_client
        self.blob_name = blob_name

    def log_event(self, event: Dict[str, Any]) -> None:
        self._executor.submit(self._write_event, dict(event))

    def _write_event(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, default=str) + "\n"

        try:
            with open(self.local_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:  # noqa: BLE001
            logger.error("Local audit trail write failed: %s", exc)

        if self.blob_container_client is not None:
            try:
                blob_client = self.blob_container_client.get_blob_client(self.blob_name)
                if not blob_client.exists():
                    blob_client.create_append_blob()
                blob_client.append_block(line)
            except Exception as exc:  # noqa: BLE001
                logger.error("Azure Blob audit trail write failed: %s", exc)

    def read_recent(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.local_path):
            return []
        events: List[Dict[str, Any]] = []
        with open(self.local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events[-limit:]


def make_audit_event(action: str, doctor_id: str, patient_id: str,
                      detail: Optional[Dict[str, Any]] = None,
                      context_hash: Optional[str] = None) -> Dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "actor_doctor_id": doctor_id,
        "subject_patient_id": patient_id,
        "action": action,
        "context_sha256": context_hash,
        "detail": detail or {},
    }
