"""
redis_memory.py — PHASE 8: Ongoing Memory Layer (Azure Cache for Redis).

Three uses, all on the same Redis instance:
  1. Session memory   — short conversational context per (doctor, patient) session
  2. Response cache   — identical query+context hash -> skip regeneration
  3. Dedup keys        — used by retrieval_rerank_dedup.py (documented there)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import redis

REDIS_HOST = os.environ.get("AZURE_REDIS_HOST")
REDIS_PORT = int(os.environ.get("AZURE_REDIS_PORT", "6380"))
REDIS_PASSWORD = os.environ.get("AZURE_REDIS_PASSWORD")

SESSION_TTL_SECONDS = 60 * 30       # 30-minute rolling session window
RESPONSE_CACHE_TTL_SECONDS = 60 * 15  # 15 minutes — clinical data changes; keep this short


def _client() -> Optional["redis.Redis"]:
    if not REDIS_HOST:
        return None
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, ssl=True, decode_responses=True)


def _session_key(doctor_id: str, patient_id: str) -> str:
    return f"session:{doctor_id}:{patient_id}"


def append_turn(doctor_id: str, patient_id: str, role: str, content: str, max_turns: int = 10) -> None:
    r = _client()
    if r is None:
        return
    key = _session_key(doctor_id, patient_id)
    r.rpush(key, json.dumps({"role": role, "content": content}))
    r.ltrim(key, -max_turns, -1)
    r.expire(key, SESSION_TTL_SECONDS)


def get_session_turns(doctor_id: str, patient_id: str) -> List[Dict[str, str]]:
    r = _client()
    if r is None:
        return []
    key = _session_key(doctor_id, patient_id)
    raw_turns = r.lrange(key, 0, -1)
    return [json.loads(t) for t in raw_turns]


def clear_session(doctor_id: str, patient_id: str) -> None:
    r = _client()
    if r is None:
        return
    r.delete(_session_key(doctor_id, patient_id))


def _cache_key(context_hash: str) -> str:
    return f"response_cache:{context_hash}"


def get_cached_response(context_hash: str) -> Optional[str]:
    r = _client()
    if r is None:
        return None
    return r.get(_cache_key(context_hash))


def set_cached_response(context_hash: str, response_text: str) -> None:
    r = _client()
    if r is None:
        return
    r.set(_cache_key(context_hash), response_text, ex=RESPONSE_CACHE_TTL_SECONDS)
