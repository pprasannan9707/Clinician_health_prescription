"""
chunking.py — PHASE 2a: Parent-Child chunking strategy.

Parent chunk  = a full logical unit (one encounter note, one care-plan
                section, one guideline sub-section) — large enough to carry
                real clinical/contextual meaning, too large to embed well.
Child chunks  = small, dense slices of a parent (1-3 sentences / ~200-400
                tokens) — what we actually embed and search over, because
                small chunks match a query's specific fact much better than
                a large one (better recall, tighter cosine similarity).

Each child stores `parent_id` and a full copy of `parent_text`, so a single
Azure AI Search hit gives us both the precise matching sentence AND its full
surrounding context in one round-trip — no second fetch needed.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

# Tune to your embedding model's effective context and your reranker's input limit.
CHILD_CHUNK_TARGET_TOKENS = 300
CHILD_CHUNK_OVERLAP_TOKENS = 40


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    parent_text: str
    child_text: str
    metadata: dict = field(default_factory=dict)


def _approx_token_count(text: str) -> int:
    # Cheap approximation (~4 chars/token for English clinical text) — swap
    # for a real tokenizer (tiktoken) if you need exact budgeting.
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> List[str]:
    # Clinical-note-aware sentence splitter: keeps "Dr." / "mg." / "e.g." intact.
    abbrev_guard = re.sub(r"\b(Dr|Mr|Mrs|Ms|vs|e\.g|i\.e|mg|mL|approx)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+", abbrev_guard)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def chunk_parent_into_children(parent_id: str, parent_text: str, metadata: dict | None = None) -> List[ChildChunk]:
    """
    Greedy sentence-packing chunker: walks sentence-by-sentence, packing into
    a child chunk until the token budget is hit, then starts a new child with
    a small sentence-level overlap for continuity across the boundary.
    """
    metadata = metadata or {}
    sentences = _split_sentences(parent_text)
    if not sentences:
        return []

    children: List[ChildChunk] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if not current:
            return
        child_text = " ".join(current)
        children.append(ChildChunk(
            child_id=str(uuid.uuid4()),
            parent_id=parent_id,
            parent_text=parent_text,
            child_text=child_text,
            metadata=metadata,
        ))
        # Overlap: keep the tail sentences that fit within the overlap budget.
        overlap: List[str] = []
        overlap_tokens = 0
        for s in reversed(current):
            t = _approx_token_count(s)
            if overlap_tokens + t > CHILD_CHUNK_OVERLAP_TOKENS:
                break
            overlap.insert(0, s)
            overlap_tokens += t
        current = overlap
        current_tokens = overlap_tokens

    for sentence in sentences:
        t = _approx_token_count(sentence)
        if current_tokens + t > CHILD_CHUNK_TARGET_TOKENS and current:
            flush()
        current.append(sentence)
        current_tokens += t

    flush()
    return children


def chunk_patient_bundle(patient_id: str, section_texts: dict[str, str]) -> List[ChildChunk]:
    """
    section_texts: {"allergies": "...", "medications": "...", "conditions": "...", ...}
    One PARENT per section per patient (e.g. "all of this patient's medication
    history" is one parent) — keeps parent granularity coarse enough to be a
    genuinely useful unit of context once reassembled.
    """
    all_children: List[ChildChunk] = []
    for section, text in section_texts.items():
        if not text or not text.strip():
            continue
        parent_id = f"{patient_id}::{section}"
        all_children.extend(
            chunk_parent_into_children(parent_id, text, metadata={"patient_id": patient_id, "section": section})
        )
    return all_children


def chunk_guideline_document(doc_id: str, disease: str, source_name: str, full_text: str,
                              publication_year: int | None = None) -> List[ChildChunk]:
    """One parent per guideline document/sub-section; children are the searchable slices."""
    return chunk_parent_into_children(
        parent_id=doc_id,
        parent_text=full_text,
        metadata={"disease": disease, "source_name": source_name, "publication_year": publication_year},
    )
