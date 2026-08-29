"""
context_assembler.py — PHASE 6: reassemble deduped parent+child chunks into
the final U-shaped prompt.

After retrieval -> rerank -> parent-dedup, we have a short list of
RetrievedChild objects, each carrying both the precise matching `child_text`
AND the full `parent_text` it came from. This module:

  1. Classifies each surviving parent by section (guideline vs. general
     history vs. safety-critical: allergy / organ-dysfunction lab)
  2. SHA-256 de-dupes at the TEXT level too (belt-and-suspenders — Redis
     dedup was by parent_id, this catches near-identical parent text that
     happened to have different parent_ids, e.g. two encounters that
     produced an identical note)
  3. Places them in the same strict U-shape used by the pandas-only
     pipeline: guidelines TOP, general history MIDDLE, safety flags BOTTOM
"""

from __future__ import annotations

import hashlib
from typing import List

from retrieval_rerank_dedup import RetrievedChild

SAFETY_SECTIONS = {"allergies", "organ_dysfunction_labs"}


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _dedupe_text(chunks: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in chunks:
        normalized = " ".join(c.split())
        if not normalized:
            continue
        h = sha256_hash(normalized)
        if h in seen:
            continue
        seen.add(h)
        out.append(normalized)
    return out


def assemble_u_shaped_context(guideline_children: List[RetrievedChild],
                               patient_children: List[RetrievedChild]) -> str:
    """
    guideline_children -> from medical-knowledge-index (already deduped)
    patient_children    -> from patient-index (already deduped); this
                            function further splits them into
                            "safety" vs "general history" by the `section`
                            metadata each child carries.
    """
    guideline_parents = _dedupe_text([c.parent_text for c in guideline_children])

    safety_parents: List[str] = []
    history_parents: List[str] = []
    for c in patient_children:
        section = (c.metadata or {}).get("section", "")
        target = safety_parents if section in SAFETY_SECTIONS else history_parents
        target.append(c.parent_text)

    safety_parents = _dedupe_text(safety_parents)
    history_parents = _dedupe_text(history_parents)

    lines: List[str] = []
    lines.append("################  TOP — EXTERNAL CLINICAL PRACTICE GUIDELINES  ################")
    lines.extend(f"- {p}" for p in (guideline_parents or ["No guideline chunks retrieved for this query."]))
    lines.append("")
    lines.append("################  MIDDLE — PATIENT CLINICAL HISTORY (retrieved & reranked)  ################")
    lines.extend(f"- {p}" for p in (history_parents or ["No additional history chunks retrieved."]))
    lines.append("")
    lines.append("################  BOTTOM (PINNED) — CRITICAL SAFETY FLAGS  ################")
    lines.append("### Allergy and organ-dysfunction flags below OVERRIDE any conflicting guidance above.")
    lines.extend(f"- {p}" for p in (safety_parents or ["No allergy or organ-dysfunction chunks retrieved."]))

    return "\n".join(lines)
