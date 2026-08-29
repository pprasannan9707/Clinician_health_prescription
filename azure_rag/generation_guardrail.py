"""
generation_guardrail.py — PHASE 9: Generation & Safety Guardrail.

Two parts:
  1. generate_summary()     — the actual Azure OpenAI GPT-4o call
  2. verify_output_safety() — a DETERMINISTIC (non-LLM) rule-based checker
     that runs on every generation before it's shown to the clinician. This
     is not another model call — it's plain string/regex matching against
     the same pinned safety flags that were placed at the bottom of the
     U-shaped prompt, so a guardrail failure can never itself hallucinate.

If verify_output_safety() finds a conflict, the summary is NOT auto-blocked
(a clinician should always see it) — it's flagged with a hard, visually
distinct warning banner, and the conflict is written to the audit trail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from config import CLINICAL_SYSTEM_PROMPT, get_secret


@dataclass
class SafetyVerificationResult:
    passed: bool
    conflicts: List[str] = field(default_factory=list)


def generate_summary(azure_openai_client, compiled_context: str, chat_deployment: str | None = None) -> str:
    deployment = chat_deployment or get_secret("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
    if azure_openai_client is None:
        return "[MOCK MODE] Compiled context ready for review; Azure OpenAI not configured."

    response = azure_openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
            {"role": "user", "content": compiled_context},
        ],
        temperature=0.1,
        max_tokens=900,
    )
    return response.choices[0].message.content or "(empty response from model)"


def _extract_allergens(compiled_context: str) -> List[str]:
    """Pull allergen names out of the pinned '*** ALLERGY FLAG ***' lines."""
    allergens = []
    for line in compiled_context.splitlines():
        m = re.search(r"\*\*\* ALLERGY FLAG \*\*\* (.+?) —", line)
        if m:
            allergens.append(m.group(1).strip().lower())
    return allergens


def _has_ckd_flag(compiled_context: str) -> bool:
    return "CKD-RANGE" in compiled_context


def verify_output_safety(compiled_context: str, generated_summary: str) -> SafetyVerificationResult:
    """
    Deterministic checks:
      A. The model must not recommend a drug that matches (substring, case-
         insensitive) a pinned allergen name.
      B. If a CKD-range eGFR flag is pinned, the summary must at least
         mention renal/dose adjustment language when it discusses medication.
    Extend this with your organization's actual drug-interaction rule set —
    this is intentionally simple and auditable, not a substitute for a real
    pharmacology engine.
    """
    conflicts: List[str] = []
    summary_lower = generated_summary.lower()

    for allergen in _extract_allergens(compiled_context):
        if allergen and allergen in summary_lower:
            # crude but auditable: flag any co-occurrence of allergen name with
            # a recommendation verb nearby
            if re.search(rf"(recommend|prescribe|start|continue|administer)[^.]{{0,80}}{re.escape(allergen)}",
                          summary_lower):
                conflicts.append(f"Generated summary appears to recommend '{allergen}', which is a pinned allergy flag.")

    if _has_ckd_flag(compiled_context):
        mentions_medication = bool(re.search(r"(medication|drug|dose|mg\b)", summary_lower))
        mentions_renal = bool(re.search(r"(renal|kidney|egfr|dose[- ]adjust)", summary_lower))
        if mentions_medication and not mentions_renal:
            conflicts.append("Patient has a pinned CKD-range eGFR flag, but the summary discusses "
                              "medication without mentioning renal dose adjustment.")

    return SafetyVerificationResult(passed=(len(conflicts) == 0), conflicts=conflicts)
