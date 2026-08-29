"""
ragas_eval.py — PHASE 10: Custom RAGAS evals.

Re-implements the four core RAGAS metrics as direct Azure OpenAI judge calls,
so you're not dependent on the `ragas` package's own default OpenAI (non-Azure)
client wiring, and so every metric prompt is auditable clinical-domain text
you control:

  - Faithfulness       — is every claim in the answer actually supported by
                          the retrieved context? (hallucination check)
  - Context Precision   — of the chunks retrieved, how many were relevant?
  - Context Recall      — of the facts needed to answer, how many did the
                          retrieved context actually contain?
  - Answer Relevance    — does the answer actually address the question?

Each returns a 0.0-1.0 score. Run this as a nightly Azure Function / DevOps
pipeline step over a held-out eval set (e.g. the 300-case clinical fact
retrieval suite) to catch retrieval or prompt regressions before they reach
clinicians.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List

from config import get_secret

JUDGE_DEPLOYMENT_DEFAULT = "gpt-4o"


@dataclass
class RagasResult:
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevance: float

    @property
    def overall(self) -> float:
        return round((self.faithfulness + self.context_precision + self.context_recall
                       + self.answer_relevance) / 4, 4)


def _judge_call(azure_openai_client, deployment: str, system: str, user: str) -> dict:
    response = azure_openai_client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Defensive: strip markdown fences if the model added them despite json_object mode.
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned)


def score_faithfulness(azure_openai_client, answer: str, retrieved_context: str, deployment: str) -> float:
    system = ("You are a strict clinical fact-checker. Break the ANSWER into individual factual "
              "claims. For each claim, decide if it is directly supported by the CONTEXT. "
              "Return JSON: {\"total_claims\": int, \"supported_claims\": int}.")
    user = f"CONTEXT:\n{retrieved_context}\n\nANSWER:\n{answer}"
    result = _judge_call(azure_openai_client, deployment, system, user)
    total = max(1, int(result.get("total_claims", 1)))
    supported = int(result.get("supported_claims", 0))
    return round(min(1.0, supported / total), 4)


def score_context_precision(azure_openai_client, question: str, retrieved_chunks: List[str], deployment: str) -> float:
    system = ("For each numbered CONTEXT CHUNK, decide if it is relevant to answering the QUESTION. "
              "Return JSON: {\"relevant_indices\": [ints]}.")
    numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(retrieved_chunks))
    user = f"QUESTION:\n{question}\n\nCONTEXT CHUNKS:\n{numbered}"
    if not retrieved_chunks:
        return 0.0
    result = _judge_call(azure_openai_client, deployment, system, user)
    relevant = len(result.get("relevant_indices", []))
    return round(relevant / len(retrieved_chunks), 4)


def score_context_recall(azure_openai_client, ground_truth_answer: str, retrieved_context: str, deployment: str) -> float:
    system = ("Break the GROUND TRUTH ANSWER into individual factual claims needed to answer the "
              "question fully. For each, decide if that fact is present somewhere in the CONTEXT. "
              "Return JSON: {\"total_facts\": int, \"facts_found_in_context\": int}.")
    user = f"CONTEXT:\n{retrieved_context}\n\nGROUND TRUTH ANSWER:\n{ground_truth_answer}"
    result = _judge_call(azure_openai_client, deployment, system, user)
    total = max(1, int(result.get("total_facts", 1)))
    found = int(result.get("facts_found_in_context", 0))
    return round(min(1.0, found / total), 4)


def score_answer_relevance(azure_openai_client, question: str, answer: str, deployment: str) -> float:
    system = ("Rate 0.0-1.0 how directly and completely the ANSWER addresses the QUESTION asked "
              "(ignore factual correctness — that's scored elsewhere; only judge relevance/on-topic-ness). "
              "Return JSON: {\"relevance_score\": float}.")
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    result = _judge_call(azure_openai_client, deployment, system, user)
    return round(max(0.0, min(1.0, float(result.get("relevance_score", 0.0)))), 4)


def evaluate(azure_openai_client, question: str, answer: str, retrieved_chunks: List[str],
             ground_truth_answer: str, deployment: str | None = None) -> RagasResult:
    deployment = deployment or get_secret("AZURE_OPENAI_JUDGE_DEPLOYMENT", JUDGE_DEPLOYMENT_DEFAULT)
    retrieved_context = "\n".join(retrieved_chunks)

    return RagasResult(
        faithfulness=score_faithfulness(azure_openai_client, answer, retrieved_context, deployment),
        context_precision=score_context_precision(azure_openai_client, question, retrieved_chunks, deployment),
        context_recall=score_context_recall(azure_openai_client, ground_truth_answer, retrieved_context, deployment),
        answer_relevance=score_answer_relevance(azure_openai_client, question, answer, deployment),
    )
