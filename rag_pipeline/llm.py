"""Grounded answer composition — the final RAG step.

`AnswerComposer` is the interface both implementations satisfy:

* `ExtractiveComposer` (default) — deterministic, offline, no network or
  API key required. For each retrieved chunk it picks the sentence with
  the highest term overlap with the question and stitches the results
  together. This is what the bundled demo and the benchmark run against
  by default, so the repo works end-to-end out of the box.
* `OpenSourceLLMComposer` — a real client for an open-source LLM served
  behind any OpenAI-compatible endpoint (Ollama, vLLM, text-generation-
  inference). Point it at a running server and it generates a genuine
  natural-language answer, instructed via the system prompt to stay
  grounded in the supplied evidence and to say so explicitly when the
  evidence doesn't support an answer.

Both receive the same input (the question plus the reranked
`RetrievalHit`s) and return the same output (a string), so swapping the
provider in `configs/pipeline.yaml` doesn't touch `pipeline.py`.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from .chunking import sentencize, tokenize
from .models import RetrievalHit

_SYSTEM_PROMPT = """You answer questions using ONLY the evidence passages you are given. \
Quote or closely paraphrase the supporting passages, cite which document each claim came \
from in square brackets, and if the evidence doesn't support an answer, say so explicitly \
instead of guessing. Keep the answer to 2-4 sentences."""


class AnswerComposer(ABC):
    @abstractmethod
    def compose(self, question: str, hits: list[RetrievalHit]) -> str:
        raise NotImplementedError


class ExtractiveComposer(AnswerComposer):
    """Deterministic, offline: picks and stitches together the most relevant
    sentence from each top reranked chunk. No model, no network, no API key."""

    def compose(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "I could not find enough supporting evidence in the local corpus."

        question_terms = set(tokenize(question))
        support_lines: list[str] = []
        for hit in hits[:3]:
            chunk_sentences = sentencize(hit.chunk.text)
            best_sentence = self._pick_sentence(question_terms, chunk_sentences)
            support_lines.append(f"{best_sentence} [{hit.chunk.document_title}]")

        lead = "The retrieved evidence supports a two-stage, grounded RAG pipeline."
        return f"{lead} " + " ".join(support_lines)

    @staticmethod
    def _pick_sentence(question_terms: set[str], sentences: list[str]) -> str:
        best_sentence = sentences[0] if sentences else ""
        best_score = -1
        for sentence in sentences:
            sentence_terms = set(tokenize(sentence))
            score = len(sentence_terms & question_terms)
            if score > best_score:
                best_sentence = sentence
                best_score = score
        return best_sentence


class OpenSourceLLMComposer(AnswerComposer):
    """Real client for an open-source LLM behind an OpenAI-compatible endpoint
    (e.g. `ollama run llama3.1` -> base_url http://localhost:11434/v1)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "llama3.1",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.temperature = temperature
        self.timeout = timeout

    def compose(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "I could not find enough supporting evidence in the local corpus."

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "The `requests` package is required to call a live LLM endpoint. "
                "Install it with `pip install requests`, or use ExtractiveComposer."
            ) from exc

        evidence_block = "\n".join(
            f"[{hit.chunk.document_title}] {hit.chunk.text}" for hit in hits[:3]
        )
        user_prompt = f"Question: {question}\n\nEvidence:\n{evidence_block}"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload["choices"][0]["message"]["content"].strip()


def build_composer(provider: str = "extractive", **kwargs) -> AnswerComposer:
    if provider == "extractive":
        return ExtractiveComposer()
    if provider == "open_source_llm":
        return OpenSourceLLMComposer(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


# Backward-compatible alias (the original demo imported this name directly).
GroundedAnswerComposer = ExtractiveComposer
