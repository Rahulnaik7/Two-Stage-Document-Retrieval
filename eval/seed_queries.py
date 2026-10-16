"""Generates the custom QA benchmark used by `eval/run_benchmark.py`.

Two natural-language questions per corpus document (68 total), each
labeled with the document it should retrieve. Questions are written as
paraphrases of the source text rather than keyword-for-keyword lifts —
several deliberately reuse a word that also appears in a lexically
confusable neighbor (e.g. "caching" appears in both the HTTP-caching and
CPU-cache documents) specifically to give a pure term-overlap ranker room
to pick the wrong one, which is what the benchmark is designed to probe.

    python -m eval.seed_queries
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "qa_benchmark.jsonl"

# (query_id, ground_truth document_id, question)
_QUERIES: List[tuple[str, str, str]] = [
    ("q01", "semantic-chunking", "How does semantic drop chunking decide where to split a document?"),
    ("q02", "semantic-chunking", "Why does keeping related sentences in one chunk help retrieval accuracy?"),

    ("q03", "stage-one", "What role does the vector index play in the first retrieval stage?"),
    ("q04", "stage-one", "Why are the query and each chunk embedded separately before searching?"),

    ("q05", "stage-two", "How does the second stage re-score candidates from the first stage?"),
    ("q06", "stage-two", "What's the benefit of scoring the question and passage together instead of comparing embeddings?"),

    ("q07", "answering", "What rule keeps the generated answer from making up unsupported claims?"),
    ("q08", "answering", "How should a grounded response treat the evidence it was given?"),

    ("q09", "http-caching", "Which response header lets a browser avoid re-downloading a page that hasn't changed?"),
    ("q10", "http-caching", "What status code tells a client its cached copy of a web page is still valid?"),

    ("q11", "cpu-caching", "Why do processors keep a small, fast memory close to each core?"),
    ("q12", "cpu-caching", "What happens to a CPU's pipeline when the data it needs isn't in cache?"),

    ("q13", "message-queues", "How do systems like RabbitMQ let a producer keep working without waiting on the consumer?"),
    ("q14", "message-queues", "What happens to a job that keeps failing to process from a queue?"),

    ("q15", "priority-queue", "What data structure is typically used to always return the most urgent item first?"),
    ("q16", "priority-queue", "Why is a binary heap a good fit for Dijkstra's shortest-path algorithm?"),

    ("q17", "db-sharding", "What problem can happen if rows aren't spread evenly across a sharded database?"),
    ("q18", "db-sharding", "Why do cross-shard transactions get expensive once a table is split across servers?"),

    ("q19", "css-flexbox", "Which CSS properties control how items are spaced out inside a flex container?"),
    ("q20", "css-flexbox", "Why is flexbox well suited to building something like a navigation bar?"),

    ("q21", "oauth2", "How does a user let an app access their account without handing over a password?"),
    ("q22", "oauth2", "What lets a client get a new access token after the old one has expired?"),

    ("q23", "jwt", "Why shouldn't sensitive information be stored inside a signed token's payload?"),
    ("q24", "jwt", "How can a server verify a bearer token without checking a database?"),

    ("q25", "websockets", "What kind of connection lets a server push updates to a client without being polled?"),
    ("q26", "websockets", "Why would a chat application prefer this protocol over regular HTTP requests?"),

    ("q27", "rate-limiting", "Which algorithm allows short bursts of traffic while still enforcing an average limit?"),
    ("q28", "rate-limiting", "Why would an API cap how many requests a single client can make?"),

    ("q29", "cap-theorem", "Which two guarantees does a distributed store have to choose between during a network partition?"),
    ("q30", "cap-theorem", "Why can't a distributed database promise consistency, availability, and partition tolerance all at once?"),

    ("q31", "raft-consensus", "How does a cluster pick a new leader if the current one stops responding?"),
    ("q32", "raft-consensus", "When is a replicated log entry considered safely committed?"),

    ("q33", "load-balancing", "How does a load balancer stop sending traffic to a server that's down?"),
    ("q34", "load-balancing", "What are some strategies for spreading requests across a pool of servers?"),

    ("q35", "git-rebase", "What's the difference between replaying commits on a new base and merging two histories?"),
    ("q36", "git-rebase", "Why can rewriting shared commit history cause trouble for collaborators?"),

    ("q37", "python-asyncio", "How does a single-threaded event loop handle many I/O tasks concurrently?"),
    ("q38", "python-asyncio", "What happens if a coroutine never yields control back to the event loop?"),

    ("q39", "gil", "Why doesn't adding more threads speed up CPU-bound Python code?"),
    ("q40", "gil", "What's an alternative to threads for parallelizing CPU-heavy work in Python?"),

    ("q41", "b-tree-index", "Why can a lookup in a large table touch only a few disk pages?"),
    ("q42", "b-tree-index", "What kind of tree structure do most relational databases use for indexing?"),

    ("q43", "bloom-filter", "What kind of answer does a space-efficient membership structure give — definite or probabilistic?"),
    ("q44", "bloom-filter", "How do databases use this structure to avoid unnecessary disk reads?"),

    ("q45", "docker-layers", "Why does the order of instructions in a Dockerfile affect build speed?"),
    ("q46", "docker-layers", "What gets added on top of the read-only layers when a container actually runs?"),

    ("q47", "kubernetes-pods", "What gives a set of ephemeral pods a stable address that other services can rely on?"),
    ("q48", "kubernetes-pods", "What's the smallest unit you can deploy in this container orchestration system?"),

    ("q49", "regex-backtracking", "What can cause a pattern match to take exponentially longer as input grows?"),
    ("q50", "regex-backtracking", "How does a backtracking engine recover after a chosen alternative fails to match?"),

    ("q51", "unicode-encoding", "Why can a string's length in bytes differ from its length in characters?"),
    ("q52", "unicode-encoding", "Which encoding stays backward compatible with the first 128 ASCII characters?"),

    ("q53", "cron-scheduling", "What problem can occur if a scheduled job takes longer to run than its interval?"),
    ("q54", "cron-scheduling", "What fields make up a recurring schedule expression?"),

    ("q55", "dns-resolution", "What's cached at each step while a domain name is being resolved?"),
    ("q56", "dns-resolution", "What trade-off comes with setting a low TTL on a DNS record?"),

    ("q57", "tcp-handshake", "What three messages are exchanged before a reliable connection is considered open?"),
    ("q58", "tcp-handshake", "Why do both sides need to synchronize sequence numbers before sending data?"),

    ("q59", "garbage-collection", "Why are young objects collected more often than old ones in a generational collector?"),
    ("q60", "garbage-collection", "What happens to an object that survives enough collection cycles?"),

    ("q61", "circuit-breaker", "What does a resilience pattern do after a dependency fails repeatedly?"),
    ("q62", "circuit-breaker", "How does short-circuiting calls to a failing service help it recover?"),

    ("q63", "idempotency-keys", "How can a client safely retry a payment request without risking a double charge?"),
    ("q64", "idempotency-keys", "What does a server store so it can return the same result for a repeated request?"),

    ("q65", "vector-quantization", "How does splitting a vector into sub-vectors help compress a large embedding index?"),
    ("q66", "vector-quantization", "What's the trade-off of using an approximate search technique instead of an exact one?"),

    ("q67", "bm25-ranking", "How does a lexical ranking function keep extremely long documents from dominating results?"),
    ("q68", "bm25-ranking", "Which factors does a classical scoring function combine besides term frequency?"),

    ("q69", "connection-pooling", "Why is reusing an already-open database connection cheaper than opening a new one each time?"),
    ("q70", "connection-pooling", "What does a request do with a pooled connection once it's finished?"),
]


def build_dataset() -> List[dict]:
    return [
        {"id": qid, "document_id": doc_id, "question": question}
        for qid, doc_id, question in _QUERIES
    ]


def main() -> None:
    examples = build_dataset()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} questions to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
