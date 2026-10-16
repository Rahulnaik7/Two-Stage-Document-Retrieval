"""Sample corpus used by the local demo and the retrieval benchmark.

34 short documents: 4 describing the pipeline itself (kept from the
original demo), plus 30 general technical-knowledge documents spanning
distinct areas of software engineering. Several pairs are deliberately
lexically confusable (two "caching" docs, two "queue" docs, two
"sharding"/"partitioning" docs) so that keyword-overlap retrieval alone
has real room to rank the wrong one first — which is what the benchmark
in `eval/` is designed to measure.
"""
from __future__ import annotations

from .models import Document


def build_sample_corpus() -> list[Document]:
    return [
        Document(
            id="semantic-chunking",
            title="Semantic Drop Chunking",
            tags=("chunking", "retrieval"),
            text=(
                "Semantic drop chunking keeps related sentences together until the"
                " local topic changes. When sentence similarity falls below the"
                " threshold, the chunk closes and a new chunk begins. This reduces"
                " fragmentation and preserves the context that retrieval needs to"
                " answer questions accurately."
            ),
        ),
        Document(
            id="stage-one",
            title="Stage One Retrieval",
            tags=("retrieval", "faiss"),
            text=(
                "The first retrieval stage uses a FAISS vector index over bi-encoder"
                " embeddings to find candidate chunks. The bi-encoder embeds the"
                " query and each chunk independently, so the index can be built once"
                " and searched with a single nearest-neighbor lookup, which narrows"
                " the search space quickly before any expensive re-ranking happens."
            ),
        ),
        Document(
            id="stage-two",
            title="Stage Two Re-ranking",
            tags=("reranking", "cross-encoder"),
            text=(
                "The second stage re-scores the stage-one candidates with a"
                " cross-encoder-style joint ranker that reads the question and each"
                " candidate passage together, rather than comparing independent"
                " embeddings. That joint scoring removes noise that a pure vector"
                " search would keep, since lexical and positional overlap between"
                " the query and passage can be weighed directly."
            ),
        ),
        Document(
            id="answering",
            title="Grounded Answer Generation",
            tags=("generation", "llm"),
            text=(
                "Grounded generation should only use the retrieved evidence. The"
                " final response quotes the most relevant claims, summarizes them"
                " clearly, and refuses to invent details that are not supported by"
                " the source chunks."
            ),
        ),

        # -- distinct technical topics (30 docs) --------------------------
        Document(
            id="http-caching", title="HTTP Caching Headers", tags=("http", "caching"),
            text=(
                "HTTP caching is controlled by response headers like Cache-Control,"
                " ETag, and Last-Modified. A client can send a conditional request"
                " with If-None-Match, and the server replies 304 Not Modified when"
                " the cached copy is still valid, avoiding a full response body"
                " transfer and reducing bandwidth."
            ),
        ),
        Document(
            id="cpu-caching", title="CPU Cache Hierarchy", tags=("hardware", "caching"),
            text=(
                "Modern CPUs use a multi-level cache hierarchy — L1, L2, and L3 —"
                " to hide the latency of main memory access. L1 cache is small and"
                " fast, private to each core, while L3 is larger and shared. Cache"
                " misses stall the pipeline, so cache-friendly data layout matters"
                " for performance-sensitive code."
            ),
        ),
        Document(
            id="message-queues", title="Message Queues for Async Processing", tags=("messaging", "architecture"),
            text=(
                "Message queues like RabbitMQ and SQS decouple producers from"
                " consumers, letting a service publish work without waiting for it"
                " to be processed. Consumers pull messages at their own pace, which"
                " smooths out traffic spikes and lets failed jobs be retried or"
                " moved to a dead-letter queue."
            ),
        ),
        Document(
            id="priority-queue", title="Priority Queues and Heaps", tags=("data-structures",),
            text=(
                "A priority queue returns the element with the highest priority"
                " first, and is commonly implemented with a binary heap. Insert and"
                " extract-min both run in O(log n) time, which makes heaps the"
                " standard choice for algorithms like Dijkstra's shortest path and"
                " event-driven simulation."
            ),
        ),
        Document(
            id="db-sharding", title="Database Sharding", tags=("databases", "scaling"),
            text=(
                "Sharding splits a database horizontally across multiple servers by"
                " a shard key, so each shard holds a disjoint subset of rows. This"
                " lets write throughput scale beyond a single machine, but cross-shard"
                " joins and transactions become expensive, and a poorly chosen shard"
                " key can create hot spots."
            ),
        ),
        Document(
            id="css-flexbox", title="CSS Flexbox Layout", tags=("css", "frontend"),
            text=(
                "Flexbox arranges items along a single axis — row or column — inside"
                " a flex container. Properties like justify-content and align-items"
                " control spacing and alignment, while flex-grow and flex-shrink let"
                " items expand or shrink to fill available space, making it well"
                " suited for one-dimensional layouts like navigation bars."
            ),
        ),
        Document(
            id="oauth2", title="OAuth 2.0 Authorization", tags=("security", "auth"),
            text=(
                "OAuth 2.0 lets a user grant a third-party application limited"
                " access to their resources without sharing a password. The"
                " authorization-code flow exchanges a short-lived code for an access"
                " token via a back-channel request, and refresh tokens let the"
                " client obtain new access tokens after the old one expires."
            ),
        ),
        Document(
            id="jwt", title="JSON Web Tokens", tags=("security", "auth"),
            text=(
                "A JSON Web Token encodes claims in a signed, base64url-encoded"
                " payload that a server can verify without a database lookup. JWTs"
                " are commonly used as bearer tokens for stateless authentication,"
                " but because the payload is only signed, not encrypted, sensitive"
                " data should not be placed in the claims."
            ),
        ),
        Document(
            id="websockets", title="WebSocket Protocol", tags=("networking", "realtime"),
            text=(
                "WebSockets upgrade an HTTP connection into a persistent, full-duplex"
                " channel, letting a server push messages to a client without the"
                " client polling for updates. This makes WebSockets well suited for"
                " chat applications, live dashboards, and multiplayer games where"
                " low-latency, bidirectional messaging matters."
            ),
        ),
        Document(
            id="rate-limiting", title="API Rate Limiting", tags=("api", "reliability"),
            text=(
                "Rate limiting caps how many requests a client can make in a time"
                " window, protecting a service from abuse or accidental overload."
                " Common algorithms include the token bucket, which refills at a"
                " steady rate and allows short bursts, and the sliding-window"
                " counter, which smooths the limit across overlapping windows."
            ),
        ),
        Document(
            id="cap-theorem", title="The CAP Theorem", tags=("distributed-systems",),
            text=(
                "The CAP theorem states that a distributed data store can only"
                " guarantee two of consistency, availability, and partition"
                " tolerance at the same time. Since network partitions are"
                " unavoidable in practice, real systems choose between favoring"
                " consistency (CP) or availability (AP) when a partition occurs."
            ),
        ),
        Document(
            id="raft-consensus", title="Raft Consensus Algorithm", tags=("distributed-systems", "consensus"),
            text=(
                "Raft achieves consensus among replicated servers by electing a"
                " leader that handles all client writes and replicates a log to"
                " followers. If the leader fails, servers hold a randomized-timeout"
                " election to pick a new one, and a log entry is considered"
                " committed once it is replicated to a majority of servers."
            ),
        ),
        Document(
            id="load-balancing", title="Load Balancing Strategies", tags=("networking", "scaling"),
            text=(
                "A load balancer distributes incoming requests across a pool of"
                " backend servers using a strategy like round robin, least"
                " connections, or weighted routing based on server capacity. Health"
                " checks remove unresponsive backends from rotation automatically,"
                " which keeps traffic away from instances that are down."
            ),
        ),
        Document(
            id="git-rebase", title="Git Rebase vs. Merge", tags=("git", "workflow"),
            text=(
                "Rebasing replays a branch's commits on top of a new base,"
                " producing a linear history, whereas merging creates a new commit"
                " that joins two histories together. Rebase is useful for cleaning"
                " up a feature branch before review, but rewriting history that"
                " others have already pulled can cause painful conflicts."
            ),
        ),
        Document(
            id="python-asyncio", title="Python asyncio Event Loop", tags=("python", "concurrency"),
            text=(
                "Python's asyncio runs coroutines on a single-threaded event loop,"
                " switching between tasks whenever one awaits an I/O operation."
                " This gives high concurrency for I/O-bound workloads without the"
                " overhead of threads, but a CPU-bound coroutine that never awaits"
                " will block the entire event loop."
            ),
        ),
        Document(
            id="gil", title="The Python Global Interpreter Lock", tags=("python", "concurrency"),
            text=(
                "CPython's Global Interpreter Lock allows only one thread to execute"
                " Python bytecode at a time, even on a multi-core machine. This"
                " means threading helps with I/O-bound work but not CPU-bound work,"
                " for which the multiprocessing module or a C-extension that"
                " releases the GIL is typically used instead."
            ),
        ),
        Document(
            id="b-tree-index", title="B-Tree Database Indexes", tags=("databases", "indexing"),
            text=(
                "A B-tree index keeps keys sorted in a balanced tree with high"
                " fan-out, so a lookup, insert, or range scan touches only a few"
                " disk pages even for a large table. Most relational databases use"
                " a B-tree (or B+tree) variant as the default index type because it"
                " handles both equality and range queries well."
            ),
        ),
        Document(
            id="bloom-filter", title="Bloom Filters", tags=("data-structures", "probabilistic"),
            text=(
                "A Bloom filter is a space-efficient probabilistic data structure"
                " that answers 'definitely not present' or 'possibly present' for"
                " set membership, with a tunable false-positive rate and no false"
                " negatives. Databases use Bloom filters to skip disk reads for"
                " keys that are guaranteed not to be in a given file."
            ),
        ),
        Document(
            id="docker-layers", title="Docker Image Layers", tags=("docker", "devops"),
            text=(
                "A Docker image is built from a stack of read-only layers, one per"
                " instruction in the Dockerfile, with a thin writable layer added at"
                " container runtime. Layers are cached and shared across images, so"
                " ordering instructions from least to most frequently changing keeps"
                " rebuilds fast by maximizing cache reuse."
            ),
        ),
        Document(
            id="kubernetes-pods", title="Kubernetes Pods and Services", tags=("kubernetes", "devops"),
            text=(
                "A Kubernetes Pod is the smallest deployable unit, usually wrapping"
                " one container plus any sidecars that share its network namespace."
                " Pods are ephemeral, so a Service provides a stable virtual IP and"
                " DNS name that load-balances traffic across the current set of"
                " healthy pods backing it."
            ),
        ),
        Document(
            id="regex-backtracking", title="Regular Expression Backtracking", tags=("regex", "algorithms"),
            text=(
                "Backtracking regex engines try one alternative at a time and undo"
                " their choice when a match fails further along. Certain patterns"
                " with nested quantifiers can cause catastrophic backtracking, where"
                " matching time grows exponentially with input length — a common"
                " source of regex-based denial-of-service vulnerabilities."
            ),
        ),
        Document(
            id="unicode-encoding", title="Unicode and UTF-8 Encoding", tags=("encoding", "strings"),
            text=(
                "Unicode assigns every character a code point, while UTF-8 encodes"
                " those code points as one to four bytes, staying backward"
                " compatible with ASCII for the first 128 characters. Because"
                " UTF-8 is variable-width, string length in bytes and in characters"
                " can differ, which is a common source of off-by-one bugs."
            ),
        ),
        Document(
            id="cron-scheduling", title="Cron Job Scheduling", tags=("ops", "scheduling"),
            text=(
                "A cron expression specifies minute, hour, day-of-month, month, and"
                " day-of-week fields to describe a recurring schedule. Cron daemons"
                " walk the crontab each minute and run any job whose schedule"
                " matches, but they don't track whether the previous run of a long"
                " job finished, which can cause overlapping executions."
            ),
        ),
        Document(
            id="dns-resolution", title="DNS Resolution", tags=("networking", "dns"),
            text=(
                "Resolving a domain name walks a chain of resolvers: a recursive"
                " resolver queries root, then top-level-domain, then authoritative"
                " name servers in turn, caching each answer for its stated TTL."
                " Low TTLs let operators change records quickly, at the cost of"
                " more frequent lookups across the chain."
            ),
        ),
        Document(
            id="tcp-handshake", title="TCP Three-Way Handshake", tags=("networking", "tcp"),
            text=(
                "TCP establishes a connection with a three-way handshake: the"
                " client sends a SYN, the server replies SYN-ACK, and the client"
                " confirms with an ACK. This exchange synchronizes initial sequence"
                " numbers on both sides before any application data is sent,"
                " letting TCP detect and retransmit lost segments reliably."
            ),
        ),
        Document(
            id="garbage-collection", title="Generational Garbage Collection", tags=("runtime", "memory"),
            text=(
                "Generational garbage collectors separate objects into young and"
                " old generations, based on the observation that most objects die"
                " young. The young generation is collected frequently and cheaply,"
                " while surviving objects are promoted to an older generation that"
                " is scanned less often, reducing average collection pause time."
            ),
        ),
        Document(
            id="circuit-breaker", title="The Circuit Breaker Pattern", tags=("reliability", "architecture"),
            text=(
                "A circuit breaker wraps a call to a potentially failing dependency"
                " and trips open after repeated failures, short-circuiting further"
                " calls for a cooldown period instead of letting them queue up and"
                " time out. This gives a failing downstream service room to recover"
                " and prevents cascading failures upstream."
            ),
        ),
        Document(
            id="idempotency-keys", title="Idempotency Keys for APIs", tags=("api", "reliability"),
            text=(
                "An idempotency key lets a client safely retry a request — like"
                " charging a payment — without risking a duplicate side effect. The"
                " server stores the result keyed by the client-supplied token on"
                " first execution and returns the same cached response for any"
                " retry carrying the same key."
            ),
        ),
        Document(
            id="vector-quantization", title="Vector Quantization for ANN Search", tags=("retrieval", "ann"),
            text=(
                "Approximate nearest-neighbor search over large embedding"
                " collections often uses product quantization, which splits each"
                " vector into sub-vectors and replaces each with the id of its"
                " nearest centroid in a small codebook. This compresses the index"
                " dramatically and speeds up distance computation, at a small cost"
                " in recall compared to an exact search."
            ),
        ),
        Document(
            id="bm25-ranking", title="BM25 Ranking Function", tags=("retrieval", "ranking"),
            text=(
                "BM25 scores a document's relevance to a query by combining term"
                " frequency, inverse document frequency, and document-length"
                " normalization, saturating the contribution of repeated terms so"
                " that extremely long documents don't dominate purely by virtue of"
                " length. It remains a strong, fast baseline for lexical search."
            ),
        ),
        Document(
            id="connection-pooling", title="Database Connection Pooling", tags=("databases", "performance"),
            text=(
                "Opening a new database connection involves a TCP handshake and"
                " authentication, which is expensive to repeat per request. A"
                " connection pool keeps a set of already-open connections ready to"
                " reuse, checking one out for the duration of a request and"
                " returning it to the pool afterward instead of closing it."
            ),
        ),
    ]
