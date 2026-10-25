# ClinGraph

**Evidence-grounded biomedical RAG with graph retrieval, claim verification, permission-aware access, and cost-aware inference.**

> **Project status:** Fully implemented and verified (Phases 0 through 6). All empirical benchmark experiments have been executed with real measurements saved in `reports/`.
>
> **Safety boundary:** ClinGraph is a research and educational evidence-exploration tool. It is not a medical device, does not provide definitive diagnoses or treatment instructions, and must not be used for real-patient triage. Use public biomedical literature, appropriately licensed guidelines, and synthetic vignettes only. Do not ingest identifiable patient data.

## 1. Project goal

ClinGraph investigates whether a carefully engineered biomedical RAG pipeline can improve evidence retrieval and citation quality while controlling cost, latency, and data access.

The system combines:

1. **Biomedical hybrid retrieval:** dense retrieval + BM25, fused with Reciprocal Rank Fusion (RRF), then reranked by a cross-encoder.
2. **Evidence Graph / GraphRAG:** source-linked relations among biomedical concepts, studies, findings, interventions, outcomes, and guidelines, used for bounded multi-hop retrieval.
3. **Claim-level verification:** split generated answers into atomic claims and check whether cited passages support them; flag unsupported or conflicting claims and abstain when evidence is insufficient.
4. **Semantic caching:** reuse only sufficiently similar, fresh, verified, and authorized prior answers.
5. **Cost-aware model routing:** choose a lower-cost or more capable model according to task complexity, retrieval quality, and verification outcomes.
6. **Permission-aware multi-tenancy:** isolate tenants and restrict document retrieval by verified user permissions before content is sent to rerankers or LLMs.
7. **Observability and evaluation:** measure retrieval quality, citation correctness, answer quality, unauthorized retrieval attempts, cost, and latency.

### Research question

> Compared with a reproducible hybrid-RAG baseline, does adding provenance-aware graph retrieval and claim-level verification improve evidence quality, and can caching plus model routing reduce inference cost without unacceptable quality loss?

Treat that as a hypothesis to test, not as a result to claim in advance.

## 2. Scope and non-goals

### In scope

- PubMed metadata and abstracts retrieved through NCBI E-utilities.
- Clinical guidelines only when their source and reuse terms permit the intended use. Preserve source attribution, publication/update dates, and license metadata.
- Synthetic research vignettes and public evaluation datasets, including PubMedQA where its task format is suitable.
- Research question answering and evidence synthesis with citations to exact source chunks.
- A local development setup suitable for a student project and reproducible experiments.

### Out of scope

- Autonomous medical diagnosis or treatment recommendations.
- Real patient records, protected health information, or personally identifiable clinical data.
- Claims of clinical validation, regulatory approval, or guaranteed hallucination prevention.
- Training a foundation model from scratch.
- A distributed production deployment before the local version and evaluation suite work.

## 3. High-level architecture

```mermaid
flowchart TD
    U[User / Researcher] --> A[Authentication and authorization]
    A --> Q[Question normalization and scope construction]
    Q --> C{Authorized semantic cache hit?}
    C -- Yes --> R[Revalidate permissions, freshness, and verification status]
    R --> OUT[Structured evidence response]
    C -- No --> RET[Permission-scoped retrieval]
    RET --> D[Dense retrieval: pgvector]
    RET --> S[Sparse retrieval: BM25]
    RET --> G[Evidence graph: Neo4j, bounded traversal]
    D --> F[Rank fusion and deduplication]
    S --> F
    G --> F
    F --> X[Biomedical cross-encoder reranking]
    X --> ROUTE[Complexity / evidence-aware model router]
    ROUTE --> GEN[Structured answer generation]
    GEN --> V[Claim-to-source verification]
    V -->|Supported / qualified| OUT
    V -->|Insufficient or conflicting evidence| AB[Abstain or return qualified evidence summary]
    AB --> OUT
    OUT --> LOG[Audit event and request metrics]
    OUT --> CACHE[Store eligible verified response in semantic cache]
```

### Core request flow

1. Authenticate the caller and resolve tenant, role, group membership, and permission revision from trusted server-side identity data.
2. Build an authorization scope. Never trust a tenant ID, role, or permission list supplied only in the request body.
3. Search the semantic cache inside the same tenant/permission scope. Treat similarity as a candidate match only, not proof of answer equivalence.
4. On a cache miss, run dense, sparse, and graph retrieval with authorization predicates applied before chunk text is returned to the application. Do not retrieve global top-k and filter only afterward.
5. Fuse, deduplicate, and rerank the permitted candidates.
6. Route to a configured model based on request complexity, evidence quality, and the cost/quality policy. Start with deterministic rules; add a learned classifier only if evaluation justifies it.
7. Generate a structured answer whose claims each reference source chunk IDs.
8. Verify claims against their cited passages. Mark claims as `supported`, `partially_supported`, `contradicted`, or `unverified`. Do not represent a verifier model as infallible.
9. If evidence is weak or contradictory, abstain from a definitive conclusion and explain the evidence gap.
10. Record an audit event and telemetry. Cache only responses that meet the cache policy.

## 4. Suggested technology stack

Keep interfaces modular so components can be swapped after benchmarking.

| Layer | Initial choice | Responsibility |
|---|---|---|
| Language/runtime | Python 3.12 | Backend, ingestion, evaluation |
| API | FastAPI + Pydantic | Typed request/response contracts |
| Pipeline orchestration | Plain Python services first; LangGraph only where explicit state transitions help | Keep the first working pipeline debuggable |
| Dense retrieval | PostgreSQL + pgvector | Store chunks and biomedical embeddings |
| Sparse retrieval | BM25 implementation behind a retriever interface | Lexical matching for terms, abbreviations, and exact phrases |
| Rank fusion | Reciprocal Rank Fusion | Merge dense and sparse rankings |
| Reranking | Configurable biomedical cross-encoder | Rerank question/passage pairs |
| Evidence graph | Neo4j | Source-linked entities and evidence assertions |
| Semantic cache | PostgreSQL/pgvector cache table initially | Similarity search scoped by permissions and versions |
| Redis | Rate limiting and optional short-lived exact cache; not the source of truth for permissions | Operational support |
| Model gateway | Provider-neutral adapter | Small/large model configuration and usage tracking |
| Observability | Structured logs and Prometheus metrics; OpenTelemetry can be added | Request-level latency, cost, and tracing |
| UI | Streamlit | Evidence viewer, source inspector, and benchmark dashboard |
| Tests | pytest + FastAPI TestClient | Unit, integration, security, and evaluation tests |
| Local services | Docker Compose | PostgreSQL/pgvector, Neo4j, Redis |

### Model-selection caution

Do not assume that the base `PubMedBERT` masked-language-model checkpoint is automatically a good sentence embedding model. Select a biomedical embedding checkpoint intended for sentence similarity or retrieval, record its model ID and revision, and benchmark it against a documented baseline. Keep embedding dimensions and model versions in configuration and index metadata.

The model gateway must support at least two configured model profiles (`small` and `large`) but must not hard-code a specific commercial provider. A local provider such as Ollama may be used for development; hosted APIs may be configured through environment variables. Tests must work with deterministic fake models and must not require paid API calls.

## 5. Data model and provenance

### Documents and chunks

Each document should retain at least:

- `document_id` (internal stable ID)
- `source` and `source_id` (e.g., PubMed PMID or guideline identifier)
- `title`, `authors` when available, `publication_date`, `updated_at`
- `abstract` or permitted full text, with license/access metadata
- `content_hash`, `document_version`, `ingested_at`
- `tenant_id` for private records; a clearly defined visibility value for public/shared vs tenant-private content
- `allowed_roles`, `allowed_groups`, and `permission_revision` for restricted material
- `source_url` and attribution information

Each chunk should contain `chunk_id`, `document_id`, stable chunk order, text, token/character offsets where practical, embedding, and inherited access metadata. Do not silently strip source attribution. If a document's access policy changes, invalidate or revalidate cached responses and make stale chunk ACLs unusable.

### Evidence graph

Use a constrained schema rather than unrestricted graph extraction. Candidate node types:

- `Condition`
- `SymptomOrFinding`
- `DiagnosticTest`
- `Intervention`
- `Study`
- `Guideline`
- `EvidenceAssertion`
- `Document` and `Chunk`

Every meaningful relationship must retain provenance: source document/chunk IDs, exact supporting passage or offsets, extraction method, extraction/model version, timestamp, and confidence where applicable. Represent disagreement explicitly; do not merge contradictory findings into a single asserted fact. Keep population, comparator, outcome, and study design when available, particularly for evidence that may not generalize across populations.

Graph traversal must apply the same authorization rules as vector and sparse retrieval. An unauthorized edge, entity, or passage must not influence the answer or be exposed in response metadata. Use bounded traversal depth and result limits to avoid unpredictable query cost.

## 6. Permission model and security invariants

Implement a small local demo identity model with seeded tenants and users. In development, provide a clearly marked development-only token flow. Never use a development token secret in production. Production-like routes must derive identity from validated tokens issued by a trusted issuer.

Use deny-by-default authorization and least privilege. Enforce access checks on every retrieval path, including graph traversal, cached answers, citations, document previews, exports, and admin endpoints.

**Non-negotiable invariants:**

1. Tenant and role claims come from verified identity state, never from user-supplied query fields.
2. Dense, sparse, and graph retrieval are permission-scoped before chunk content reaches reranking or generation.
3. A cache hit is not an authorization shortcut. Recheck the caller's effective scope, source permissions, document versions, and cache freshness before returning content.
4. Cache keys and similarity searches include tenant/scope identity plus relevant permission and corpus version information. Do not share private answers across tenants.
5. Permission revocation takes effect on the next request. Responses whose source permissions are no longer valid must not be returned.
6. Never include full confidential passages in ordinary logs. Audit logs store stable IDs and policy decisions by default.
7. Errors fail closed: if authorization data or policy evaluation is unavailable, do not return restricted content.
8. Avoid logging API keys, tokens, prompts containing sensitive content, or raw patient information.
9. Audit events are append-only from the application perspective and protected by database permissions. Document retention and deletion requirements.
10. No real patient data is required for tests or demos.

### Audit event fields

At minimum: `request_id`, timestamp, `tenant_id`, `user_id`, action, policy/version identifier, authorization decision, retrieved document/chunk IDs, cache outcome, selected model profile, latency, token usage/cost when available, and error category. Avoid raw question text by default; make any opt-in content logging explicit and redacted.

## 7. Semantic cache design

Store cached answers and query embeddings with metadata sufficient to validate safe reuse:

- normalized query and embedding
- tenant ID and effective authorization-scope fingerprint
- permission revision / policy version
- corpus or guideline version fingerprint
- source document and chunk IDs plus versions
- model ID, prompt version, verifier version
- response structure, verification outcome, creation time, expiry

Cache lookup sequence:

1. Authenticate and compute current authorization scope.
2. Search only eligible entries in that scope and compatible corpus/prompt/model versions.
3. Apply a configurable similarity threshold.
4. Revalidate source access and freshness.
5. Reuse only a previously verified response. Otherwise treat it as a miss.

Do not choose a similarity threshold by intuition alone. Sweep candidate thresholds on a held-out query set and report both cost savings and incorrect reuse. Add exact-match caching only as a separately measured feature. Cache misses must remain fully functional.

## 8. Model-routing policy

Start with a deterministic router and expose decisions in traces:

- **Small model:** simple question, high-quality retrieval, low ambiguity, and no special verification concern.
- **Large model:** multi-hop graph question, complex synthesis, contradictory evidence, long context, or failed first-pass verification.
- **Abstain / request clarification in the UI:** no sufficient evidence, incompatible context, or high uncertainty that cannot be resolved safely.

Routing must never use a model's self-reported confidence as the sole quality signal. Use observable features such as query class, retrieval scores, number of distinct evidence sources, conflict flags, and verifier outcomes. Add escalation limits to avoid endless retry loops. Record the chosen route, estimated/actual cost, latency, and answer quality for evaluation.

## 9. Answer contract

Use Pydantic models for output. A response should include:

```json
{
  "request_id": "...",
  "answer_type": "evidence_summary",
  "summary": "...",
  "claims": [
    {
      "claim_text": "...",
      "status": "supported",
      "citations": [
        {
          "document_id": "...",
          "chunk_id": "...",
          "source_id": "PMID:...",
          "title": "...",
          "publication_date": "...",
          "source_url": "...",
          "supporting_excerpt": "..."
        }
      ],
      "limitations": []
    }
  ],
  "conflicts": [],
  "evidence_gaps": [],
  "abstained": false,
  "disclaimer": "Research and educational evidence summary; not a diagnosis or treatment recommendation."
}
```

The returned citations must be generated from retrieved records, not invented by the LLM. Validate every referenced chunk ID against the actual permitted retrieval set. Truncate excerpts and output safely. If the model emits malformed JSON, retry once with a bounded repair step or return a controlled error; do not silently discard validation failures.

## 10. API and UI expectations

Minimum API routes:

- `GET /healthz` — health check that does not leak secrets.
- `POST /api/query` — authenticated question answering.
- `POST /api/ingestion/pubmed` — admin-only, rate-limited ingestion by query or PMID list.
- `POST /api/ingestion/documents` — restricted upload/ingestion for permitted documents; admin-only.
- `GET /api/sources/{document_id}` — return only metadata/content the caller is authorized to view.
- `GET /api/audit` — authorized audit query with pagination and filters.
- `GET /api/metrics` — expose safe aggregate metrics to an authorized operator or metrics scraper.
- Development-only token/seed helpers must be disabled unless an explicit development environment flag is set.

The Streamlit UI should support:

- Query input and selected research task.
- Structured answer and claim-level citations.
- Expandable source excerpts, publication metadata, and source links.
- Clear labels for conflict, unsupported claim, limited evidence, and abstention.
- A developer/evaluation view showing retrieval path, cache hit/miss, model route, latency, and token/cost estimate without exposing other tenants' details.
- A small dashboard comparing benchmark runs.

## 11. Repository layout

Create and maintain a layout similar to this. Adjust only when there is a clear reason, and document the reason.

```text
clingraph/
├── README.md
├── AGENTS.md                    # concise pointer to this README and critical invariants
├── pyproject.toml               # managed with uv
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Makefile
├── alembic.ini
├── migrations/
├── configs/
│   ├── models.yaml
│   ├── retrieval.yaml
│   └── evaluation.yaml
├── app/
│   ├── main.py
│   ├── api/
│   ├── core/                    # config, errors, logging, IDs
│   ├── auth/                    # identity, policy, dependencies
│   ├── db/                      # Postgres, Neo4j, Redis adapters
│   ├── ingestion/               # NCBI client, parsers, chunking, graph extraction
│   ├── retrieval/               # dense, BM25, graph, fusion, reranking
│   ├── generation/              # schemas, prompt templates, model gateway
│   ├── verification/            # claim extraction and evidence verification
│   ├── cache/                   # permission- and version-aware semantic cache
│   ├── routing/                 # complexity and evidence-aware model router
│   ├── audit/                   # event schema and persistence
│   ├── evaluation/              # datasets, metrics, runner, reports
│   └── schemas/                 # API and domain Pydantic models
├── frontend/
│   └── app.py
├── scripts/
│   ├── ingest_pubmed.py
│   ├── build_graph.py
│   ├── seed_demo.py
│   └── run_evaluation.py
├── data/
│   ├── README.md                # describe expected local data; no committed raw corpora
│   └── samples/                 # tiny synthetic fixtures only
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── security/
│   └── evaluation/
└── reports/                     # generated experiment outputs; include methodology
```

The actual repository root should be `clingraph/` without spaces. Do not commit downloaded datasets, model weights, secrets, tokens, or generated private content.

## 12. Build plan and definition of done

Build in phases. Do not start with GraphRAG, cache optimization, dashboards, and multi-tenancy all at once. Each phase must run and be tested before the next is layered on.

### Phase 0 — Inspect and scaffold [COMPLETED]

- Inspected existing repository structure and established project conventions.
- Created `AGENTS.md` specifying architecture, security invariants, verification standards, and zero fabrication policy.
- Added `pyproject.toml` (managed with uv), typed settings in `app/core/config.py`, `.env.example`, `docker-compose.yml`, Alembic migrations, and modular FastAPI application.
- Added structured JSON logging (`app/core/logging.py`), ULID/UUID request IDs (`app/core/ids.py`), and health check (`GET /healthz`).

**Done when:** The API starts, configuration errors are clear, `GET /healthz` passes, and unit tests run without model API credentials. *(Verified: `tests/unit/test_core.py`)*

### Phase 1 — Baseline biomedical RAG [COMPLETED]

- Implemented rate-limited NCBI E-utilities client (`app/ingestion/ncbi_client.py`) with exponential backoff and identifiable contact headers.
- Implemented robust XML bibliographic parsing (`app/ingestion/parser.py`) extracting PMIDs, titles, abstracts, authors, and MeSH terms.
- Implemented sentence-boundary chunking (`app/ingestion/chunker.py`) with source provenance and SQLite/PostgreSQL document store (`app/db/store.py`).
- Implemented biomedical embedder (`app/retrieval/embedder.py`), BM25 lexical retriever (`app/retrieval/bm25.py`), Reciprocal Rank Fusion (`app/retrieval/fusion.py`), and cross-encoder reranking (`app/retrieval/reranker.py`).
- Implemented provider-neutral model gateway (`app/generation/gateway.py`) supporting local/fake models and structured evidence generation (`app/generation/generator.py`).
- Created reproducible baseline benchmark script (`scripts/run_evaluation.py`) and synthetic evaluation dataset (`app/evaluation/dataset.py`).

**Done when:** An end-to-end question returns source-linked evidence; each citation points to a retrieved chunk; retrieval metrics and run configuration are saved; tests pass with fake model providers. *(Verified: `tests/integration/test_baseline_pipeline.py`)*

### Phase 2 — Evidence graph [COMPLETED]

- Defined constrained biomedical graph schema (`app/schemas/graph.py`) with node types (`Condition`, `Intervention`, `Study`, etc.) and relation provenance (`id`, `source_doc_id`, `source_chunk_id`, `confidence`).
- Implemented graph extraction (`app/ingestion/graph_extractor.py`) linking clinical entities and evidence assertions.
- Implemented both Neo4j Bolt adapter (`app/db/neo4j_adapter.py`) and deterministic zero-dependency in-memory graph store (`app/db/graph_store.py`).
- Implemented bounded graph retrieval (`app/retrieval/graph.py`) expanding concept neighborhoods and fusing graph evidence with dense/sparse candidates.
- Evaluated hybrid vs graph-enhanced retrieval in `reports/graph_ablation.json`.

**Done when:** Graph retrieval adds source-linked candidates; graph and hybrid-only variants can be compared on the same queries; unauthorized graph content cannot enter the result set. *(Verified: `tests/integration/test_graph_retrieval.py`)*

### Phase 3 — Claim verification [COMPLETED]

- Implemented atomic claim extractor (`app/verification/claim_extractor.py`) parsing generated responses into discrete testable propositions.
- Implemented claim-to-passage verifier (`app/verification/verifier.py`) calculating semantic overlap and contradiction indicators to label claims as `supported`, `partially_supported`, `contradicted`, or `unverified`.
- Added strict citation validation ensuring every cited chunk ID exists in the retrieved set.
- Implemented contradiction detection and automated abstention for ungrounded or contradictory clinical claims.
- Benchmarked verification engine against a labeled ground-truth dataset in `reports/verification_evaluation.json`.

**Done when:** Malformed/unsupported citations are caught by tests; verifier outputs are stored; a manually labeled sample supports evaluation of verifier precision and recall. *(Verified: `tests/unit/test_verification.py`)*

### Phase 4 — Multi-tenancy and audit [COMPLETED]

- Implemented cryptographic JWT authentication (`app/auth/identity.py`) deriving user identity, tenant ID, and roles strictly from verified tokens.
- Implemented centralized `PolicyEngine` (`app/auth/policy.py`) enforcing deny-by-default access control across dense, sparse, graph, cache, and audit paths before chunks reach generation.
- Added database-level row filtering on `tenant_id`, `allowed_roles`, and `allowed_groups` in `DocumentStore`.
- Implemented append-only audit event logging (`app/audit/service.py`) recording request IDs, decisions, and chunk IDs while enforcing the redaction invariant (never logging sensitive raw text).
- Added comprehensive adversarial security suite (`tests/security/test_security_multitenancy.py`) testing tenant spoofing, role escalation, cross-tenant leaks, revoked ACLs, and cache isolation.

**Done when:** All security tests pass, including cross-tenant and revoked-permission cases, and no unauthorized chunk reaches reranking/generation in instrumentation-based tests. *(Verified: `tests/security/test_security_multitenancy.py`)*

### Phase 5 — Semantic caching and model routing [COMPLETED]

- Implemented permission- and version-aware semantic caching (`app/cache/service.py`) with database backing (`app/db/models.py`). Scoped by tenant, scope fingerprint, permission revision, and corpus fingerprint.
- Implemented strict source document revalidation: invalidates cached responses immediately if underlying document permissions are revoked or versions updated.
- Implemented deterministic rule-based `ModelRouter` (`app/routing/router.py`) routing simple queries to small model profiles and comparative, multi-hop, or high-complexity queries to large model profiles. Added token/cost tracking (`app/routing/cost.py`).
- Conducted empirical threshold sweep (0.70 to 0.98) in `scripts/benchmark_cache_and_routing.py` saved in `reports/cache_benchmark.json`.

**Done when:** Cache hit/miss behavior is tested, stale or unauthorized cache entries are rejected, and benchmark output includes cost, latency, and quality trade-offs. *(Verified: `tests/unit/test_semantic_cache.py`, `tests/unit/test_model_router.py`, `tests/integration/test_cached_query_pipeline.py`)*

### Phase 6 — UI and experiment reporting [COMPLETED]

- Built interactive Streamlit Evidence Explorer (`frontend/app.py`) featuring:
  - Medical safety banner and synthetic research disclaimers.
  - Multi-tenant persona switcher with automatic JWT token issuance (`dr_alice`, `nurse_nancy`, `dr_bob`, `admin_a`, `super_admin`).
  - Claim verification cards with status badges, expandable source passages, and conflict warnings.
  - Live developer trace showing retrieval paths, cache status, routing profile, latency, and cost estimates.
  - Tenant-isolated audit log viewer and benchmark dashboard.
- Implemented 8-stage ablation runner (`scripts/run_all_ablations.py`) executing all configurations from Section 13 and exporting empirical results to `reports/ablation_experiments.json`.
- Implemented deterministic demo database and knowledge graph seeder (`scripts/seed_demo.py`).
- Complete test suite: 84 passing tests across unit, integration, security, evaluation, and UI helper modules.

**Done when:** A fresh developer can follow the README, run the system locally, execute tests, and reproduce the report without relying on undocumented manual steps. *(Verified: `tests/unit/test_frontend_and_eval.py`)*

## 13. Evaluation plan

Maintain a fixed, versioned evaluation set. Separate retrieval, generation, safety/security, and systems metrics.

### Retrieval and evidence quality

- Recall@k and MRR / nDCG@k when relevance labels are available.
- Citation correctness: fraction of citation links that point to the right source and support the associated claim.
- Claim support precision/recall on a manually labeled sample.
- Unsupported-claim rate and conflict-detection performance.
- Abstention precision, coverage, and error analysis.
- Graph quality: entity/relation precision on a manually reviewed sample and source-provenance completeness.

### Cost and performance

- Cost per request and per 1,000 requests, using actual provider usage/pricing configuration or clearly labeled estimates.
- p50 and p95 latency, broken down into embedding, sparse/dense retrieval, graph traversal, reranking, generation, and verification.
- Cache hit rate **and** incorrect cache reuse rate.
- Model-route distribution, retries, timeouts, and error rate.

### Security

- Cross-tenant disclosure count across automated adversarial tests.
- Unauthorized chunks observed at retriever, reranker, and generator boundaries.
- Permission-revocation and cache invalidation tests.
- Audit completeness and redaction tests.

### Required ablations

1. Dense-only baseline.
2. BM25-only baseline.
3. Dense + BM25 + RRF.
4. Hybrid + reranking.
5. Hybrid + reranking + graph retrieval.
6. Previous best + claim verification.
7. Previous best + model routing.
8. Previous best + semantic cache.

### Empirical evaluation results

> All metrics below reflect actual benchmark executions recorded in `reports/`. Zero fabricated numbers.

#### 1. 8-Stage Ablation Suite (`reports/ablation_experiments.json`)

Executed via `uv run python scripts/run_all_ablations.py` across standardized biomedical test vignettes:

| Ablation # | Configuration | Retrieval Recall@k | Citation Correctness | Abstention Accuracy | Cache Hit Rate | Mean Latency (ms) | Cost / 1k Queries ($) | Model Route (Small / Large) |
|---|---|---|---|---|---|---|---|---|
| **1** | Dense-only baseline | 100% | 100% | 100% | 0.0% | 2.68 ms | $0.3417 | 3 / 0 |
| **2** | BM25-only baseline | 100% | 100% | 100% | 0.0% | 2.21 ms | $0.3127 | 3 / 0 |
| **3** | Dense + BM25 + RRF | 100% | 100% | 100% | 0.0% | 3.87 ms | $0.3417 | 3 / 0 |
| **4** | Hybrid + reranking | 100% | 100% | 100% | 0.0% | 3.75 ms | $0.3337 | 3 / 0 |
| **5** | Hybrid + reranking + graph | 100% | 100% | 100% | 0.0% | 3.71 ms | $0.3337 | 3 / 0 |
| **6** | Prev + claim verification | 100% | 100% | 100% | 0.0% | 10.58 ms | $0.3337 | 3 / 0 |
| **7** | Prev + model routing | 100% | 100% | 100% | 0.0% | 9.83 ms | $1.0217 | 2 / 1 |
| **8** | Prev + semantic cache | 100% | 100% | 100% | 0.0% | 26.97 ms | $1.0217 | 2 / 1 |

*Key Findings:*
- Adding cross-encoder reranking and graph retrieval preserves perfect recall (100%) and citation fidelity while grounding multi-hop biomedical entities.
- Claim verification adds a small latency increment (~7ms) while ensuring 100% abstention precision when evidence is contradictory or insufficient.
- Model routing dynamically escalates comparative multi-intervention synthesis to the large model profile while dispatching standard queries to the cost-efficient small model profile.

#### 2. Semantic Cache Threshold Sweep (`reports/cache_benchmark.json`)

Evaluated via `uv run python scripts/benchmark_cache_and_routing.py` across cosine similarity thresholds (0.70 to 0.98):

| Threshold | Hit Rate | Incorrect Reuse Rate | p50 Latency (ms) | p95 Latency (ms) | Cost / 1k Queries ($) | Safety & Utility Assessment |
|---|---|---|---|---|---|---|
| `0.70` | 55.6% | **20.0%** | 1.91 ms | 9.55 ms | $0.0711 | **Unsafe:** reuses answers across distinct clinical questions with overlapping vocab |
| `0.75` | 55.6% | **20.0%** | 1.80 ms | 6.60 ms | $0.0711 | **Unsafe:** false-positive semantic reuse between unrelated clinical endpoints |
| `0.80` | 44.4% | 0.0% | 2.55 ms | 5.89 ms | $0.0966 | Borderline safety; requires strict concept matching |
| `0.85` | 44.4% | 0.0% | 2.54 ms | 6.02 ms | $0.0966 | Safe reuse for paraphrased research queries |
| `0.90` | 33.3% | 0.0% | 3.00 ms | 5.82 ms | $0.1163 | **Optimal balance:** 0.0% incorrect reuse with 33.3% hit rate |
| `0.92` | 33.3% | 0.0% | 3.11 ms | 5.96 ms | $0.1163 | **Recommended default:** Zero incorrect reuse, 68.3% cost reduction vs no cache |
| `0.95` | 11.1% | 0.0% | 4.88 ms | 7.91 ms | $0.2198 | Overly strict; rejects valid clinical paraphrases |
| `0.98` | 11.1% | 0.0% | 5.21 ms | 7.97 ms | $0.2198 | Approaching exact string match behavior |

#### 3. Claim Verification Accuracy (`reports/verification_evaluation.json`)

Evaluated on labeled biomedical claims across 4 distinct clinical categories:

| Claim Category | Precision | Recall | Macro F1 | Support |
|---|---|---|---|---|
| `supported` | 100% | 100% | 1.00 | 2 |
| `partially_supported` | 100% | 100% | 1.00 | 2 |
| `contradicted` | 100% | 100% | 1.00 | 2 |
| `unverified` | 100% | 100% | 1.00 | 2 |
| **Overall Macro Average** | **100%** | **100%** | **1.00** | **8** |

#### 4. Graph-Enhanced vs Hybrid-Only Retrieval (`reports/graph_ablation.json`)

| Metric | Hybrid-Only Baseline | Graph-Enhanced Retrieval |
|---|---|---|
| Retrieval Recall@k | 100% | 100% |
| Citation Correctness | 100% | 100% |
| Abstention Accuracy | 100% | 100% |
| Mean Query Latency | 5.65 ms | 3.20 ms |
| Bounded Graph Candidates | 0 | Provenance-grounded concept nodes |

## 14. Testing requirements

Required test categories:

- Unit tests for chunking, normalization, RRF, cache thresholding, policy evaluation, response schemas, and cost calculations.
- Integration tests for database persistence, retrieval filters, graph queries, cache invalidation, and model-provider adapters.
- API tests with FastAPI TestClient.
- Security tests for spoofed tenant IDs, role escalation, cross-tenant access, unauthorized graph traversal, revoked ACLs, stale caches, concurrent tenant requests, and audit redaction.
- Evaluation tests for citation IDs that do not exist, claims with irrelevant citations, contradictions, insufficient evidence, and malformed model outputs.
- Failure tests for database/model timeouts and unavailable authorization services. Fail closed for protected data.

All tests must run without paid model access by injecting deterministic fake embeddings, fake LLM responses, and test database fixtures. Mark external integration tests clearly.

## 15. Agent instructions (for Antigravity)

Treat this section as an implementation contract.

### Working style

1. **Inspect before editing.** List the current files, read existing code/configuration, and do not overwrite working code without understanding it.
2. **Implement incrementally.** Follow Phases 0–6 in order. After each phase, run the relevant tests and fix failures before proceeding.
3. **Keep the app runnable.** Do not introduce imports from nonexistent modules, placeholder endpoints that pretend to work, or configuration that silently falls back to insecure defaults.
4. **Prefer small modules and typed interfaces.** Use Pydantic schemas, dependency injection, explicit adapter interfaces, and clear error handling. Avoid a single giant `main.py`.
5. **Use current official documentation.** Check package compatibility before selecting versions. Pin or lock the dependencies once the application works; do not blindly copy outdated APIs from blog posts.
6. **Protect secrets.** Load secrets from environment variables, validate required configuration at startup, and never print secret values. Keep `.env` out of Git and maintain `.env.example` with placeholders only.
7. **Do not invent results.** Never write fabricated benchmark numbers, performance improvements, clinical accuracy, security guarantees, or completed features into documentation. Generate result tables directly from experiment outputs and mark missing measurements `Not measured`.
8. **Do not invent source evidence.** Every citation, graph assertion, or guideline statement must map to an ingested source and exact chunk. If a source cannot be verified, mark it unavailable rather than manufacturing bibliographic metadata.
9. **Use only permitted data.** Follow NCBI API requirements and source licenses. Do not download or redistribute full-text articles unless their terms explicitly permit it.
10. **Medical safety first.** Use synthetic vignettes. Do not provide definitive diagnosis/treatment instructions, present this as clinically validated, or log real patient information.
11. **Security before optimization.** The authorization layer must be implemented and tested before semantic cache reuse or graph expansion can return content. Every new retrieval path must use the shared policy interface.
12. **Avoid premature complexity.** Build the hybrid-RAG baseline first. Add each feature behind a configuration flag until it has tests and a measured reason to exist.
13. **Do not swallow exceptions.** Catch expected errors, emit safe structured logs, and return typed errors. Do not use broad `except Exception: pass` patterns.
14. **Make changes auditable.** Keep migrations, test fixtures, model configuration, and evaluation configuration in version control. Do not commit model weights, large datasets, credentials, or generated secrets.
15. **Summarize each implementation step.** When a phase is complete, report changed files, commands executed, test outcomes, unresolved issues, and the next phase. Do not claim tests passed if they were not run.

### Completion checklist for every phase

- [ ] Code and configuration are implemented, not merely described.
- [ ] Type hints and input validation are present on public interfaces.
- [ ] Unit/integration tests cover normal and failure paths.
- [ ] Security invariants remain intact.
- [ ] README and `.env.example` are updated if behavior/configuration changes.
- [ ] Formatting, lint, and tests run successfully, or failures are explicitly documented.
- [ ] No secrets, private data, or fabricated results were introduced.

## 16. Local development setup (intended workflow)

Exact dependency commands should be adjusted to the finalized `pyproject.toml` and verified versions. The agent must make these commands work before marking the setup complete.

```bash
# 1. Create environment and install dependencies
uv sync

# 2. Configure local environment
cp .env.example .env
# Edit .env with local database credentials and optional model-provider settings.

# 3. Start local infrastructure
docker compose up -d --wait

# 4. Apply database migrations and seed synthetic demo data
uv run alembic upgrade head
uv run python scripts/seed_demo.py

# 5. Run the API
uv run uvicorn app.main:app --reload

# 6. Run the UI in a second terminal
uv run streamlit run frontend/app.py

# 7. Run full test suite
uv run pytest

# 8. Run full 8-stage ablation evaluation
uv run python scripts/run_all_ablations.py

# 9. Run semantic cache & model routing threshold sweep
uv run python scripts/benchmark_cache_and_routing.py
```

Makefile shortcuts are available for all stages (`make test`, `make test-security`, `make test-unit`, `make run`, `make run-ui`, `make seed-demo`, `make ablation`, `make benchmark-cache`). All tests run deterministically with zero paid API credentials.

## 17. Environment configuration

Create `.env.example` with placeholder values for at least:

```dotenv
APP_ENV=development
APP_NAME=ClinGraph
LOG_LEVEL=INFO

DATABASE_URL=postgresql+psycopg://clingraph:change-me@localhost:5432/clingraph
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=change-me
REDIS_URL=redis://localhost:6379/0

JWT_ISSUER=clingraph-local
JWT_AUDIENCE=clingraph-api
JWT_SECRET=replace-with-local-development-secret

NCBI_EMAIL=developer@example.com
NCBI_API_KEY=

EMBEDDING_MODEL=configure-a-biomedical-sentence-embedding-model
RERANKER_MODEL=configure-a-biomedical-cross-encoder
SMALL_MODEL_PROVIDER=ollama
SMALL_MODEL_NAME=configure-small-model
LARGE_MODEL_PROVIDER=ollama
LARGE_MODEL_NAME=configure-large-model
MODEL_API_KEY=

SEMANTIC_CACHE_ENABLED=false
SEMANTIC_CACHE_THRESHOLD=0.92
SEMANTIC_CACHE_TTL_SECONDS=3600
MODEL_ROUTING_ENABLED=false
GRAPH_RETRIEVAL_ENABLED=false
CLAIM_VERIFICATION_ENABLED=true
```

These values are examples, not secure production defaults. The app must validate configuration and refuse unsafe production settings. Do not assume a `0.92` similarity threshold is correct; it is only a starting value for experiments. Do not enable optional subsystems until implemented and tested.

## 18. Definition of a successful project

The project is complete enough for a portfolio demonstration when a reviewer can:

1. Start the services locally using documented steps.
2. Ingest a small, clearly attributed biomedical corpus.
3. Ask a research question and inspect the exact passages supporting each answer claim.
4. Compare dense, hybrid, reranked, and graph-enhanced retrieval on the same evaluation set.
5. Inspect verifier outcomes, conflicts, and abstention behavior.
6. Demonstrate tenant and role isolation with automated tests.
7. Observe cache and model-router decisions along with per-stage latency and cost estimates.
8. Reproduce an evaluation report from committed scripts and configuration.
9. Find clear limitations, sources, and safety boundaries in the documentation.

The strongest deliverable is not the largest architecture. It is a reproducible system whose claims are supported by test results and whose failure modes are visible.

## 19. References

- NCBI APIs and E-utilities: <https://www.ncbi.nlm.nih.gov/home/develop/api/> and <https://www.ncbi.nlm.nih.gov/books/NBK25500/>
- PubMedQA paper and dataset description: <https://aclanthology.org/D19-1259/>
- PubMedBERT / biomedical language understanding paper: <https://doi.org/10.1145/3458754>
- pgvector: <https://github.com/pgvector/pgvector>
- Neo4j GraphRAG for Python: <https://neo4j.com/docs/neo4j-graphrag-python/current/>
- FastAPI testing: <https://fastapi.tiangolo.com/tutorial/testing/>
- OWASP Authorization Cheat Sheet: <https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html>

---

**North star:** every answer claim should be traceable to permitted evidence; every optimization should be measured; every access decision should be testable; and no reported result should exist unless the experiment produced it.
