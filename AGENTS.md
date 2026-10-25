# ClinGraph Agent Instructions

All development on ClinGraph must adhere strictly to [README.md](README.md).

## Critical Invariants

1. **Follow the Architecture and Phased Plan:**
   Implement features strictly in phased order (Phases 0 through 6). Do not jump ahead or implement placeholder stubs. Keep the application runnable after every phase.
2. **Security & Authorization First:**
   Enforce deny-by-default access control on all retrieval paths (dense, sparse, graph), caches, and APIs. Identity, tenant ID, and roles must be derived from trusted tokens, never user-supplied query bodies. Retrieval filtering must happen before chunk text is sent to rerankers or LLMs.
3. **No Fabricated Data or Metrics:**
   Never invent or fabricate benchmark results, clinical validation claims, performance numbers, or citations. Mark unmeasured items as `Not measured`. All citations and evidence assertions must map directly to ingested, permitted chunks.
4. **Medical Safety Boundary:**
   ClinGraph is a research and educational evidence-exploration system, not a medical device. Never use it for real-patient triage, diagnoses, or prescriptions. Use synthetic vignettes and public literature only.
5. **Fail Closed and Auditable:**
   If authorization data is unavailable, deny access. Audit logs must record decisions and IDs without leaking full confidential passages, credentials, or raw patient data.
6. **Testing and Verification:**
   Run relevant tests after every phase and fix all failures before proceeding. Tests must run deterministically without requiring paid external model APIs.
