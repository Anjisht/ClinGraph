"""ClinGraph Streamlit Evidence Exploration and Evaluation Dashboard.

Features:
- Evidence Explorer: authenticated query execution, structured claims,
  expandable source citations, conflict detection, and safety abstentions.
- Developer View: model routing rationale, cache hit/miss status,
  token usage, cost estimation, and latency breakdown.
- Benchmark Dashboard: 8-stage ablation comparisons, cache similarity sweeps,
  and live system metrics.
- Multi-tenant Identity Switcher: test tenant isolation across seeded users.
"""

import json
import sys
from pathlib import Path

# Fix Streamlit path collision: Streamlit prepends frontend/ to sys.path[0].
# Since this script is named app.py, Python's import system resolves 'import app'
# to this file rather than the root 'app' package. Ensure repo root takes precedence
# and remove frontend/ from sys.path.
_frontend_dir = str(Path(__file__).resolve().parent)
_repo_root = str(Path(__file__).resolve().parent.parent)

if _frontend_dir in sys.path:
    sys.path.remove(_frontend_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
else:
    sys.path.remove(_repo_root)
    sys.path.insert(0, _repo_root)

# If 'app' was registered in sys.modules as a single-file module from this script,
# clear it so 'from app...' resolves to the project's root 'app' package.
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st  # noqa: E402

from app.auth.demo_users import SEEDED_DEMO_USERS  # noqa: E402
from app.auth.tokens import create_jwt_token  # noqa: E402
from app.core.config import Settings  # noqa: E402


def is_ollama_available() -> bool:
    """Check if local Ollama daemon is reachable on localhost:11434."""
    import httpx

    try:
        with httpx.Client(timeout=0.3) as client:
            resp = client.get("http://localhost:11434/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def get_active_settings() -> Settings:
    provider = st.session_state.get("model_provider", "fake")
    return Settings(
        APP_ENV="development",
        DEV_TOKEN_ENABLED=True,
        SMALL_MODEL_PROVIDER=provider,
        LARGE_MODEL_PROVIDER=provider,
        SEMANTIC_CACHE_ENABLED=st.session_state.get("enable_cache", True)
        if "enable_cache" in st.session_state
        else True,
        SEMANTIC_CACHE_THRESHOLD=st.session_state.get("cache_threshold", 0.92)
        if "cache_threshold" in st.session_state
        else 0.92,
        MODEL_ROUTING_ENABLED=st.session_state.get("enable_routing", True)
        if "enable_routing" in st.session_state
        else True,
        GRAPH_RETRIEVAL_ENABLED=st.session_state.get("enable_graph", False)
        if "enable_graph" in st.session_state
        else False,
        CLAIM_VERIFICATION_ENABLED=True,
    )


def execute_query_backend(query_text: str, token: str, settings: Settings, top_k: int = 5):
    """Execute authenticated query via direct FastAPI test client or HTTP."""
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import app

    app.dependency_overrides[get_settings] = lambda: settings
    try:
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "query": query_text,
            "top_k": top_k,
            "use_reranker": True,
            "use_graph": settings.GRAPH_RETRIEVAL_ENABLED,
        }
        return client.post("/api/query", headers=headers, json=payload)
    finally:
        app.dependency_overrides.pop(get_settings, None)


def fetch_system_metrics(token: str):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}
    return client.get("/api/metrics", headers=headers)


def fetch_audit_logs(token: str):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}
    return client.get("/api/audit?limit=25", headers=headers)


def render_ui():
    """Render the full Streamlit UI interface."""
    st.set_page_config(
        page_title="ClinGraph | Biomedical Evidence Explorer",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .safety-banner {
            background-color: #FEF3C7;
            border-left: 5px solid #F59E0B;
            padding: 0.8rem 1.2rem;
            border-radius: 4px;
            color: #92400E;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }
        .claim-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .badge-supported {
            background-color: #DEF7EC;
            color: #03543F;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.75rem;
        }
        .badge-partially {
            background-color: #FEF08A;
            color: #854D0E;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.75rem;
        }
        .badge-contradicted {
            background-color: #FDE8E8;
            color: #9B1C1C;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.75rem;
        }
        .badge-unverified {
            background-color: #F3F4F6;
            color: #4B5563;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Identity & Configuration
    with st.sidebar:
        st.markdown("### 🧬 **ClinGraph**")
        st.caption("Evidence-Grounded Biomedical Research & GraphRAG")
        st.markdown("---")

        st.markdown("#### 👤 **User & Tenant Identity**")
        demo_options = list(SEEDED_DEMO_USERS.keys())
        selected_user_key = st.selectbox(
            "Active Persona",
            options=demo_options,
            index=0,
            help="Select demo user to inspect tenant boundaries and access control",
        )
        user_context = SEEDED_DEMO_USERS[selected_user_key]

        settings = get_active_settings()
        active_token = create_jwt_token(user_context, settings)

        st.info(
            f"**Tenant:** `{user_context.tenant_id}`\n\n"
            f"**Roles:** `{', '.join(user_context.roles)}`\n\n"
            f"**Groups:** `{', '.join(user_context.groups) or 'none'}`"
        )

        st.markdown("---")
        st.markdown("#### ⚙️ **Retrieval & Inference Options**")

        ollama_online = is_ollama_available()
        provider_options = ["fake", "ollama"]
        provider_labels = {
            "fake": "Deterministic Mock (Zero Dependency)",
            "ollama": f"Local Ollama ({'Online ✅' if ollama_online else 'Offline ⚠️'})",
        }
        current_choice = st.session_state.get("model_provider")
        if current_choice in provider_options:
            default_idx = provider_options.index(current_choice)
        else:
            default_idx = 1 if ollama_online else 0

        selected_provider = st.selectbox(
            "Model Provider Profile",
            options=provider_options,
            index=default_idx,
            format_func=lambda p: provider_labels.get(p, p),
            help=(
                "Select 'Deterministic Mock' for instant testing without external servers, "
                "or 'Local Ollama' if running Ollama locally."
            ),
        )
        st.session_state["model_provider"] = selected_provider

        if selected_provider == "ollama" and not ollama_online:
            st.warning(
                "⚠️ Local Ollama is unreachable on `localhost:11434`. Start Ollama with "
                "`ollama run llama3.2:3b` or switch to 'Deterministic Mock'."
            )

        st.session_state["enable_graph"] = st.checkbox(
            "GraphRAG Traversal",
            value=False,
            help="Traverse evidence assertion and concept links in Neo4j",
        )
        st.session_state["enable_cache"] = st.checkbox(
            "Semantic Caching",
            value=True,
            help="Permission- and staleness-aware semantic response caching",
        )
        st.session_state["enable_routing"] = st.checkbox(
            "Adaptive Model Routing",
            value=True,
            help="Deterministic complexity routing between small and large profiles",
        )
        st.session_state["cache_threshold"] = st.slider(
            "Cache Cosine Threshold",
            min_value=0.70,
            max_value=0.98,
            value=0.92,
            step=0.01,
            help="Minimum similarity required to reuse cached answers",
        )
        top_k = st.slider("Top Evidence Chunks (k)", min_value=1, max_value=10, value=5)

    # Main Header
    st.markdown('<div class="main-title">ClinGraph Evidence Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="safety-banner">
            ⚠️ <strong>RESEARCH & EDUCATIONAL DEMONSTRATION SYSTEM ONLY</strong><br>
            ClinGraph is an evidence-exploration research prototype, not a medical device.
            It is never intended for real-patient diagnosis, treatment, or clinical triage.
            All statements must be validated against primary peer-reviewed medical literature.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_explorer, tab_benchmarks, tab_audit = st.tabs(
        ["🔍 Evidence Explorer", "📊 Evaluation & Benchmarks", "🛡️ Audit & Security Log"]
    )

    # 1. Evidence Explorer Tab
    with tab_explorer:
        col_input, col_examples = st.columns([3, 2])

        with col_input:
            st.selectbox(
                "Research Objective",
                [
                    "Treatment Efficacy & Outcomes",
                    "Comparative Effectiveness",
                    "Adverse Events & Safety Warnings",
                    "Biochemical Mechanism & Pathways",
                ],
            )
            default_query = "Does metformin reduce cardiovascular mortality in type 2 diabetes?"
            query_input = st.text_area(
                "Biomedical Research Query",
                value=default_query,
                height=90,
                help="Enter clinical inquiry or evidence question",
            )
            submit_btn = st.button("🔎 Search & Synthesize Evidence", type="primary")

        with col_examples:
            st.markdown("**Sample Evaluation Queries**")
            if st.button("Cardiovascular: Metformin vs Sulfonylureas"):
                query_input = (
                    "Compare the cardiovascular benefits of metformin "
                    "versus sulfonylureas in diabetes"
                )
                st.rerun()
            if st.button("Heart Failure: Dapagliflozin Mortality"):
                query_input = "Does dapagliflozin reduce cardiovascular death in heart failure?"
                st.rerun()
            if st.button("Safety Boundary: Astronaut Surgery (Abstention Test)"):
                query_input = (
                    "What are the surgical outcomes of laparoscopic appendectomy in "
                    "astronauts during orbital spaceflight?"
                )
                st.rerun()
            if st.button("Multi-Tenant Boundary: Hospital A Secret Protocol"):
                query_input = "What is the secret protocol for acute heart failure stabilization?"
                st.rerun()

        if submit_btn and query_input.strip():
            with st.spinner(
                "Retrieving permitted literature and synthesizing grounded evidence..."
            ):
                resp = execute_query_backend(
                    query_text=query_input.strip(),
                    token=active_token,
                    settings=settings,
                    top_k=top_k,
                )

            if resp.status_code == 200:
                data = resp.json()

                st.markdown("### 📝 **Evidence Summary**")
                if data.get("abstained"):
                    st.warning(
                        f"**System Abstained from Answering:**\n\n{data.get('summary')}\n\n"
                        f"**Identified Gaps:** {', '.join(data.get('evidence_gaps', []))}"
                    )
                else:
                    st.success(data.get("summary"))

                if data.get("conflicts"):
                    st.error("⚠️ **Evidence Contradiction Detected Across Sources:**")
                    for conf in data["conflicts"]:
                        competing = ", ".join(conf.get("competing_chunk_ids", []))
                        st.markdown(
                            f"- **Conflict:** {conf['description']}\n"
                            f"  **Competing Chunks:** `{competing}`"
                        )

                st.markdown("### 🔬 **Claim-Level Grounded Verification**")
                claims = data.get("claims", [])
                if not claims:
                    st.info("No atomic claims extracted for this query.")

                for idx, claim in enumerate(claims):
                    status = claim.get("status", "unverified")
                    badge_class = {
                        "supported": "badge-supported",
                        "partially_supported": "badge-partially",
                        "contradicted": "badge-contradicted",
                        "unverified": "badge-unverified",
                    }.get(status, "badge-unverified")

                    with st.container():
                        st.markdown(
                            f"""
                            <div class="claim-card">
                                <span class="{badge_class}">{status.upper()}</span>
                                <strong> Claim {idx + 1}:</strong> {claim.get("claim_text")}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        citations = claim.get("citations", [])
                        if citations:
                            with st.expander(
                                f"📚 View {len(citations)} Supporting Source Passage(s)"
                            ):
                                for cit in citations:
                                    src_id = cit.get("source_id")
                                    chk_id = cit.get("chunk_id")
                                    doc_id = cit.get("document_id")
                                    st.markdown(
                                        f"**Title:** {cit.get('title')}\n\n"
                                        f"**Source:** `{src_id}` | "
                                        f"**Chunk:** `{chk_id}` | "
                                        f"**Doc:** `{doc_id}`\n\n"
                                        f'> *"{cit.get("supporting_excerpt")}"*'
                                    )
                        if claim.get("limitations"):
                            st.caption(f"**Limitations:** {', '.join(claim['limitations'])}")

                st.markdown("---")
                with st.expander("🛠️ **Developer & Inference Execution Trace**", expanded=False):
                    meta = data.get("metadata", {})
                    cache_meta = meta.get("cache", {})
                    routing_meta = meta.get("routing", {})
                    tokens_meta = meta.get("tokens", {})

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        outcome = cache_meta.get("outcome", "miss").upper()
                        st.metric("Cache Outcome", outcome)
                    with c2:
                        st.metric("Model Profile", routing_meta.get("profile", "standard"))
                    with c3:
                        st.metric("Latency", f"{meta.get('latency_ms', 0)} ms")
                    with c4:
                        st.metric("Estimated Cost", f"${meta.get('cost_usd', 0.0):.6f}")

                    st.json(
                        {
                            "request_id": data.get("request_id"),
                            "cache_details": cache_meta,
                            "model_routing": routing_meta,
                            "token_breakdown": tokens_meta,
                            "medical_disclaimer": data.get("disclaimer"),
                        }
                    )
            elif resp.status_code == 503 and "Ollama" in resp.text:
                st.error(
                    "⚠️ **Ollama Server Unavailable (HTTP 503)**\n\n"
                    f"**Details:** `{resp.text}`\n\n"
                    "**How to Fix:**\n"
                    "- **Zero-Dependency Demo:** In the left sidebar under "
                    "**Retrieval & Inference Options**, switch **Model Provider Profile** "
                    "to **`Deterministic Mock (Zero Dependency)`**.\n"
                    "- **Local Ollama:** Ensure Ollama is running (`ollama serve` or "
                    "`ollama run llama3.2:3b`)."
                )
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")

    # 2. Evaluation & Benchmarks Dashboard Tab
    with tab_benchmarks:
        st.markdown("### 📊 **Empirical Experiment Reports & Ablations**")
        st.caption(
            "All figures generated from repeatable script execution without fabricated metrics."
        )

        sub_ablation, sub_cache, sub_graph = st.tabs(
            ["1. 8-Stage Ablation Report", "2. Semantic Cache Sweep", "3. Knowledge Graph Ablation"]
        )

        with sub_ablation:
            ablation_file = Path("reports/ablation_experiments.json")
            if ablation_file.exists():
                ab_data = json.loads(ablation_file.read_text(encoding="utf-8"))
                st.caption(
                    f"Evaluation timestamp: `{ab_data.get('timestamp')}` | "
                    f"Vignettes: `{ab_data.get('total_vignettes')}`"
                )

                rows = []
                for item in ab_data.get("ablations", []):
                    m = item["metrics"]
                    rows.append(
                        {
                            "Ablation Configuration": item["name"],
                            "Recall@k": f"{m['retrieval_recall']:.1%}",
                            "Citation Validity": f"{m['citation_correctness']:.1%}",
                            "Abstention Accuracy": f"{m['abstention_accuracy']:.1%}",
                            "Avg Latency (ms)": m["avg_latency_ms"],
                            "Cost / 1k ($)": f"${m['cost_per_1k_usd']:.4f}",
                        }
                    )
                st.dataframe(rows, use_container_width=True)

                st.markdown("#### **Performance Trends Across Pipeline Stages**")
                chart_data = {
                    item["name"][:20]: item["metrics"]["avg_latency_ms"]
                    for item in ab_data.get("ablations", [])
                }
                st.bar_chart(chart_data)
            else:
                st.info(
                    "Run `python scripts/run_all_ablations.py` to generate the 8-stage "
                    "ablation report."
                )

        with sub_cache:
            cache_file = Path("reports/cache_benchmark.json")
            if cache_file.exists():
                c_data = json.loads(cache_file.read_text(encoding="utf-8"))
                st.caption(f"Methodology: {c_data.get('methodology')}")

                sweep = c_data.get("threshold_sweep", [])
                sweep_rows = []
                for s in sweep:
                    sweep_rows.append(
                        {
                            "Similarity Threshold": s["threshold"],
                            "Hit Rate": f"{s['hit_rate']:.1%}",
                            "Incorrect Reuse Rate": f"{s['incorrect_reuse_rate']:.1%}",
                            "p50 Latency (ms)": s["p50_latency_ms"],
                            "Cost / 1k ($)": f"${s['cost_per_1k_queries_usd']:.4f}",
                        }
                    )
                st.dataframe(sweep_rows, use_container_width=True)

                st.markdown("#### **Routing Distribution & Savings**")
                routing_eval = c_data.get("routing_evaluation", {})
                dist = routing_eval.get("distribution", {})
                costs = routing_eval.get("cost_comparison", {})

                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Small Profile Ratio", f"{dist.get('small_ratio', 0):.1%}")
                with col_r2:
                    st.metric("Large Profile Ratio", f"{dist.get('large_ratio', 0):.1%}")
                with col_r3:
                    st.metric(
                        "Cost Savings vs All-Large",
                        f"{costs.get('cost_savings_vs_all_large_percent', 0)}%",
                    )
            else:
                st.info(
                    "Run `python scripts/benchmark_cache_and_routing.py` to generate "
                    "threshold benchmarks."
                )

        with sub_graph:
            graph_file = Path("reports/graph_ablation.json")
            if graph_file.exists():
                g_data = json.loads(graph_file.read_text(encoding="utf-8"))
                vars_data = g_data.get("variants", {})
                h_var = vars_data.get("hybrid_only", {})
                g_var = vars_data.get("graph_enhanced", {})

                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.markdown("#### 🔹 **Hybrid Retrieval Only**")
                    st.json(h_var)
                with g_col2:
                    st.markdown("#### 🌐 **Graph-Enhanced Retrieval**")
                    st.json(g_var)

    # 3. Audit & Security Log Tab
    with tab_audit:
        st.markdown("### 🛡️ **Tenant Audit Trail & Operational Metrics**")
        st.caption("Immutable append-only audit trail filtered strictly to active tenant boundary.")

        metrics_resp = fetch_system_metrics(active_token)
        if metrics_resp.status_code == 200:
            m_json = metrics_resp.json()
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.metric("Total Queries", m_json.get("total_queries", 0))
            with a2:
                st.metric("Cache Hit Rate", f"{m_json.get('cache_hit_rate', 0):.1%}")
            with a3:
                st.metric("Allowed Decisions", m_json.get("allowed_decisions", 0))
            with a4:
                st.metric("Avg Latency", f"{m_json.get('average_latency_ms', 0):.1f} ms")

        st.markdown("#### **Recent Tenant Audit Events**")
        audit_resp = fetch_audit_logs(active_token)
        if audit_resp.status_code == 200:
            events = audit_resp.json().get("events", [])
            if events:
                audit_table = []
                for e in events:
                    audit_table.append(
                        {
                            "Timestamp": e.get("timestamp"),
                            "Action": e.get("action"),
                            "User": e.get("user_id"),
                            "Tenant": e.get("tenant_id"),
                            "Decision": e.get("decision"),
                            "Cache": e.get("cache_outcome"),
                            "Profile": e.get("selected_model_profile"),
                            "Latency (ms)": e.get("latency_ms"),
                            "Retrieved Chunks": len(e.get("retrieved_chunk_ids", [])),
                        }
                    )
                st.dataframe(audit_table, use_container_width=True)
            else:
                st.info("No audit events recorded for this tenant yet.")
        elif audit_resp.status_code == 403:
            st.warning(
                "Access Denied: Current user role does not possess 'auditor' or 'admin' "
                "privileges to view the audit log."
            )
        else:
            st.error(f"Failed to fetch audit events: {audit_resp.text}")


if __name__ == "__main__":
    render_ui()
