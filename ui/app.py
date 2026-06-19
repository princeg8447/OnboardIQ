"""
ui/app.py — OnboardIQ Streamlit UI
Clean, professional chat interface with document upload and guardrail visibility.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from main import ask
from rag.ingestor import ingest_documents

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="OnboardIQ",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .stApp { background-color: #0f1117; }

    /* Chat bubbles */
    .user-bubble {
        background: #1e3a5f;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        color: #e8eaf6;
        font-size: 15px;
    }
    .bot-bubble {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 85%;
        color: #e2e8f0;
        font-size: 15px;
        line-height: 1.6;
    }
    .source-tag {
        display: inline-block;
        background: #1e3a5f;
        color: #90cdf4;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 2px 3px;
    }
    .warning-box {
        background: #2d2000;
        border-left: 3px solid #d69e2e;
        border-radius: 4px;
        padding: 8px 12px;
        margin-top: 8px;
        font-size: 13px;
        color: #faf089;
    }
    .guard-badge {
        display: inline-block;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-pass { background: #1c4532; color: #68d391; }
    .badge-warn { background: #2d2000; color: #f6e05e; }
    .badge-fail { background: #3d0000; color: #fc8181; }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value { font-size: 22px; font-weight: 700; color: #90cdf4; }
    .metric-label { font-size: 12px; color: #718096; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "docs_ingested"  not in st.session_state: st.session_state.docs_ingested  = False
if "total_queries"  not in st.session_state: st.session_state.total_queries  = 0
if "blocked_count"  not in st.session_state: st.session_state.blocked_count  = 0
if "avg_grounded"   not in st.session_state: st.session_state.avg_grounded   = []


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 OnboardIQ")
    st.markdown("*Intelligent Company Onboarding Assistant*")
    st.divider()

    # Document upload
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload your company's HR policies, handbooks, SOPs, etc.",
    )

    url_input = st.text_area(
        "Or paste URLs (one per line)",
        placeholder="https://company.com/handbook\nhttps://...",
        height=80,
    )

    if st.button("⚡ Ingest Documents", use_container_width=True, type="primary"):
        pdf_paths, text_paths, urls = [], [], []

        with st.spinner("Processing documents..."):
            for f in uploaded_files:
                suffix = "." + f.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                if suffix == ".pdf":
                    pdf_paths.append(tmp_path)
                else:
                    text_paths.append(tmp_path)

            if url_input.strip():
                urls = [u.strip() for u in url_input.strip().split("\n") if u.strip()]

            # Always ingest sample docs
            sample_dir = "../data/sample_docs"
            if os.path.exists(sample_dir):
                for fname in os.listdir(sample_dir):
                    fp = os.path.join(sample_dir, fname)
                    if fname.endswith(".pdf"):   pdf_paths.append(fp)
                    elif fname.endswith((".txt",".md")): text_paths.append(fp)

            result = ingest_documents(
                pdf_paths=pdf_paths or None,
                urls=urls or None,
                text_paths=text_paths or None,
            )

        if result["status"] == "success":
            st.session_state.docs_ingested = True
            st.success(f"✅ Ingested {result['child_chunks']} chunks from {result['total_docs']} docs")
        else:
            st.warning("No documents found. Using pre-loaded sample docs.")

    st.divider()

    # Stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{st.session_state.total_queries}</div>
            <div class='metric-label'>Queries</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        avg = (sum(st.session_state.avg_grounded) / len(st.session_state.avg_grounded)
               if st.session_state.avg_grounded else 0)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{avg:.0%}</div>
            <div class='metric-label'>Avg Ground.</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Example queries
    st.markdown("### 💡 Try asking...")
    examples = [
        "How many sick leaves do I get?",
        "How do I claim travel reimbursement?",
        "What should I do on my first day?",
        "What are the IT setup steps?",
        "What benefits am I entitled to?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_query = ex

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────
st.markdown("## 🏢 OnboardIQ — Company Onboarding Assistant")
st.markdown("Ask me anything about company policies, benefits, IT setup, or your first week.")
st.divider()

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>👤 {msg['content']}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>🤖 {msg['content']}</div>",
                    unsafe_allow_html=True)

        # Sources
        if msg.get("sources"):
            src_html = "".join(f"<span class='source-tag'>📄 {s}</span>" for s in msg["sources"])
            st.markdown(f"<div style='margin-top:6px;'>{src_html}</div>",
                        unsafe_allow_html=True)

        # Warnings
        for w in msg.get("warnings", []):
            st.markdown(f"<div class='warning-box'>{w}</div>", unsafe_allow_html=True)

        # Guardrail badges
        g_score = msg.get("groundedness_score", 0)
        g_class = "badge-pass" if g_score >= 0.7 else ("badge-warn" if g_score >= 0.5 else "badge-fail")
        hall    = msg.get("hallucination_risk", False)
        h_class = "badge-fail" if hall else "badge-pass"

        st.markdown(
            f"<div style='margin-top:8px;'>"
            f"<span class='guard-badge {g_class}'>Ground: {g_score:.0%}</span>"
            f"<span class='guard-badge {h_class}'>{'⚠ Hallucination Risk' if hall else '✓ Grounded'}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Chat input ────────────────────────────────────────────────────
query = st.chat_input("Ask about leave, reimbursements, IT setup, benefits...")

# Handle example query clicks
if hasattr(st.session_state, "pending_query"):
    query = st.session_state.pending_query
    del st.session_state.pending_query

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🔍 Searching company documents..."):
        response = ask(query)

    st.session_state.total_queries += 1
    st.session_state.avg_grounded.append(response.groundedness_score)

    if response.blocked:
        bot_msg = {
            "role":    "assistant",
            "content": f"🚫 {response.block_reason}",
            "sources": [],
            "warnings": [],
            "groundedness_score": 0.0,
            "hallucination_risk": False,
        }
        st.session_state.blocked_count += 1
    else:
        bot_msg = {
            "role":               "assistant",
            "content":            response.answer,
            "sources":            response.sources,
            "warnings":           response.warnings,
            "groundedness_score": response.groundedness_score,
            "hallucination_risk": response.hallucination_risk,
        }

    st.session_state.messages.append(bot_msg)
    st.rerun()
