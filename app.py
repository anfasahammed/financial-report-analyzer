"""
Financial Report Analyzer — Streamlit UI
=========================================
RAG-powered financial document analysis system.
Analyzes SEC filings (10-K / 20-F) from IT/consulting companies.
"""

import os
import streamlit as st
import chromadb
from groq import Groq

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Financial Report Analyzer",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# CONSTANTS
# ============================================================
CHROMA_DB_DIR = os.path.join("data", "chroma_db")
COLLECTION_NAME = "sec_filings"

COMPANIES = {
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "ACN": "Accenture",
    "IBM": "IBM",
    "CTSH": "Cognizant",
    "INFY": "Infosys",
    "WIT": "Wipro",
}

SYSTEM_PROMPT = """You are an expert financial analyst assistant. Answer questions about 
company SEC filings (10-K and 20-F annual reports) based ONLY on the provided context.

Rules:
1. Answer ONLY based on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, say so clearly.
3. Always cite which company's filing the information comes from.
4. When discussing financial figures, be precise — include exact numbers.
5. Structure your response clearly."""


# ============================================================
# INITIALIZE SERVICES (cached so they don't reload every time)
# ============================================================
@st.cache_resource
def load_chroma():
    """Load ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def get_groq_client():
    """Get Groq client using API key from secrets."""
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ============================================================
# RAG PIPELINE
# ============================================================
def retrieve_and_answer(question, collection, groq_client, model_name,
                         ticker_filter=None, n_results=5):
    """
    Full RAG pipeline: retrieve chunks → build prompt → call LLM → return answer + sources.
    """
    # Retrieve
    where_filter = {"ticker": ticker_filter} if ticker_filter else None
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where_filter,
    )

    if not results['documents'][0]:
        return "No relevant information found in the filings.", []

    # Build context
    context_parts = []
    sources = []
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
    )):
        source_label = f"{meta['ticker']} ({meta['company_name']}) - {meta['form_type']}"
        context_parts.append(f"[Source {i+1}: {source_label}]\n{doc}")
        sources.append({
            'company': meta['company_name'],
            'ticker': meta['ticker'],
            'form': meta['form_type'],
            'chunk': meta['chunk_index'],
            'distance': round(dist, 4),
        })

    context = "\n\n---\n\n".join(context_parts)

    # Generate
    user_prompt = f"""Context from SEC filings:

{context}

---

Question: {question}

Provide a detailed answer based on the context above."""

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model_name,
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    return answer, sources, tokens_used


# ============================================================
# UI
# ============================================================
def main():
    # Header
    st.title("📊 Financial Report Analyzer")
    st.markdown(
        "Ask questions about SEC filings from IT/consulting companies. "
        "Powered by RAG + Groq (Llama 3.3 70B)."
    )

    # Check for API key
    groq_client = get_groq_client()
    if not groq_client:
        st.error(
            "Groq API key not found. Add `GROQ_API_KEY` to your Streamlit secrets. "
            "See the README for instructions."
        )
        st.stop()

    # Load ChromaDB
    try:
        collection = load_chroma()
    except Exception as e:
        st.error(f"Failed to load vector database: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        model_name = st.selectbox(
            "LLM Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0,
            help="70B is more capable but uses more tokens. 8B is faster and cheaper.",
        )

        ticker_filter = st.selectbox(
            "Filter by company",
            ["All Companies"] + [f"{t} — {n}" for t, n in COMPANIES.items()],
            index=0,
        )

        n_results = st.slider(
            "Chunks to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="More chunks = broader context but more noise.",
        )

        st.divider()
        st.markdown("**Available Companies**")
        for ticker, name in COMPANIES.items():
            st.markdown(f"- **{ticker}** — {name}")

        st.divider()
        st.caption(f"Vector DB: {collection.count()} chunks")

    # Parse ticker filter
    selected_ticker = None
    if ticker_filter != "All Companies":
        selected_ticker = ticker_filter.split(" — ")[0]

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"- **{s['ticker']}** ({s['company']}) — "
                            f"{s['form']}, chunk #{s['chunk']}, "
                            f"distance: {s['distance']}"
                        )

    # Chat input
    if question := st.chat_input("Ask about the SEC filings..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching filings and generating answer..."):
                try:
                    answer, sources, tokens = retrieve_and_answer(
                        question=question,
                        collection=collection,
                        groq_client=groq_client,
                        model_name=model_name,
                        ticker_filter=selected_ticker,
                        n_results=n_results,
                    )

                    st.markdown(answer)

                    with st.expander("📎 Sources"):
                        for s in sources:
                            st.markdown(
                                f"- **{s['ticker']}** ({s['company']}) — "
                                f"{s['form']}, chunk #{s['chunk']}, "
                                f"distance: {s['distance']}"
                            )

                    st.caption(f"Tokens used: {tokens} | Model: {model_name}")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"Error: {e}")

    # Example questions
    if not st.session_state.messages:
        st.markdown("### Try these questions:")
        cols = st.columns(2)

        examples = [
            "What is Microsoft's total revenue?",
            "Compare cloud strategies of Microsoft and Google",
            "What are IBM's key risk factors?",
            "How many employees does Infosys have?",
        ]

        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.rerun()


if __name__ == "__main__":
    main()
