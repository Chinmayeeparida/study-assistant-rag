import html
import io
import time

import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from rag_core import (
    EMBED_MAX_TOKENS,
    EMBED_MODEL,
    LLM_MODEL,
    MAX_COMPARE_CHARS,
    RERANK_MODEL,
    STRATEGIES,
    answer,
    balance,
    chunk_text,
    get_embeddings,
    get_tokenizer,
    overflow_check,
    rerank,
    token_lengths,
)

st.set_page_config(page_title="Detailed Analysis", page_icon="🔬", layout="wide")

# Hide the left sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

top_left, _, top_right = st.columns([5, 2, 2])
with top_left:
    st.title("🔬 Detailed Analysis")
    st.caption(
        "A RAG lab: see how your material is chunked, how tokens are read, how "
        "retrieval works step by step, and how chunking methods compare."
    )
with top_right:
    if st.button("← Back to home"):
        st.switch_page("app.py")

if "store" not in st.session_state or "docs" not in st.session_state:
    st.warning("Build the study assistant on the home page first.")
    st.stop()

docs = st.session_state["docs"]
store = st.session_state["store"]
raw = st.session_state.get("raw", {})
stats = st.session_state.get("stats", {})
settings = st.session_state.get("settings", {"chunk_size": 450, "overlap": 45})
sources = list(dict.fromkeys(d.metadata["source"] for d in docs))
emb = get_embeddings()

PALETTE = [
    "rgba(255, 99, 132, 0.28)",
    "rgba(54, 162, 235, 0.28)",
    "rgba(255, 206, 86, 0.35)",
    "rgba(75, 192, 192, 0.28)",
    "rgba(153, 102, 255, 0.28)",
]
CURRENT = "Current index (as built on the home page)"


def safe(text):
    """Escape text so it can sit inside our HTML snippets."""
    return html.escape(text).replace("\n", " ").replace("$", "&#36;")


def approx_tokens(text):
    return max(1, len(text) // 4)


def source_text(src):
    return raw.get(src) or "\n".join(
        d.page_content for d in docs if d.metadata["source"] == src
    )


def colored_chunks(texts, limit=25):
    shown = texts[:limit] if limit else texts
    parts = []
    for i, t in enumerate(shown):
        color = PALETTE[i % len(PALETTE)]
        parts.append(
            f"<span style='background:{color};padding:2px 4px;border-radius:4px;'>"
            f"<b>#{i + 1}</b> {safe(t)}</span>"
        )
    st.markdown(" ".join(parts), unsafe_allow_html=True)


tabs = st.tabs(
    [
        "Overview & chunks",
        "Token inspection",
        "Query & retrieval",
        "Compare strategies",
        "Why this method",
    ]
)

# =================================================== 1. Overview & chunks
with tabs[0]:
    avg_tok = round(sum(len(d.page_content) for d in docs) / len(docs) / 4)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sources", len(sources))
    c2.metric("Chunks", len(docs))
    c3.metric("Avg tokens (approx.)", avg_tok)
    c4.metric("Chunk size", f"{settings.get('chunk_size', 900)} chars")
    c5.metric("Overlap", f"{settings.get('overlap', 90)} chars")

    st.markdown("**The pipeline your material goes through**")
    size_now = settings.get("chunk_size", 900)
    dot = (
        'digraph G { rankdir=LR; node [shape=box, style="rounded"]; '
        '"Sources\\nvideo, docs, blogs, code" -> '
        '"Chunking\\n' + str(size_now) + ' chars" -> '
        '"Embedding\\nmultilingual MiniLM" -> "Vector store\\nin memory" -> '
        '"Retrieve\\ntop-N" -> "Rerank\\noptional" -> '
        '"LLM answer\\n' + LLM_MODEL.split("/")[-1] + '"; }'
    )
    st.graphviz_chart(dot)

    st.markdown("---")
    st.markdown("**Chunk visualization**")
    cc1, cc2 = st.columns(2)
    src = cc1.selectbox("Source", sources, key="vis_src")
    strategy = cc2.selectbox("Chunking strategy", [CURRENT] + STRATEGIES, key="vis_strat")

    if strategy == CURRENT:
        texts = [d.page_content for d in docs if d.metadata["source"] == src]
    else:
        s1, s2 = st.columns(2)
        vsize = s1.slider("Chunk size (characters)", 200, 2000, 450, 50, key="vis_size")
        voverlap = s2.slider("Overlap (characters)", 0, 400, 50, 10, key="vis_overlap")
        memo = st.session_state.setdefault("chunk_memo", {})
        key = (src, strategy, vsize, voverlap)
        if key not in memo:
            with st.spinner(f"Chunking with the {strategy} strategy..."):
                text = source_text(src)[:MAX_COMPARE_CHARS]
                items = chunk_text(text, strategy, vsize, voverlap, embeddings=emb)
                memo[key] = [i["text"] for i in items]
        texts = memo[key]
        if len(source_text(src)) > MAX_COMPARE_CHARS:
            st.caption(
                f"Only the first {MAX_COMPARE_CHARS:,} characters of this source "
                "are used here, to keep the lab fast."
            )

    toks = [approx_tokens(t) for t in texts]
    if toks:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chunks", len(texts))
        m2.metric("Avg tokens (approx.)", round(sum(toks) / len(toks)))
        m3.metric("Smallest", min(toks))
        m4.metric("Largest", max(toks))
        st.bar_chart(
            pd.DataFrame({"approx. tokens per chunk": toks}, index=range(1, len(toks) + 1))
        )
        show_all = st.checkbox("Show all chunks (default: first 25)", value=False, key="vis_all")
        colored_chunks(texts, limit=None if show_all else 25)
    else:
        st.info("No chunks were produced for this source with these settings.")

    table = pd.DataFrame(
        [
            {
                "chunk": i + 1,
                "type": d.metadata["type"],
                "source": d.metadata["source"],
                "label": d.metadata["label"],
                "characters": len(d.page_content),
                "approx_tokens": approx_tokens(d.page_content),
                "text": d.page_content,
            }
            for i, d in enumerate(docs)
        ]
    )
    st.download_button(
        "⬇ Download the current index chunks (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="chunks.csv",
        mime="text/csv",
    )

# ================================================== 2. Token inspection
with tabs[1]:
    st.write(
        "Models read **tokens** (pieces of words), not words. The embedding model "
        f"used here keeps only the first **{EMBED_MAX_TOKENS} tokens** of each chunk "
        "and silently ignores the rest."
    )

    if st.button("Scan every chunk in the index"):
        with st.spinner("Counting tokens..."):
            lengths = token_lengths([d.page_content for d in docs])
        over = sum(1 for n in lengths if n > EMBED_MAX_TOKENS)
        st.session_state["scan"] = (len(lengths), over, max(lengths))
    if "scan" in st.session_state:
        n_all, n_over, longest = st.session_state["scan"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Chunks scanned", n_all)
        s2.metric(f"Longer than {EMBED_MAX_TOKENS} tokens", n_over)
        s3.metric("Longest chunk (tokens)", longest)
        if n_over:
            st.warning(
                f"{n_over} of {n_all} chunks are longer than the embedder can read, "
                "so the end of each is ignored when matching a question."
            )

    st.markdown("---")
    choice = st.selectbox(
        "Pick a chunk",
        options=list(range(len(docs))),
        format_func=lambda i: (
            f"#{i + 1} · {docs[i].metadata['label']} · {docs[i].page_content[:50]}"
        ),
        key="tok_choice",
    )
    text = st.text_area(
        "Text to inspect (you can edit it)",
        value=docs[choice].page_content,
        height=140,
        key=f"tok_text_{choice}",
    )
    if text.strip():
        try:
            tokens = get_tokenizer().tokenize(text)
        except Exception as e:
            tokens = None
            st.error(f"Could not load the tokenizer: {e}")
        if tokens is not None:
            limit = EMBED_MAX_TOKENS - 2  # 2 slots are used by special tokens
            kept, ignored = tokens[:limit], tokens[limit:]
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Tokens in this text", len(tokens))
            t2.metric("Embedder reads", len(kept))
            t3.metric("Ignored", len(ignored))
            t4.metric("Characters", len(text))

            def span(token, bad):
                label = token.lstrip("▁") or "▁"
                style = (
                    "background:rgba(248,81,73,0.30);text-decoration:line-through;"
                    if bad
                    else "background:rgba(46,160,67,0.30);"
                )
                return (
                    f"<span style='{style}padding:1px 3px;margin:1px;"
                    f"border-radius:3px;display:inline-block;'>{safe(label)}</span>"
                )

            st.markdown("🟩 read by the embedder   🟥 ignored (crossed out)")
            st.markdown(
                "".join(span(t, False) for t in kept)
                + "".join(span(t, True) for t in ignored),
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("**Overflow: before and after**")
            st.write(
                "Type a question whose answer sits near the END of this chunk. "
                "*Before* = the whole chunk is embedded (the tail is ignored). "
                "*After* = the chunk is cut into windows that fit, and each is scored."
            )
            oq = st.text_input("Question about this chunk", key="over_q")
            if oq and st.button("Compare before / after"):
                with st.spinner("Embedding..."):
                    before, windows, after = overflow_check(oq, text, emb)
                best = max(after)
                o1, o2 = st.columns(2)
                o1.metric("BEFORE: whole chunk, score", f"{before:.3f}")
                o2.metric(
                    "AFTER: best window, score",
                    f"{best:.3f}",
                    delta=f"{best - before:+.3f}",
                )
                st.dataframe(
                    pd.DataFrame(
                        {
                            "window": range(1, len(windows) + 1),
                            "score": [round(x, 3) for x in after],
                            "text": [w[:120] for w in windows],
                        }
                    )
                )
                if best > before:
                    st.success(
                        "Splitting recovered a better match that the ignored tail "
                        "was hiding."
                    )
                else:
                    st.info(
                        "The answer was already in the part the embedder reads, so "
                        "splitting did not help for this question."
                    )

# ================================================== 3. Query & retrieval
with tabs[2]:
    st.write(
        "Follow one question through the pipeline: embed it, search the vector "
        "store for candidates, optionally rerank them, then write the answer. "
        "Each stage is timed."
    )
    question = st.text_input("Question", key="ret_q")
    p1, p2, p3, p4 = st.columns(4)
    top_n = p1.slider("Vector search: candidates (top N)", 3, 30, 15, key="ret_n")
    keep = p2.slider("Chunks sent to the AI (keep)", 1, 8, 4, key="ret_keep")
    balanced = p3.checkbox("Balanced across sources", value=True, key="ret_bal")
    use_rerank = p4.checkbox(
        "Use the reranker (first use downloads a small model)", value=False, key="ret_rr"
    )

    if st.button("Run the pipeline"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            rows = []
            t = time.perf_counter()
            qvec = emb.embed_query(question)
            rows.append(("1. Embed the question", time.perf_counter() - t, f"{len(qvec)}-number vector"))

            t = time.perf_counter()
            ranked_all = store.similarity_search_with_score_by_vector(qvec, k=len(docs))
            candidates = (
                balance(ranked_all, top_n) if balanced else ranked_all[:top_n]
            )
            rows.append(("2. Vector search", time.perf_counter() - t, f"{len(candidates)} candidates of {len(docs)} chunks"))

            reranked, rerank_error = None, None
            t = time.perf_counter()
            if use_rerank:
                try:
                    reranked = rerank(question, candidates, keep)
                    rows.append(("3. Rerank", time.perf_counter() - t, f"kept {len(reranked)}"))
                except Exception as e:
                    rerank_error = str(e)
                    rows.append(("3. Rerank", time.perf_counter() - t, "failed, used vector order"))
            else:
                rows.append(("3. Rerank", 0.0, "skipped"))

            if reranked:
                final = [(d, sc) for d, sc, _ in reranked]
            else:
                final = candidates[:keep]

            t = time.perf_counter()
            reply = answer(question, final, "Step-by-step tutorial")
            rows.append(("4. Write the answer", time.perf_counter() - t, f"from {len(final)} chunks"))

            st.session_state["pipe"] = {
                "rows": rows,
                "candidates": candidates,
                "reranked": reranked,
                "final": final,
                "reply": reply,
                "rerank_error": rerank_error,
            }

    pipe = st.session_state.get("pipe")
    if pipe:
        st.markdown("**Time per stage**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"stage": a, "seconds": round(b, 2), "detail": c}
                    for a, b, c in pipe["rows"]
                ]
            )
        )
        if pipe["rerank_error"]:
            st.error(f"The reranker could not run: {pipe['rerank_error']}")

        st.markdown("**Candidates from vector search (similarity score)**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rank": i + 1,
                        "score": round(float(s), 3),
                        "type": d.metadata["type"],
                        "label": d.metadata["label"],
                        "source": d.metadata["source"],
                    }
                    for i, (d, s) in enumerate(pipe["candidates"])
                ]
            )
        )
        if pipe["reranked"]:
            st.markdown("**After reranking (reranker score, higher is better)**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "rank": i + 1,
                            "rerank score": round(sc, 3),
                            "vector score": round(vs, 3),
                            "label": d.metadata["label"],
                        }
                        for i, (d, sc, vs) in enumerate(pipe["reranked"])
                    ]
                )
            )
            st.caption(
                "The reranker reads the question and each chunk together, so it can "
                "reorder chunks that looked similar by vector search alone. Its "
                "scores are not on the same scale as similarity scores."
            )

        st.markdown("**Chunks sent to the AI**")
        for d, s in pipe["final"]:
            with st.expander(f"{d.metadata['label']} · score {float(s):.3f}"):
                st.write(d.page_content)

        st.markdown("**Answer**")
        st.markdown(pipe["reply"])

# ================================================= 4. Compare strategies
with tabs[3]:
    st.write(
        "Cut the **same text** with different chunking strategies, ask the same "
        "question, and see which method retrieves the best passages."
    )
    k1, k2 = st.columns(2)
    csrc = k1.selectbox("Source to compare on", sources, key="cmp_src")
    cquestion = k2.text_input("Test question", key="cmp_q")
    z1, z2 = st.columns(2)
    csize = z1.slider("Chunk size (characters)", 200, 2000, 450, 50, key="cmp_size")
    coverlap = z2.slider("Overlap (characters)", 0, 400, 50, 10, key="cmp_overlap")
    chosen = st.multiselect(
        "Strategies to compare", STRATEGIES, default=STRATEGIES, key="cmp_strats"
    )
    with_answers = st.checkbox(
        "Also write an answer for each strategy (uses one AI call per strategy)",
        value=False,
        key="cmp_answers",
    )
    st.caption(
        "Semantic chunking embeds every sentence, so it is the slowest. Only the "
        f"first {MAX_COMPARE_CHARS:,} characters of a source are used here."
    )

    if st.button("Run comparison"):
        if not cquestion.strip() or not chosen:
            st.warning("Type a test question and pick at least one strategy.")
        else:
            text = source_text(csrc)[:MAX_COMPARE_CHARS]
            results = {}
            for name in chosen:
                with st.spinner(f"Running {name}..."):
                    t0 = time.perf_counter()
                    items = chunk_text(text, name, csize, coverlap, embeddings=emb)
                    t1 = time.perf_counter()
                    temp = InMemoryVectorStore(emb)
                    temp.add_documents(
                        [
                            Document(page_content=i["text"], metadata={"context": i["context"]})
                            for i in items
                        ]
                    )
                    t2 = time.perf_counter()
                    hits = temp.similarity_search_with_score(cquestion, k=3)
                    t3 = time.perf_counter()
                    lens = token_lengths([i["text"] for i in items])
                    reply = None
                    if with_answers and hits:
                        pairs = [
                            (
                                Document(
                                    page_content=h.metadata["context"],
                                    metadata={"label": f"{name} #{r}", "type": "document", "source": csrc},
                                ),
                                sc,
                            )
                            for r, (h, sc) in enumerate(hits, start=1)
                        ]
                        reply = answer(cquestion, pairs, "Explain like I'm 10")
                    results[name] = {
                        "items": items,
                        "hits": hits,
                        "lens": lens,
                        "times": (t1 - t0, t2 - t1, t3 - t2),
                        "reply": reply,
                    }
            st.session_state["cmp_results"] = results

    results = st.session_state.get("cmp_results")
    if results:
        summary_rows, hit_rows = [], []
        for name, r in results.items():
            lens = r["lens"] or [0]
            over = sum(1 for n in lens if n > EMBED_MAX_TOKENS)
            best = r["hits"][0][1] if r["hits"] else float("nan")
            summary_rows.append(
                {
                    "Strategy": name,
                    "Chunks": len(r["items"]),
                    "Avg tokens": round(sum(lens) / len(lens)),
                    "Largest (tokens)": max(lens),
                    f"Over {EMBED_MAX_TOKENS} tokens": over,
                    "Chunking (s)": round(r["times"][0], 2),
                    "Embed + index (s)": round(r["times"][1], 2),
                    "Search (s)": round(r["times"][2], 3),
                    "Best match score": round(float(best), 3),
                }
            )
            for rank, (h, sc) in enumerate(r["hits"], start=1):
                hit_rows.append(
                    {
                        "Strategy": name,
                        "Rank": rank,
                        "Score": round(float(sc), 3),
                        "Matched chunk": h.page_content,
                        "Context given to AI": h.metadata["context"],
                    }
                )
        summary_df, hits_df = pd.DataFrame(summary_rows), pd.DataFrame(hit_rows)
        st.markdown("**Side by side**")
        st.dataframe(summary_df)

        for name, r in results.items():
            with st.expander(f"{name}: top 3 passages" + (" and answer" if r["reply"] else "")):
                for rank, (h, sc) in enumerate(r["hits"], start=1):
                    st.markdown(f"**Top {rank}** · score `{float(sc):.3f}`")
                    st.write(h.page_content)
                    if h.metadata["context"] != h.page_content:
                        st.caption(
                            f"Parent context sent to the AI: "
                            f"{len(h.metadata['context'])} characters"
                        )
                if r["reply"]:
                    st.markdown("**Answer from this strategy**")
                    st.markdown(r["reply"])

        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                hits_df.to_excel(writer, sheet_name="Top chunks", index=False)
            st.download_button(
                "⬇ Download comparison (Excel)",
                buf.getvalue(),
                file_name="chunking_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.download_button(
                "⬇ Download comparison (CSV)",
                summary_df.to_csv(index=False).encode("utf-8"),
                file_name="chunking_comparison.csv",
                mime="text/csv",
            )
            st.caption("For an Excel file, run: pip install openpyxl")

        st.info(
            "A higher score means closer in meaning to your question, but also "
            "read the passages: the best strategy is the one whose top passage "
            "actually answers it."
        )

# ================================================= 5. Why this method
with tabs[4]:
    size_now = settings.get("chunk_size", 900)
    overlap_now = settings.get("overlap", size_now // 10)
    cards = [
        (
            f"Chunk size: {size_now} characters (about {size_now // 4} tokens)",
            "Big enough to hold one complete idea, small enough that the match "
            "stays focused.",
            "Smaller chunks give precise matches but can cut an explanation in "
            "half. Larger chunks keep context but blur the match and may exceed "
            "what the embedding model can read.",
        ),
        (
            f"Overlap: {overlap_now} characters (10% of the chunk)",
            "Repeats the end of one chunk at the start of the next, so a sentence "
            "on a boundary is not lost.",
            "More overlap protects meaning but stores duplicate text and makes the "
            "index larger.",
        ),
        (
            "Five chunking strategies to compare",
            "Fixed-size is the simplest, recursive respects paragraphs and "
            "sentences, sentence-based keeps sentences whole, semantic cuts where "
            "the meaning changes, and hierarchical searches small chunks but gives "
            "the AI the bigger parent passage.",
            "Smarter strategies keep ideas together but cost more: semantic needs "
            "an embedding for every sentence, and hierarchical sends more text to "
            "the AI.",
        ),
        (
            "Embedding model: multilingual MiniLM (runs on your computer, free)",
            f"`{EMBED_MODEL}` works across many languages, needs no paid service, "
            "and is small enough for a free server.",
            f"It reads only the first {EMBED_MAX_TOKENS} tokens of each chunk "
            "(see Token inspection). A model with a longer limit reads more but "
            "is heavier to run.",
        ),
        (
            "Retrieval: balanced across sources",
            "Takes the 2 best chunks from every source, so the video, each "
            "document and each blog all contribute to the answer.",
            "A weakly related source can still add a less useful chunk. Pure "
            "similarity avoids that but can let one source crowd out the others.",
        ),
        (
            "Reranker (optional)",
            f"`{RERANK_MODEL}` re-reads the question together with each candidate "
            "and re-orders them, usually putting the truly relevant ones first.",
            "It adds time and a model download, and it only reorders what vector "
            "search already found.",
        ),
        (
            f"Answer model: open-source `{LLM_MODEL}` via Groq (free tier)",
            "The model is told to answer only from the retrieved excerpts and to "
            "cite them, which reduces made-up answers.",
            "Free tiers have rate limits shared by all visitors, and a model can "
            "still misread an excerpt, so check the cited sources.",
        ),
        (
            "Vector store: in memory",
            "Simple and fast, and nothing is saved on the server, so visitors' "
            "files stay private to their session.",
            "The index disappears when the page is refreshed. A persistent store "
            "such as Chroma or FAISS would keep it.",
        ),
    ]
    for title, why, consequence in cards:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.markdown(f"**Why:** {why}")
            st.markdown(f"**Trade-off / consequence:** {consequence}")
