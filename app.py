import json

import streamlit as st
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore

from history import (
    PERSIST,
    add_question,
    delete_record,
    describe_source,
    find_record,
    load_history,
    load_material,
    make_topic,
    material_exists,
    new_record,
    save_history,
    save_material,
)
from rag_core import (
    CODE_EXT,
    DOC_EXT,
    STYLES,
    answer,
    extract_video_id,
    get_embeddings,
    load_file,
    load_pasted,
    load_video,
    load_web,
    retrieve,
    split_answer,
    summarize,
)

load_dotenv()
st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")

# Hide the left sidebar (the "app / Detailed Analysis" list)
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


if "history" not in st.session_state:
    st.session_state["history"] = load_history()


def tile(col, label, value):
    with col:
        with st.container(border=True):
            st.metric(label, value)


st.title("📚 Study Assistant")

# ---------- top row (as in the wireframe) ----------
top_left, _, top_right = st.columns([5, 1, 3])

with top_left:
    with st.expander(
        "➕ Upload doc / video / code / pdf / blog",
        expanded="store" not in st.session_state,
    ):
        topic_in = st.text_input(
            "Topic (optional, the AI names it if you leave this empty)", key="topic"
        )
        url = st.text_input("YouTube link (optional)", key="url")
        blog_urls = st.text_area(
            "Blog / web page links (optional, one per line)", key="blog_urls"
        )
        pasted = st.text_area(
            "Or paste text (a blog post, notes) (optional)", key="pasted"
        )
        files = st.file_uploader(
            "Documents or code (optional)",
            type=sorted(DOC_EXT | CODE_EXT),
            accept_multiple_files=True,
            key="files",
        )
        chunk_size = st.slider(
            "Chunk size (characters)", 200, 2000, 450, step=50, key="chunk_size"
        )
        user_key = st.text_input(
            "Your own Groq API key (optional, free at console.groq.com)",
            type="password",
            key="user_groq_key",
            help="If the shared key is busy, paste your own. It is kept only for this visit.",
        )
        build = st.button("Build study assistant", type="primary")

# Remember the visitor's key for the whole visit (also on the other page)
st.session_state["groq_key_saved"] = (user_key or "").strip()

if build:
    all_docs = []
    raw = {}
    stats = {"videos": 0, "documents": 0, "code": 0, "pages": 0}
    try:
        with st.spinner("Reading your material and building the index..."):
            if url.strip():
                vid = extract_video_id(url)
                if not vid:
                    st.error("That doesn't look like a YouTube link.")
                    st.stop()
                docs_v, text_v = load_video(vid, chunk_size)
                all_docs += docs_v
                raw[f"YouTube {vid}"] = text_v
                stats["videos"] = 1

            failed = []
            for page_url in blog_urls.split():
                try:
                    docs_w, text_w = load_web(page_url.strip(), chunk_size)
                    all_docs += docs_w
                    raw[page_url.strip()] = text_w
                    stats["documents"] += 1
                except Exception:
                    failed.append(page_url)
            if failed:
                st.warning(
                    "Could not read these pages (the site may block automated "
                    "readers, or need a login): " + ", ".join(failed)
                )

            if pasted.strip():
                docs_p, text_p = load_pasted(pasted, chunk_size)
                all_docs += docs_p
                raw["Pasted text"] = text_p
                stats["documents"] += 1

            for f in files or []:
                docs_f, kind, pages, text_f = load_file(f, chunk_size)
                all_docs += docs_f
                raw[f.name] = text_f
                stats["code" if kind == "code" else "documents"] += 1
                stats["pages"] += pages

            if not all_docs:
                st.error(
                    "Nothing to index. Add a link, some text, or a file with "
                    "readable text."
                )
                st.stop()

            store = InMemoryVectorStore(get_embeddings())
            store.add_documents(all_docs)
            summary = summarize(all_docs)
        st.session_state.update(
            store=store,
            docs=all_docs,
            raw=raw,
            stats=stats,
            summary=summary,
            settings={"chunk_size": chunk_size, "overlap": chunk_size // 10},
        )
        topic = topic_in.strip() or make_topic(summary)
        record = new_record(
            topic, stats, raw.keys(), len(all_docs),
            {"chunk_size": chunk_size, "overlap": chunk_size // 10}, summary,
        )
        save_material(record["id"], all_docs, raw)
        st.session_state["history"].append(record)
        save_history(st.session_state["history"])
        st.session_state["current_id"] = record["id"]
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        if url.strip():
            st.info(
                "If the video's captions could not be downloaded (YouTube sometimes "
                "blocks cloud servers), open the video, copy its transcript, and "
                "paste it into the **Or paste text** box instead."
            )

built = "store" in st.session_state and "docs" in st.session_state
with top_right:
    if st.button("🔬 Detailed Analysis", disabled=not built):
        st.switch_page("pages/1_Detailed_Analysis.py")

# ---------- tiles, summary, questions ----------
if not built:
    st.info(
        "Add a video, documents, code, blog links or text above, then click "
        "**Build study assistant**."
    )
else:
    s = st.session_state["stats"]
    docs = st.session_state["docs"]
    total_chars = sum(len(d.page_content) for d in docs)
    avg_tokens = round(total_chars / len(docs) / 4)

    row1 = st.columns(3)
    tile(row1[0], "Video", s["videos"])
    tile(row1[1], "Code files", s["code"])
    tile(row1[2], "Avg tokens (approx.)", avg_tokens)
    row2 = st.columns(3)
    tile(row2[0], "Document", s["documents"])
    tile(row2[1], "Pages (PDF)", s["pages"])
    tile(row2[2], "Chunks", len(docs))

    with st.container(border=True):
        st.subheader("Summary")
        st.write(st.session_state["summary"])

    question = st.text_input("Ask a question about your material", key="question")
    style = st.radio("Explain it as", STYLES, horizontal=True, key="style")
    want_diagram = st.checkbox("Also draw a simple diagram", value=True)
    if question:
        ranked = retrieve(
            st.session_state["store"], len(docs), question, k=5, balanced=True
        )
        with st.spinner("Reading all your sources and writing the tutorial..."):
            reply = answer(question, ranked, style, diagram=want_diagram)
        before, dot, after = split_answer(reply)
        st.subheader("Answer")
        st.markdown(before)
        if dot:
            st.graphviz_chart(dot)
            with st.expander("Diagram source"):
                st.code(dot)
        if after:
            st.markdown(after)

        current = find_record(st.session_state["history"], st.session_state.get("current_id"))
        if current is not None:
            add_question(current, question, style, reply)
            save_history(st.session_state["history"])

        with st.expander("Sources used (best match first)"):
            for d, score in ranked:
                m = d.metadata
                if m["type"] == "video":
                    link = f"https://youtu.be/{m['video_id']}?t={int(m['start'])}"
                    head = f"**Video [{m['label']}]({link})**"
                else:
                    head = f"**{m['label']}** ({m['type']})"
                st.markdown(f"{head} · similarity `{score:.2f}`")
                st.write(d.page_content)


# ---------- Previous study (right column, under the Detailed Analysis button) ----------
with top_right:
    history = st.session_state["history"]
    st.markdown("#### 📖 Previous study")
    if not PERSIST:
        st.caption(
            "On this public site your history is kept only while this page is "
            "open. Download it below to keep it."
        )
    if not history:
        st.caption("Nothing yet. Build a study assistant and it will appear here.")
    for rec in reversed(history[-30:]):
        with st.expander(f"{rec['date'][:10]} · {rec['topic']}"):
            st.caption(
                f"🎬 {rec['videos']} video · 📄 {rec['documents']} docs · "
                f"💻 {rec['code']} code · 🧩 {rec['chunks']} chunks"
            )
            st.markdown("**Material**")
            for src in rec["sources"]:
                st.markdown("- " + describe_source(src))
            st.markdown(f"**Questions asked ({len(rec['questions'])})**")
            for item in rec["questions"]:
                st.markdown(f"- {item['q']}")
            if rec["questions"] and st.checkbox("Show answers", key=f"show_{rec['id']}"):
                for item in rec["questions"]:
                    st.markdown(f"**{item['q']}**")
                    st.markdown(item["answer"])
            if material_exists(rec["id"]) and st.button("Open again", key=f"open_{rec['id']}"):
                loaded = load_material(rec["id"])
                if loaded:
                    docs_l, raw_l = loaded
                    store_l = InMemoryVectorStore(get_embeddings())
                    store_l.add_documents(docs_l)
                    st.session_state.update(
                        store=store_l,
                        docs=docs_l,
                        raw=raw_l,
                        stats={k: rec[k] for k in ("videos", "documents", "code", "pages")},
                        summary=rec["summary"],
                        settings=rec["settings"],
                        current_id=rec["id"],
                    )
                    st.rerun()
            if st.button("Delete this study", key=f"del_{rec['id']}"):
                delete_record(history, rec["id"])
                save_history(history)
                st.rerun()

    with st.expander("Save / restore history"):
        st.download_button(
            "⬇ Download history (JSON)",
            json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="study_history.json",
            mime="application/json",
        )
        uploaded = st.file_uploader("Restore a history file", type=["json"], key="hist_up")
        if uploaded is not None and st.button("Restore", key="hist_restore"):
            try:
                restored = json.loads(uploaded.getvalue().decode("utf-8"))
                known = {r["id"] for r in history}
                history.extend(r for r in restored if r.get("id") not in known)
                save_history(history)
                st.rerun()
            except Exception as e:
                st.error(f"Could not read that file: {e}")
