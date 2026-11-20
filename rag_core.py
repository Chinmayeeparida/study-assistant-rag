"""Shared helpers for the Study Assistant (loading, chunking, retrieval, answers)."""
import io
import re
import time

import numpy as np
import streamlit as st
import trafilatura
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

from video_chunks import fmt, get_chunks

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_MAX_TOKENS = 128  # this embedding model reads only ~128 tokens per chunk
LLM_MODEL = "openai/gpt-oss-20b"
CODE_EXT = {"py", "sql", "r", "js"}
DOC_EXT = {"pdf", "docx", "txt", "md", "csv"}
STRATEGIES = [
    "Fixed-size",
    "Recursive",
    "Sentence-based",
    "Semantic",
    "Hierarchical (parent-child)",
]
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_COMPARE_CHARS = 60000


# ---------------------------------------------------------------- models
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def _build_llm(api_key):
    kwargs = {"api_key": api_key} if api_key else {}
    return ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=3000, **kwargs)


def get_llm():
    """Uses the visitor's own Groq key if they pasted one, else the app's key."""
    return _build_llm(st.session_state.get("groq_key_saved", ""))


@st.cache_resource
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBED_MODEL)


# ---------------------------------------------------------------- chunking
def extract_video_id(text):
    text = text.strip()
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", text)
    if m:
        return m.group(1)
    return text if re.fullmatch(r"[A-Za-z0-9_-]{11}", text) else None


def make_splitter(ext, size):
    overlap = size // 10
    if ext == "py":
        return RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON, chunk_size=size, chunk_overlap=overlap
        )
    if ext == "js":
        return RecursiveCharacterTextSplitter.from_language(
            Language.JS, chunk_size=size, chunk_overlap=overlap
        )
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def split_sentences(text, size):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current)
    return chunks


def split_semantic(text, size, embeddings, percentile=90, max_sentences=1200):
    """Cut where the meaning changes: embed every sentence and split at the
    biggest drops in similarity between neighbouring sentences."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    sentences = sentences[:max_sentences]
    if len(sentences) < 3:
        return [text.strip()] if text.strip() else []
    vecs = np.array(embeddings.embed_documents(sentences), dtype=float)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    distance = 1.0 - (vecs[:-1] * vecs[1:]).sum(axis=1)
    cut = np.percentile(distance, percentile)
    chunks, current = [], sentences[0]
    for i, sentence in enumerate(sentences[1:]):
        too_big = len(current) + len(sentence) + 1 > size * 2
        if (distance[i] >= cut and len(current) >= size // 3) or too_big:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}"
    chunks.append(current)
    return chunks


def chunk_text(text, strategy, size, overlap, embeddings=None):
    """Returns a list of dicts: {"text": what is embedded, "context": what the
    AI is shown}. They differ only for the hierarchical (parent-child) strategy."""
    overlap = max(0, min(overlap, size // 2))
    if strategy == "Fixed-size":
        pieces = CharacterTextSplitter(
            separator="", chunk_size=size, chunk_overlap=overlap
        ).split_text(text)
    elif strategy == "Recursive":
        pieces = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap
        ).split_text(text)
    elif strategy == "Sentence-based":
        pieces = split_sentences(text, size)
    elif strategy == "Semantic":
        pieces = split_semantic(text, size, embeddings)
    else:  # Hierarchical (parent-child)
        parents = RecursiveCharacterTextSplitter(
            chunk_size=size * 3, chunk_overlap=overlap
        ).split_text(text)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap
        )
        items = []
        for parent in parents:
            for child in child_splitter.split_text(parent):
                if child.strip():
                    items.append({"text": child, "context": parent})
        return items
    return [{"text": p, "context": p} for p in pieces if p.strip()]


def doc(text, kind, source, label, **extra):
    return Document(
        page_content=text,
        metadata={"type": kind, "source": source, "label": label, **extra},
    )


# ---------------------------------------------------------------- loaders
def load_video(video_id, size):
    """Returns (documents, raw_text)."""
    chunks = get_chunks(video_id, chunk_size=size)
    docs = [
        doc(
            c["text"],
            "video",
            f"YouTube {video_id}",
            fmt(c["start"]),
            start=c["start"],
            video_id=video_id,
        )
        for c in chunks
    ]
    raw = " ".join(c["text"] for c in chunks)
    return docs, raw


def load_web(page_url, size):
    """Returns (documents, raw_text)."""
    html = trafilatura.fetch_url(page_url)
    text = trafilatura.extract(html) if html else None
    if not text:
        raise ValueError(f"Could not read any article text from {page_url}")
    host = re.sub(r"^https?://(www\.)?", "", page_url).split("/")[0]
    splitter = make_splitter("txt", size)
    docs = [
        doc(p, "web", page_url, f"{host} #{n}")
        for n, p in enumerate(splitter.split_text(text), start=1)
    ]
    return docs, text


def load_pasted(text, size):
    splitter = make_splitter("txt", size)
    docs = [
        doc(p, "document", "Pasted text", f"Pasted text #{n}")
        for n, p in enumerate(splitter.split_text(text), start=1)
    ]
    return docs, text


def load_file(f, size):
    """Returns (documents, kind, pages, raw_text)."""
    name = f.name
    ext = name.rsplit(".", 1)[-1].lower()
    data = f.getvalue()
    splitter = make_splitter(ext, size)
    out, pages = [], 0
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = len(reader.pages)
        page_texts = []
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_texts.append(page_text)
            for piece in splitter.split_text(page_text):
                out.append(doc(piece, "document", name, f"{name} p.{i}", page=i))
        return out, "document", pages, "\n\n".join(page_texts)
    if ext == "docx":
        text = "\n".join(p.text for p in DocxDocument(io.BytesIO(data)).paragraphs)
    else:
        text = data.decode("utf-8", errors="ignore")
    kind = "code" if ext in CODE_EXT else "document"
    for n, piece in enumerate(splitter.split_text(text), start=1):
        out.append(doc(piece, kind, name, f"{name} #{n}"))
    return out, kind, pages, text


# ---------------------------------------------------------------- retrieval
def balance(ranked, k=5):
    """Keeps the 2 best chunks from EVERY source, so no source crowds out others."""
    per_source, chosen = {}, []
    for d, score in ranked:
        src = d.metadata["source"]
        if per_source.get(src, 0) < 2:
            per_source[src] = per_source.get(src, 0) + 1
            chosen.append((d, score))
    chosen.sort(key=lambda item: item[1], reverse=True)
    return chosen[: max(k, min(len(chosen), 10))]


def retrieve(store, n_docs, question, k=5, balanced=True):
    """Returns a list of (Document, similarity score), best first."""
    ranked = store.similarity_search_with_score(question, k=n_docs)
    return balance(ranked, k) if balanced else ranked[:k]


def build_context(ranked):
    return "\n\n".join(
        f"[{d.metadata['label']}] ({d.metadata['type']}) {d.page_content}"
        for d, _ in ranked
    )


STYLES = ["Step-by-step tutorial", "Explain like I'm 10", "Exam revision notes"]

BASE_RULES = (
    "You are a patient teacher helping a student learn from THEIR OWN study "
    "material. Use ONLY the excerpts provided. Each excerpt starts with a label in "
    "square brackets and its type (video, document, web or code). Never add facts "
    "that are not in the excerpts; if a section cannot be filled from them, leave "
    "that section out. After each fact taken from an excerpt, cite its label in "
    "square brackets, for example [3:25]. Use short sentences and everyday words, "
    "and explain every technical word the first time you use it. Format the reply "
    "in Markdown with the exact headings given below, in the order given. "
    "If the excerpts do not contain the answer at all, say exactly: I couldn't "
    "find that in the material.\n\n"
)

DOT_RULES = (
    "Right after that section, add the diagram: a line containing only [[DOT]], "
    "then Graphviz DOT code and nothing else (no code fences), then a line "
    "containing only [[/DOT]]. The DOT code must start with: digraph G. Set "
    "rankdir=LR. Use rounded box nodes, at most 8 nodes, labels of at most 4 words "
    "in double quotes, and arrows that show how the idea flows.\n"
)


def style_instructions(style, diagram):
    dot = DOT_RULES if diagram else ""
    if style == "Explain like I'm 10":
        return (
            "Explain it so a 10-year-old could follow, with these headings:\n"
            "## 🧒 The idea\n(2 to 3 very short sentences.)\n" + dot +
            "## 🎈 Think of it like this\n(One friendly real-life analogy in 3 to "
            "4 sentences. If there is a diagram, tie the analogy to its boxes.)\n"
            "## ➕ Add-on information\n(2 or 3 short extra facts from the "
            "excerpts.)\n"
            "## 📚 Where this comes from\n(One bullet per source that helped, "
            "naming it and citing its label.)\n"
            "## 📝 Check yourself\n(3 easy questions, each followed by a one-line "
            "answer.)"
        )
    if style == "Exam revision notes":
        return (
            "Write quick revision notes with these headings:\n"
            "## 📌 Definition\n(1 to 2 lines.)\n" + dot +
            "## 🔑 Key points\n(5 to 8 bullets, each one line. If there is a "
            "diagram, make them follow its flow.)\n"
            "## ➕ Add-on information\n(2 to 4 extra bullets from the excerpts: "
            "why it matters, limits, related ideas.)\n"
            "## 📖 Important terms\n(Bullets written as: **term**: meaning.)\n"
            "## 📚 Sources\n(One bullet per source that helped, citing its "
            "label.)\n"
            "## 📝 Self-test\n(4 short questions, each followed by a one-line "
            "answer.)"
        )
    explained = (
        "## 🔍 The diagram explained, with an example\n(Walk through the diagram "
        "box by box and arrow by arrow as numbered steps, using ONE small concrete "
        "example taken from the excerpts.)\n"
        if diagram
        else "## 🔍 Step by step, with an example\n(Explain the idea as numbered "
        "steps, using ONE small concrete example taken from the excerpts.)\n"
    )
    return (
        "Write a tutorial with exactly these headings, in this order:\n"
        "## 🧒 The idea\n(2 to 3 sentences in plain words plus one real-life "
        "analogy, such as a kitchen, a school or a post office.)\n" + dot +
        explained +
        "## ➕ Add-on information\n(2 to 4 bullets of extra useful details from "
        "the excerpts that go beyond the basic idea: why it matters, limits, "
        "related ideas.)\n"
        "## 📖 Words to know\n(3 to 6 bullets written as: **term**: simple "
        "meaning.)\n"
        "## 📚 What your sources say\n(One bullet for each source that helped: "
        "the video, each document, each blog or code file. Name the source, say "
        "what it adds, and cite its label. Say if sources agree or differ.)\n"
        "## 📝 Check yourself with more questions\n(4 questions that go from easy "
        "to harder, each followed by a one-line answer.)"
    )


def answer_prompt(style, diagram=False):
    return ChatPromptTemplate.from_messages(
        [
            ("system", BASE_RULES + style_instructions(style, diagram)),
            ("human", "Excerpts:\n{context}\n\nQuestion: {question}"),
        ]
    )


DOT_RE = re.compile(r"\[\[DOT\]\](.*?)\[\[/DOT\]\]", re.DOTALL)


def split_answer(reply):
    """Splits a reply into (text before diagram, DOT code or None, text after)."""
    m = DOT_RE.search(reply)
    if not m:
        cleaned = reply.replace("[[DOT]]", "").replace("[[/DOT]]", "")
        return cleaned.strip(), None, ""
    before, after = reply[: m.start()], reply[m.end():]
    dot = re.sub(r"^```[a-zA-Z]*\s*", "", m.group(1).strip())
    dot = re.sub(r"\s*```$", "", dot).strip()
    if not re.match(r"^(strict\s+)?(di)?graph\b", dot):
        return (before + "\n\n" + after).strip(), None, ""
    return before.strip(), dot, after.strip()


DIAGRAM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You draw simple concept diagrams as Graphviz DOT code. Reply with DOT "
            "code only: no explanation and no code fences. Begin with: digraph G. "
            "Set rankdir=LR. Use rounded box nodes, at most 8 nodes, labels of at "
            "most 4 words in double quotes, and arrows that show how the idea flows.",
        ),
        ("human", "Question: {question}\n\nExplanation to draw:\n{answer}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize what this study material covers in 5 short bullet points.",
        ),
        ("human", "{text}"),
    ]
)


def answer(question, ranked, style="Step-by-step tutorial", diagram=False):
    chain = answer_prompt(style, diagram) | get_llm() | StrOutputParser()
    return chain.invoke({"context": build_context(ranked), "question": question})


def diagram_dot(question, answer_text):
    chain = DIAGRAM_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"question": question, "answer": answer_text}).strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    return raw if re.match(r"^(strict\s+)?(di)?graph\b", raw) else None


def summarize(docs):
    step = max(1, len(docs) // 10)
    sample = docs[::step][:10]
    text = "\n\n".join(d.page_content[:800] for d in sample)
    chain = SUMMARY_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"text": text})


# ---------------------------------------------------------------- analysis helpers
@st.cache_resource
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def rerank(question, ranked, keep):
    """Re-scores candidates with a cross-encoder that reads question and chunk
    together. Returns (document, rerank score, original vector score)."""
    model = get_reranker()
    scores = model.predict([(question, d.page_content) for d, _ in ranked])
    order = sorted(zip(ranked, scores), key=lambda x: float(x[1]), reverse=True)
    return [(d, float(sc), float(vs)) for (d, vs), sc in order[:keep]]


def token_lengths(texts):
    texts = list(texts)
    try:
        tok = get_tokenizer()
        ids = tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
        return [len(x) for x in ids]
    except Exception:
        return [max(1, len(t) // 4) for t in texts]


def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def overflow_check(question, chunk, embeddings):
    """BEFORE: embed the whole chunk (the model silently ignores tokens past its
    limit). AFTER: cut the chunk into windows that fit and score each one."""
    tok = get_tokenizer()
    tokens = tok.tokenize(chunk)
    limit = EMBED_MAX_TOKENS - 2
    windows = [
        tok.convert_tokens_to_string(tokens[i : i + limit])
        for i in range(0, max(len(tokens), 1), limit)
    ]
    qvec = embeddings.embed_query(question)
    before = cosine(qvec, embeddings.embed_documents([chunk])[0])
    after = [cosine(qvec, v) for v in embeddings.embed_documents(windows)]
    return before, windows, after
