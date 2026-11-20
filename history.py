"""Study history: what you studied, when, and which questions you asked.

On your own computer it is saved to the study_data folder next to app.py.
On a Hugging Face Space (public site) nothing is saved on the server: history
lives only while the page is open, and visitors can download / restore it.
"""
import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

PERSIST = not os.getenv("SPACE_ID")  # SPACE_ID is set on Hugging Face Spaces
BASE = Path(__file__).parent / "study_data"
INDEX = BASE / "history.json"
CODE_EXT = {"py", "sql", "r", "js"}


def load_history():
    if PERSIST and INDEX.exists():
        try:
            return json.loads(INDEX.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(records):
    if PERSIST:
        BASE.mkdir(exist_ok=True)
        INDEX.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def make_topic(summary, fallback="Study session"):
    """Asks the AI for a short topic title (one small call)."""
    try:
        from rag_core import get_llm

        reply = get_llm().invoke(
            "Give a topic title of 3 to 6 words for this study material. "
            "Reply with the title only, no quotes.\n\n" + summary[:1500]
        )
        text = getattr(reply, "content", str(reply)).strip()
        title = text.strip('"').splitlines()[0][:80] if text else ""
        return title or fallback
    except Exception:
        return fallback


def new_record(topic, stats, sources, chunks, settings, summary):
    return {
        "id": uuid.uuid4().hex[:8],
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topic": topic,
        "videos": stats.get("videos", 0),
        "documents": stats.get("documents", 0),
        "code": stats.get("code", 0),
        "pages": stats.get("pages", 0),
        "chunks": chunks,
        "sources": list(sources),
        "summary": summary,
        "settings": settings,
        "questions": [],
    }


def add_question(record, question, style, reply):
    for item in record["questions"]:
        if item["q"] == question and item["style"] == style:
            item["answer"] = reply
            return
    record["questions"].append(
        {
            "q": question,
            "style": style,
            "answer": reply,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    )


def find_record(records, record_id):
    return next((r for r in records if r["id"] == record_id), None)


def save_material(record_id, docs, raw):
    """Local only: keeps the chunks and the full text so a study can be reopened."""
    if not PERSIST:
        return
    folder = BASE / record_id
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
        "raw": raw,
    }
    (folder / "material.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def material_exists(record_id):
    return PERSIST and (BASE / record_id / "material.json").exists()


def load_material(record_id):
    path = BASE / record_id / "material.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in payload["docs"]
    ]
    return docs, payload["raw"]


def delete_record(records, record_id):
    records[:] = [r for r in records if r["id"] != record_id]
    if PERSIST:
        shutil.rmtree(BASE / record_id, ignore_errors=True)


def describe_source(source):
    if source.startswith("YouTube "):
        return f"🎬 [YouTube video](https://youtu.be/{source.split(' ', 1)[1]})"
    if source.startswith("http"):
        host = re.sub(r"^https?://(www\.)?", "", source).split("/")[0]
        return f"🌐 [{host}]({source})"
    if source == "Pasted text":
        return "📝 Pasted text"
    ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
    return f"💻 {source}" if ext in CODE_EXT else f"📄 {source}"
