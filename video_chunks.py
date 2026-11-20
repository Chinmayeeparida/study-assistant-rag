from youtube_transcript_api import YouTubeTranscriptApi


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def get_chunks(video_id, chunk_size=900, languages=("en",)):
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=list(languages))
    chunks = []
    text = ""
    start = None
    for snippet in transcript:
        line = " ".join(snippet.text.split())
        if start is None:
            start = snippet.start
        if len(text) + len(line) > chunk_size and text:
            chunks.append({"start": start, "text": text.strip()})
            text, start = "", snippet.start
        text += " " + line
    if text.strip():
        chunks.append({"start": start, "text": text.strip()})
    return chunks
