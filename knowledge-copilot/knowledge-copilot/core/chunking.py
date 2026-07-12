"""
chunking.py
Splits normalised document text into retrievable chunks, each tagged with
enough metadata to produce a citation (source file, chunk position) and to
support entity-aware filtering later (e.g. by equipment tag).
"""
import re
import os

CHUNK_TARGET_CHARS = 700
CHUNK_OVERLAP_CHARS = 100

# Equipment tags in these sample docs look like P-101A, V-102, VT-101, C-201,
# SOP-22, IR-2025-089, WO-4521 etc. A simple regex is enough to demonstrate
# entity extraction for the knowledge-graph metadata layer without needing a
# trained NER model for the prototype.
TAG_PATTERN = re.compile(r"\b([A-Z]{1,4}-\d{2,6}[A-Z]?)\b")


def extract_entities(text):
    return sorted(set(TAG_PATTERN.findall(text)))


def split_into_paragraphs(text):
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts if parts else [text]


def chunk_text(text, max_chars=CHUNK_TARGET_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """Greedy paragraph-aware chunking: keep paragraphs whole where possible,
    and only fall back to a hard character split for a single very long
    paragraph. Sliding overlap keeps context from being cut mid-thought."""
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars - overlap):
                    chunks.append(para[i : i + max_chars])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def build_chunks_for_document(path, raw_text):
    doc_name = os.path.basename(path)
    pieces = chunk_text(raw_text)
    records = []
    for idx, piece in enumerate(pieces):
        records.append(
            {
                "doc_name": doc_name,
                "doc_path": path,
                "chunk_id": f"{doc_name}::chunk_{idx}",
                "chunk_index": idx,
                "text": piece,
                "entities": extract_entities(piece),
            }
        )
    return records
