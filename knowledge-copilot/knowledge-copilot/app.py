"""
app.py
Expert Knowledge Copilot - Flask backend.

Run with:
    python app.py
Then open http://localhost:5000 in a browser (or on a phone on the same
network at http://<your-ip>:5000 to try the mobile field-technician view).

On startup this ingests every document in data/sample_docs/ (and anything
in data/uploads/), builds the TF-IDF knowledge index, and serves a chat API
plus a simple web UI.
"""
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from core.loaders import discover_documents, load_document, SUPPORTED_EXTENSIONS
from core.chunking import build_chunks_for_document
from core.index import KnowledgeIndex, compute_confidence
from core.generation import generate_answer
from core import feedback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS_DIR = os.path.join(BASE_DIR, "data", "sample_docs")
UPLOADS_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__)
knowledge_index = KnowledgeIndex()
ingested_doc_names = []


def build_index():
    """(Re)build the knowledge index from every document currently sitting
    in data/sample_docs/ and data/uploads/. This is the pipeline the
    architecture diagram calls 'Ingestion & extraction' -> 'Chunking &
    embedding' -> 'Knowledge store', run end to end."""
    global ingested_doc_names
    all_chunks = []
    doc_paths = discover_documents(SAMPLE_DOCS_DIR) + discover_documents(UPLOADS_DIR)

    for path in doc_paths:
        try:
            raw_text = load_document(path)
            all_chunks.extend(build_chunks_for_document(path, raw_text))
        except Exception as e:
            print(f"[ingest] failed on {path}: {e}")

    knowledge_index.build(all_chunks)
    ingested_doc_names = sorted({c["doc_name"] for c in all_chunks})
    print(f"[ingest] indexed {len(all_chunks)} chunks from {len(ingested_doc_names)} documents")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/documents", methods=["GET"])
def list_documents():
    return jsonify({"documents": ingested_doc_names, "chunk_count": len(knowledge_index.chunks)})


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    # Check the feedback/correction log first - captures corrections made by
    # experienced staff, the "knowledge cliff" mitigation loop.
    correction = feedback.find_correction_for_query(question)

    results = knowledge_index.search(question, top_k=6)
    confidence = compute_confidence(results)
    answer = generate_answer(question, results)

    sources = [
        {
            "doc_name": r["doc_name"],
            "chunk_id": r["chunk_id"],
            "score": round(r["score"], 3),
            "excerpt": r["text"][:300],
            "entities": r["entities"],
        }
        for r in results
    ]

    return jsonify(
        {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "expert_correction": correction,
        }
    )


@app.route("/api/related/<entity>", methods=["GET"])
def related(entity):
    """Knowledge-graph style lookup: everything else that mentions this
    equipment tag or document reference."""
    chunks = knowledge_index.related_chunks(entity)
    return jsonify(
        {
            "entity": entity,
            "related": [
                {"doc_name": c["doc_name"], "excerpt": c["text"][:200]} for c in chunks
            ],
        }
    )


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json(force=True)
    feedback.record_feedback(
        query=data.get("question", ""),
        answer=data.get("answer", ""),
        rating=data.get("rating", ""),
        correction=data.get("correction"),
    )
    return jsonify({"status": "ok"})


@app.route("/api/upload", methods=["POST"])
def upload():
    """Lets a field technician upload a new document (e.g. a phone photo of
    a paper checklist) straight from the mobile UI. Re-indexes afterwards."""
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"unsupported file type {ext}"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOADS_DIR, filename)
    file.save(save_path)
    build_index()
    return jsonify({"status": "ingested", "filename": filename, "chunk_count": len(knowledge_index.chunks)})


build_index()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
