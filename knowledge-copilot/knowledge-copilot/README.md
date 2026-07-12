# Industrial Knowledge Copilot

A working prototype of the **Expert Knowledge Copilot** track: a RAG-powered
conversational assistant that answers operational, maintenance, and
engineering questions across a heterogeneous document corpus — with source
citations, a confidence score, and a mobile-friendly interface for field
technicians.

Runs fully offline out of the box (no API key required). Includes an
optional upgrade path to Claude for higher-quality grounded answers.

## Why this exists

Industrial plants spread their operational knowledge across P&IDs,
maintenance work orders, SOPs, inspection reports, and compliance
procedures — usually in 7-12 disconnected systems. This prototype ingests
all of those formats into one searchable index, so a question like *"why
does pump P-101A keep failing?"* pulls the answer from a work order, an
inspection report, and an SOP simultaneously, and shows exactly which
document each part of the answer came from.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Tesseract OCR must be installed separately (for scanned docs/photos):
#   macOS:   brew install tesseract
#   Ubuntu:  sudo apt-get install tesseract-ocr
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki

python app.py
```

Open **http://localhost:5000** in a browser. To try the mobile field-tech
view, open the same address on a phone connected to the same Wi-Fi network,
using your computer's local IP instead of `localhost`.

### Try these sample questions
The included sample documents (`data/sample_docs/`) tell one connected story
— a pump that failed, why, and what the fix was — so cross-document answers
are easy to demo:

- "Why did pump P-101A fail?"
- "What's the alignment tolerance for the feed pump coupling?"
- "What compliance requirements apply to maintenance on rotating equipment?"
- "What did the last inspection report say about P-101A?"
- "Has the SOP been updated with the second technician sign-off requirement?"

That last question is deliberately unresolved in the sample docs (SOP-22
Section 6.2 says the update is "pending Process Engineering approval") — a
good example of the copilot correctly reporting a **medium/low confidence**,
partially-answered state instead of fabricating a clean answer.

## Architecture

```
Document sources (PDF, DOCX, CSV, scanned images/photos, .txt)
        │
        ▼
core/loaders.py        — normalises every format to plain text (OCR for scans/photos)
        │
        ▼
core/chunking.py        — paragraph-aware chunking + equipment-tag entity extraction
        │
        ▼
core/index.py           — TF-IDF vector index + tag co-reference graph (knowledge store)
        │
        ▼
core/index.py search()  — hybrid lexical + entity-boosted retrieval, confidence scoring
        │
        ▼
core/generation.py      — extractive answer by default, or grounded LLM answer if
                           ANTHROPIC_API_KEY is set
        │
        ▼
app.py (Flask API) ─── templates/index.html (mobile-responsive chat UI)
```

This mirrors the full target architecture from the concept deck: Universal
Ingestion → Knowledge Store → Retrieval/Orchestration → Grounded Generation →
Field & Desktop Interfaces. The prototype swaps in lightweight, dependency-
free components at each layer so it runs anywhere without cloud services,
while keeping the same interfaces so each layer can be upgraded independently.

## Where the differentiators live in the code

| Brief requirement | Where it's implemented |
|---|---|
| Multi-format ingestion | `core/loaders.py` — PDF, DOCX, CSV, TXT, and OCR for scanned images/photos |
| Source citations | Every API response includes `sources[]` with document name and excerpt |
| Confidence scores | `core/index.py: compute_confidence()` — based on match strength + corroboration across distinct documents |
| Mobile-first for field technicians | `templates/index.html` is a single responsive layout; works unmodified on a phone browser, includes a camera/photo upload button |
| Knowledge graph seed | `core/index.py: entity_to_chunks` + `/api/related/<tag>` — "what else references this equipment?" |
| Capturing tacit knowledge before the "knowledge cliff" | `core/feedback.py` — expert corrections on wrong answers are stored and automatically resurfaced next time the same question is asked |

## Scaling this prototype

- **Retrieval**: swap `TfidfVectorizer` in `core/index.py` for a real
  embedding model (e.g. `sentence-transformers`, or a hosted embeddings API)
  and a vector DB (Qdrant, pgvector, Pinecone) — the `search()` interface
  doesn't need to change.
- **Knowledge graph**: promote `entity_to_chunks` into a real graph database
  (Neo4j) once relationships (equipment → work orders → procedures →
  regulations) need multi-hop queries.
- **Generation**: set `ANTHROPIC_API_KEY` to switch from extractive answers
  to full grounded LLM synthesis (`core/generation.py: llm_answer`).
- **Entity extraction**: replace the regex tagger in `core/chunking.py` with
  a trained NER model for messier real-world documents.

## Project structure

```
knowledge-copilot/
├── app.py                    # Flask API + server
├── core/
│   ├── loaders.py             # multi-format document loading + OCR
│   ├── chunking.py            # chunking + entity extraction
│   ├── index.py               # TF-IDF retrieval + confidence scoring
│   ├── generation.py          # extractive / LLM answer generation
│   └── feedback.py            # expert correction capture
├── templates/index.html       # mobile-responsive chat UI
├── data/sample_docs/          # 5 synthetic industrial documents
├── requirements.txt
└── README.md
```

## Pushing this to GitHub

```bash
cd knowledge-copilot
git init
git add .
git commit -m "Industrial Knowledge Copilot prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

(Create the empty repository on github.com first — click **New repository**,
give it a name, don't initialise it with a README, then use the URL it gives
you in the `git remote add` command above.)
