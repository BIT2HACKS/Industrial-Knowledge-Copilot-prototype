"""
index.py
The knowledge store + retrieval/orchestration layer.

For a hackathon prototype this uses TF-IDF cosine similarity (fast, no GPU,
no external API needed to stand the demo up). The retrieval interface is
deliberately kept generic (`search(query, top_k)` returning ranked chunk
dicts) so it's a drop-in swap to a real embedding model (e.g. sentence-
transformers or a hosted embedding API) for a production version - see
README "Scaling this prototype" section.

It also builds a tiny in-memory entity graph: which chunks/documents share
an equipment tag or document reference (e.g. WO-4521 mentioned inside
IR-2025-089). This is the seed of the "knowledge graph" layer in the full
architecture - enough to answer "what else references this equipment?"
without needing a graph database for the prototype.
"""
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeIndex:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.matrix = None
        self.entity_to_chunks = defaultdict(set)

    def build(self, chunk_records):
        self.chunks = chunk_records
        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_df=0.9
        )
        self.matrix = self.vectorizer.fit_transform(texts)

        self.entity_to_chunks = defaultdict(set)
        for i, chunk in enumerate(self.chunks):
            for entity in chunk["entities"]:
                self.entity_to_chunks[entity].add(i)

    def is_ready(self):
        return self.matrix is not None and len(self.chunks) > 0

    def search(self, query, top_k=5):
        if not self.is_ready():
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()

        # Boost chunks whose entities directly match tags mentioned in the
        # query (e.g. user asks about "P-101A" - exact tag match is a
        # stronger signal than pure lexical overlap).
        query_upper = query.upper()
        for i, chunk in enumerate(self.chunks):
            for entity in chunk["entities"]:
                if entity in query_upper:
                    sims[i] += 0.15

        ranked_idx = sims.argsort()[::-1][:top_k]
        results = []
        for i in ranked_idx:
            if sims[i] <= 0:
                continue
            results.append({**self.chunks[i], "score": float(sims[i])})
        return results

    def related_chunks(self, entity, exclude_chunk_id=None, limit=5):
        """Find other chunks that mention the same equipment tag / doc ref -
        powers the 'what else references this?' knowledge-graph style query."""
        idxs = self.entity_to_chunks.get(entity.upper(), set())
        related = [
            self.chunks[i]
            for i in idxs
            if self.chunks[i]["chunk_id"] != exclude_chunk_id
        ]
        return related[:limit]


def compute_confidence(results):
    """Confidence heuristic for the prototype:
    - top similarity score (how well the best chunk matches)
    - corroboration: how many *distinct documents* appear in the top results
      (an answer backed by 3 independent documents is more trustworthy than
      one backed by 3 chunks of the same document)
    This is intentionally simple and explainable - swap for a calibrated
    model once you have labeled query/answer pairs.
    """
    if not results:
        return {"level": "none", "score": 0.0, "reason": "No matching content found."}

    top_score = results[0]["score"]
    distinct_docs = len({r["doc_name"] for r in results})

    # normalise: TF-IDF cosine scores rarely exceed ~0.6 in practice
    normalised = min(top_score / 0.5, 1.0)
    corroboration_bonus = min((distinct_docs - 1) * 0.15, 0.3)
    final = min(normalised * 0.7 + corroboration_bonus + 0.3, 1.0)

    if final >= 0.75:
        level = "high"
    elif final >= 0.5:
        level = "medium"
    else:
        level = "low"

    reason = (
        f"Best match score {top_score:.2f}, corroborated by {distinct_docs} "
        f"distinct source document(s)."
    )
    return {"level": level, "score": round(final, 2), "reason": reason}
