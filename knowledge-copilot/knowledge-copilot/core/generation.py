"""
generation.py
The "grounded generation" layer.

Default mode: extractive - stitches together the most relevant sentences
from retrieved chunks into a readable answer. This needs no API key and no
network access, so the prototype runs fully offline out of the box.

Optional mode: if an ANTHROPIC_API_KEY environment variable is set, answers
are synthesised by Claude, strictly instructed to only use the retrieved
context and to say so when the context doesn't cover the question. This is
the upgrade path for a production deployment.
"""
import os
import re

MAX_CONTEXT_CHUNKS_FOR_LLM = 6


def _clean_sentences(text, limit=2):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences[:limit]


def extractive_answer(query, results):
    if not results:
        return (
            "I couldn't find anything in the indexed documents that answers "
            "this. Try rephrasing, or check whether the relevant document has "
            "been ingested yet."
        )

    lines = []
    for r in results[:4]:
        snippet_sentences = _clean_sentences(r["text"], limit=2)
        snippet = " ".join(snippet_sentences) if snippet_sentences else r["text"][:220]
        lines.append(f"- {snippet} [{r['doc_name']}]")

    return (
        "Based on the indexed documents, here's what's most relevant:\n\n"
        + "\n".join(lines)
    )


def llm_answer(query, results):
    """Calls the Anthropic API to synthesise a grounded answer. Only used if
    ANTHROPIC_API_KEY is set - falls back to extractive_answer otherwise."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return extractive_answer(query, results)

    try:
        import anthropic
    except ImportError:
        return extractive_answer(query, results)

    context_blocks = []
    for r in results[:MAX_CONTEXT_CHUNKS_FOR_LLM]:
        context_blocks.append(f"[Source: {r['doc_name']}]\n{r['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are an industrial knowledge copilot for plant operators, "
        "maintenance engineers, and field technicians. Answer the question "
        "using ONLY the provided source excerpts. Every claim must be "
        "traceable to a source. Cite sources inline like [document_name]. "
        "If the excerpts don't contain enough information to answer "
        "confidently, say so explicitly rather than guessing."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Source excerpts:\n\n{context}\n\nQuestion: {query}",
                }
            ],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
    except Exception as e:
        return extractive_answer(query, results) + f"\n\n(LLM call failed: {e})"


def generate_answer(query, results):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return llm_answer(query, results)
    return extractive_answer(query, results)
