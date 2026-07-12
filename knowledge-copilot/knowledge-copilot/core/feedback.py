"""
feedback.py
Captures thumbs up/down and free-text corrections on answers. This is the
prototype's answer to the "knowledge cliff" problem described in the
challenge brief: when an experienced engineer corrects or annotates an
answer, that correction is stored and surfaced for future identical/similar
queries - capturing tacit knowledge before it's lost to retirement or
attrition, instead of relying on documents alone.

Stored as a simple JSON file for the prototype; swap for a real database
in production.
"""
import json
import os
import time

FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feedback.json")


def _load():
    if not os.path.exists(FEEDBACK_PATH):
        return []
    with open(FEEDBACK_PATH, "r") as f:
        return json.load(f)


def _save(records):
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(records, f, indent=2)


def record_feedback(query, answer, rating, correction=None):
    records = _load()
    records.append(
        {
            "query": query,
            "answer": answer,
            "rating": rating,  # "up" or "down"
            "correction": correction,
            "timestamp": time.time(),
        }
    )
    _save(records)


def find_correction_for_query(query):
    """If a prior correction exists for a very similar query, surface it -
    a stand-in for a proper semantic match against the feedback log."""
    records = _load()
    query_lower = query.lower().strip()
    for r in reversed(records):
        if r.get("correction") and r["query"].lower().strip() == query_lower:
            return r["correction"]
    return None


def all_feedback():
    return _load()
