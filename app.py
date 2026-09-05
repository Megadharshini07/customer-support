"""
NexusTiQ 24 — Customer Support track
Ticket Resolution Assistant

python app.py -> http://localhost:8000

Single command, starts backend + frontend together (Flask serves the
static/index.html UI and the JSON API from the same process).
"""

import os
import json
import re
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory

from rag import Retriever

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GENERATION_MODEL = os.environ.get("GEMINI_GEN_MODEL", "gemini-3.5-flash-lite")
TOP_K = 4
# Below this cosine-similarity score, we treat the ticket as "not covered by
# any policy" (the null case) rather than letting the model guess.
NULL_CASE_THRESHOLD = 0.55

ESCALATION_KEYWORDS = [
    "unauthorized", "hacked", "compromised", "someone else logged in",
    "chargeback", "lawyer", "sue", "legal action", "regulator", "complaint to",
    "delete all my data", "delete my personal data", "merge my accounts",
    "transfer my order history",
]

app = Flask(__name__, static_folder="static", static_url_path="")

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are a customer support resolution-drafting assistant.

You will be given a customer ticket and a set of retrieved policy excerpts.

Rules you must follow exactly:
1. Base your resolution ONLY on the retrieved policy excerpts provided below.
   Never invent a policy, number, or timeframe that is not in the excerpts.
2. Every recommendation you make must cite the specific policy id and point
   number it comes from (e.g. "POL-RR-01, point 3").
3. If the excerpts do not clearly cover the customer's situation, or if the
   situation matches an escalation trigger, do NOT resolve it yourself.
   Instead recommend escalation and say which team, if the excerpts specify
   one.
4. Separate what the customer explicitly reported from anything you are
   inferring or that remains unknown/unconfirmed.
5. Never diagnose intent, assume facts not stated by the customer, or promise
   an outcome the policy excerpts don't support.

Respond with ONLY a JSON object (no markdown fences, no preamble) with this
exact shape:
{
  "customer_reported": "<one or two sentences, in the customer's own situation, no inference>",
  "resolution_draft": "<the resolution or next step, written for the customer>",
  "policy_citations": ["<policy id + point, e.g. POL-RR-01 point 3>", ...],
  "escalate": true|false,
  "escalation_target": "<team name, or null if escalate is false>",
  "unknowns": "<what remains unconfirmed/unknown, or empty string>",
  "confidence": "high"|"medium"|"low"
}
"""


def _extract_json(raw_text):
    """Model output should already be pure JSON, but strip code fences
    defensively in case the model adds them anyway."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def draft_resolution(ticket_text, retrieved_chunks):
    context_block = "\n\n".join(
        f"[{c['doc']} | {c['chunk_id']} | similarity={c['score']:.2f}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"RETRIEVED POLICY EXCERPTS:\n{context_block}\n\n"
        f"CUSTOMER TICKET:\n{ticket_text}\n\n"
        f"JSON response:"
    )

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2},
    )
    return _extract_json(response.text)


def keyword_escalation_hit(ticket_text):
    lowered = ticket_text.lower()
    return any(k in lowered for k in ESCALATION_KEYWORDS)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/resolve", methods=["POST"])
def resolve():
    body = request.get_json(silent=True) or {}
    ticket_text = (body.get("ticket") or "").strip()

    if not ticket_text:
        return jsonify({"error": "Field 'ticket' is required and cannot be empty."}), 400

    retriever = get_retriever()
    chunks = retriever.retrieve(ticket_text, top_k=TOP_K)
    top_score = chunks[0]["score"] if chunks else 0.0

    # Null case: nothing in the KB is actually relevant to this ticket.
    if top_score < NULL_CASE_THRESHOLD:
        return jsonify({
            "customer_reported": ticket_text,
            "resolution_draft": (
                "This request doesn't match any existing support policy in our "
                "knowledge base. Rather than guess, this should be routed to a "
                "human Tier 2 agent to review."
            ),
            "policy_citations": [],
            "escalate": True,
            "escalation_target": "Tier 2 support (human)",
            "unknowns": "No policy document covers this situation.",
            "confidence": "low",
            "retrieved_context": chunks,
            "null_case": True,
        })

    try:
        result = draft_resolution(ticket_text, chunks)
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 502

    # Belt-and-suspenders: force escalation on hard trigger keywords even if
    # the model didn't flag it, matching the Escalation Matrix policy.
    if keyword_escalation_hit(ticket_text) and not result.get("escalate"):
        result["escalate"] = True
        result.setdefault("escalation_target", "Tier 2 support (human)")
        result["unknowns"] = (
            (result.get("unknowns") or "")
            + " Escalation trigger keyword detected; routing overridden to escalate."
        ).strip()

    result["retrieved_context"] = chunks
    result["null_case"] = False
    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
