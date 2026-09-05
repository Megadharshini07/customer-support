TRACK_ID=PS04

# Ticket Resolution Assistant — Customer Support track

> ⚠️ **Before you submit:** replace `TRACK_ID=PS04` on the very first line above
> with the exact track ID from your actual Customer Support problem statement.
> The orientation deck only published the full brief for Healthcare (`PS01`);
> the Customer Support ID wasn't shown, so `PS04` here is a placeholder based
> on track ordering, not a confirmed value. Get the real ID from your Devfolio
> problem-statement page and swap it in — a wrong ID means judges compare your
> project to the wrong problem.

## What it does

Takes a customer support ticket in plain language and produces a **resolution
draft**: the recommended next step, the specific policy clause(s) it's based
on, what the customer explicitly reported vs. what's still unknown, and
whether the ticket should be escalated to a human team instead of resolved
directly.

It never invents a policy. If nothing in the knowledge base covers the
ticket, it says so and routes to a human rather than guessing (the null
case).

## How it works

1. **Knowledge base** (`data/*.md`) — 10 synthetic support policy documents
   (refunds, shipping delays, billing disputes, account access,
   cancellations, warranty claims, an escalation matrix, data privacy,
   promo codes, address changes). Includes edge cases (ambiguous warranty
   claims), explicit escalation triggers, and deliberately has no policy
   for some out-of-scope topics, to exercise the null case.
2. **Indexing** (`rag.py` / `build_index.py`) — chunks each doc (~800 chars,
   150 overlap), embeds every chunk with Gemini (`gemini-embedding-001`),
   stores vectors as a flat numpy array on disk.
3. **Retrieval** — embeds the incoming ticket with the same model, does
   cosine similarity against the stored vectors, returns the top 4 chunks.
   If the best match is below a similarity threshold, we skip generation
   entirely and return a "no matching policy, escalate" response.
4. **Generation** (`app.py`) — the retrieved chunks + the ticket are sent to
   `gemini-3.5-flash-lite` (the evaluation model) with instructions to cite
   policy IDs, separate reported facts from unknowns, and recommend
   escalation rather than guessing when unsure. A keyword safety net also
   force-escalates on hard triggers (compromise, chargeback, legal threats,
   data deletion, account merge) even if the model misses them.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here

# One-time: build the vector index and commit the result so app startup
# doesn't have to re-embed the whole KB inside the 90s startup window.
python build_index.py

python app.py   # -> http://localhost:8000
```

Open `http://localhost:8000` for the demo UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/api/resolve \
  -H "Content-Type: application/json" \
  -d '{"ticket": "My order was supposed to arrive 12 days ago and tracking hasn'"'"'t moved. Where is it?"}'
```

## Data you generated

The 10 policy documents in `data/` were authored for this project (no real
company policy). They intentionally include:
- Time-based branching rules (e.g. refund policy changes at 30/90 days)
- An explicit escalation matrix that overrides otherwise-resolvable cases
- An ambiguous case requiring human judgement (warranty: defect vs. damage)
- Topics with **no** matching policy at all, to test the null case (e.g.
  international in-store availability questions)

## Demo video

`<add your 5-minute Devfolio demo video link here before submitting>`
