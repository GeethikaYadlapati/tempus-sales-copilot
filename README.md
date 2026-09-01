# Tempus Sales Copilot — prototype

Ranks a rep's territory by **Impact**, and for each provider drafts an
objection handler and a 30-second pitch grounded in Tempus's published
capabilities.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. No API key needed — the sample notes and all
generated text have verified fallbacks, so the demo works fully offline.

All tooling is free: Streamlit for the app, and Groq's no-card free tier
(open-weight models) for the optional live model calls.

To run extraction and generation live (free, no card):

```bash
export GROQ_API_KEY=...        # console.groq.com/keys
streamlit run app.py
```

Two models are used, for the reason the deck gives: extraction is short,
structured and runs on every note, so it takes the small fast model; handlers
and pitches are low-volume and read aloud, so they take the larger one.

Model names change over time. To see what your key can reach:

```bash
python3 list_models.py
export GROQ_MODEL_FAST=<small model>     # extraction
export GROQ_MODEL_STRONG=<larger model>  # handlers and pitches
```

Hitting a rate limit is not fatal: the sample notes fall back to their verified
extractions, and generated text falls back to reviewed templates.

## Deploy (free)

Push to a public GitHub repo, then at
[share.streamlit.io](https://share.streamlit.io) point a new app at `app.py`.
Add `GROQ_API_KEY` under *Advanced settings → Secrets* for live generation.

## Files

| File | Role |
|---|---|
| `app.py` | UI only — presentation, no logic |
| `scoring.py` | The impact model. Pure functions, no I/O, no model calls |
| `data_layer.py` | File reads + KB retrieval. The seam that would become Postgres + a vector store |
| `llm.py` | The only place a model writes text. Three prompts, validators, reviewed fallbacks, model selection |
| `ingest.py` | The reading half: note extraction, MI parse, KB cascade |
| `data/market_intelligence.csv` | 10 providers, 3 hospital systems |
| `data/crm_notes.txt` | Raw notes — the unstructured input |
| `data/crm_extracted.json` | What the ingestion call produces from those notes, cached |
| `data/product_kb.md` | Tempus capabilities, chunked and tagged |

## How it works

**Impact = Opportunity × Fit × Winnability**, with Trigger sorting within tiers.

Multiplied rather than added: a large account we can't win is worth nothing,
and a perfect fit with no way in is worth nothing. Adding would let one strong
dimension mask a fatal weak one.

- **Opportunity** — addressable volume × role attribution × specialty attach.
  Base layer: always computes from market data alone.
- **Fit** — how the barrier was stated × how well the KB answers it. Read off
  the *same* retrieval that feeds the objection handler, so the score and the
  pitch can't disagree.
- **Winnability** — incumbent × ease of first yes × receptivity, bounded
  0.3–1.3 so weak sub-scores don't compound to near-zero.
- **Trigger** — a badge, not a factor. A hot signal moves a strong account
  forward in the queue rather than rescuing a weak one.

Missing CRM data scores **neutral, never a penalty**. Silence is absence of
signal, not a bad signal.

## KB retrieval — two paths

- **Glossary answers** use a stored tag → chunk lookup. Shared and reused, so
  reviewed once and never varies.
- **Provider handlers and pitches** use semantic retrieval, scoped within the
  tag's section, with a similarity floor. Below the floor returns nothing and
  the prompt leaves the fact out rather than inventing one.


## Functional uploads

All three upload paths are live, matching the design-steps slide:

| Path | Where | LLM calls | What happens |
|---|---|---|---|
| CRM note | ＋ Note on any provider card | 1 (or 0 for bundled test notes) | Extraction → validation → re-score → regenerate handler + pitch |
| Market intelligence | Upload data → CSV | 0 | Schema-checked parse → Opportunity recomputes; unknown provider_ids become new providers on the neutral path |
| Knowledge base | Upload data → MD | derive attach + regens | Re-chunk → diff → attach re-derived → Fit recomputed → handlers/pitches refreshed where chunks changed |

**With `GROQ_API_KEY` set, extraction and generation are live** — any note you
write is really extracted, and handlers and pitches are really generated with
the regenerate-on-fail loop. Groq's free tier runs open-weight models, needs no
card, and has generous daily limits: get a key at
[console.groq.com/keys](https://console.groq.com/keys).

The verified extractions in `data/test_extractions.json` are a fallback — they
catch a rate limit or a malformed response mid-demo rather than surfacing a
stack trace.

Without a key, the sample notes in `data/test_uploads/` still ingest from those
cached extractions, and generated text falls back to reviewed templates. An
arbitrary note without a key returns a clear message. Nothing is ever
heuristically guessed.

## Test data — data/test_uploads/

Each file exercises one edge case from the deck's hallucination tables:

| File | Edge case | Expected behaviour |
|---|---|---|
| note_1_satisfied_trap.txt | Chen praises his vendor's turnaround, complains about cost | Turnaround NOT tagged (mention is not objection); Cost tagged; sentiment satisfied |
| note_2_new_tag.txt | Epic ordering gap — no existing tag fits | Mints a new tag; with no stored mapping, retrieval widens to global search and finds the ordering-channels fact |
| note_3_duplicate_phrasing.txt | "TAT is brutal" | Matches the existing Turnaround tag; no TAT tag minted; Cost kept |
| note_4_thin.txt | Three unreturned voicemails | Nothing extracted; receptivity not_looking; previous tags overwritten; Fit stays neutral at 1.0 |
| note_5_multi_objection.txt | Turnaround and cost raised in one meeting | Both tagged, each with its own grounded context |
| mi_update.csv | Refreshed market data plus a new provider, P11 | 0 LLM calls; P11 ranks on the base layer alone, every enrichment factor neutral |
| kb_update_faster_turnaround.md | xF turnaround improves from 7 to 5 days | Only the turnaround chunk diffs; attach re-derives; affected handlers and pitches refresh quoting 5 days |

## Automated tests

```bash
python3 test_pipeline.py
```

The suite runs offline by design, against the verified cached extractions,
even if `GROQ_API_KEY` is set. A live model is non-deterministic — it may mint
`Ordering Pathway` on one run and `EHR Integration` on the next, both correct —
so asserting exact output against a live call would make the suite flaky. What
is under test is the pipeline: validation, scoring, retrieval and the cascades.
Those must be reproducible.

48 checks across 11 groups: the five notes above, grounding rejection of
invented contexts, enum defaulting, all six generation validators (invented
number, competitor name, inferred feeling, word cap, digits in pitch,
commitment words), MI schema rejection, the KB cascade, and the
no-KB-answer path (below the similarity floor the handler declines rather
than inventing a fact).

## Where model calls happen

Rendering **never** calls a model. Generation runs once — at note ingestion,
or during a KB cascade — and the result is stored in session state. A card
that generated on every render would make the list unusable (fourteen
sequential calls per page) and would mean the text a rep reads could change
between glances at the same screen.

A provider whose note has been replaced will not fall back to the reviewed
seed text, because that text was written for the superseded note. Without a
key the handler is withheld and marked stale rather than showing something
factually mismatched.

## Guardrail: unsourced claims

Numeric and named-entity hallucinations were caught from the start — a digit
must trace to a supplied input, competitor names are banned. Softer claims were
not, and testing surfaced a pitch asserting the product was *available across
major oncology centers*, *aligned with guideline recommendations* and could
*improve outcomes*. None of that is in the KB.

Two changes closed it. Regulatory, availability, guideline, reimbursement and
efficacy language is now rejected unless it appears in the supplied fact. And a
provider with no logged objection is given a published specialty-level fact to
work from, so the model never has to fill empty space.

## Known limitations

- Similarity uses token overlap as a stand-in for embeddings. It demonstrates
  the mechanism — score, floor, empty-on-weak-match — but under-scores synonyms
  ("quantity not sufficient" vs "tissue is insufficient").
- One note per provider at a time; a new note overwrites (newest wins), which
  is the designed conflict rule. Merging across multiple retained notes is not
  built.
- The FDA approvals feed for the "New approval" trigger is specified, not wired.
- Weights are directional hypotheses, not calibrated against closed-won data.

All provider names and CRM notes are fictional. Product facts are paraphrased
from public Tempus sources and would need medical-affairs review before
shipping rep-facing language.
