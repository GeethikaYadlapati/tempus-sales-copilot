"""
Ingestion engine — the reading half of the LLM design.

Three upload paths, mirroring the deck:

  CRM note   -> one constrained-JSON extraction call (contexts, tags, enums),
                validated, then re-score + regenerate for that provider.
  MI CSV     -> deterministic parse, Opportunity recomputes. 0 LLM calls.
  KB markdown-> re-chunk, re-derive attach, re-map tags; Fit + generated
                text refresh where chunks changed.

Guardrails implemented here (from the deck):
  - Closed tag vocabulary passed in-prompt; model matches before minting.
  - "Mention is not objection": satisfied mentions must return no tag.
  - Every enum has an unknown option; missing scores neutral downstream.
  - Extracted contexts must be grounded in the note text or they are rejected.
  - Without an API key, only the bundled test notes ingest (their verified
    extractions are cached); arbitrary notes require a key. Nothing is ever
    heuristically guessed.
"""

import csv
import hashlib
import io
import json
import re
from pathlib import Path

DATA = Path(__file__).parent / "data"

ALLOWED_SENTIMENT = {"satisfied", "dissatisfied", "unknown"}
ALLOWED_EASE = {"add_on", "new_order", "new_workflow", "committee", "unknown"}
ALLOWED_RECEPTIVITY = {"warm", "neutral", "passive", "not_looking", "unknown"}
ALLOWED_BARRIER = {"stated", "unrecognised", "none"}

MI_REQUIRED = ["provider_id", "name", "title", "specialty", "hospital_system",
               "territory", "annual_patient_volume", "testing_rate_pct",
               "current_vendor", "referring_network_size", "role_type"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def note_key(text: str) -> str:
    return hashlib.sha1(_norm(text).encode()).hexdigest()[:16]


def load_test_extractions() -> dict:
    p = DATA / "test_extractions.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------------------------------------------------------------------------
# CRM note extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """
You extract structured fields from a pharma sales rep's CRM note about one
oncology provider, for a call-planning tool.

OBJECTIONS
An objection is a STATED BARRIER to adopting Tempus testing — not a topic that
came up. If the provider is satisfied with something (their turnaround, their
vendor), that is NOT an objection; do not tag it. Classify by the barrier, not
the topic.

TAGS
GLOSSARY below is the existing tag vocabulary. For each objection, reuse the
glossary tag that covers the same barrier even if worded differently ("TAT",
"slow results" and "turnaround" are one barrier). Mint a new tag ONLY when no
glossary tag covers the barrier; a new tag is one or two Title Case words
naming the barrier. Never mint a synonym of an existing tag.

FIELDS
- objection_contexts: map of tag -> one sentence quoting or closely
  paraphrasing the specific instance in the note. Only what the note states —
  never add numbers, names or events the note does not contain. Empty map if
  no objection.
- new_tags: list of tags you minted (subset of the context keys).
- barrier_type: "stated" if the provider raised a barrier outright,
  "unrecognised" if the note describes a real gap the provider does not see
  as solvable, "none" if there is no barrier.
- entire_note_context: one or two sentences summarising the note for a pitch
  writer. Facts from the note only.
- vendor_sentiment: satisfied | dissatisfied | unknown  (about their CURRENT
  vendor, only if the note says so).
- ease_signal: add_on | new_order | new_workflow | committee | unknown —
  how big the first yes is, only if the note indicates it.
- receptivity: warm | neutral | passive | not_looking | unknown.

If the note contains no usable information beyond logistics (rescheduling,
voicemails, small talk), return an empty context map, barrier_type "none" and
unknown for every enum. That is a correct output, not a failure.

Return ONLY JSON:
{"objection_contexts": {}, "new_tags": [], "barrier_type": "",
 "entire_note_context": "", "vendor_sentiment": "", "ease_signal": "",
 "receptivity": ""}
""".strip()


def _grounded(context: str, note: str) -> bool:
    """Reject extracted contexts that are not traceable to the note."""
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
            "with", "that", "this", "their", "they", "she", "he", "her",
            "his", "before", "after", "has", "had", "have", "was", "were"}
    ctx = {w for w in re.findall(r"[a-z0-9]+", context.lower())
           if w not in stop and len(w) > 2}
    src = set(re.findall(r"[a-z0-9]+", note.lower()))
    if not ctx:
        return False
    return len(ctx & src) / len(ctx) >= 0.5


def validate_extraction(out: dict, note: str, glossary: list):
    """Schema + guardrail checks. Returns (clean_dict, problems)."""
    problems = []
    contexts = out.get("objection_contexts") or {}
    new_tags = [t for t in (out.get("new_tags") or []) if t]

    clean_ctx = {}
    for tag, ctx in contexts.items():
        tag = str(tag).strip()
        if not ctx or not str(ctx).strip():
            problems.append(f"empty context for {tag}")
            continue
        if not _grounded(str(ctx), note):
            problems.append(f"context for {tag} not grounded in the note")
            continue
        known = tag in glossary
        minted = tag in new_tags
        if not known and not minted:
            problems.append(f"tag {tag} neither in glossary nor declared new")
            continue
        if minted and any(_norm(tag) == _norm(g) for g in glossary):
            # exact duplicate mint — silently fold into the existing tag
            tag = next(g for g in glossary if _norm(g) == _norm(tag))
            minted = False
        clean_ctx[tag] = str(ctx).strip()

    new_tags = [t for t in new_tags if t in clean_ctx and t not in glossary]

    barrier = out.get("barrier_type", "none")
    if barrier not in ALLOWED_BARRIER:
        barrier = "none"
        problems.append("barrier_type out of range, defaulted to none")
    if not clean_ctx:
        barrier = "none"

    def enum(field, allowed):
        v = out.get(field, "unknown")
        if v not in allowed:
            problems.append(f"{field}='{v}' out of range, defaulted to unknown")
            return "unknown"
        return v

    clean = {
        "objection_tags": list(clean_ctx.keys()),
        "objection_contexts": clean_ctx,
        "new_tags": new_tags,
        "barrier_type": barrier,
        "entire_note_context": str(out.get("entire_note_context", "")).strip(),
        "vendor_sentiment": enum("vendor_sentiment", ALLOWED_SENTIMENT),
        "ease_signal": enum("ease_signal", ALLOWED_EASE),
        "receptivity": enum("receptivity", ALLOWED_RECEPTIVITY),
    }
    return clean, problems


MIN_NOTE_WORDS = 5


def extract_note(note_text: str, glossary: list):
    """Returns (extraction, source, problems).

    LIVE FIRST when a key is present, so any note a user writes is really
    extracted — the bundled test notes are examples, not a special path.
    The verified cached extractions are a fallback: they catch an API error
    or a malformed response mid-demo instead of surfacing a stack trace.

    Order of attempts:
      1. GROQ_API_KEY set -> live constrained-JSON call ("live").
      2. Bundled test note      -> verified cached extraction ("cached").
      3. Neither                -> (None, "unavailable", [reason]).
    Nothing is ever heuristically guessed from the text.
    """
    # A note too short to contain a meeting is rejected rather than extracted.
    # Sending it to the model would spend a call to learn nothing, and storing
    # the empty result would wipe whatever the provider's previous note held.
    if len(note_text.split()) < MIN_NOTE_WORDS:
        return None, "too_short", [
            f"Note is too short to ingest ({len(note_text.split())} words). "
            f"At least {MIN_NOTE_WORDS} are needed. Nothing was changed."]

    cached = load_test_extractions().get(note_key(note_text))

    from llm import _client, call_model, MODEL_FAST  # late import: avoid cycle

    if _client() is not None:
        user = f"GLOSSARY: {', '.join(glossary)}\n\nNOTE:\n{note_text}"
        out = call_model(EXTRACTION_SYSTEM, user,
                         model=MODEL_FAST, max_tokens=900)
        if out is not None:
            clean, problems = validate_extraction(out, note_text, glossary)
            return clean, "live", problems
        if not cached:
            from llm import LAST_ERROR
            return None, "error", [LAST_ERROR.get("msg")
                                   or "extraction call failed"]
        # fall through to the verified extraction rather than erroring

    if cached:
        clean, problems = validate_extraction(
            cached["extraction"], note_text, glossary)
        return clean, "cached", problems

    return None, "unavailable", [
        "Live ingestion needs GROQ_API_KEY (free, "
        "console.groq.com/keys). The sample notes in "
        "data/test_uploads/ ingest offline — their verified extractions "
        "ship with the prototype."]


def apply_note(provider, extraction: dict, glossary: list):
    """Newest note wins: overwrite this provider's CRM enrichment.
    Returns the updated glossary (minted tags appended)."""
    provider.objection_tags = extraction["objection_tags"]
    provider.objection_contexts = extraction["objection_contexts"]
    provider.barrier_type = extraction["barrier_type"]
    provider.entire_note_context = extraction["entire_note_context"]
    provider.vendor_sentiment = extraction["vendor_sentiment"]
    provider.ease_signal = extraction["ease_signal"]
    provider.receptivity = extraction["receptivity"]
    provider.has_crm_notes = True
    for t in extraction["new_tags"]:
        if t not in glossary:
            glossary.append(t)
    return glossary


# ---------------------------------------------------------------------------
# MI upload — deterministic, 0 LLM calls
# ---------------------------------------------------------------------------

def parse_mi_csv(file_bytes: bytes):
    """Returns (rows, problems). Rejects rather than guessing on bad schema."""
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, ["file is not UTF-8 text"]
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in MI_REQUIRED if c not in (reader.fieldnames or [])]
    if missing:
        return None, [f"missing required columns: {', '.join(missing)}"]
    rows, problems = [], []
    for i, r in enumerate(reader, start=2):
        try:
            r["annual_patient_volume"] = int(r["annual_patient_volume"])
            r["testing_rate_pct"] = int(r["testing_rate_pct"])
            r["referring_network_size"] = int(r["referring_network_size"])
        except (ValueError, TypeError):
            problems.append(f"row {i}: non-numeric volume/rate, skipped")
            continue
        if not 0 <= r["testing_rate_pct"] <= 100:
            problems.append(f"row {i}: testing_rate_pct out of range, skipped")
            continue
        rows.append(r)
    if not rows:
        problems.append("no valid rows")
        return None, problems
    return rows, problems


def apply_mi(providers: list, rows: list):
    """Update matching providers' market fields; add unmatched rows as new
    providers with no CRM enrichment (the neutral path). Returns
    (providers, updated_ids, added_ids)."""
    from scoring import Provider
    by_id = {p.provider_id: p for p in providers}
    updated, added = [], []
    for r in rows:
        pid = r["provider_id"].strip()
        if pid in by_id:
            p = by_id[pid]
            p.name = r["name"]; p.title = r["title"]
            p.specialty = r["specialty"]
            p.hospital_system = r["hospital_system"]
            p.territory = r["territory"]
            p.annual_patient_volume = r["annual_patient_volume"]
            p.testing_rate_pct = r["testing_rate_pct"]
            p.current_vendor = r["current_vendor"]
            p.referring_network_size = r["referring_network_size"]
            p.role_type = r["role_type"]
            updated.append(pid)
        else:
            p = Provider(provider_id=pid, name=r["name"], title=r["title"],
                         specialty=r["specialty"],
                         hospital_system=r["hospital_system"],
                         territory=r["territory"],
                         annual_patient_volume=r["annual_patient_volume"],
                         testing_rate_pct=r["testing_rate_pct"],
                         current_vendor=r["current_vendor"],
                         referring_network_size=r["referring_network_size"],
                         role_type=r["role_type"])
            providers.append(p)
            added.append(pid)
    return providers, updated, added


# ---------------------------------------------------------------------------
# KB upload — re-chunk, re-derive attach, find changed chunks
# ---------------------------------------------------------------------------

def parse_kb(text: str):
    """Chunking is deterministic: sections are marked [chunk: name]."""
    chunks = {}
    for m in re.finditer(r"## \[chunk: (\w+)\][^\n]*\n+(.*?)(?=\n## |\Z)",
                         text, re.S):
        chunks[m.group(1)] = m.group(2).strip()
    return chunks


def derive_attach(chunks: dict):
    """Attach needs the whole document at once. Live: one LLM call over the
    full KB. Offline: the bundles chunk is structured enough to parse
    deterministically — and if it is absent, attach stays unchanged rather
    than being guessed."""
    bundles = chunks.get("bundles", "")
    if not bundles:
        return None, "KB has no bundles section — attach table left unchanged"
    counts = {}
    patterns = {
        "Breast / Gynecologic": r"[Bb]reast and gyn\w* oncology typically orders ([^.]+)\.",
        "Pancreatic / GI": r"[Pp]ancreatic oncology typically orders ([^.]+)\.",
        "Thoracic (Lung)": r"[Tt]horacic and GI oncology typically order ([^.]+)\.",
    }
    for spec, pat in patterns.items():
        m = re.search(pat, bundles)
        if m:
            n = len(re.findall(r"\bx[TFR]\w*|\bHRD\b|\bPurIST\b|\bIPS\b",
                               m.group(1)))
            counts[spec] = max(n, 1)
    if not counts:
        return None, "could not read bundle structure — attach unchanged"
    attach = {}
    base = min(counts.values())
    for spec, n in counts.items():
        attach[spec] = round(1.0 + 0.15 * (n - base), 2)
    attach["GI / Colorectal"] = attach.get("Thoracic (Lung)", 1.15)
    for spec in ["Prostate / GU", "Sarcoma / Rare Tumors",
                 "Hematologic Malignancies"]:
        attach[spec] = 1.00
    attach["Multi-Tumor / Network Oversight"] = 1.10
    return attach, None


def changed_chunks(old: dict, new: dict) -> set:
    keys = set(old) | set(new)
    return {k for k in keys if _norm(old.get(k, "")) != _norm(new.get(k, ""))}
