"""
Data access + KB retrieval.

Everything here reads flat files. In production the same function signatures
would read Postgres (providers, CRM) and a vector store (KB chunks) — the rest
of the app doesn't know or care which, which is the point of keeping this
behind one module.
"""

import csv
import json
import re
from pathlib import Path

from scoring import Provider

DATA = Path(__file__).parent / "data"


# --------------------------------------------------------------------------
# Market intelligence — the always-present structured layer
# --------------------------------------------------------------------------

def load_providers() -> list:
    with open(DATA / "market_intelligence.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    extracted = json.loads((DATA / "crm_extracted.json").read_text(encoding="utf-8"))
    crm = extracted["providers"]

    providers = []
    for r in rows:
        pid = r["provider_id"]
        notes = crm.get(pid)
        p = Provider(
            provider_id=pid,
            name=r["name"],
            title=r["title"],
            specialty=r["specialty"],
            hospital_system=r["hospital_system"],
            territory=r["territory"],
            annual_patient_volume=int(r["annual_patient_volume"]),
            testing_rate_pct=int(r["testing_rate_pct"]),
            current_vendor=r["current_vendor"],
            referring_network_size=int(r["referring_network_size"]),
            role_type=r["role_type"],
        )
        if notes:
            p.objection_tags = notes["objection_tags"]
            p.objection_contexts = notes["objection_contexts"]
            p.barrier_type = notes["barrier_type"]
            p.entire_note_context = notes["entire_note_context"]
            p.vendor_sentiment = notes["vendor_sentiment"]
            p.ease_signal = notes["ease_signal"]
            p.receptivity = notes["receptivity"]
            p.open_loop = notes["open_loop"]
            p.last_contact = notes["last_contact"]
            p.has_crm_notes = True
        providers.append(p)
    return providers


def load_glossary_tags() -> list:
    extracted = json.loads((DATA / "crm_extracted.json").read_text(encoding="utf-8"))
    return extracted["glossary"]


def load_raw_notes() -> dict:
    """Raw CRM text, keyed by provider id — shown in the UI for transparency."""
    text = (DATA / "crm_notes.txt").read_text(encoding="utf-8")
    notes = {}
    for block in text.split("### ")[1:]:
        header, _, body = block.partition("\n")
        pid = header.split("|")[0].strip()
        notes[pid] = body.strip()
    return notes


# --------------------------------------------------------------------------
# KB — chunked at ingestion, retrieved two ways
# --------------------------------------------------------------------------

def load_kb_chunks() -> dict:
    """Parse the KB into tag-keyed chunks. In production this runs once at
    KB upload and the result is stored, not re-parsed per request."""
    text = (DATA / "product_kb.md").read_text(encoding="utf-8")
    chunks = {}
    for match in re.finditer(
        r"## \[chunk: (\w+)\][^\n]*\n+(.*?)(?=\n## |\Z)", text, re.S
    ):
        chunks[match.group(1)] = match.group(2).strip()
    return chunks


# Stored tag -> chunk mapping, built once at ingestion and human-reviewed.
# This is the LOOKUP path: deterministic, used for the shared glossary answer.
TAG_TO_CHUNK = {
    "Turnaround": "turnaround",
    "Tissue": "tissue",
    "Referral": "referral",
    "Outcomes": "outcomes",
    "Cost": "cost",
    "Actionability": "actionability",
    "Incumbent": "incumbent",
}


def lookup_chunk(tag: str, chunks: dict) -> str:
    """Glossary path — stored mapping, no search."""
    return chunks.get(TAG_TO_CHUNK.get(tag, ""), "")


SIMILARITY_FLOOR = 0.06
STRONG_MATCH = 0.15


def _similarity(context: str, chunk: str) -> float:
    """Stand-in for embedding cosine similarity.

    A real implementation embeds both and compares vectors. Token overlap is
    enough to demonstrate the mechanism that matters: a score, a floor, and
    the fact that a weak match returns NOTHING rather than a bad fact.

    Its known weakness is synonymy — "quantity not sufficient" and "tissue is
    insufficient" mean the same thing but share few tokens, so this proxy
    under-scores them where real embeddings would not.
    """
    stop = {
        "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of",
        "in", "on", "for", "with", "at", "by", "from", "that", "this", "it",
        "as", "has", "had", "have", "not", "no", "she", "he", "her", "his",
        "before", "after", "than", "which", "been", "can", "will", "would",
    }
    def toks(s):
        return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in stop and len(w) > 2}
    a, b = toks(context), toks(chunk)
    if not a or not b:
        return 0.0
    # Overlap coefficient, not Jaccard: we care what fraction of the provider's
    # context the chunk covers, not how long the chunk happens to be.
    return len(a & b) / len(a)


def semantic_retrieve(tag: str, context: str, chunks: dict):
    """Provider path — search scoped WITHIN the tag's section.

    A minted tag has no stored mapping yet, so its scope widens to the whole
    KB — semantic search is exactly what covers the tail the mapping doesn't.
    Below the similarity floor returns ("", "below_floor") so the generation
    prompt leaves the fact out rather than inventing one.
    """
    if not context:
        return "", "below_floor"
    section = lookup_chunk(tag, chunks)
    if not section:
        section = " ".join(chunks.values())   # minted tag: global search

    # Score each sentence in the tag's section against the provider's context.
    sentences = [s.strip() for s in re.split(r"(?<=[.])\s+", section) if s.strip()]
    scored = sorted(
        ((_similarity(context, s), s) for s in sentences), reverse=True
    )
    best_score, best_sentence = scored[0]

    if best_score < SIMILARITY_FLOOR:
        return "", "below_floor"
    strength = "strong" if best_score >= STRONG_MATCH else "middling"
    # Return the top two sentences so the handler has room to work.
    top = " ".join(s for _, s in scored[:2])
    return top, strength


def attach_table() -> dict:
    """Derived from the KB at upload time; Opportunity reads it as a lookup.
    Kept here to show the dependency — it is not called per provider."""
    from scoring import ATTACH
    return ATTACH


SPECIALTY_CHUNK = {
    "Breast / Gynecologic": "referral",
    "Pancreatic / GI": "actionability",
    "GI / Colorectal": "incumbent",
    "Thoracic (Lung)": "incumbent",
    "Prostate / GU": "tissue",
    "Sarcoma / Rare Tumors": "incumbent",
    "Hematologic Malignancies": "turnaround",
    "Multi-Tumor / Network Oversight": "outcomes",
}


def specialty_fact(specialty: str, chunks: dict) -> str:
    """Fallback KB fact when a provider has no logged objection.

    The pitch prompt is priority-ordered — objection first, then interest,
    then workflow, then the specialty bundle. This is that last tier: a real
    published capability relevant to what they treat, so the model always has
    something true to say instead of inventing a claim.
    """
    key = SPECIALTY_CHUNK.get(specialty)
    text = chunks.get(key, "") if key else ""
    if not text:
        text = chunks.get("bundles", "")
    # first two sentences keep the prompt tight
    parts = [x.strip() for x in re.split(r"(?<=[.])\s+", text) if x.strip()]
    return " ".join(parts[:2])
