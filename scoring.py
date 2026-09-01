"""
Impact scoring engine.

Impact = Opportunity x Fit x Winnability

Every function here is pure: same inputs, same output, no I/O, no model calls.
That is deliberate — a rep or manager has to be able to challenge a ranking
("why is this provider third?") and get an answer that is reproducible.
All LLM work happens upstream in ingestion; nothing in this file calls a model.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# --------------------------------------------------------------------------
# Weight tables. These are the assumptions the model makes explicit.
# Each is a testable hypothesis, not a measured coefficient.
# --------------------------------------------------------------------------

# What share of stated volume this role's "yes" actually converts.
# A CMO's endorsement enables adoption but does not itself generate orders —
# physician-level selling still happens downstream. Without this correction
# administrators top every list by construction (~6x head start).
ATTRIBUTION = {
    "administrator": 0.25,
    "department_lead": 0.60,
    "physician": 1.00,
}

# Some specialties pull several tests per order. Inferred from how the KB
# describes test bundling, NOT from Tempus pricing (which we don't have).
ATTACH = {
    "Breast / Gynecologic": 1.30,
    "Pancreatic / GI": 1.25,
    "Thoracic (Lung)": 1.15,
    "GI / Colorectal": 1.15,
    "Prostate / GU": 1.00,
    "Sarcoma / Rare Tumors": 1.00,
    "Hematologic Malignancies": 1.00,
    "Multi-Tumor / Network Oversight": 1.10,
}

# Patients already tested by a competitor are winnable but harder.
DISPLACEMENT_DISCOUNT = 0.40

# Fit inputs -------------------------------------------------------------
# How the barrier was stated (from CRM ingestion)
BARRIER_TYPE = {
    "stated": 1.8,        # they named it outright as a problem
    "unrecognised": 1.4,  # a real gap they don't see as solvable
    "none": 1.0,          # no barrier -> neutral, never a penalty
}

# How well the KB answers it (from the same retrieval that feeds the handler)
RETRIEVAL_STRENGTH = {
    "strong": 1.0,
    "middling": 0.7,
    "below_floor": 0.4,
}

FIT_CAP = 1.8

# Winnability inputs ------------------------------------------------------
INCUMBENT = {
    ("none", None): 1.0,
    ("basic", None): 0.9,
    ("competitor", "dissatisfied"): 0.8,
    ("competitor", "satisfied"): 0.6,
    ("competitor", "unknown"): 0.7,   # fallback: sentiment not in the notes
}

EASE = {
    "add_on": 1.2,        # attaches to an order they already place
    "new_order": 1.0,
    "new_workflow": 0.8,
    "committee": 0.6,
    "unknown": 1.0,       # missing data is neutral, never a penalty
}

RECEPTIVITY = {
    "warm": 1.2,
    "neutral": 1.0,
    "passive": 0.7,
    "not_looking": 0.5,
    "unknown": 1.0,
}

# Multiplying compounds: three weak sub-scores would drive a provider to
# near-zero and overstate how unreachable they are. Three optimistic ones
# would inflate the top. Bounds keep the spread interpretable.
WINNABILITY_FLOOR = 0.30
WINNABILITY_CAP = 1.30



@dataclass
class Provider:
    """One row of market intelligence, enriched with CRM ingestion output."""
    provider_id: str
    name: str
    title: str
    specialty: str
    hospital_system: str
    territory: str
    annual_patient_volume: int
    testing_rate_pct: int
    current_vendor: str
    referring_network_size: int
    role_type: str

    # From CRM ingestion. Defaults are the no-notes case.
    objection_tags: list = field(default_factory=list)
    objection_contexts: dict = field(default_factory=dict)
    barrier_type: str = "none"
    entire_note_context: str = ""
    vendor_sentiment: str = "unknown"
    ease_signal: str = "unknown"
    receptivity: str = "unknown"
    open_loop: bool = False
    last_contact: Optional[str] = None

    # From KB retrieval (set at scoring time by the retrieval layer)
    retrieval_strength: str = "below_floor"

    # Computed
    opportunity: float = 0.0
    fit: float = 0.0
    winnability: float = 0.0
    impact: float = 0.0
    has_crm_notes: bool = False


# --------------------------------------------------------------------------
# Factor 1 — Opportunity. "Is there enough business here?"
# Base layer: always computes from market data alone, never falls back.
# --------------------------------------------------------------------------

def addressable_volume(volume: int, testing_rate_pct: int, role_type: str) -> float:
    """Untested patients count fully; patients at a competitor count at 40%."""
    untested = (100 - testing_rate_pct) / 100
    tested = testing_rate_pct / 100
    attribution = ATTRIBUTION.get(role_type, 1.0)
    return volume * attribution * (untested + tested * DISPLACEMENT_DISCOUNT)


def opportunity_score(provider: Provider, territory_max: float) -> float:
    raw = addressable_volume(
        provider.annual_patient_volume,
        provider.testing_rate_pct,
        provider.role_type,
    )
    normalised = raw / territory_max if territory_max > 0 else 0.0
    return round(normalised * ATTACH.get(provider.specialty, 1.0), 3)


# --------------------------------------------------------------------------
# Factor 2 — Fit. "Do we solve a problem they actually have?"
# Read off the same retrieval that feeds the objection handler, so the score
# and the pitch cannot disagree.
# --------------------------------------------------------------------------

def fit_score(barrier_type: str, retrieval_strength: str) -> float:
    if barrier_type == "none":
        return 1.0  # no tag -> neutral, not a penalty
    raw = BARRIER_TYPE.get(barrier_type, 1.0) * RETRIEVAL_STRENGTH.get(
        retrieval_strength, 0.4
    )
    return round(min(raw, FIT_CAP), 3)


# --------------------------------------------------------------------------
# Factor 3 — Winnability. "Will this call go anywhere?"
# --------------------------------------------------------------------------

def incumbent_score(current_vendor: str, vendor_sentiment: str) -> float:
    vendor = (current_vendor or "").strip().lower()
    if vendor in ("", "none"):
        return INCUMBENT[("none", None)]
    if "in-house" in vendor or "mixed" in vendor:
        return INCUMBENT[("basic", None)]
    sentiment = vendor_sentiment if vendor_sentiment in (
        "satisfied", "dissatisfied") else "unknown"
    return INCUMBENT[("competitor", sentiment)]


def winnability_score(provider: Provider) -> float:
    raw = (
        incumbent_score(provider.current_vendor, provider.vendor_sentiment)
        * EASE.get(provider.ease_signal, 1.0)
        * RECEPTIVITY.get(provider.receptivity, 1.0)
    )
    return round(max(WINNABILITY_FLOOR, min(raw, WINNABILITY_CAP)), 3)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def score_all(providers: list, today: date = None) -> list:
    """Score and rank. Deterministic: same input, same order, every time."""
    volumes = [
        addressable_volume(p.annual_patient_volume, p.testing_rate_pct, p.role_type)
        for p in providers
    ]
    territory_max = max(volumes) if volumes else 1.0

    for p in providers:
        p.opportunity = opportunity_score(p, territory_max)
        p.fit = fit_score(p.barrier_type, p.retrieval_strength)
        p.winnability = winnability_score(p)
        p.impact = round(p.opportunity * p.fit * p.winnability, 3)

    return sorted(providers, key=lambda p: (-p.impact, p.provider_id))


def objection_counts(providers: list) -> list:
    """Count providers per objection tag, most raised first."""
    counts = {}
    for p in providers:
        for tag in p.objection_tags:
            counts.setdefault(tag, []).append(p.name.replace("Dr. ", "").split()[-1])
    return sorted(counts.items(), key=lambda kv: (-len(kv[1]), kv[0]))
