"""
Edge-case tests for the ingestion + generation pipeline.

Run:  python3 test_pipeline.py

Each test pushes a bundled note (or upload file) through the SAME code paths
the app uses — extract_note, apply_note, rescore, validators — and asserts
the guardrail behaviour the deck claims.
"""

import json
import os
import sys
from pathlib import Path

# Tests run OFFLINE by design, against the verified cached extractions.
# A live model is non-deterministic — it may mint "Ordering Pathway" on one
# run and "EHR Integration" on the next, both correct — so asserting exact
# output against a live call would make the suite flaky and meaningless.
# What is being tested here is the pipeline: validation, scoring, retrieval,
# the cascades. Those must be reproducible.
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

import data_layer as dl
import ingest
import llm
from scoring import score_all, fit_score

T = Path("data/test_uploads")
PASS, FAIL = 0, []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def fresh():
    providers = dl.load_providers()
    chunks = dl.load_kb_chunks()
    glossary = dl.load_glossary_tags()
    return providers, chunks, glossary


def rescore(providers, chunks):
    for p in providers:
        p._facts, p._strengths = {}, {}
        for tag in p.objection_tags:
            fact, s = dl.semantic_retrieve(
                tag, p.objection_contexts.get(tag, ""), chunks)
            p._facts[tag] = fact
            p._strengths[tag] = s
        p.retrieval_strength = (p._strengths[p.objection_tags[0]]
                                if p.objection_tags else "below_floor")
        p._kb_fact = (p._facts[p.objection_tags[0]]
                      if p.objection_tags else "")
        p.fit = fit_score(p.barrier_type, p.retrieval_strength)
    return score_all(providers)


def by_id(providers, pid):
    return next(p for p in providers if p.provider_id == pid)


# ---------------------------------------------------------------------------
print("\n1 · Satisfied mention is NOT an objection (Chen)")
providers, chunks, glossary = fresh()
note = (T / "note_1_satisfied_trap.txt").read_text()
ex, src, problems = ingest.extract_note(note, glossary)
check("extraction available offline", ex is not None and src == "cached")
check("Turnaround NOT tagged despite being discussed",
      "Turnaround" not in ex["objection_tags"], str(ex["objection_tags"]))
check("Cost IS tagged (the real barrier)",
      "Cost" in ex["objection_tags"])
check("vendor_sentiment = satisfied", ex["vendor_sentiment"] == "satisfied")
p = by_id(providers, "P02")
glossary = ingest.apply_note(p, ex, glossary)
providers = rescore(providers, chunks)
check("incumbent stays 0.6 (satisfied competitor)",
      abs(p.winnability - max(0.3, min(0.6 * 0.6 * 1.2, 1.3))) < 1e-6,
      f"win={p.winnability}")

# ---------------------------------------------------------------------------
print("\n2 · Unmatchable barrier mints a new tag (Siddiqui)")
providers, chunks, glossary = fresh()
note = (T / "note_2_new_tag.txt").read_text()
ex, src, _ = ingest.extract_note(note, glossary)
check("exactly one new tag minted", len(ex["new_tags"]) == 1,
      str(ex["new_tags"]))
check("minted tag is not a synonym of an existing one",
      not any(t.lower() in {g.lower() for g in glossary}
              for t in ex["new_tags"]), str(ex["new_tags"]))
p = by_id(providers, "P07")
g2 = ingest.apply_note(p, ex, list(glossary))
check("glossary grew by exactly the minted tag",
      set(g2) - set(glossary) == set(ex["new_tags"]))
providers = rescore(providers, chunks)
check("minted tag with no mapping still retrieves via global search",
      p.retrieval_strength in ("middling", "strong"),
      p.retrieval_strength)
check("retrieved fact is the ordering-channels chunk",
      "Hub portal" in p._kb_fact, p._kb_fact[:60])

# ---------------------------------------------------------------------------
print("\n3 · 'TAT' phrasing matches Turnaround — no duplicate tag (Kim)")
providers, chunks, glossary = fresh()
note = (T / "note_3_duplicate_phrasing.txt").read_text()
ex, src, _ = ingest.extract_note(note, glossary)
check("mapped to existing Turnaround tag",
      "Turnaround" in ex["objection_tags"])
check("no 'TAT' tag minted", ex["new_tags"] == [], str(ex["new_tags"]))
check("Cost barrier retained alongside", "Cost" in ex["objection_tags"])

# ---------------------------------------------------------------------------
print("\n4 · Thin note degrades to neutral, not to garbage (Zhang)")
providers, chunks, glossary = fresh()
note = (T / "note_4_thin.txt").read_text()
ex, src, _ = ingest.extract_note(note, glossary)
check("no tags from three voicemails", ex["objection_tags"] == [])
check("receptivity reads not_looking", ex["receptivity"] == "not_looking")
check("barrier none", ex["barrier_type"] == "none")
check("vendor sentiment stays unknown", ex["vendor_sentiment"] == "unknown")
p = by_id(providers, "P09")
glossary = ingest.apply_note(p, ex, glossary)
providers = rescore(providers, chunks)
check("newest-note-wins: previous tags overwritten", p.objection_tags == [])
check("Fit neutral 1.0 — no barrier is not a penalty", p.fit == 1.0,
      f"fit={p.fit}")
check("still ranks on the base layer alone", p.impact > 0)

# ---------------------------------------------------------------------------
print("\n5 · Multi-objection note yields one context per barrier (Marsh)")
providers, chunks, glossary = fresh()
note = (T / "note_5_multi_objection.txt").read_text()
ex, src, _ = ingest.extract_note(note, glossary)
check("both barriers tagged",
      set(ex["objection_tags"]) == {"Turnaround", "Cost"},
      str(ex["objection_tags"]))
check("each has its own grounded context",
      all(len(ex["objection_contexts"][t]) > 20
          for t in ("Turnaround", "Cost")))

# ---------------------------------------------------------------------------
print("\n6 · Grounding check rejects invented context")
fake = {"objection_contexts":
        {"Turnaround": "A MolecularDx order took 45 days last spring."},
        "new_tags": [], "barrier_type": "stated",
        "entire_note_context": "x", "vendor_sentiment": "unknown",
        "ease_signal": "unknown", "receptivity": "unknown"}
clean, problems = ingest.validate_extraction(
    fake, "Brief chat, nothing specific discussed.", ["Turnaround"])
check("ungrounded context stripped", clean["objection_tags"] == [])
check("problem reported", any("not grounded" in p for p in problems))

# ---------------------------------------------------------------------------
print("\n7 · Enum out of range defaults to unknown, never invents")
fake = {"objection_contexts": {}, "new_tags": [], "barrier_type": "maybe",
        "entire_note_context": "", "vendor_sentiment": "angry",
        "ease_signal": "easy", "receptivity": "unknown"}
clean, problems = ingest.validate_extraction(fake, "note", [])
check("barrier defaulted", clean["barrier_type"] == "none")
check("sentiment defaulted", clean["vendor_sentiment"] == "unknown")
check("ease defaulted", clean["ease_signal"] == "unknown")

# ---------------------------------------------------------------------------
print("\n8 · Generation validators catch the deck's failure modes")
check("invented number caught",
      any("not in supplied" in x for x in
          llm.validate_handler("Results in 4 days.", "typically 7 days", "")))
check("competitor name caught",
      any("competitor" in x for x in
          llm.validate_handler("Beats Guardant easily.", "", "")))
check("inferred feeling caught",
      any("feeling" in x for x in
          llm.validate_handler("You seem frustrated with them.", "", "")))
check("short pitch caught",
      any("words" in x for x in llm.validate_pitch("Sam, hello.", "Sam")))
good = "Sam, " + " ".join(["word"] * 68) + " end."
check("digit in pitch caught",
      any("digits" in x for x in
          llm.validate_pitch(good.replace("end.", "7 days."), "Sam")))
check("commitment caught",
      any("commits" in x for x in
          llm.validate_pitch(good.replace("end.", "want a demo?"), "Sam")))

# ---------------------------------------------------------------------------
print("\n9 · MI upload: schema rejection + neutral new provider")
bad = b"provider_id,name\nP99,Dr. X\n"
rows, problems = ingest.parse_mi_csv(bad)
check("bad schema rejected", rows is None)
good_csv = (T / "mi_update.csv").read_bytes()
rows, problems = ingest.parse_mi_csv(good_csv)
check("valid MI parses", rows is not None and len(rows) == 11)
providers, chunks, glossary = fresh()
providers, upd, added = ingest.apply_mi(providers, rows)
check("existing providers updated", len(upd) == 10)
check("new provider P11 added", added == ["P11"])
providers = rescore(providers, chunks)
p11 = by_id(providers, "P11")
check("P11 has no CRM enrichment → Fit neutral", p11.fit == 1.0)
check("P11 winnability from MI fallback only (vendor None → 1.0, "
      "rest neutral)", p11.winnability == min(1.3, 1.0 * 1.0 * 1.0))
check("P11 still ranks — base layer alone", p11.impact > 0)

# ---------------------------------------------------------------------------
print("\n10 · KB upload cascade: changed chunk → Fit + text refresh")
providers, chunks, glossary = fresh()
v2_text = (T / "kb_update_faster_turnaround.md").read_text()
new_chunks = ingest.parse_kb(v2_text)
changed = ingest.changed_chunks(chunks, new_chunks)
check("only the turnaround chunk changed", changed == {"turnaround"},
      str(changed))
affected = {t for t, c in dl.TAG_TO_CHUNK.items() if c in changed}
check("maps to the Turnaround tag", affected == {"Turnaround"})
providers = rescore(providers, new_chunks)
ok4 = by_id(providers, "P04")
check("Okafor's retrieved fact now cites 5 days",
      "5 days" in ok4._kb_fact, ok4._kb_fact[:80])
regens = json.loads((Path("data/kb_regens.json")).read_text())
import hashlib, re
h = hashlib.sha1(re.sub(
    r"\s+", " ", new_chunks["turnaround"].strip().lower())
    .encode()).hexdigest()[:16]
check("verified regenerated text exists for the new chunk", h in regens)
check("regenerated handler quotes the new number, not the old",
      "5 days" in regens[h]["handlers"]["P04|Turnaround"]
      and "7 days" not in regens[h]["handlers"]["P04|Turnaround"])
check("attach re-derivation runs on the new KB",
      ingest.derive_attach(new_chunks)[0] is not None)

# ---------------------------------------------------------------------------
print("\n11 · No-KB-answer path: floor returns nothing, prompt admits it")
fact, strength = dl.semantic_retrieve(
    "Turnaround", "completely unrelated gardening discussion", chunks)
check("below-floor retrieval returns empty", fact == "" and
      strength == "below_floor")


class Stub:
    provider_id = "PX"
    name = "Dr. Test Case"
    specialty = "GI / Colorectal"
    role_type = "physician"
    objection_contexts = {"Turnaround": "unrelated"}
    entire_note_context = ""


text, src = llm.generate_handler(Stub(), "Turnaround", "")
check("handler without a fact declines rather than invents",
      src == "no-kb-answer" and "flagged" in text)

# ---------------------------------------------------------------------------
print(f"\n{'='*56}\n{PASS} passed, {len(FAIL)} failed")
if FAIL:
    print("Failed:", *FAIL, sep="\n  - ")
    sys.exit(1)
