"""
Tempus Sales Copilot — functional prototype.

  data_layer.py  file reads + KB retrieval (the Postgres/vector-store seam)
  scoring.py     pure functions, no I/O, no model calls
  ingest.py      the reading half: note extraction, MI parse, KB cascade
  llm.py         the writing half: three prompts + validators
  app.py         presentation and state only

Uploads are live. CRM notes run one extraction call (or use the verified
cached extraction for the bundled test notes, so the demo works offline).
MI recomputes Opportunity deterministically. KB re-chunks, re-derives attach,
and cascades: Fit and generated text refresh where chunks changed.
"""

import hashlib
import json
import re
from pathlib import Path

import streamlit as st

import data_layer as dl
import ingest
import llm
from scoring import score_all, objection_counts, fit_score

st.set_page_config(page_title="Tempus Sales Copilot", layout="wide",
                   initial_sidebar_state="collapsed")


def html(block):
    if hasattr(st, "html"):
        st.html(block)
    else:
        st.markdown(block, unsafe_allow_html=True)


CSS = """
<style>
  [data-testid="stToolbar"] {display: none;}
  .stApp {background: #F4F6F5;}
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1340px;}
  .eyebrow {font-family: ui-monospace, Menlo, monospace; font-size: 11px;
            letter-spacing: .14em; text-transform: uppercase;
            color: #1F6F62; font-weight: 600;}
  .title {font-family: Georgia, 'Times New Roman', serif; font-size: 32px;
          font-weight: 700; color: #16233D; margin: 4px 0 2px;}
  .sub {color: #8A9290; font-size: 13.5px; margin: 0;}
  .track {display:flex; gap:3px; height:6px; margin:14px 0 18px;
          border-radius:3px; overflow:hidden;}
  .track i {flex:1; display:block;}
  .sec {font-family: Georgia, serif; font-size: 19px; font-weight: 600;
        color:#16233D; margin: 0 0 2px;}
  .sechint {font-size: 13px; color:#8A9290; margin: 0 0 10px;}
  .pill-lock {display:inline-block; font-size:13px; padding:6px 14px;
        border-radius:20px; background:#E6F0EE; color:#1F6F62;
        border:1px solid #BFD8D2; font-weight:600;}
  .pname {font-size: 15px; font-weight: 600; color:#16233D;}
  .pmeta {font-size: 13px; color:#3D4C63; margin-top: 2px;}
  .pspec {font-size: 13px; color:#8A9290; margin-top: 1px;}
  .tag {display:inline-block; font-size:11.5px; padding:3px 10px;
        border-radius:20px; background:#EDF1EF; color:#3D4C63;
        margin:8px 5px 0 0;}
  .tag-new {background:#FAEEDA; color:#8A5E1F;}
  .lab {font-family: ui-monospace, Menlo, monospace; font-size:10px;
        letter-spacing:.09em; text-transform:uppercase; color:#8A9290;
        margin: 0 0 6px;}
  .objwrap {background:#FFFFFF; border:1px solid #E1E5E3; border-radius:10px;
        padding:11px 12px 3px; margin-bottom:12px;}
  .obj {background:#F4F6F5; border-radius:8px; padding:11px 13px;
        margin-bottom:8px;}
  .objtag {font-size:11px; font-weight:700; color:#1F6F62;
           letter-spacing:.05em; text-transform:uppercase;}
  .objctx {font-size:13px; color:#6B7772; line-height:1.5; margin-top:5px;
           font-style:italic;}
  .objans {font-size:14px; color:#16233D; line-height:1.6; margin-top:8px;}
  .pitch {background:#E6F0EE; border-radius:8px; padding:12px 14px;
          font-size:14px; line-height:1.6; color:#16233D;}
  .prov {font-size:11px; color:#8A9290; margin-top:8px;}
  .stale {font-size:11px; color:#8A5E1F; background:#FAEEDA;
          border-radius:6px; padding:4px 9px; display:inline-block;
          margin-top:6px;}
  .banner {background:#E6F0EE; border-radius:8px; padding:9px 13px;
           font-size:13px; color:#1F6F62;}
  .gtag {font-size:14px; font-weight:600; color:#16233D;}
  .gcount {font-size:12px; color:#8A9290;}
  .gwho {font-size:12px; color:#8A9290; margin:3px 0 6px;}
  .gans {font-size:13px; line-height:1.55; color:#3D4C63;}
  div[data-testid="stExpander"] details {border:none !important;
       background:transparent !important;}
  div[data-testid="stExpander"] summary {font-size:13px; color:#1F6F62;
       font-weight:600; padding-left:0 !important;}
  div[data-testid="stVerticalBlockBorderWrapper"] {background:#FFFFFF;
       border-radius:12px;}
  .stButton button {border-radius:20px; font-size:12px; font-weight:600;
       padding:4px 14px; border:1px solid #E1E5E3; white-space:nowrap;}
  .stButton button p {white-space:nowrap; margin:0;}
  div[data-testid="stPopover"] button {white-space:nowrap;}
</style>
"""
html(CSS)


# ---------------------------------------------------------------------------
# Session DB — mutable state that uploads modify
# ---------------------------------------------------------------------------

def _chunk_hash(text):
    return hashlib.sha1(
        re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()[:16]


def rescore(db):
    """Re-run retrieval + scoring for every provider. Deterministic."""
    for p in db["providers"]:
        p._facts, p._strengths = {}, {}
        for tag in p.objection_tags:
            fact, strength = dl.semantic_retrieve(
                tag, p.objection_contexts.get(tag, ""), db["chunks"])
            p._facts[tag] = fact
            p._strengths[tag] = strength
        if p.objection_tags:
            p.retrieval_strength = p._strengths[p.objection_tags[0]]
            p._kb_fact = p._facts[p.objection_tags[0]]
        else:
            p.retrieval_strength = "below_floor"
            p._kb_fact = ""
        p.fit = fit_score(p.barrier_type, p.retrieval_strength)
    db["providers"] = score_all(db["providers"])


def init_db():
    providers = dl.load_providers()
    chunks = dl.load_kb_chunks()
    db = {
        "providers": providers,
        "glossary": dl.load_glossary_tags(),
        "chunks": chunks,
        "raw_notes": dl.load_raw_notes(),
        "gen_override": {},        # (pid, tag) -> handler text
        "pitch_override": {},      # pid -> pitch text
        "glossary_override": {},   # tag -> glossary answer
        "stale": set(),            # (pid, tag) pairs awaiting regeneration
        "no_kb": set(),            # (pid, tag) the KB genuinely cannot answer
        "gen_reason": {},          # (pid, tag) -> why the draft was rejected
        "note_replaced": set(),    # pids whose seed note was overwritten —
                                   # their reviewed fallbacks were written for
                                   # the OLD context and must not be reused
        "log": [],
        "last_ingest": None,       # survives the rerun so the change is visible
    }
    rescore(db)
    return db


if "db" not in st.session_state:
    st.session_state.db = init_db()
db = st.session_state.db

ALL_TERR = sorted({p.territory for p in db["providers"]})
st.session_state.setdefault("obj_filter", None)
st.session_state.setdefault("terr_on", False)
st.session_state.setdefault("terr_sel", set(ALL_TERR))

KB_REGENS = (json.loads((Path("data/kb_regens.json")).read_text())
             if Path("data/kb_regens.json").exists() else {})


# Rendering NEVER calls a model. Generation happens once, at ingestion or on
# a KB cascade, and the result is stored. A card that hit the API on every
# render would make the list unusable — fourteen sequential calls per page —
# and would also mean the text a rep reads could change between glances at
# the same screen. Same reason scoring is deterministic: reproducibility.

def get_handler(p, tag):
    if (p.provider_id, tag) in db["gen_override"]:
        return db["gen_override"][(p.provider_id, tag)], "generated"
    if p.provider_id not in db["note_replaced"]:
        fb = llm.FALLBACK_HANDLERS.get((p.provider_id, tag), "")
        if fb:
            return fb, "template"
    if not p._facts.get(tag, ""):
        return ("No published Tempus capability answers this concern yet — "
                "flagged. Acknowledge it honestly rather than improvising "
                "a fact."), "no-kb-answer"
    return "", "pending"


def get_pitch(p):
    if p.provider_id in db["pitch_override"]:
        return db["pitch_override"][p.provider_id], "generated"
    if p.provider_id in db["note_replaced"]:
        return "", "pending"     # seed pitch describes a superseded note
    return llm.FALLBACK_PITCHES.get(p.provider_id, ""), "template"


def get_glossary(tag):
    if tag in db["glossary_override"]:
        return db["glossary_override"][tag], "generated"
    return llm.FALLBACK_GLOSSARY.get(tag, ""), "template"


def regenerate_provider(p, cached_gen=None):
    """Refresh a provider's handler(s) and pitch after their inputs changed.
    Cached (verified) text first, then live generation, else mark stale."""
    live = llm._client() is not None
    for tag in p.objection_tags:
        if cached_gen and tag in cached_gen.get("handlers", {}):
            db["gen_override"][(p.provider_id, tag)] = \
                cached_gen["handlers"][tag]
            db["stale"].discard((p.provider_id, tag))
        elif live:
            text, src = llm.generate_handler(p, tag, p._facts.get(tag, ""))
            if src in ("live", "no-kb-answer"):
                # no-kb-answer is a correct, final result: the KB genuinely
                # has nothing for this barrier. Store it; it is not stale.
                db["gen_override"][(p.provider_id, tag)] = text
                db["stale"].discard((p.provider_id, tag))
                if src == "no-kb-answer":
                    db["no_kb"].add((p.provider_id, tag))
            else:
                db["stale"].add((p.provider_id, tag))
                db["gen_reason"][(p.provider_id, tag)] = list(
                    getattr(llm, "LAST_GEN", {}).get("problems") or [])
        else:
            db["gen_override"].pop((p.provider_id, tag), None)
            db["stale"].add((p.provider_id, tag))
    if cached_gen and cached_gen.get("pitch"):
        db["pitch_override"][p.provider_id] = cached_gen["pitch"]
    elif live:
        text, src = llm.generate_pitch(p, p._kb_fact)
        if src == "live":
            db["pitch_override"][p.provider_id] = text
            db["stale"].discard((p.provider_id, "__pitch__"))
        else:
            db["stale"].add((p.provider_id, "__pitch__"))
            db["gen_reason"][(p.provider_id, "__pitch__")] = list(
                getattr(llm, "LAST_GEN", {}).get("problems") or [])
    else:
        db["pitch_override"].pop(p.provider_id, None)
        if p.objection_tags:
            db["stale"].add((p.provider_id, "__pitch__"))
    for tag in cached_gen.get("glossary", {}) if cached_gen else {}:
        db["glossary_override"][tag] = cached_gen["glossary"][tag]


# ---------------------------------------------------------------------------
# Header + seasonal uploads
# ---------------------------------------------------------------------------

head, upload = st.columns([2.6, 1])
with head:
    html('<div class="eyebrow">Tempus / Field enablement</div>'
         '<div class="title">Sales Copilot</div>'
         '<p class="sub">Ohio territory · Riverbend, Lakeside &amp; Heartland'
         '</p>')

with upload:
    # nudges the control down to sit level with the title block
    html('<div style="height:34px"></div>')
    with st.popover("Upload data", use_container_width=True):
        st.caption("Seasonal sources. Try the files in data/test_uploads/.")

        mi = st.file_uploader("Market intelligence (CSV)", type=["csv"],
                              key="up_mi")
        if mi and st.button("Upload MI", use_container_width=True):
            rows, problems = ingest.parse_mi_csv(mi.getvalue())
            if rows is None:
                st.error("Rejected: " + "; ".join(problems))
            else:
                db["providers"], upd, added = ingest.apply_mi(
                    db["providers"], rows)
                rescore(db)
                msg = f"Opportunity recomputed for {len(upd)} providers"
                if added:
                    msg += (f"; {len(added)} new provider(s) added with no "
                            "CRM enrichment — every enrichment factor sits "
                            "at neutral")
                db["log"].append("MI upload: " + msg)
                for w in problems:
                    db["log"].append("MI warning: " + w)
                st.success(msg + ". 0 LLM calls.")
                st.rerun()

        kb = st.file_uploader("Product knowledge base (MD)",
                              type=["md", "txt"], key="up_kb")
        if kb and st.button("Upload KB", use_container_width=True):
            new_chunks = ingest.parse_kb(kb.getvalue().decode("utf-8"))
            if not new_chunks:
                st.error("Rejected: no [chunk: name] sections found.")
            else:
                changed = ingest.changed_chunks(db["chunks"], new_chunks)
                db["chunks"] = new_chunks
                attach, err = ingest.derive_attach(new_chunks)
                if attach:
                    import scoring
                    scoring.ATTACH.update(attach)
                affected_tags = {t for t, c in dl.TAG_TO_CHUNK.items()
                                 if c in changed}
                for t in affected_tags:
                    db["glossary_override"].pop(t, None)
                    # regenerate the shared glossary answer once per changed
                    # tag — not per provider, and not on render
                    if llm._client() is not None:
                        txt, src = llm.generate_glossary(
                            t, dl.lookup_chunk(t, new_chunks))
                        if src == "live":
                            db["glossary_override"][t] = txt
                rescore(db)
                regens = KB_REGENS.get(
                    _chunk_hash(new_chunks.get("turnaround", "")), {})
                for t, txt in regens.get("glossary", {}).items():
                    db["glossary_override"][t] = txt
                n_regen = 0
                for p in db["providers"]:
                    hit = [t for t in p.objection_tags if t in affected_tags]
                    if not hit:
                        continue
                    cached = {"handlers": {}, "pitch": None}
                    for t in hit:
                        key = f"{p.provider_id}|{t}"
                        if key in regens.get("handlers", {}):
                            cached["handlers"][t] = regens["handlers"][key]
                    cached["pitch"] = regens.get("pitches", {}).get(
                        p.provider_id)
                    regenerate_provider(p, cached)
                    n_regen += 1
                msg = (f"{len(changed)} chunk(s) changed → attach re-derived, "
                       f"tags re-mapped, Fit recomputed; handlers and pitches "
                       f"refreshed for {n_regen} affected provider(s)")
                db["log"].append("KB upload: " + msg)
                st.success(msg + (f". Attach: {err}" if err else "."))
                st.rerun()

        if db["log"]:
            st.caption("Activity")
            for line in db["log"][-4:]:
                st.caption("· " + line)

seq = "".join(f'<i style="background:{c};opacity:{o}"></i>'
              for c, o in [("#1F6F62", .85), ("#C97A1A", .85),
                           ("#8A9290", .5), ("#16233D", .6)] * 12)
html(f'<div class="track">{seq}</div>')

left, right = st.columns([1.45, 1], gap="large")


# ---------------------------------------------------------------------------
# Left — ranked list
# ---------------------------------------------------------------------------

with left:
    html('<div class="sec">Ranked call list</div>'
         '<p class="sechint">Ordered by impact. Click a provider for their '
         'objections and pitch.</p>')

    f1, f2, f3, _ = st.columns([1.3, 1.25, 1.35, 1.6])
    with f1:
        html('<span class="pill-lock">&#128274; Impact</span>')
    with f2:
        if st.button("Overall", use_container_width=True,
                     type="primary" if not st.session_state.terr_on
                     else "secondary"):
            st.session_state.terr_on = False
            st.rerun()
    with f3:
        if st.button("Territory", use_container_width=True,
                     type="primary" if st.session_state.terr_on
                     else "secondary"):
            st.session_state.terr_on = True
            st.rerun()

    if st.session_state.terr_on:
        cols = st.columns(len(ALL_TERR) + 1)
        for col, t in zip(cols, ALL_TERR):
            on = t in st.session_state.terr_sel
            if col.button(t, key=f"t_{t}", use_container_width=True,
                          type="primary" if on else "secondary"):
                st.session_state.terr_sel.symmetric_difference_update({t})
                st.rerun()

    terr = (st.session_state.terr_sel if st.session_state.terr_on
            else set(ALL_TERR))
    rows = [p for p in db["providers"] if p.territory in terr]

    if st.session_state.obj_filter:
        rows = [p for p in rows
                if st.session_state.obj_filter in p.objection_tags]
        c1, c2 = st.columns([4, 1])
        with c1:
            html(f'<div class="banner">Filtered to '
                 f'<b>{st.session_state.obj_filter}</b></div>')
        if c2.button("Clear", use_container_width=True):
            st.session_state.obj_filter = None
            st.rerun()

    li = db.get("last_ingest")
    if li:
        order = [x.provider_id for x in db["providers"]]
        rank = order.index(li["pid"]) + 1 if li["pid"] in order else None
        before = ", ".join(li["before"]) or "none"
        after = ", ".join(li["after"]) or "none"
        mint = (" · minted <b>" + ", ".join(li["minted"]) + "</b>"
                if li["minted"] else "")
        pos = f" · now ranked <b>#{rank}</b>" if rank else ""
        c1, c2 = st.columns([5, 1])
        with c1:
            html(f'<div class="banner"><b>{li["name"]}</b> re-ingested '
                 f'({li["src"]}) — objections <i>{before}</i> → '
                 f'<b>{after}</b>{mint}{pos}. Handler and pitch regenerated.'
                 f'</div>')
        if c2.button("Dismiss", use_container_width=True):
            db["last_ingest"] = None
            st.rerun()

    html(f'<p class="sechint">{len(rows)} of {len(db["providers"])} '
         'providers</p>')
    if not rows:
        st.info("No providers match. Clear a filter to see more.")

    with st.container(height=640, border=False):
        for p in rows:
            minted = {t for t in p.objection_tags
                      if t not in dl.TAG_TO_CHUNK}
            tags = "".join(
                f'<span class="tag{" tag-new" if t in minted else ""}">'
                f'{t}</span>' for t in p.objection_tags)
            with st.container(border=True):
                nm, up = st.columns([4, 1])
                with nm:
                    html(f'<div class="pname">{p.name}</div>'
                         f'<div class="pmeta">{p.hospital_system}</div>'
                         f'<div class="pspec">{p.specialty}</div>{tags}')
                with up:
                    with st.popover("＋ Note", use_container_width=True):
                        st.caption(f"Add a CRM note for {p.name} "
                                   "(newest note wins)")
                        note = st.file_uploader(
                            "Note file", type=["txt", "md"],
                            key=f"nf_{p.provider_id}",
                            label_visibility="collapsed")
                        typed = st.text_area(
                            "Or paste", height=100,
                            key=f"nt_{p.provider_id}",
                            label_visibility="collapsed",
                            placeholder="Paste meeting notes here…")
                        if st.button("Upload note", key=f"ni_{p.provider_id}",
                                     use_container_width=True):
                            text = (note.getvalue().decode("utf-8")
                                    if note else typed).strip()
                            if not text:
                                st.warning("Add a file or paste a note first.")
                            else:
                                ex, src, problems = ingest.extract_note(
                                    text, db["glossary"])
                                if ex is None:
                                    st.error(problems[0])
                                else:
                                    before_tags = list(p.objection_tags)
                                    db["glossary"] = ingest.apply_note(
                                        p, ex, db["glossary"])
                                    db["raw_notes"][p.provider_id] = text
                                    db["note_replaced"].add(p.provider_id)
                                    for t in list(db["gen_override"]):
                                        if t[0] == p.provider_id:
                                            db["gen_override"].pop(t)
                                    db["pitch_override"].pop(
                                        p.provider_id, None)
                                    rescore(db)
                                    cached = (ingest.load_test_extractions()
                                              .get(ingest.note_key(text), {})
                                              .get("generated"))
                                    regenerate_provider(p, cached)
                                    minted_now = ex["new_tags"]
                                    msg = (f"Extraction ({src}): "
                                           f"{len(ex['objection_tags'])} "
                                           f"objection(s)")
                                    if minted_now:
                                        msg += (", minted tag "
                                                + ", ".join(minted_now))
                                    if problems:
                                        msg += (" · guardrails: "
                                                + "; ".join(problems))
                                    db["log"].append(
                                        f"Note for {p.name}: {msg}")
                                    db["last_ingest"] = {
                                        "pid": p.provider_id,
                                        "name": p.name,
                                        "before": before_tags,
                                        "after": list(p.objection_tags),
                                        "src": src,
                                        "minted": minted_now,
                                    }
                                    st.rerun()

                with st.expander("Call prep"):
                    html('<div class="lab">Objection handler</div>')
                    if p.objection_tags:
                        blocks = ""
                        for tag in p.objection_tags:
                            answer, src = get_handler(p, tag)
                            ctx = p.objection_contexts.get(tag, "")
                            if (p.provider_id, tag) in db["no_kb"]:
                                stale = ('<div class="stale">No KB match — '
                                         'retrieval fell below the similarity '
                                         'floor, so no fact was supplied and '
                                         'none was invented. A rising count '
                                         'here is a signal for product '
                                         'marketing.</div>')
                            elif (p.provider_id, tag) in db["stale"]:
                                why = db["gen_reason"].get(
                                    (p.provider_id, tag)) or []
                                detail = ("; ".join(why) if why
                                          else "no API key set")
                                stale = ('<div class="stale">Not regenerated '
                                         f'({detail}) — showing nothing rather '
                                         'than text written for the previous '
                                         'note.</div>')
                            else:
                                stale = ""
                            blocks += (f'<div class="obj">'
                                       f'<span class="objtag">{tag}</span>'
                                       f'<div class="objctx">\u201c{ctx}\u201d'
                                       f'</div>'
                                       f'<div class="objans">{answer}</div>'
                                       f'{stale}</div>')
                        html(f'<div class="objwrap">{blocks}</div>')
                    else:
                        html('<div class="objwrap"><div class="obj">'
                             '<div class="objctx">No objection logged in '
                             'this provider\'s notes. Enrichment factors sit '
                             'at neutral — the pitch below is the '
                             'specialty-level version.</div></div></div>')

                    pitch, psrc = get_pitch(p)
                    if ((p.provider_id, "__pitch__") in db["stale"]
                            and psrc != "generated"):
                        why = db["gen_reason"].get(
                            (p.provider_id, "__pitch__")) or []
                        detail = ("draft rejected: " + "; ".join(why)) if why \
                            else "no API key set"
                        stale_p = ('<div class="stale">Not regenerated '
                                   f'({detail}).</div>')
                    else:
                        stale_p = ""
                    html('<div class="lab">30-second pitch</div>'
                         f'<div class="pitch">{pitch if pitch else "No pitch shown — this provider's note was replaced, so the earlier pitch describes a superseded conversation and is withheld rather than displayed."}</div>'
                         f'{stale_p}'
                         f'<div class="prov">Pitch source: <b>{psrc}</b>'
                         f'{" · written from the current note" if psrc == "generated" else " · seed text, not regenerated"}'
                         f'</div>')

                    with st.popover("Source note", use_container_width=True):
                        st.text(db["raw_notes"].get(
                            p.provider_id, "No note on file."))


# ---------------------------------------------------------------------------
# Right — glossary
# ---------------------------------------------------------------------------

with right:
    html('<div class="sec">Objection glossary</div>'
         '<p class="sechint">Most raised first · select to filter the '
         'list</p>')

    counts = dict(objection_counts(db["providers"]))
    # include minted tags that exist in the glossary even before anyone
    # carries them (shouldn't happen, but cheap to be safe)
    with st.container(height=640, border=False):
        for tag, who in counts.items():
            answer, _ = get_glossary(tag)
            if not answer:
                answer = ("No published Tempus capability maps to this tag "
                          "yet — a rising count here is a signal for product "
                          "marketing, not a prompt to improvise an answer.")
            active = st.session_state.obj_filter == tag
            new_badge = ("" if tag in dl.TAG_TO_CHUNK
                         else ' <span class="tag tag-new">new</span>')
            with st.container(border=True):
                html(f'<span class="gtag">{tag}</span>{new_badge} '
                     f'<span class="gcount">· {len(who)} '
                     f'{"providers" if len(who) > 1 else "provider"}</span>'
                     f'<div class="gwho">{" · ".join(who)}</div>'
                     f'<div class="gans">{answer}</div>')
                if st.button("Clear filter" if active else "Filter list",
                             key=f"b_{tag}", use_container_width=True,
                             type="primary" if active else "secondary"):
                    st.session_state.obj_filter = None if active else tag
                    st.rerun()

html('<p class="prov" style="text-align:center;margin-top:22px">'
     'Prototype · provider names and CRM notes are fictional mock data. '
     'Product facts are paraphrased from public Tempus sources and would '
     'need medical-affairs review before shipping rep-facing language.</p>')
