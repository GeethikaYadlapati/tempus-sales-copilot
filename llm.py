"""
Generation layer — the only place a model writes text a human reads.

Three prompts:
  1. Provider Objection Handler  — per provider, per tag
  2. Objection Glossary          — per tag (shared, generic)
  3. Elevator Pitch              — per provider

Each ships with a reviewed fallback so the demo never breaks without an API
key. Set GROQ_API_KEY (free, console.groq.com/keys) to run live.

Note what is NOT here: no scoring, no ranking, no retrieval. The model reads
and writes; it does not rank.
"""

import json
import os
import re

# Free tier, Google AI Studio. Fast model for high-volume classification;
# stronger model for the low-volume text a rep reads aloud.
# Groq free tier — open-weight models, no card, high daily limits.
#
# The split is deliberate. Extraction is short, structured and runs on EVERY
# note, so latency compounds across a territory: it gets the small fast model.
# Handlers and pitches are low-volume and read aloud to a physician, so they
# can afford the larger one.
#
# Model names change; run `python3 list_models.py` to see what your key can
# reach, then override with GROQ_MODEL_FAST / GROQ_MODEL_STRONG (or GROQ_MODEL
# to force both to the same one).
_BOTH = os.environ.get("GROQ_MODEL")
MODEL_FAST = _BOTH or os.environ.get("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
MODEL_STRONG = _BOTH or os.environ.get("GROQ_MODEL_STRONG",
                                       "openai/gpt-oss-120b")

SHARED_GUARDRAILS = """
Use only facts supplied in KB_FACT. Never state a number, test name, turnaround
time or capability that does not appear there. Never supply a number the notes
do not state. Never infer a feeling ("you seem frustrated"). Never name the
competitor. Never mention internal scoring or ranking.
Return only JSON, no markdown, no preamble.
""".strip()




# --------------------------------------------------------------------------
# Post-generation validation — the deck's "regenerate on fail" step.
# Deterministic checks; a draft that fails never reaches the rep.
# --------------------------------------------------------------------------

COMPETITORS = ["guardant", "foundationone", "foundation one", "caris",
               "neogenomics"]
COMMIT_WORDS = ["demo", "trial run", "pricing", "quote", "discount",
                "timeline", "contract"]
MARKETING = ["revolutionary", "cutting-edge", "game-changing", "world-class",
             "best-in-class", "state-of-the-art", "industry-leading"]

# Claims a model reaches for when it has no fact to cite. None of these are
# in the KB, so none may appear unless the supplied fact contains them.
# Numbers and competitor names were already checked; this closes the softer
# gap — regulatory, availability, guideline and efficacy assertions.
UNSOURCED_CLAIMS = [
    "guideline", "fda", "approved", "clearance", "reimburs", "covered by",
    "improve outcomes", "improves outcomes", "better outcomes",
    "available across", "major oncology centers", "nationwide",
    "industry standard", "gold standard", "proven to", "clinically proven",
    "trusted by", "leading provider",
]


def _unsourced(text, *sources):
    """Claims that appear in the draft but in none of the supplied inputs."""
    low = (text or "").lower()
    src = " ".join(s.lower() for s in sources if s)
    return [c for c in UNSOURCED_CLAIMS if c in low and c not in src]


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def validate_handler(text, kb_fact, context):
    """Every digit must trace to the supplied fact or the provider's own
    context; no competitor names; no dates; no inferred feelings."""
    problems = []
    allowed = _numbers(kb_fact) | _numbers(context)
    for n in _numbers(text):
        if n not in allowed:
            problems.append(f"number {n} not in supplied inputs")
    low = (text or "").lower()
    for c in COMPETITORS:
        if c in low:
            problems.append(f"names competitor '{c}'")
    if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
                 r"[a-z]*\.? \d|\d{4}-\d{2}-\d{2}|\d+ (?:jan|feb|mar)", low):
        problems.append("cites a date")
    if "you seem" in low or "you must be" in low:
        problems.append("infers a feeling")
    for c in _unsourced(text, kb_fact, context):
        problems.append(f"unsourced claim ('{c}') — not in the supplied fact")
    if not (text or "").strip():
        problems.append("empty")
    return problems


def validate_pitch(text, first_name, kb_fact="", crm_detail=""):
    """60-90 words, opens with the first name, numbers as words (no digits),
    full sentence close, no commitments, no marketing adjectives."""
    problems = []
    t = (text or "").strip()
    words = len(t.split())
    if not 60 <= words <= 90:
        problems.append(f"{words} words, outside 60-90")
    if not t.startswith(first_name):
        problems.append("does not open with first name")
    if re.search(r"\d", t):
        problems.append("contains digits — numbers must be spoken words")
    if t and t[-1] not in ".?!":
        problems.append("ends mid-sentence")
    low = t.lower()
    for w in COMMIT_WORDS:
        if w in low:
            problems.append(f"commits the rep ('{w}')")
    for w in MARKETING:
        if w in low:
            problems.append(f"marketing adjective ('{w}')")
    for c in _unsourced(t, kb_fact, crm_detail):
        problems.append(f"unsourced claim ('{c}') — not in the supplied fact")
    for c in COMPETITORS:
        if c in low:
            problems.append(f"names competitor '{c}'")
    return problems


LAST_ERROR = {"msg": ""}   # surfaced in the UI instead of a generic failure
LAST_GEN = {"problems": []}  # why the most recent draft was rejected


def _client():
    """Groq free tier — open-weight models, no card required.

    Set GROQ_API_KEY from console.groq.com/keys. Returns None when no key is
    configured, which is the offline path: verified cached extractions and
    reviewed templates take over.
    """
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except ImportError:
        return None


def call_model(system: str, user: str, model: str = MODEL_STRONG,
               max_tokens: int = 800):
    """One place every model call goes through. Returns parsed JSON or None.

    json_object response format is requested where supported; the prompt also
    demands JSON, and the parse is defensive, because open models are looser
    about format than a native JSON mode.
    """
    client = _client()
    if client is None:
        return None

    def _attempt(strict: bool, effort: str = "low"):
        kwargs = dict(
            model=model,
            # Reasoning models spend tokens thinking before they emit content.
            # Too small a budget returns an empty message, so give headroom.
            max_tokens=max(max_tokens, 2048),
            temperature=0.4,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        if strict:
            kwargs["response_format"] = {"type": "json_object"}
        if effort:
            kwargs["reasoning_effort"] = effort
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            # older/other models reject reasoning_effort
            if effort and "reasoning_effort" in str(e):
                kwargs.pop("reasoning_effort")
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
        msg = resp.choices[0].message
        return (getattr(msg, "content", None) or "")

    try:
        # Strict JSON mode first. Reasoning-style open models sometimes emit
        # tokens the strict validator rejects, so fall back to prompt-only
        # JSON and parse defensively rather than losing the call.
        try:
            text = _attempt(strict=True)
        except Exception as inner:
            if "json_validate_failed" in str(inner) or "json" in str(
                    inner).lower() and "400" in str(inner):
                text = _attempt(strict=False)
            else:
                raise
        if not text.strip():
            # reasoning consumed the budget — retry once with more headroom
            try:
                text = _attempt(strict=False, effort="low")
            except Exception:
                text = ""
        if not text.strip():
            LAST_ERROR["msg"] = (
                "model returned no text — reasoning likely consumed the token "
                "budget. Try a non-reasoning model: "
                "export GROQ_MODEL_STRONG=qwen/qwen3.8-27b")
            return None
        cleaned = re.sub(r"```json|```", "", text).strip()
        # open models sometimes wrap or prepend prose; take the JSON object
        if not cleaned.startswith("{"):
            m = re.search(r"\{.*\}", cleaned, re.S)
            if m:
                cleaned = m.group(0)
        return json.loads(cleaned)
    except Exception as e:
        msg = str(e)
        if "rate_limit" in msg.lower() or "429" in msg:
            LAST_ERROR["msg"] = (
                "Groq rate limit reached. Wait a moment and retry — cached "
                "extractions and reviewed templates still work meanwhile.")
        elif "model" in msg.lower() and ("not found" in msg.lower()
                                         or "404" in msg
                                         or "decommission" in msg.lower()):
            LAST_ERROR["msg"] = ("model not available to this key — run "
                                 "`python3 list_models.py` and set "
                                 "GROQ_MODEL_FAST / GROQ_MODEL_STRONG")
        elif "401" in msg or "invalid_api_key" in msg.lower():
            LAST_ERROR["msg"] = "API key rejected — check GROQ_API_KEY"
        else:
            LAST_ERROR["msg"] = f"{type(e).__name__}: {msg[:200]}"
        return None


# kept as an alias so existing call sites read unchanged
_call = call_model


# --------------------------------------------------------------------------
# 1. Provider Objection Handler
# --------------------------------------------------------------------------

HANDLER_SYSTEM = f"""
You write objection handlers for Tempus sales reps. The handler answers a
concern the physician has ALREADY raised, logged in their CRM notes.

FORM
- Two or three sentences.
- Open by mirroring their stated concern in a single clause, not a retelling.
- Then give the Tempus fact that answers it.
- Do not cite dates.

{SHARED_GUARDRAILS}
Shape: {{"handler": "..."}}
""".strip()


def generate_handler(provider, tag, kb_fact, attempts=3):
    context = provider.objection_contexts.get(tag, "")
    if _client() is None:
        LAST_GEN["problems"] = ["no API key — GROQ_API_KEY not set in this "
                                "process"]
    elif not kb_fact:
        LAST_GEN["problems"] = ["no KB fact above the similarity floor"]
    else:
        LAST_GEN["problems"] = []
    if kb_fact:
        payload = (
            f"PHYSICIAN: {provider.name}, {provider.specialty}\n"
            f"OBJECTION_TAG: {tag}\n"
            f"OBJECTION_CONTEXT: {context}\n"
            f"KB_FACT: {kb_fact}"
        )
        for i in range(attempts):
            out = _call(HANDLER_SYSTEM, payload)
            if not (out and out.get("handler")):
                break
            draft = out["handler"]
            problems = validate_handler(draft, kb_fact, context)
            LAST_GEN["problems"] = problems
            if not problems:
                return draft, "live"
            payload += ("\n\nYour previous draft failed validation: "
                        + "; ".join(problems) + ". Rewrite and fix this.")
        # no valid live draft
    fb = FALLBACK_HANDLERS.get((provider.provider_id, tag), "")
    if fb:
        return fb, "template"
    if not kb_fact:
        return ("No published Tempus capability answers this concern yet — "
                "flagged. Acknowledge it honestly rather than improvising "
                "a fact."), "no-kb-answer"
    return "", "unavailable"


# --------------------------------------------------------------------------
# 2. Objection Glossary — generic, shared across everyone with the tag
# --------------------------------------------------------------------------

GLOSSARY_SYSTEM = f"""
You write entries for a Tempus rep's objection glossary. A rep reads this
mid-call when a concern comes up unexpectedly.

FORM
- One or two sentences, written TO THE REP as strategy.
- Cite only tag-level KB facts.
- No physician name, no personal history, no specific account details.
- This answer is reused by every provider carrying this tag.

{SHARED_GUARDRAILS}
Shape: {{"answer": "..."}}
""".strip()


def generate_glossary(tag, kb_chunk):
    if kb_chunk:
        out = _call(
            GLOSSARY_SYSTEM,
            f"OBJECTION_TAG: {tag}\nKB_FACT: {kb_chunk}",
            max_tokens=400,
        )
        if out and out.get("answer"):
            return out["answer"], "live"
    return FALLBACK_GLOSSARY.get(tag, ""), "template"


# --------------------------------------------------------------------------
# 3. Elevator Pitch
# --------------------------------------------------------------------------

PITCH_SYSTEM = f"""
You write 30-second opening scripts for Tempus sales reps. The rep reads your
output aloud at the start of a meeting.

PROVIDER CONTEXT
SPECIALTY constrains which tests you may name. ROLE sets the register:
physician means one patient's treatment decision; administrator means network
adoption and board-facing evidence. Never state a treating physician's own
patient volume or testing rate — it reads as an audit of their practice.

FORM
- 60-90 words. Never end mid-sentence.
- Open with their first name. One fact only.
- Numbers as spoken words: "about seven days", not "~7d".
- No sentence longer than 20 words.
- Close by offering a short conversation. Commit the rep to nothing — no
  demos, trials, pricing, quotes or timelines.
- No greeting beyond the name, no sign-off, no marketing adjectives.

NEVER claim regulatory status, guideline inclusion, reimbursement, market
availability or improved outcomes unless those exact facts appear in KB_FACT.
If KB_FACT is thin, say less — a shorter, plainer opener is correct. Inventing
a capability to fill space is the worst possible failure here.

{SHARED_GUARDRAILS}
Shape: {{"pitch": "..."}}
""".strip()


def generate_pitch(provider, kb_fact, attempts=3):
    first_name = provider.name.replace("Dr. ", "").split()[0]
    payload = (
        f"FIRST_NAME: {first_name}\n"
        f"SPECIALTY: {provider.specialty}\n"
        f"ROLE: {provider.role_type}\n"
        f"CRM_DETAIL: {provider.entire_note_context}\n"
        f"KB_FACT: {kb_fact}"
    )
    if _client() is None:
        LAST_GEN["problems"] = ["no API key — GROQ_API_KEY not set in this "
                                "process"]
    else:
        LAST_GEN["problems"] = []
        for i in range(attempts):
            out = _call(PITCH_SYSTEM, payload)
            if not (out and out.get("pitch")):
                LAST_GEN["problems"] = (LAST_GEN["problems"]
                                        or [LAST_ERROR.get("msg")
                                            or "model returned no pitch"])
                break
            draft = out["pitch"]
            problems = validate_pitch(draft, first_name, kb_fact,
                                      provider.entire_note_context)
            LAST_GEN["problems"] = problems
            if not problems:
                return draft, "live"
            payload += ("\n\nYour previous draft failed validation: "
                        + "; ".join(problems) + ". Rewrite and fix this.")
    return FALLBACK_PITCHES.get(provider.provider_id, ""), "template"


# --------------------------------------------------------------------------
# Reviewed fallbacks. Every one was read for tone and factual grounding
# before being treated as a default.
# --------------------------------------------------------------------------

FALLBACK_HANDLERS = {
    ("P01", "Outcomes"): "Two of your affiliated practices are testing well below your network average. Our Immune Profile Score has shown a measurable real-world survival difference on checkpoint inhibitors beyond what PD-L1 or TMB show alone, which is the kind of evidence that maps onto value-based contract reporting.",
    ("P02", "Incumbent"): "You are not looking to replace your current workflow, and this is not a pitch to. The gap is patients where a repeat biopsy carries real risk — Tempus xF is a blood draw with results typically back in about seven days, so it sits alongside what you already do.",
    ("P03", "Referral"): "An outside genetics referral is adding three to four weeks before you can finalise a PARP-inhibitor decision. Tempus HRD is derived from data an xT order already generates, so it attaches with no new tissue and no separate referral.",
    ("P04", "Turnaround"): "You asked for a real number rather than a marketing one: Tempus xF returns results typically within seven days of specimen retrieval, and xF+ within seven to nine. That is roughly a third of the twenty-one day wait you described.",
    ("P05", "Cost"): "This does not have to land as a new standing expense. Financial assistance is income-based with decisions typically returned at submission, so cost is handled per patient — and a hotspot PCR panel is likely missing alterations outside its gene list.",
    ("P06", "Outcomes"): "You asked for outcomes rather than gene counts. In our real-world data, IPS-High patients showed higher overall survival on checkpoint-inhibitor regimens than IPS-Low, which is a more precise signal than PD-L1 or TMB alone.",
    ("P07", "Actionability"): "Pancreatic often feels like there is little to test for. PurIST classifies a tumour as basal or classical from RNA expression, and that classification has been linked to which patients see more consistent benefit from FOLFIRINOX in first line.",
    ("P08", "Tissue"): "You mentioned a quantity-not-sufficient result after a multi-week wait. With Tempus, if submitted tissue is insufficient the order converts automatically to xF liquid biopsy — no second physician order and no repeat biopsy conversation.",
    ("P09", "Incumbent"): "Your current panel covers the broad-profiling need. Where Tempus differs is trial matching — roughly 96% of patients matched to at least one trial when clinical data was combined with our NGS results, which matters most for the rare-mutation cases you described.",
    ("P10", "Turnaround"): "You told me results usually arrive after you have already started empiric chemo. Tempus xF is a blood draw returning results typically within seven days of specimen retrieval — built to land before the first treatment decision rather than after it.",
    ("P05", "Actionability"): "A hotspot PCR panel only reports what it was designed to look for. Tempus xT covers 648 genes with matched-normal calling, so alterations outside a small hotspot list are actually seen rather than assumed absent.",
    ("P06", "Cost"): "Cost is a fair question for a board. Financial assistance is income-based with decisions typically returned at submission, so adoption is handled per patient rather than as a standing line item across the network.",
    ("P08", "Turnaround"): "The wait made the quantity-not-sufficient result worse, because the time was lost before you knew there was a problem. Auto-conversion to xF means an insufficient sample does not restart the clock — the blood draw returns typically within seven days.",
    ("P09", "Actionability"): "That is exactly where trial matching earns its place. Roughly 96% of patients matched to at least one trial when clinical data was combined with Tempus NGS results, which matters most when local options are thin.",
}

FALLBACK_GLOSSARY = {
    "Turnaround": "Anchor to the wait they described. xF returns typically within 7 days of specimen retrieval, xF+ within 7 to 9, xT Heme within 9 days of receipt.",
    "Tissue": "xT CDx converts automatically to xF liquid biopsy when tissue is insufficient — no second order and no repeat biopsy conversation with the patient.",
    "Referral": "HRD, IPS and TO are derived from data an existing xT/xR order already generates. No new tissue, no separate referral step.",
    "Outcomes": "Lead with IPS rather than panel size. IPS-High patients showed higher real-world survival on checkpoint inhibitors than IPS-Low.",
    "Cost": "Frame as patient access, not a practice expense. Financial assistance is income-based with decisions typically returned at submission.",
    "Actionability": "Point at the algorithmic layer. PurIST subtypes pancreatic tumours basal or classical from RNA and informs first-line regimen choice.",
    "Incumbent": "There is no pain to counter here, so do not pitch displacement. Concede the broad-panel need is met, then find the slice the incumbent leaves open — trial matching, or liquid biopsy where re-biopsy is risky.",
}

FALLBACK_PITCHES = {
    "P01": "Elena, across your network only about fifty-eight percent of eligible patients are getting comprehensive biomarker testing, and a couple of your practices sit well below that. Our Immune Profile Score has shown a real survival difference on checkpoint inhibitors beyond PD-L1 or TMB alone. That is the kind of evidence your value-based contracts are asking for. Worth a short conversation about where the gap sits?",
    "P02": "Marcus, I know your NSCLC workflow is working and I am not here to change it. For the patients where a repeat biopsy carries real risk, Tempus xF is a blood draw with results typically back in about seven days. It sits alongside what you already order rather than replacing it. Worth a short conversation about that specific subset?",
    "P03": "Priya, you mentioned outside genetics referrals add three to four weeks before you can finalise a PARP decision. Tempus HRD comes from data an order you are likely already placing generates, so there is no new tissue and no separate referral. It removes that step entirely. Worth a short conversation about how that fits your ovarian patients?",
    "P04": "Sam, last time we spoke your patient had already started second-line chemo before that twenty-one day result came back. Tempus xF is a blood draw with results typically in about seven days. That is roughly a third of the wait you described. Worth a short conversation about how that maps onto your treatment decision window?",
    "P05": "Grace, I know any new vendor goes through your finance committee first. Tempus has an income-based financial assistance programme with decisions typically returned at submission, so this is handled per patient rather than as a standing cost. And it is a far broader panel than a hotspot approach. Worth a short conversation before you take it to them?",
    "P06": "Aaron, you asked for outcomes data rather than gene counts. In our real-world data, IPS-High patients showed higher overall survival on checkpoint-inhibitor regimens than IPS-Low patients. That is the evidence I would bring to your board. Worth a short conversation about what that would look like at network level?",
    "P07": "Fatima, I know pancreatic often feels like there is not much to test for. PurIST classifies a tumour as basal or classical from RNA expression, and that has been linked to which patients benefit more consistently from FOLFIRINOX. It informs a first-line decision you are already making. Worth five minutes on how that fits?",
    "P08": "Robert, you mentioned getting a quantity-not-sufficient result after a multi-week wait. With Tempus, if the tissue comes back insufficient the order converts automatically to a blood-based test. No second order, no re-biopsy conversation, no restarting the clock. Worth a short conversation about how that fits your prostate workflow?",
    "P09": "Wei, I know your current panel covers the broad-profiling need. Where Tempus is different is trial matching — about ninety-six percent of patients matched to at least one trial when clinical data was combined with our sequencing results. For rare-mutation sarcoma that could open options beyond what is locally available. Worth a short conversation?",
    "P10": "Linda, you told me results usually land after you have already started empiric chemo, which defeats the purpose. Tempus xF is a blood draw with results typically back in about seven days. It is built to arrive before your first treatment decision rather than after it. Worth a short conversation about how that would fit your workflow?",
}
