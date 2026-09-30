"""Streamlit interface — Personalized Research Paper Triage.

    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import random
from datetime import datetime, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from triage import config
from triage.embeddings import sbert_available
from triage import export
from triage.explain import explain_many, llm_available
from triage.models import InterestProfile, TriageResult
from triage.pipeline import TriageEngine, TriageRun
from triage.profile import parse_list
from triage.ranking import Cutoffs, estimated_minutes, read_budget, summarize
from triage.demo import DEMO_NAME, is_demo, seed_demo
from triage.evaluate import MIN_LABELS
from triage.relevance import FEATURES, PRIOR_WEIGHTS
from triage.sources import ARXIV_CATEGORIES, SourceError
from triage.transfer import export_profile, load_json, parse_papers, restore_profile

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="Paper Triage", page_icon=":material/menu_book:", layout="wide")

PAGE_SIZE = 20
LABEL_TEXT = {"READ": "Read", "SKIM": "Skim", "SKIP": "Skip"}
LABEL_HELP = {"READ": "Highly relevant", "SKIM": "Useful but peripheral", "SKIP": "Low relevance"}
FEATURE_NAMES = {
    "semantic": "Similarity to your project",
    "focus": "Similarity to current focus",
    "keyword_sem": "Closest keyword (semantic)",
    "keyword_lex": "Keyword matches (exact)",
    "recency": "Recency",
    "avoid": "Excluded-topic penalty",
    "feedback": "Similarity to your feedback",
}
SOURCES = {
    "arxiv_new": "arXiv — new listings",
    "semantic_scholar": "Semantic Scholar — keyword search",
    "arxiv": "arXiv — keyword search",
    "both": "arXiv new listings + Semantic Scholar",
    "sample": "Sample collection (offline)",
}

st.markdown(
    """
<style>
.block-container { max-width: 1080px; padding-top: 2.2rem; }
h1 { font-weight: 600 !important; letter-spacing: -0.01em; }
.subtle { color: #5f6773; font-size: .9rem; }
.brand { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 600; margin-bottom: 0; }
.chip { display: inline-block; font-size: .7rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
        padding: .08rem .45rem; border-radius: 3px; margin-right: .5rem; vertical-align: 2px; }
.chip.READ { color: #17603a; background: #e6f2eb; }
.chip.SKIM { color: #8a5a00; background: #fbf1dc; }
.chip.SKIP { color: #5f6773; background: #eef0f3; }
.chip.NEW  { color: #1F3A5F; background: #e8eef6; }
.paper-title { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.12rem; font-weight: 600; line-height: 1.35;
               color: #1b1f24 !important; text-decoration: none; }
.paper-title:hover { text-decoration: underline; }
.paper-meta { color: #5f6773; font-size: .84rem; margin: .2rem 0 .45rem 0; }
.paper-reason { color: #333a44; font-size: .92rem; line-height: 1.45; }
.paper-note { color: #8a5a00; font-size: .8rem; margin-top: .25rem; }
.card-top { display: flex; align-items: center; gap: .1rem; }
.status { color: #5f6773; font-size: .78rem; margin-left: auto; margin-right: .8rem; }
.score { color: #8a919c; font-size: .8rem; font-variant-numeric: tabular-nums; cursor: help; }
.status:empty + .score { margin-left: auto; }
.card-title { margin-top: .4rem; }
div[data-testid="stExpander"] details summary p { font-size: .88rem; }
.page-title { font-family: 'Source Serif 4', Georgia, serif; font-size: 2rem; font-weight: 600; letter-spacing: -.01em;
              line-height: 1.2; margin: .2rem 0 .15rem 0; color: #1b1f24; }
.pill { font-family: Inter, sans-serif; font-size: .68rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
        color: #5f6773; border: 1px solid #d5d9de; border-radius: 999px; padding: .1rem .5rem; margin-left: .7rem;
        vertical-align: middle; cursor: help; }
.summary { color: #5f6773; font-size: .88rem; margin: 0 0 1rem 0; font-variant-numeric: tabular-nums; }
.section { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.15rem; font-weight: 600; color: #1b1f24;
           margin: 1.4rem 0 .5rem 0; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem 1.5rem; margin: .4rem 0 .8rem 0; }
.stat { border-left: 2px solid #e3e6ea; padding-left: .8rem; }
.stat .l { color: #5f6773; font-size: .8rem; }
.stat .v { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.6rem; font-weight: 600; line-height: 1.25; color: #1b1f24; }
.stat .s { color: #8a919c; font-size: .76rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- resources


@st.cache_resource
def get_engine() -> TriageEngine:
    return TriageEngine()


engine = get_engine()
store = engine.store

# A demo profile (sample papers + simulated labels/feedback) is created once, on
# first launch. Deleting it is respected; it can be recreated from Settings.
if not store.get_setting(0, "demo_seeded", False):
    if not store.list_profiles():
        with st.spinner("Preparing the demo profile…"):
            seed_demo(engine)
    store.set_setting(0, "demo_seeded", True)


@st.cache_resource(max_entries=6, show_spinner=False)  # a spinner above the tabs would reset them
def cached_run(
    profile_json: str, pool_key: str, model_version: str, read: float, skim: float, budget: bool, group: bool, good: str
) -> TriageRun:
    prof = InterestProfile(**json.loads(profile_json))
    return engine.run(prof, Cutoffs(read=read, skim=skim), respect_budget=budget, group=group)


def pool_key(p: InterestProfile) -> str:
    ids = store.pool_ids(p.id)
    rev = store.get_setting(p.id, "content_rev", "")  # bumped when full text is added
    return f"{len(ids)}:{hashlib.sha1('|'.join(ids).encode()).hexdigest()[:12]}:{rev}"


def profile_json(p: InterestProfile) -> str:
    return json.dumps(p.__dict__, sort_keys=True)


def use_llm(profile: InterestProfile | None) -> bool:
    return bool(not config.BROWSER_MODE and profile and store.get_setting(profile.id, "use_llm", False) and llm_available())


@st.cache_data(max_entries=16, show_spinner=False, persist="disk")
def cached_eval(profile_json: str, profile_id: int, version: str, read: float, skim: float, pool: str, good: str):
    """Evaluation results survive reruns, tab switches and page refreshes; recomputed only when inputs change."""
    prof = InterestProfile(**json.loads(profile_json))
    prof.id = profile_id
    return engine.evaluate_all(prof, Cutoffs(read=read, skim=skim), good=good)


def get_run(profile: InterestProfile) -> TriageRun | None:
    cut = engine.cutoffs(profile)
    budget = store.get_setting(profile.id, "budget", True)
    fv = store.feedback_version(profile.id)
    mv_key = f"model_version_{profile.id}"
    if store.get_setting(profile.id, "auto_update", True) or mv_key not in st.session_state:
        st.session_state[mv_key] = fv
    try:
        return cached_run(
            profile_json(profile), pool_key(profile), st.session_state[mv_key], cut.read, cut.skim, budget,
            store.get_setting(profile.id, "group_similar", True), store.get_setting(profile.id, "good_mode", "read"),
        )
    except ValueError as e:
        st.info(str(e))
        return None


def user_state(profile: InterestProfile) -> dict[str, dict]:
    """Latest per-paper feedback state from the log."""
    state: dict[str, dict] = {}
    for f in store.get_feedback(profile.id):
        s = state.setdefault(f["paper_id"], {})
        a = f["action"]
        if a in ("useful", "not_useful"):
            s["vote"] = a
        elif a == "correct":
            s["corrected"] = f["value"]
        elif a == "dismiss":
            s["dismissed"] = True
        elif a == "undismiss":
            s["dismissed"] = False
    return state


def log(profile: InterestProfile, r: TriageResult, action: str, value: str | None = None) -> None:
    store.log_feedback(profile.id, r.paper.id, action, value, r.label, r.score)


def esc(s: str) -> str:
    return html.escape(s or "")


def chip(label: str) -> str:
    return f"<span class='chip {label}'>{LABEL_TEXT[label]}</span>"


# ------------------------------------------------------------------ sidebar


def sidebar() -> InterestProfile | None:
    sb = st.sidebar
    sb.markdown("<p class='brand'>Paper Triage</p>", unsafe_allow_html=True)
    if config.BROWSER_MODE:
        sb.caption("Saved in this browser · No account needed")

    profiles = store.list_profiles()
    names = [p.name for p in profiles]
    NEW = "New profile…"
    options = names + [NEW]
    if "goto_profile" in st.session_state:  # set by Save / example buttons, applied before the widget renders
        st.session_state["profile_select"] = st.session_state.pop("goto_profile")
    if st.session_state.get("profile_select") not in options:
        from_url = st.query_params.get("profile")
        st.session_state["profile_select"] = from_url if from_url in names else (names[0] if names else NEW)
    choice = sb.selectbox("Profile", options, key="profile_select")
    if choice != NEW:
        st.query_params["profile"] = choice
    elif "profile" in st.query_params:
        del st.query_params["profile"]
    profile = store.get_profile(choice) if choice != NEW else None

    with sb.expander("Research interests", expanded=profile is None):
        profile_form(profile, names)
        if profile is None and DEMO_NAME not in names:
            if st.button("Open demo profile", width="stretch", type="tertiary"):
                demo = seed_demo(engine)
                st.session_state["goto_profile"] = demo.name
                st.rerun()

    with sb.expander("Restore profile backup"):
        backup = st.file_uploader("Profile backup (.json)", type=["json"], key="profile_backup")
        if st.button("Restore as a new profile", disabled=backup is None, width="stretch"):
            try:
                restored = restore_profile(store, backup.getvalue())
            except ValueError as e:
                st.error(str(e))
            else:
                st.session_state["goto_profile"] = restored.name
                st.rerun()

    if profile is not None:
        start_visit(profile)
        auto_refresh(profile)
        with sb.expander("Add papers", expanded=not store.pool_ids(profile.id)):
            fetch_form(profile)
        with sb.expander("Seed papers"):
            seed_form(profile)

    return profile


def profile_form(profile: InterestProfile | None, names: list[str]) -> None:
    base = profile or InterestProfile(name="")
    with st.form(f"profile_{base.id or 'new'}", border=False):
        name = st.text_input("Name", value=base.name, placeholder="e.g. Thesis — RAG evaluation")
        desc = st.text_area("Project description", value=base.description, height=120,
                            placeholder="A few sentences about your current research.")
        kws = st.text_area("Keywords", value=", ".join(base.keywords), height=68,
                           help="Comma-separated. Matched exactly and used as the search query.")
        focus = st.text_input("Current focus", value=base.focus, help="Weighted most heavily.")
        avoid = st.text_input("Exclude topics", value=", ".join(base.avoid))
        hours = st.number_input("Reading time per week (hours)", 0.5, 40.0, float(base.hours_per_week), 0.5)
        if st.form_submit_button("Save", type="primary", width="stretch"):
            new = InterestProfile(
                name=name.strip(), description=desc.strip(), keywords=parse_list(kws),
                focus=focus.strip(), avoid=parse_list(avoid), hours_per_week=float(hours),
            )
            if not new.name:
                st.error("Please enter a name.")
            elif new.is_empty():
                st.error("Add a description, keywords, or a current focus.")
            elif new.name in names and (profile is None or new.name != profile.name):
                st.error("A profile with that name already exists.")
            else:
                saved = store.save_profile(new)
                if profile is not None and saved.id != profile.id:  # renamed
                    _move_profile(profile.id, saved.id)
                st.session_state["goto_profile"] = new.name
                st.rerun()


def _move_profile(old_id: int, new_id: int) -> None:
    with store._conn() as c:
        for table in ("profile_papers", "labels", "feedback", "settings"):
            c.execute(f"UPDATE OR IGNORE {table} SET profile_id=? WHERE profile_id=?", (new_id, old_id))
        c.execute("DELETE FROM profiles WHERE id=?", (old_id,))


def start_visit(profile: InterestProfile) -> None:
    """Remember the previous visit (for "New since your last visit"), once per browser session."""
    key = f"prev_visit_{profile.id}"
    if key not in st.session_state:
        st.session_state[key] = store.get_setting(profile.id, "last_visit", None)
        store.set_setting(profile.id, "last_visit", datetime.now(timezone.utc).isoformat())


def auto_refresh(profile: InterestProfile) -> None:
    """Once per session: fetch today's arXiv listings if the last fetch is older than ~a day."""
    if config.BROWSER_MODE:
        return
    key = f"refreshed_{profile.id}"
    if key in st.session_state:
        if st.session_state[key]:
            st.sidebar.caption(st.session_state[key])
        return
    msg = ""
    with st.sidebar:
        with st.spinner("Checking for new papers…"):
            try:
                res = engine.refresh_if_due(profile)
            except Exception as e:  # never let a background refresh break the page
                res = {"ok": False, "message": str(e)}
    if res:
        msg = f"{res['added']} new papers today" if res["ok"] else "Daily update failed — see Settings › Fetch history"
    st.session_state[key] = msg
    if msg:
        st.sidebar.caption(msg)


def seed_form(profile: InterestProfile) -> None:
    if config.BROWSER_MODE:
        st.caption("Choose essential papers from your collection to guide the ranking.")
        papers = {p.id: p for p in [*engine.pool(profile), *engine.seed_papers(profile)]}
        with st.form(f"browser_seeds_{profile.id}"):
            selected = st.multiselect("Essential papers", list(papers), default=store.seed_ids(profile.id),
                                      format_func=lambda pid: papers[pid].title)
            if st.form_submit_button("Save seed papers", width="stretch"):
                for pid in set(store.seed_ids(profile.id)) - set(selected):
                    store.remove_seed(profile.id, pid)
                store.add_seeds(profile.id, selected)
                st.rerun()
        return
    st.caption("Papers you consider essential. arXiv IDs, links or DOIs, or a BibTeX file.")
    with st.form(f"seeds_{profile.id}", border=False, clear_on_submit=True):
        text = st.text_area("Identifiers", height=80, placeholder="2005.11401\n10.18653/v1/2020.emnlp-main.550",
                            label_visibility="collapsed")
        bib = st.file_uploader("BibTeX", type=["bib", "txt"], label_visibility="collapsed")
        if st.form_submit_button("Add", width="stretch"):
            bib_text = bib.getvalue().decode("utf-8", "replace") if bib else ""
            if not text.strip() and not bib_text.strip():
                st.warning("Paste identifiers or choose a BibTeX file.")
            else:
                with st.spinner("Looking up papers…"):
                    n, warns = engine.add_seed_papers(profile, text, bib_text)
                for w in warns:
                    st.warning(w)
                if n:
                    st.success(f"Added {n}.")
    for p in engine.seed_papers(profile):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f"<span class='subtle' style='font-size:.82rem'>{esc(p.title)}</span>", unsafe_allow_html=True)
        if c2.button("", key=f"rmseed_{p.id}", icon=":material/close:", type="tertiary", help="Remove seed"):
            store.remove_seed(profile.id, p.id)
            st.rerun()


def fetch_form(profile: InterestProfile) -> None:
    if config.BROWSER_MODE:
        st.caption("Use the sample collection or import papers from a JSON file. Live searches are available in the Python app.")
        if st.button("Add sample papers", type="primary", width="stretch"):
            engine.fetch_into_pool(profile, "sample")
            st.rerun()
        with st.form(f"paper_import_{profile.id}", clear_on_submit=True):
            uploaded = st.file_uploader("Paper collection (.json)", type=["json"])
            st.caption('A list of papers with "id", "title" and preferably "abstract". Up to 2,000 papers / 10 MB.')
            if st.form_submit_button("Import papers", width="stretch"):
                if uploaded is None:
                    st.warning("Choose a JSON file first.")
                else:
                    try:
                        papers = parse_papers(load_json(uploaded.getvalue()))
                        store.upsert_papers(papers)
                        store.add_to_pool(profile.id, [p.id for p in papers])
                        store.set_setting(profile.id, "content_rev", datetime.now(timezone.utc).isoformat())
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        st.rerun()
        st.caption(f"{len(store.pool_ids(profile.id))} papers in this profile")
        return
    src = st.selectbox("Source", list(SOURCES), format_func=SOURCES.get)
    cats, days, n = [], None, 100
    if src in ("arxiv_new", "arxiv", "both"):
        cats = st.multiselect(
            "arXiv categories", list(ARXIV_CATEGORIES),
            default=store.get_setting(profile.id, "categories", ["cs.CL", "cs.IR", "cs.LG"]),
            format_func=lambda c: f"{c} · {ARXIV_CATEGORIES[c]}",
        )
    if src in ("arxiv", "semantic_scholar", "both"):
        n = st.number_input("Maximum results", 20, 300, 100, 20)
        days = st.selectbox("Published within", [30, 90, 365, 1825], index=2,
                            format_func=lambda d: {30: "Last month", 90: "Last 3 months", 365: "Last year", 1825: "Last 5 years"}[d])
    if st.button("Fetch papers", type="primary", width="stretch"):
        store.set_setting(profile.id, "categories", cats)
        with st.spinner("Fetching…"):
            try:
                got, added, warns = engine.fetch_into_pool(profile, src, cats, int(n), days)
            except (SourceError, ValueError) as e:
                st.error(str(e))
                return
        for w in warns:
            st.warning(w)
        if got:
            st.success(f"{added} new papers.")
        if added and store.get_setting(profile.id, "full_text", True):
            with st.spinner("Reading full text of borderline papers…"):
                n_ft = engine.enrich_borderline(profile)
    auto = st.checkbox(
        "Update daily",
        value=store.get_setting(profile.id, "auto_fetch", True),
        help="Runs when you open the app (at most once a day) for the categories above. "
             "For fetching while the app is closed, schedule scripts/daily.py.",
    )
    if auto != store.get_setting(profile.id, "auto_fetch", True):
        store.set_setting(profile.id, "auto_fetch", auto)
    last = store.last_successful_fetch(profile.id)
    hist = store.fetch_history(profile.id, 1)
    health = f"updated {_ago(last)}" if last else "never updated"
    if hist and not hist[0]["ok"]:
        health += " · last attempt failed"
    st.caption(f"{len(store.pool_ids(profile.id))} papers · {health}")


def _ago(ts: str | None) -> str:
    if not ts:
        return "never"
    delta = datetime.now(timezone.utc) - datetime.fromisoformat(ts)
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return "just now"
    if hours < 24:
        return f"{int(hours)} h ago"
    return f"{int(hours // 24)} d ago"


# ------------------------------------------------------------ reading list


def reading_list(profile: InterestProfile, run: TriageRun) -> None:
    states = user_state(profile)
    visible = [r for r in run.results if not states.get(r.paper.id, {}).get("dismissed")]
    counts = summarize(visible)
    dismissed = len(run.results) - len(visible)
    new_ids = engine.new_since(profile, st.session_state.get(f"prev_visit_{profile.id}"))
    new_worth = [r for r in visible if r.paper.id in new_ids and r.label in ("READ", "SKIM")]

    # Option values stay fixed (only their captions carry counts), so the selection
    # survives feedback, reruns and refreshes (it is also kept in the URL).
    options = {
        "NEW": f"New ({len(new_worth)})",
        "READ": f"Read ({counts['READ']})",
        "SKIM": f"Skim ({counts['SKIM']})",
        "SKIP": f"Skip ({counts['SKIP']})",
        "ALL": f"All ({len(visible)})",
        "DISMISSED": f"Dismissed ({dismissed})",
    }
    # Keep the stable option value (e.g. "SKIM") in the URL, not its caption with a count.
    if st.session_state.get("view") not in options:
        from_url = st.query_params.get("view")
        st.session_state["view"] = from_url if from_url in options else ("NEW" if new_worth else "READ")
    view = st.segmented_control(
        "View", list(options), format_func=options.get, key="view", persist_state="session",
        label_visibility="collapsed",
    ) or "READ"
    if st.query_params.get("view") != view:
        st.query_params["view"] = view
    bar = st.container(horizontal=True, vertical_alignment="center", gap="small")
    with bar:
        query = st.text_input(
            "Search", placeholder="Search titles and abstracts", label_visibility="collapsed",
            key="search", persist_state="session",
        )
        export_menu(profile, run, visible, new_ids)

    fv = store.feedback_version(profile.id)
    if st.session_state.get(f"model_version_{profile.id}") != fv:
        if st.button("Your feedback has changed — update ranking", type="tertiary", icon=":material/refresh:"):
            st.session_state[f"model_version_{profile.id}"] = fv
            st.rerun()

    if view == "DISMISSED":
        rows = [r for r in run.results if states.get(r.paper.id, {}).get("dismissed")]
    elif view == "NEW":
        rows = new_worth
        if not rows:
            since = st.session_state.get(f"prev_visit_{profile.id}")
            st.markdown(
                f"<p class='subtle'>Nothing new since your last visit ({_ago(since)}).</p>" if since else
                "<p class='subtle'>Nothing new yet — see Read and Skim.</p>",
                unsafe_allow_html=True,
            )
            return
    else:
        rows = [r for r in visible if view == "ALL" or r.label == view]
    # Near-duplicates: show one card per cluster; the rest appear under "Similar papers".
    shown_ids = {r.paper.id for r in rows}
    followers = {r.paper.id for r in rows if r.group_lead and r.group_lead in shown_ids}
    rows = [r for r in rows if r.paper.id not in followers]
    if query:
        q = query.lower()
        rows = [r for r in rows if q in f"{r.paper.title} {r.paper.abstract}".lower()]
    if not rows:
        st.markdown("<p class='subtle'>No papers in this view.</p>", unsafe_allow_html=True)
        return

    shown_key = f"shown_{profile.id}_{view}_{query}"
    shown = st.session_state.get(shown_key, PAGE_SIZE)
    page = rows[:shown]
    with st.spinner("Preparing explanations…"):
        expl = explain_many(page, profile, run.papers, use_llm=use_llm(profile), store=store)
    by_id = run.by_id
    for r in page:
        similar = [by_id[i] for i in r.similar if i in followers]
        paper_row(profile, r, expl.get(r.paper.id, {}), states.get(r.paper.id, {}), similar, r.paper.id in new_ids)
    if len(rows) > shown:
        if st.button(f"Show more ({len(rows) - shown} remaining)", width="stretch"):
            st.session_state[shown_key] = shown + PAGE_SIZE
            st.rerun()


def export_menu(profile: InterestProfile, run: TriageRun, visible: list[TriageResult], new_ids: set[str]) -> None:
    reads = [r.paper for r in visible if r.label == "READ"]
    worth = [r.paper for r in visible if r.label in ("READ", "SKIM")]
    slug = profile.name.replace(" ", "_")
    with st.popover("Export", icon=":material/download:", type="tertiary"):
        st.download_button(f"BibTeX — Read ({len(reads)})", export.to_bibtex(reads), f"{slug}_read.bib", "application/x-bibtex", width="stretch")
        st.download_button(f"BibTeX — Read and Skim ({len(worth)})", export.to_bibtex(worth), f"{slug}_read_skim.bib", "application/x-bibtex", width="stretch")
        st.download_button(f"RIS — Read and Skim ({len(worth)})", export.to_ris(worth), f"{slug}.ris", "application/x-research-info-systems", width="stretch")
        st.divider()
        digest_scope = st.radio("Digest", ["New since last visit", "Everything"], horizontal=True, key=f"dscope_{profile.id}")
        scope_ids = new_ids if digest_scope.startswith("New") else None
        items = [r for r in visible if r.label == "READ" and (scope_ids is None or r.paper.id in scope_ids)]  # reasons shown for Read only
        reasons = {k: v["reason"] for k, v in explain_many(items, profile, run.papers, use_llm=use_llm(profile), store=store).items()}
        md = export.digest_markdown(profile.name, visible, reasons, scope_ids)
        st.download_button("Download digest", md, f"digest_{slug}_{datetime.now():%Y%m%d}.md", "text/markdown", width="stretch")
        ch = export.delivery_channels()
        if ch["slack"] and st.button("Post digest to Slack", width="stretch"):
            try:
                export.send_slack(md)
                st.success("Posted.")
            except Exception as e:
                st.error(str(e))
        if ch["email"] and st.button("Email digest", width="stretch"):
            try:
                export.send_email(md, f"Reading digest — {profile.name}")
                st.success("Sent.")
            except Exception as e:
                st.error(str(e))


def paper_row(
    profile: InterestProfile, r: TriageResult, ex: dict, state: dict, similar: list[TriageResult] | None = None, is_new: bool = False
) -> None:
    p, pid = r.paper, r.paper.id
    label = state.get("corrected") or r.label
    status = {"useful": "Relevant", "not_useful": "Not relevant"}.get(state.get("vote", ""), "")
    if state.get("dismissed"):
        status = "Dismissed"
    venue = p.venue if p.venue and p.venue != "arXiv preprint" else ("arXiv" if pid.startswith("arxiv:") else "")
    meta = [p.author_str, venue, str(p.year or p.published[:4] or "")]
    if similar:
        meta.append(f"+{len(similar)} similar")
    title = (f"<a class='paper-title' href='{esc(p.url)}' target='_blank'>{esc(p.title)}</a>"
             if p.url else f"<span class='paper-title'>{esc(p.title)}</span>")
    new_chip = "<span class='chip NEW'>New</span>" if is_new else ""
    corrected = " title='Your label'" if state.get("corrected") else ""

    with st.container(border=True):
        st.markdown(
            f"""<div class='card-top'><span{corrected}>{chip(label)}</span>{new_chip}
            <span class='status'>{esc(status)}</span><span class='score' title='Relevance score'>{r.score:.2f}</span></div>
            <div class='card-title'>{title}</div>
            <div class='paper-meta'>{esc(' · '.join(m for m in meta if m))}</div>
            <div class='paper-reason'>{esc(ex.get('reason', ''))}</div>""",
            unsafe_allow_html=True,
        )
        with st.container(horizontal=True, gap="medium", vertical_alignment="center"):
            if st.button("Relevant", key=f"up_{pid}", type="tertiary", icon=":material/thumb_up:"):
                log(profile, r, "useful")
                st.rerun()
            if st.button("Not relevant", key=f"down_{pid}", type="tertiary", icon=":material/thumb_down:"):
                log(profile, r, "not_useful")
                st.rerun()
            with st.popover("Reclassify", type="tertiary", icon=":material/swap_vert:", key=f"pop_{pid}"):
                for lab in config.LABELS:
                    if st.button(LABEL_TEXT[lab], key=f"re_{lab}_{pid}", type="tertiary", disabled=lab == label, help=LABEL_HELP[lab]):
                        log(profile, r, "correct", lab)
                        st.rerun()
            if state.get("dismissed"):
                if st.button("Restore", key=f"undis_{pid}", type="tertiary", icon=":material/undo:"):
                    log(profile, r, "undismiss")
                    st.rerun()
            elif st.button("Dismiss", key=f"dis_{pid}", type="tertiary", icon=":material/close:"):
                log(profile, r, "dismiss")
                st.rerun()

        with st.expander("Details"):
            st.write(p.abstract or "No abstract available.")
            links = [f"[Abstract page]({p.url})"] if p.url else []
            if p.pdf_url:
                links.append(f"[PDF]({p.pdf_url})")
            if links:
                st.markdown(" · ".join(links))
            if similar:
                st.markdown("**Similar papers**")
                for sr in similar:
                    sp = sr.paper
                    link = f"<a href='{esc(sp.url)}' target='_blank'>{esc(sp.title)}</a>" if sp.url else esc(sp.title)
                    st.markdown(f"<div style='margin:.1rem 0 .35rem 0'>{chip(sr.label)}{link}</div>", unsafe_allow_html=True)
            st.markdown("**Scoring**")
            df = pd.DataFrame({
                "Signal": [FEATURE_NAMES[f] for f in FEATURES],
                "Value": [round(r.features[f], 2) for f in FEATURES],
                "Weight": [PRIOR_WEIGHTS[f] for f in FEATURES],
            })
            st.dataframe(df, hide_index=True, width="stretch")
            notes = [n for n in (r.note if not state.get("corrected") else "", "Scored using full text." if p.full_text else "",
                                 ex.get("warning", "")) if n]
            if ex.get("rejected_llm_reason"):
                notes.append(f"Discarded unverifiable LLM explanation: “{ex['rejected_llm_reason']}”")
            for n in notes:
                st.caption(n)


# ---------------------------------------------------------------- labeling


def labeling(profile: InterestProfile) -> None:
    pool = engine.pool(profile)
    labels = store.get_labels(profile.id)
    counts = {l: sum(1 for v in labels.values() if v == l) for l in config.LABELS}
    unlabeled = [p for p in pool if p.id not in labels]
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.markdown(
            f"<p class='summary' style='margin:0'>{len(labels)} labelled · {counts['READ']} read · {counts['SKIM']} skim · "
            f"{counts['SKIP']} skip · {len(unlabeled)} remaining</p>",
            unsafe_allow_html=True,
        )
        strategy = st.selectbox(
            "Sampling", ["Balanced across relevance", "Random", "Closest to the cutoffs"],
            help="Predictions are hidden while labelling. Balanced sampling gives a more useful test set than random.",
            key="sampling", persist_state="session", label_visibility="collapsed", width=240,
        )
    if not unlabeled:
        st.success("Every paper in this profile has a label.")
    else:
        qkey = f"queue_{profile.id}_{strategy}"
        queue = [pid for pid in st.session_state.get(qkey, []) if pid not in labels] or _queue(profile, unlabeled, strategy)
        st.session_state[qkey] = queue
        p = next((x for x in unlabeled if x.id == queue[0]), unlabeled[0])
        with st.container(border=True):
            title = f"<a class='paper-title' href='{esc(p.url)}' target='_blank'>{esc(p.title)}</a>" if p.url else f"<span class='paper-title'>{esc(p.title)}</span>"
            st.markdown(f"<div class='card-title'>{title}</div><div class='paper-meta'>{esc(p.author_str)} · {esc(p.venue or 'arXiv')} · {p.year or ''}</div>", unsafe_allow_html=True)
            st.write(p.abstract)
            b = st.columns(4)
            for i, lab in enumerate(config.LABELS):
                if b[i].button(LABEL_TEXT[lab], key=f"lab_{lab}_{p.id}", help=LABEL_HELP[lab], width="stretch"):
                    store.set_label(profile.id, p.id, lab)
                    st.session_state[qkey] = queue[1:]
                    st.rerun()
            if b[3].button("Not sure", key=f"lab_skip_{p.id}", type="tertiary", width="stretch"):
                st.session_state[qkey] = queue[1:] + queue[:1]
                st.rerun()

    with st.expander("Manage labels"):
        if labels:
            by_id = {p.id: p for p in store.get_papers(list(labels))}
            df = pd.DataFrame([{"id": pid, "Title": by_id[pid].title if pid in by_id else pid, "Label": lab, "Remove": False} for pid, lab in labels.items()])
            edited = st.data_editor(
                df, hide_index=True, width="stretch", disabled=["id", "Title"], key=f"editor_{profile.id}",
                column_config={"id": None, "Label": st.column_config.SelectboxColumn(options=list(config.LABELS), required=True)},
            )
            c1, c2 = st.columns(2)
            if c1.button("Save changes"):
                for _, row in edited.iterrows():
                    if row["Remove"]:
                        store.remove_label(profile.id, row["id"])
                    elif row["Label"] != labels.get(row["id"]):
                        store.set_label(profile.id, row["id"], row["Label"])
                st.rerun()
            c2.download_button(
                "Export labels", json.dumps({"profile": profile.name, "labels": labels}, indent=1),
                file_name=f"labels_{profile.name.replace(' ', '_')}.json", mime="application/json",
            )
        up = st.file_uploader("Import labels (JSON)", type="json", key=f"imp_{profile.id}")
        if up is not None and st.button("Import"):
            try:
                data = json.load(up)
                imported = data.get("labels", data)
                known = set(store.pool_ids(profile.id)) | {p.id for p in store.get_papers(list(imported))}
                n = 0
                for pid, lab in imported.items():
                    if lab in config.LABELS and pid in known:
                        store.set_label(profile.id, pid, lab)
                        n += 1
                st.success(f"Imported {n} labels; {len(imported) - n} skipped.")
            except (json.JSONDecodeError, AttributeError) as e:
                st.error(f"Could not read that file: {e}")


def _queue(profile: InterestProfile, unlabeled, strategy: str) -> list[str]:
    rng = random.Random(f"{profile.id}-{strategy}-{len(unlabeled)}")
    if strategy == "Random":
        ids = [p.id for p in unlabeled]
        rng.shuffle(ids)
        return ids
    try:
        run = engine.run(profile, engine.cutoffs(profile), respect_budget=False, learn=False)
    except ValueError:
        return [p.id for p in unlabeled]
    ids = {p.id for p in unlabeled}
    scored = [(r.score, r.paper.id) for r in run.results if r.paper.id in ids]
    if strategy.startswith("Closest"):
        cut = engine.cutoffs(profile)
        scored.sort(key=lambda t: min(abs(t[0] - cut.read), abs(t[0] - cut.skim)))
        return [pid for _, pid in scored]
    # Balanced: split by score into 5 bands and interleave, highest band first.
    scored.sort(reverse=True)
    size = max(1, -(-len(scored) // 5))
    bands = [[pid for _, pid in scored[i : i + size]] for i in range(0, len(scored), size)]
    for b in bands:
        rng.shuffle(b)
    out = []
    while any(bands):
        for b in bands:
            if b:
                out.append(b.pop())
    return out


# -------------------------------------------------------------- evaluation


ABOUT_EVAL = """
**How these numbers are computed**

- All metrics use your hand-labelled papers. Each paper is scored by a model that never saw its label
  (5-fold cross-validation).
- **Speed** — walking down the ranked list, how soon you reach the good papers, compared with a random
  order. Screening time assumes about 2 minutes per abstract.
- **Learning curve** — your labels and ratings replayed in the order you gave them; the band is a 90% range.
- **Ranking quality** — Spearman ρ and NDCG@10 don't depend on the cutoffs; macro-F1 does.
- Ranges are 90% bootstrap intervals. With few labels, overlapping ranges mean a difference may be noise.
- Balanced sampling on the Labeling tab over-represents good papers compared with a real feed, so
  compare with the random baseline rather than reading absolute values.
"""


def evaluation(profile: InterestProfile) -> None:
    labels = store.get_labels(profile.id)
    if len(labels) < MIN_LABELS:
        st.markdown(f"<p class='subtle'>Label at least {MIN_LABELS} papers to see evaluation ({len(labels)} so far).</p>",
                    unsafe_allow_html=True)
        explanation_review(profile)
        return

    good_opts = {"read": "Read", "read_skim": "Read or Skim"}
    saved_good = store.get_setting(profile.id, "good_mode", "read")
    with st.container(horizontal=True, vertical_alignment="center"):
        good = st.segmented_control(
            "Good paper", list(good_opts), format_func=good_opts.get, default=saved_good,
            key=f"good_{profile.id}", persist_state="session", label_visibility="collapsed",
            help="Which labels count as a good paper",
        ) or saved_good
        with st.popover("About these numbers", type="tertiary", icon=":material/info:"):
            st.markdown(ABOUT_EVAL)
    if good != saved_good:
        store.set_setting(profile.id, "good_mode", good)

    cut = engine.cutoffs(profile)
    with st.spinner("Evaluating…"):
        rep, speed = cached_eval(
            profile_json(profile), profile.id, store.feedback_version(profile.id), cut.read, cut.skim, pool_key(profile), good
        )

    speed_section(speed)
    if not rep.ok:
        st.info(rep.message)
        explanation_review(profile)
        return

    st.markdown("<div class='section'>Ranking quality</div>", unsafe_allow_html=True)
    rows = [
        {
            "Method": s.name.replace(" (cross-validated)", ""),
            "Spearman ρ": _with_ci(s.spearman, s.spearman_ci),
            "NDCG@10": _with_ci(s.ndcg10, s.ndcg10_ci),
            "Macro-F1 (tuned)": f"{s.at_best['macro_f1']:.2f}",
        }
        for s in rep.systems
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    full = rep.systems[0]
    cur = full.at_current
    st.markdown(
        f"<p class='summary'>At current cutoffs: macro-F1 {cur['macro_f1']:.2f} · accuracy {cur['accuracy']:.2f} · "
        f"{cur['severe_errors']} Read↔Skip errors · learning weight {rep.blend.get('weight') or 0:.0%}</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='section'>Confusion matrix</div>", unsafe_allow_html=True)
        cm = pd.DataFrame(full.at_current["confusion"], index=[f"Labelled {LABEL_TEXT[l]}" for l in config.LABELS],
                          columns=[f"Predicted {LABEL_TEXT[l]}" for l in config.LABELS])
        st.dataframe(cm, width="stretch")
    with c2:
        st.markdown("<div class='section'>Suggested cutoffs</div>", unsafe_allow_html=True)
        bc = full.best_cutoffs
        f1_best, f1_now = full.at_best["macro_f1"], cur["macro_f1"]
        st.markdown(
            "<div class='stats'>"
            + _stat("Read at", f"{bc.read:.2f}", f"now {cut.read:.2f}")
            + _stat("Skim at", f"{bc.skim:.2f}", f"now {cut.skim:.2f}")
            + _stat("Macro-F1", f"{f1_best:.2f}", f"now {f1_now:.2f}")
            + "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Apply", help="Tuned on the same labels, so the gain is optimistic."):
            engine.save_cutoffs(profile, bc)
            for k in (f"read_{profile.id}", f"skim_{profile.id}"):
                st.session_state.pop(k, None)
            st.rerun()

    wrong = [(pid, t, p, s) for pid, t, p, s in zip(rep.paper_ids, rep.truth, rep.cv_predictions, full.scores) if t != p]
    with st.expander("Disagreements"):  # constant label: a changing label would re-create (collapse) it
        st.caption(f"{len(wrong)} papers where the prediction differs from your label.")
        by_id = {p.id: p for p in store.get_papers([w[0] for w in wrong])}
        st.dataframe(
            pd.DataFrame([{"Title": by_id[w[0]].title if w[0] in by_id else w[0], "Your label": LABEL_TEXT[w[1]],
                           "Predicted": LABEL_TEXT[w[2]], "Score": w[3]} for w in wrong]),
            hide_index=True, width="stretch", column_config={"Score": st.column_config.NumberColumn(format="%.2f")},
        )
    explanation_review(profile)


SCREEN_MINUTES = 2  # time to read a title + abstract and decide
SERIES = {  # categorical slots 1-2 of the validated reference palette; references in neutral gray
    "Personalised model": ("#2a78d6", [1, 0]),
    "Profile only": ("#eb6834", [1, 0]),
    "Random order": ("#8a919c", [5, 4]),
    "Perfect ranking": ("#c4c8cf", [2, 3]),
}


def _with_ci(v: float, ci) -> str:
    if v is None or np.isnan(v):
        return "—"
    return f"{v:.2f} ({ci[0]:.2f}–{ci[1]:.2f})" if ci else f"{v:.2f}"


def _range(ci, fmt) -> str:
    return f"90% range {fmt(ci[0])}–{fmt(ci[1])}" if ci else ""


def _stat(label: str, value: str, sub: str, tip: str = "") -> str:
    t = f" title='{esc(tip)}'" if tip else ""
    return (f"<div class='stat'{t}><div class='l'>{esc(label)}</div><div class='v'>{esc(value)}</div>"
            f"<div class='s'>{esc(sub)}</div></div>")


def speed_section(sp) -> None:
    st.markdown("<div class='section'>Speed</div>", unsafe_allow_html=True)
    if not sp.ok:
        st.info(sp.message)
        return
    n, g = sp.n_papers, sp.n_good
    model = sp.curves[0]
    rand_first = (n + 1) / (g + 1)
    rand_half = int(np.searchsorted(sp.random, 0.5 - 1e-9)) + 1
    rand_80 = int(np.searchsorted(sp.random, 0.8 - 1e-9)) + 1
    rand_hit10 = min(10, n) * g / n / max(1, min(10, g))
    saved = (rand_80 - model.to_80) * SCREEN_MINUTES if model.to_80 else None
    fmt_pos = lambda v: f"#{v}" if v else "—"
    st.markdown(
        "<div class='stats'>"
        + _stat("First good paper", fmt_pos(model.first_good), f"random #{rand_first:.0f}")
        + _stat("Half of good papers", f"top {model.to_half}" if model.to_half else "—", f"random top {rand_half}",
                _range(model.to_half_ci, lambda v: f"{v:.0f}"))
        + _stat("80% of good papers", f"top {model.to_80}" if model.to_80 else "—", f"random top {rand_80}",
                _range(model.to_80_ci, lambda v: f"{v:.0f}"))
        + _stat("Good in top 10", f"{model.hit_rate_at_10:.0%}", f"random {rand_hit10:.0%}",
                _range(model.hit_rate_ci, lambda v: f"{v:.0%}"))
        + _stat("Screening saved", f"{max(saved, 0)} min" if saved is not None else "—", "to reach 80%")
        + "</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.altair_chart(discovery_chart(sp), width="stretch")
    with c2:
        if len(sp.learning) > 1:
            st.altair_chart(learning_chart(sp), width="stretch")


def _title(text: str) -> alt.TitleParams:
    return alt.TitleParams(text, anchor="start", fontSize=13, fontWeight=600, color="#1b1f24", font="Inter", offset=10)


def _axis(**kw):
    return alt.Axis(labelColor="#5f6773", titleColor="#5f6773", titleFontWeight="normal", domainColor="#d5d9de",
                    tickColor="#d5d9de", gridColor="#eef0f3", **kw)


def discovery_chart(sp) -> alt.Chart:
    n = len(sp.random)
    wide = pd.DataFrame({
        "k": np.arange(1, n + 1),
        "Personalised model": sp.curves[0].found * 100,
        "Profile only": sp.curves[1].found * 100,
        "Random order": sp.random * 100,
        "Perfect ranking": sp.ideal * 100,
    })
    long = wide.melt("k", var_name="series", value_name="found")
    names = list(SERIES)
    color_scale = alt.Scale(domain=names, range=[SERIES[s][0] for s in names])
    color = alt.Color("series:N", scale=color_scale,
                      legend=alt.Legend(orient="bottom", title=None, labelColor="#333a44", symbolType="stroke", symbolStrokeWidth=2))
    dash = alt.StrokeDash("series:N", scale=alt.Scale(domain=names, range=[SERIES[s][1] for s in names]), legend=None)
    x = alt.X("k:Q", title="Papers reviewed", scale=alt.Scale(domain=[1, n], nice=False), axis=_axis(grid=False, tickMinStep=1))
    y = alt.Y("found:Q", title="Good papers found (%)", scale=alt.Scale(domain=[0, 100]), axis=_axis(tickCount=5))
    hover = alt.selection_point(fields=["k"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=x, y=y, color=color, strokeDash=dash)
    points = (
        alt.Chart(long).mark_point(filled=True, size=70, stroke="white", strokeWidth=2)
        .encode(x=x, y=y, color=alt.Color("series:N", scale=color_scale, legend=None),
                opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    )
    rule = (
        alt.Chart(wide).mark_rule(color="#9aa1ab", strokeWidth=1)
        .encode(
            x="k:Q",
            opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
            tooltip=[alt.Tooltip("k:Q", title="Papers reviewed")]
            + [alt.Tooltip(f"{s}:Q", title=s, format=".0f") for s in names],
        )
        .add_params(hover)
    )
    return (lines + points + rule).properties(height=260, title=_title("Good papers found as you read down the list")).configure_view(stroke=None)


def learning_chart(sp) -> alt.Chart:
    df = pd.DataFrame(sp.learning)
    df["hit"] = df["hit_rate_at_10"] * 100
    df["lo100"], df["hi100"] = df["lo"] * 100, df["hi"] * 100
    n, g = sp.n_papers, sp.n_good
    rand = min(10, n) * g / n / max(1, min(10, g)) * 100
    x = alt.X("signals:Q", title="Feedback given", axis=_axis(grid=False, tickMinStep=1))
    y = alt.Y("hit:Q", title="Good in top 10 (%)", scale=alt.Scale(domain=[0, 100]), axis=_axis(tickCount=5))
    band = alt.Chart(df).mark_area(opacity=0.12, color=SERIES["Personalised model"][0]).encode(
        x=x, y=alt.Y("lo100:Q", scale=alt.Scale(domain=[0, 100])), y2="hi100:Q"
    )
    line = alt.Chart(df).mark_line(strokeWidth=2, color=SERIES["Personalised model"][0]).encode(x=x, y=y)
    pts = alt.Chart(df).mark_point(filled=True, size=70, stroke="white", strokeWidth=2, color=SERIES["Personalised model"][0]).encode(
        x=x, y=y,
        tooltip=[alt.Tooltip("signals:Q", title="Feedback given"), alt.Tooltip("hit:Q", title="Good in top 10 (%)", format=".0f"),
                 alt.Tooltip("lo100:Q", title="Range from (%)", format=".0f"), alt.Tooltip("hi100:Q", title="Range to (%)", format=".0f"),
                 alt.Tooltip("ndcg10:Q", title="NDCG@10", format=".2f")],
    )
    ref = pd.DataFrame({"y": [rand], "label": ["Random order"]})
    ref_rule = alt.Chart(ref).mark_rule(color=SERIES["Random order"][0], strokeDash=[5, 4]).encode(y="y:Q")
    ref_text = alt.Chart(ref).mark_text(align="left", baseline="bottom", dx=4, dy=-3, color="#5f6773", fontSize=11).encode(
        y="y:Q", x=alt.value(0), text="label:N"
    )
    return (band + ref_rule + ref_text + line + pts).properties(height=260, title=_title("Top-10 quality as feedback accumulates")).configure_view(stroke=None)


def explanation_review(profile: InterestProfile) -> None:
    latest: dict[str, str] = {}
    for f in store.get_feedback(profile.id):
        if f["action"] in ("explanation_ok", "explanation_bad"):
            latest[f["paper_id"]] = f["action"]
    ok = sum(1 for v in latest.values() if v == "explanation_ok")
    with st.expander("Review explanations"):
        if latest:
            st.caption(f"{ok} of {len(latest)} judged accurate.")
        run = get_run(profile)
        if run is None:
            return
        skey = f"review_{profile.id}"
        if st.button("New sample", type="tertiary", icon=":material/refresh:") or skey not in st.session_state:
            top = run.results[:40]
            st.session_state[skey] = [r.paper.id for r in random.sample(top, min(5, len(top)))]
        rows = [run.by_id[pid] for pid in st.session_state[skey] if pid in run.by_id]
        expl = explain_many(rows, profile, run.papers, use_llm=use_llm(profile), store=store)
        for r in rows:
            ex = expl[r.paper.id]
            verdict = latest.get(r.paper.id)
            st.markdown(
                f"{chip(r.label)}<b>{esc(r.paper.title)}</b><div class='paper-reason' style='margin:.3rem 0'>{esc(ex['reason'])}</div>",
                unsafe_allow_html=True,
            )
            c = st.container(horizontal=True, gap="medium")
            if c.button("Accurate", key=f"eok_{r.paper.id}", type="tertiary", icon=":material/check:", disabled=verdict == "explanation_ok"):
                log(profile, r, "explanation_ok", ex["source"])
                st.rerun()
            if c.button("Overstates overlap", key=f"ebad_{r.paper.id}", type="tertiary", icon=":material/close:", disabled=verdict == "explanation_bad"):
                log(profile, r, "explanation_bad", ex["source"])
                st.rerun()
            st.divider()


# ---------------------------------------------------------------- settings


def _toggle(profile: InterestProfile, key: str, label: str, default: bool = True, help: str | None = None) -> bool:
    saved = bool(store.get_setting(profile.id, key, default))
    val = st.toggle(label, value=saved, help=help, key=f"set_{key}_{profile.id}")
    if val != saved:
        store.set_setting(profile.id, key, val)
        st.rerun()
    return val


def settings(profile: InterestProfile, run: TriageRun | None) -> None:
    st.markdown("<div class='section' style='margin-top:.4rem'>Triage</div>", unsafe_allow_html=True)
    saved = engine.cutoffs(profile)
    c1, c2 = st.columns(2, gap="large")
    read_c = c1.slider("Read at", 0.0, 1.0, saved.read, 0.01, key=f"read_{profile.id}", help="Minimum relevance for Read")
    skim_c = c2.slider("Skim at", 0.0, 1.0, min(saved.skim, read_c), 0.01, key=f"skim_{profile.id}", help="Minimum relevance for Skim")
    cut = Cutoffs(read=read_c, skim=skim_c)
    if (cut.read, cut.skim) != (saved.read, saved.skim):
        engine.save_cutoffs(profile, cut)
        st.rerun()
    _toggle(profile, "budget", f"Cap Read at {read_budget(profile.hours_per_week)} papers a week",
            help="Based on your reading time; extra papers move to Skim.")
    _toggle(profile, "auto_update", "Re-rank after each feedback")
    _toggle(profile, "group_similar", "Group near-identical papers")
    full = False if config.BROWSER_MODE else _toggle(profile, "full_text", "Use full text for borderline papers",
                   help="Re-scores papers near a cutoff using their introduction and conclusion.")
    if full and run is not None:
        pending = len(engine.borderline(profile, run))
        if pending and st.button(f"Fetch full text now ({min(pending, 12)})", type="tertiary", icon=":material/article:"):
            with st.spinner("Fetching full text…"):
                engine.enrich_borderline(profile)
            st.rerun()

    st.markdown("<div class='section'>Explanations</div>", unsafe_allow_html=True)
    available = not config.BROWSER_MODE and llm_available()
    saved_llm = bool(store.get_setting(profile.id, "use_llm", False))
    if config.BROWSER_MODE:
        st.caption("Explanations use evidence from the paper. This browser edition uses TF-IDF ranking and runs without API keys.")
    llm_on = False if config.BROWSER_MODE else st.toggle(
        "Write explanations with Claude", value=saved_llm and available, disabled=not available,
        help=(f"Uses {config.LLM_MODEL}. Each explanation is checked against the paper; unverifiable ones are replaced."
              if available else "Set ANTHROPIC_API_KEY to enable."),
    )
    if available and llm_on != saved_llm:
        store.set_setting(profile.id, "use_llm", llm_on)
        st.rerun()

    if run is not None:
        st.markdown("<div class='section'>Model</div>", unsafe_allow_html=True)
        info, blend = run.model.train_info, run.blend or {}
        w = run.model.learned_weight if info.get("learned") else 0.0
        status = "validated" if blend.get("validated") else ("not validated" if info.get("learned") else "not trained")
        st.markdown(
            "<div class='stats'>"
            + _stat("Learning weight", f"{w:.0%}", status,
                    "Chosen by cross-validation on your labels; off unless learning beats the profile-only ranking.")
            + _stat("Training signals", str(info.get("n_examples", 0)), f"{len(store.seed_ids(profile.id))} seed papers")
            + _stat("Embeddings", "MiniLM" if run.embedder_name.startswith("sbert") else "TF-IDF", run.embedder_name.split(":")[-1])
            + "</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Model details"):
            sc = blend.get("scores", {})
            if sc:
                st.dataframe(
                    pd.DataFrame({"Learning weight": [f"{float(k):.0%}" for k in sc], "Average precision": list(sc.values()),
                                  "Chosen": ["✓" if float(k) == blend.get("weight") else "" for k in sc]}),
                    hide_index=True, width="stretch",
                    column_config={"Average precision": st.column_config.NumberColumn(format="%.3f")},
                )
            coef = info.get("coefficients", {})
            st.dataframe(
                pd.DataFrame({
                    "Signal": [FEATURE_NAMES[f] for f in FEATURES],
                    "Profile weight": [PRIOR_WEIGHTS[f] for f in FEATURES],
                    "Learned weight": [coef.get(f) for f in FEATURES],
                }),
                hide_index=True, width="stretch",
            )

    st.markdown("<div class='section'>Data</div>", unsafe_allow_html=True)
    if config.BROWSER_MODE:
        st.caption("Profiles and feedback are saved on this device. Clearing site data removes them. Download a backup to keep a copy or move to another browser.")
    st.download_button("Download profile backup", export_profile(store, profile),
                       f"paper_triage_profile_{profile.id}.json", "application/json", icon=":material/save:")
    st.download_button("Download paper collection", json.dumps([p.to_dict() for p in engine.pool(profile)], ensure_ascii=False),
                       f"paper_collection_{profile.id}.json", "application/json", icon=":material/download:")
    with st.expander("Fetch history"):
        hist = store.fetch_history(profile.id, 30)
        if hist:
            df = pd.DataFrame(hist)
            df["status"] = df["ok"].map({1: "ok", 0: "failed"})
            st.dataframe(df[["created_at", "source", "status", "n_papers", "n_new", "message"]], hide_index=True, width="stretch")
    with st.expander("Feedback history"):
        fb = store.get_feedback(profile.id)
        if fb:
            by_id = {p.id: p.title for p in store.get_papers(list({f["paper_id"] for f in fb}))}
            df = pd.DataFrame(fb)
            df["paper"] = df["paper_id"].map(by_id)
            st.dataframe(df[["created_at", "action", "value", "predicted_label", "score", "paper"]].iloc[::-1], hide_index=True, width="stretch")
    with st.container(horizontal=True, gap="small"):
        if run is not None:
            res = pd.DataFrame([{
                "rank": r.rank, "label": r.label, "score": round(r.score, 4), "title": r.paper.title,
                "authors": "; ".join(r.paper.authors), "venue": r.paper.venue, "year": r.paper.year, "url": r.paper.url,
            } for r in run.results])
            st.download_button("Export CSV", res.to_csv(index=False), f"reading_list_{datetime.now():%Y%m%d}.csv", "text/csv",
                               icon=":material/download:")
        if is_demo(engine, profile) and st.button("Reset demo", icon=":material/restart_alt:",
                                                   help="Recreate the demo with fresh simulated labels and feedback."):
            seed_demo(engine)
            st.rerun()
        with st.popover("Delete…", icon=":material/delete:"):
            if st.button("Remove all papers", help="Labels and feedback are kept."):
                store.clear_pool(profile.id)
                st.rerun()
            confirm = st.text_input("Type the profile name to delete it")
            if st.button("Delete profile", type="primary", disabled=confirm != profile.name):
                store.delete_profile(profile.id)
                st.session_state.pop("profile_select", None)
                st.query_params.clear()
                st.rerun()


# ------------------------------------------------------------------- main


def main() -> None:
    profile = sidebar()
    if profile is None:
        st.markdown("<div class='page-title'>Paper Triage</div>", unsafe_allow_html=True)
        st.markdown("<p class='subtle'>Create a profile in the sidebar to get a ranked Read / Skim / Skip list of new papers.</p>",
                    unsafe_allow_html=True)
        return

    demo = (
        "<span class='pill' title='Real arXiv papers with simulated labels and feedback. Reset under Settings.'>Demo</span>"
        if is_demo(engine, profile) else ""
    )
    st.markdown(f"<div class='page-title'>{esc(profile.name)}{demo}</div>", unsafe_allow_html=True)
    if not store.pool_ids(profile.id):
        st.markdown("<p class='subtle'>No papers yet. Add some from the sidebar.</p>", unsafe_allow_html=True)
        return
    run = get_run(profile)
    if run is not None:
        states = user_state(profile)
        visible = [r for r in run.results if not states.get(r.paper.id, {}).get("dismissed")]
        c = summarize(visible)
        new_ids = engine.new_since(profile, st.session_state.get(f"prev_visit_{profile.id}"))
        n_new = sum(r.paper.id in new_ids for r in visible)
        parts = [f"{len(visible)} papers", f"{c['READ']} to read", f"{c['SKIM']} to skim"] + ([f"{n_new} new"] if n_new else [])
        st.markdown(f"<p class='summary'>{' · '.join(parts)}</p>", unsafe_allow_html=True)

    # Stateful tabs: the active tab is tracked by Streamlit (not just the browser),
    # mirrored into the URL so it survives refreshes, and only the open tab runs.
    TABS = ["Reading list", "Labeling", "Evaluation", "Settings"]
    # `default` is part of the widget's identity, so it must not change during a session
    # (otherwise the next click lands on a "new" widget and is lost). Fix it once, from the URL.
    if "tab_default" not in st.session_state:
        from_url = st.query_params.get("tab")
        st.session_state["tab_default"] = from_url if from_url in TABS else TABS[0]
    tabs = st.tabs(TABS, key="main_tab", on_change="rerun", default=st.session_state["tab_default"])
    active = st.session_state.get("main_tab") or TABS[0]
    if st.query_params.get("tab") != active:
        st.query_params["tab"] = active
    t1, t2, t3, t4 = tabs
    with t1:
        if t1.open and run is not None:
            reading_list(profile, run)
    with t2:
        if t2.open:
            labeling(profile)
    with t3:
        if t3.open:
            evaluation(profile)
    with t4:
        if t4.open:
            settings(profile, run)

main()
