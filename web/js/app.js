// Paper Triage — browser app. Everything personal stays in this browser.
import * as account from "./account.js";
import { bindCharts, discoveryChart, learningChart } from "./chart.js";
import { fromRecord, loadCatalog, parseImport, toRecord } from "./data.js";
import { Embedder, paperText } from "./embedder.js";
import {
  DEFAULT_CUTOFFS, DenseSpace, FEATURES, FEATURE_NAMES, LABELS, MIN_LABELS, MINUTES_PER_READ, PRIOR_WEIGHTS, SparseSpace,
  buildProfileVectors, documentFrequencies, evidence, highlightTerms, isEmptyProfile, profileTexts, rank, readBudget, reason,
} from "./engine.js";
import { evaluate, learningCurve, SCREEN_MINUTES } from "./evaluate.js";
import { digestMarkdown, download, slug, toBibtex, toCsv, toRis } from "./export.js";
import { DEFAULT_SOURCES, SOURCES, parseIdentifiers, resolveIdentifiers, searchAll } from "./live.js";
import { badges, loadRules, signals } from "./quality.js";
import * as store from "./store.js";
import { parseList, tokenize } from "./text.js";
import { $, $$, ago, authors, date, esc, fix, highlight, icon, pct, safeUrl, toast } from "./ui.js";

const ROOT = new URL("../", import.meta.url);
const LABEL_TEXT = { READ: "Read", SKIM: "Skim", SKIP: "Skip" };
const LABEL_HELP = { READ: "Highly relevant: worth reading in full", SKIM: "Useful but peripheral: skim it", SKIP: "Low relevance: skip it" };
const PAGE = 20;
const MIN_RATINGS = 5; // ratings before the algorithm sorts papers into Read / Skim / Skip
const DATASET_GOAL = 500;
const EXAMPLES = [
  { name: "RAG & LLM evaluation", description: "I build retrieval-augmented generation systems and study how to evaluate whether LLM answers are faithful to the retrieved evidence.", keywords: ["retrieval augmented generation", "large language models", "evaluation", "hallucination"], fields: ["ai", "computing"] },
  { name: "Climate adaptation", description: "Climate change adaptation in coastal cities: flood risk, urban resilience and how communities plan for sea-level rise.", keywords: ["flood risk", "climate adaptation", "resilience"], fields: ["environment", "society"] },
  { name: "Cancer immunotherapy", description: "Cancer immunotherapy, especially T-cell therapies and tumour immune evasion, and clinical evidence for new treatments.", keywords: ["immunotherapy", "tumor", "T cells"], fields: ["life"] },
  { name: "Household finance", description: "Behavioural economics of household saving and borrowing decisions, financial literacy and policy nudges.", keywords: ["household", "savings", "financial decisions"], fields: ["society"] },
  { name: "Digital humanities", description: "Computational methods for history and literature: digitised archives, manuscripts and cultural heritage collections.", keywords: ["archives", "manuscripts", "cultural heritage"], fields: ["humanities", "society"] },
  { name: "Materials discovery", description: "Machine learning for materials and molecules: predicting properties, interatomic potentials and accelerating discovery.", keywords: ["materials", "molecular", "machine learning"], fields: ["physical", "ai"] },
];
const SIGNAL_FILTERS = {
  peer: { text: "Peer-reviewed", test: s => s.peer_reviewed === true },
  code: { text: "Code released", test: s => s.code },
  data: { text: "Data released", test: s => s.data },
  oa: { text: "Open access", test: s => s.open_access },
  review: { text: "Reviews & surveys", test: s => s.review },
  strong: { text: "Strong signals", test: s => s.level === "strong" },
};
const SORTS = { match: "Best match", trust: "Best match + trust signals", newest: "Newest first" };

// ------------------------------------------------------------------- state

const A = {
  ready: false,
  view: "feed",
  tab: "READ",
  sort: "match",
  query: "",
  signalFilters: new Set(),
  shown: PAGE,
  expanded: new Set(),
  focusId: null,
  run: null,
  explained: new Map(),
  labelQueue: null,
  labelStrategy: "balanced",
  labelUndo: [],
  report: null,
  curve: null,
  embedder: new Embedder(),
  vecCache: new Map(),
  newIds: new Set(),
};

const defaultState = () => ({ version: 1, active: null, profiles: {}, settings: { semantic: true } });
const profile = () => A.state?.profiles?.[A.state.active] || null; // no state yet on the sign-in screen
/** Tab, sort and signal filters are remembered per profile, so the feed looks the same when you come back. */
const VIEW_DEFAULT = { tab: "READ", sort: "match", signals: [] };
function applyView(prof) {
  const v = { ...VIEW_DEFAULT, ...(prof?.ui || {}) };
  A.tab = ["NEW", "READ", "SKIM", "SKIP", "ALL", "RATED", "HIDDEN"].includes(v.tab) ? v.tab : VIEW_DEFAULT.tab;
  A.sort = v.sort in SORTS ? v.sort : VIEW_DEFAULT.sort;
  A.signalFilters = new Set((v.signals || []).filter(k => k in SIGNAL_FILTERS));
  A.query = "";
}
function saveView() {
  const prof = profile();
  if (!prof) return;
  prof.ui = { tab: A.tab, sort: A.sort, signals: [...A.signalFilters] };
  save();
}

const save = () => { store.saveState(A.state); account.queue(A.state); };
const now = () => new Date().toISOString();

function newProfile(fields = {}) {
  const id = `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  return {
    id, name: "My research", description: "", keywords: [], focus: "", avoid: [], hours: 3, fields: [],
    cutoffs: { ...DEFAULT_CUTOFFS }, budget: true, group: true, seeds: [], labels: {}, feedback: [], extra: [],
    seen: [], lastVisit: null, created: now(), ...fields,
  };
}

// ------------------------------------------------------------------- corpus

function buildCorpus() {
  const byId = new Map(A.catalog.papers.map(p => [p.id, p]));
  for (const p of A.visitor) if (!byId.has(p.id)) byId.set(p.id, p);
  A.byId = byId;
  A.all = [...byId.values()];
  for (const p of A.all) p._sig ||= signals(p);
  A.sparse = new SparseSpace(A.all);
  A.dense = new DenseSpace();
  A.df = documentFrequencies(A.all);
}

function poolFor(prof) {
  const extra = new Set(prof.extra);
  const seeds = new Set(prof.seeds);
  // Catalog papers are shared by every profile; papers a visitor added belong to the profile that added them.
  return A.all.filter(p => !seeds.has(p.id) && (!p.added_at || extra.has(p.id)));
}

async function addVisitorPapers(papers, prof, { asSeeds = false } = {}) {
  const fresh = papers.filter(p => !A.byId.has(p.id));
  if (A.embedder.status === "ready" && fresh.length) await embedPapers(fresh);
  for (const p of fresh) { p.added_at = now(); p.area ||= "imported"; }
  await store.putPapers(fresh.map(toRecord));
  A.visitor.push(...fresh);
  buildCorpus();
  const ids = papers.map(p => p.id);
  const list = asSeeds ? prof.seeds : prof.extra;
  for (const id of ids) if (!list.includes(id)) list.push(id);
  save();
  return fresh.length;
}

async function embedPapers(papers) {
  const vecs = await A.embedder.embed(papers.map(paperText));
  papers.forEach((p, i) => { p._dense = vecs[i]; });
}

async function embedMissing() {
  const missing = A.visitor.filter(p => !p._dense);
  if (!missing.length) return;
  await embedPapers(missing);
  await store.putPapers(missing.map(toRecord));
}

// ------------------------------------------------------------------ ranking

function useDense(prof) {
  if (A.embedder.status !== "ready") return false;
  return poolFor(prof).every(p => p._dense) && prof.seeds.every(id => A.byId.get(id)?._dense);
}

async function encode(texts, space) {
  if (space === A.sparse) return texts.map(t => A.sparse.vectorize(t));
  const missing = [...new Set(texts.filter(t => !A.vecCache.has(t)))];
  if (missing.length) (await A.embedder.embed(missing)).forEach((v, i) => A.vecCache.set(missing[i], v));
  return texts.map(t => A.vecCache.get(t));
}

async function contextFor(prof) {
  const space = useDense(prof) ? A.dense : A.sparse;
  const encoded = await encode(profileTexts(prof).map(t => t[1]), space);
  const seedPapers = prof.seeds.map(id => A.byId.get(id)).filter(Boolean);
  // Without a written description, the papers you liked describe your interests.
  const taste = isEmptyProfile(prof) ? likedIds(prof).map(id => A.byId.get(id)).filter(Boolean) : [];
  if (isEmptyProfile(prof) && !seedPapers.length && !taste.length) return null;
  const pv = buildProfileVectors(prof, space, encoded, [...seedPapers, ...taste]);
  const pool = poolFor(prof);
  const byId = new Map([...pool, ...seedPapers].map(p => [p.id, p]));
  for (const id of Object.keys(prof.labels)) if (A.byId.has(id)) byId.set(id, A.byId.get(id));
  return {
    // Snapshots, so a report computed from this run matches the model trained in it.
    papers: pool, byId, space, pv, labels: { ...prof.labels }, feedback: [...prof.feedback], seeds: prof.seeds.filter(id => byId.has(id)),
    cutoffs: prof.cutoffs, hours: prof.hours, budget: prof.budget, group: prof.group, today: Date.now(),
    adaptive: prof.adaptive === false ? null : { read: readBudget(prof.hours) },
    exclude: decidedIds(prof), // already decided, so they don't use up reading time
  };
}

let rankSeq = 0;
async function rerank({ quiet = true } = {}) {
  const prof = profile();
  if (!prof) return;
  const seq = ++rankSeq;
  const before = A.run ? new Set(A.run.results.filter(r => r.label === "READ").map(r => r.paper.id)) : null;
  try {
    const ctx = await contextFor(prof);
    if (seq !== rankSeq) return;
    const out = ctx ? rank(ctx) : discover(prof);
    A.run = { ...out, space: ctx?.space ?? A.sparse, ctx, byId: new Map(out.results.map(r => [r.paper.id, r])) };
    A.explained.clear();
    A.report = null;
    A.curve = null;
    if (!quiet && before) {
      const after = out.results.filter(r => r.label === "READ").map(r => r.paper.id);
      const moved = after.filter(id => !before.has(id)).length;
      toast(moved ? `Ranking updated · ${moved} new paper${moved > 1 ? "s" : ""} in Read` : "Ranking updated from your feedback");
    }
  } catch (e) {
    console.error(e);
    A.run = { error: e.message || String(e) };
  }
  render();
}

/**
 * Before there is anything to rank against (fields only, nothing rated yet):
 * a varied sample that interleaves the chosen fields, papers with strong
 * signals and recent papers first. Nothing is labelled.
 */
function discover(prof) {
  const fields = prof.fields.length ? prof.fields : A.catalog.areas.map(a => a.id);
  const by = Object.fromEntries(fields.map(f => [f, []]));
  for (const p of poolFor(prof)) (by[p.area] ||= []).push(p);
  for (const list of Object.values(by)) list.sort((a, b) => b._sig.points - a._sig.points || (b.published || "").localeCompare(a.published || ""));
  const order = [];
  const lists = Object.values(by);
  for (let i = 0; lists.some(l => i < l.length); i++) for (const l of lists) if (l[i]) order.push(l[i]);
  const zeros = FEATURES.map(() => 0);
  return {
    results: order.map((p, i) => ({ paper: p, score: 0, prior: 0, learned: null, f: zeros, label: "SKIP", rank: i + 1, note: "", lead: "", similar: [] })),
    model: { info: { n_examples: 0, learned: false }, learnedWeight: 0 }, blend: {}, discover: true,
  };
}

let rerankTimer = null;
function scheduleRerank() {
  clearTimeout(rerankTimer);
  rerankTimer = setTimeout(() => rerank({ quiet: false }), 350);
}

/** Latest per-paper state from the feedback log. */
function states(prof) {
  const s = new Map();
  for (const f of prof.feedback) {
    const x = s.get(f.pid) || {};
    if (f.action === "useful" || f.action === "not_useful") x.vote = f.action;
    else if (f.action === "clear_vote") x.vote = undefined;
    else if (f.action === "correct") x.corrected = f.value;
    else if (f.action === "dismiss") x.hidden = true;
    else if (f.action === "undismiss") x.hidden = false;
    else if (f.action === "save") x.saved = true;
    else if (f.action === "unsave") x.saved = false;
    s.set(f.pid, x);
  }
  return s;
}

function logFeedback(pid, action, value = null) {
  const prof = profile();
  const r = A.run?.byId?.get(pid);
  prof.feedback.push({ pid, action, value, at: now(), predicted: r?.label ?? null, score: r ? Math.round(r.score * 1e4) / 1e4 : null });
  save();
}

function explain(r) {
  const prof = profile();
  const key = `${r.paper.id}|${r.label}`;
  if (!A.explained.has(key)) {
    const ev = evidence(r.paper, prof, A.df);
    A.explained.set(key, { ev, reason: reason(r.label, ev, r.f, prof.focus) });
  }
  return A.explained.get(key);
}

/** Papers you have judged in the feed (a vote or a save). */
function ratedIds(prof) {
  return [...states(prof)].filter(([, x]) => x.vote || x.saved).map(([id]) => id);
}
/** Everything you have already decided: feed ratings, saves and hand labels. These leave the Read/Skim/Skip lists and live under Reviewed. */
function decidedIds(prof) {
  return new Set([...ratedIds(prof), ...Object.keys(prof.labels)]);
}
function likedIds(prof) {
  return [...states(prof)].filter(([, x]) => x.vote === "useful" || (x.saved && x.vote !== "not_useful")).map(([id]) => id);
}
const ratingsCount = prof => ratedIds(prof).length;
/** Hand labels tell the algorithm what you like just as well as feed ratings, so they also end the rating phase. */
const hasEnoughSignal = prof => ratingsCount(prof) + Object.keys(prof.labels).length >= MIN_RATINGS;

// -------------------------------------------------------------------- shell

function render() {
  if (!A.ready) return;
  const prof = profile();
  // A review set can be started straight from #/label, before having any profile.
  const view = prof ? A.view : A.view === "label" && A.proposals?.sets.length ? "review-start" : "welcome";
  document.body.dataset.view = view;
  renderHeader(prof);
  const main = $("#main");
  if (view === "welcome") main.innerHTML = welcomeView();
  else if (view === "review-start") main.innerHTML = `<section class="label-view"><header class="page-head"><h1>Label papers</h1>
      <p>Pick a set to review. The app creates a profile for it and walks you through each paper.</p></header>
      ${reviewPanel(null)}</section>`;
  else if (A.run?.error) main.innerHTML = `<section class="empty"><h2>Couldn’t rank papers</h2><p>${esc(A.run.error)}</p><button class="btn primary" data-act="edit-profile">Edit research interests</button></section>`;
  else if (!A.run) main.innerHTML = `<section class="empty"><div class="spinner"></div><p>Ranking papers…</p></section>`;
  else if (view === "saved") main.innerHTML = savedView(prof);
  else if (view === "label") main.innerHTML = labelView(prof);
  else if (view === "insights") main.innerHTML = insightsView(prof);
  else main.innerHTML = feedView(prof);
  bindCharts(main);
  if (view === "insights" && !A.report && A.run && !A.run.error) setTimeout(computeReport, 30);
}

function renderHeader(prof) {
  const accountBtn = $("#account-btn");
  accountBtn.hidden = !account.acct.available;
  accountBtn.innerHTML = account.acct.user ? `${icon("user")}${esc(account.acct.user.username)}` : "Sign in";
  const nav = $("#nav");
  nav.hidden = !prof;
  if (prof) {
    const s = states(prof);
    const saved = [...s.values()].filter(x => x.saved).length;
    const labels = Object.keys(prof.labels).length;
    const items = [["feed", "For you", ""], ["saved", "Saved", saved || ""], ["label", "Label", labels || ""], ["insights", "Insights", ""]];
    nav.innerHTML = items.map(([v, t, n]) => `<a href="#/${v}" class="${A.view === v ? "active" : ""}" ${A.view === v ? 'aria-current="page"' : ""}>${t}${n !== "" ? `<span class="count">${n}</span>` : ""}</a>`).join("");
  }
  const names = Object.values(A.state.profiles);
  $("#profile-menu").hidden = !prof;
  if (prof) {
    $("#profile-name").textContent = prof.name;
    $("#profile-list").innerHTML = names.map(p => `<button type="button" role="menuitem" data-act="switch-profile" data-id="${esc(p.id)}" class="${p.id === prof.id ? "current" : ""}">${icon(p.id === prof.id ? "check" : "user")}${esc(p.name)}</button>`).join("")
      + `<hr><button type="button" role="menuitem" data-act="edit-profile">${icon("edit")}Edit research interests</button><button type="button" role="menuitem" data-act="new-profile">${icon("plus")}New profile</button>`;
  }
  renderModelPill();
}

function renderModelPill() {
  const el = $("#model-pill");
  const e = A.embedder;
  const semanticOn = A.state?.settings.semantic !== false;
  let text, cls, tip;
  if (!semanticOn) { text = "Keyword ranking"; cls = "off"; tip = "Semantic model is off (Settings). Ranking uses TF-IDF keywords."; }
  else if (e.status === "ready") { text = "Semantic model ready"; cls = "ready"; tip = "MiniLM sentence embeddings run in your browser."; }
  else if (e.status === "error") { text = "Keyword ranking"; cls = "off"; tip = `Semantic model unavailable (${e.error}). Ranking uses TF-IDF keywords.`; }
  else { text = `Loading semantic model ${e.progress ? Math.round(e.progress * 100) + "%" : "…"}`; cls = "loading"; tip = "Downloading MiniLM (~23 MB, once). Keyword ranking is used meanwhile."; }
  el.className = `pill model ${cls}`;
  el.title = tip;
  el.innerHTML = `<span class="dot"></span>${esc(text)}`;
}

// ------------------------------------------------------------------ welcome

const SHORT_AREA = { ai: "AI & ML", computing: "Computing", physical: "Maths & physics", life: "Biology & medicine", environment: "Earth & environment", society: "Society & economics", humanities: "Arts & humanities" };

function fieldChips(selected, name = "fields", { noneMeansAll = true, short = false } = {}) {
  const sel = new Set(selected.length || !noneMeansAll ? selected : A.catalog.areas.map(a => a.id));
  return A.catalog.areas.map(a => `<label class="chip-toggle" title="${esc(a.label)} · ${a.count} papers"><input type="checkbox" name="${name}" value="${esc(a.id)}" ${sel.has(a.id) ? "checked" : ""}><span>${esc(short ? SHORT_AREA[a.id] || a.label : a.label)}${short ? "" : ` <small>${a.count}</small>`}</span></label>`).join("");
}

function welcomeView() {
  const m = A.catalog.manifest;
  const total = A.catalog.papers.length;
  return `<section class="welcome">
    <div class="hero">
      <p class="eyebrow">${icon("spark")} Research paper triage</p>
      <h1>Too many papers.<br><em>Read the right ones.</em></h1>
      <p class="lede">Papers are cheaper to produce than ever, and your reading time hasn’t grown. Pick your fields and rate a few papers. Paper Triage then sorts the rest into <b class="t-read">Read</b>, <b class="t-skim">Skim</b> and <b class="t-skip">Skip</b>, says why, shows whether each one is worth trusting, and keeps learning as you rate.</p>
    </div>
    <form class="start card-surface" id="start-form">
      <label class="field"><span>What are you working on? <small>optional, helps the first suggestions</small></span>
        <textarea name="description" rows="3" placeholder="e.g. I study how retrieval-augmented generation systems can be evaluated for faithfulness."></textarea></label>
      <div class="examples" role="group" aria-label="Examples"><span>Try:</span>${EXAMPLES.map((e, i) => `<button type="button" class="example" data-act="example" data-i="${i}">${esc(e.name)}</button>`).join("")}</div>
      <label class="field"><span>Keywords <small>optional, comma-separated</small></span><input name="keywords" placeholder="e.g. RAG, hallucination, LLM evaluation"></label>
      <fieldset class="field"><legend>Fields to include <small>optional, leave empty for all</small></legend><div class="chips one-line">${fieldChips([], "fields", { noneMeansAll: false, short: true })}</div></fieldset>
      <div class="start-actions"><button class="btn primary big" type="submit">Show papers ${icon("chevron", "rot")}</button>
      <label class="btn ghost file">${icon("upload")}Restore a backup<input type="file" accept=".json,application/json" data-act="restore" hidden></label></div>
    </form>
    <ol class="how">
      <li><b>Pick</b><span>Choose your fields, and describe your research if you like. You get 20 papers to start.</span></li>
      <li><b>Rate</b><span>Mark at least ${MIN_RATINGS} as relevant or not. Load more papers from OpenAlex, Europe PMC and Crossref whenever you want.</span></li>
      <li><b>Triage</b><span>The algorithm sorts every other paper into Read, Skim and Skip with a reason and trust signals, and re-sorts as you keep rating.</span></li>
    </ol>
    ${A.proposals?.sets.length ? `<p class="fine">${icon("check")}<span>Labelling papers for the project dataset? <a href="#/label">Review suggested labels</a>.</span></p>` : ""}
    <p class="fine">${icon("shield")}<span>Runs entirely in your browser. No account, and your interests and ratings never leave this device. Catalog: ${total.toLocaleString()} papers from <a href="https://openalex.org" target="_blank" rel="noopener">OpenAlex</a> (CC0), updated ${esc(date(m.updated_at))}.</span></p>
  </section>`;
}

// --------------------------------------------------------------------- feed

function rowsFor(prof, { tab = A.tab, ignoreTab = false } = {}) {
  const s = states(prof);
  const fields = new Set(prof.fields.length ? prof.fields : [...A.catalog.areas.map(a => a.id), "imported"]);
  fields.add("imported");
  const q = A.query.trim().toLowerCase();
  const decided = decidedIds(prof);
  let rows = A.run.results.filter(r => {
    const st = s.get(r.paper.id) || {};
    if (!fields.has(r.paper.area)) return false;
    for (const f of A.signalFilters) if (!SIGNAL_FILTERS[f].test(r.paper._sig)) return false;
    if (q && !`${r.paper.title} ${r.paper.abstract} ${(r.paper.authors || []).join(" ")}`.toLowerCase().includes(q)) return false;
    const done = decided.has(r.paper.id);
    if (ignoreTab) return !st.hidden && !done;
    if (tab === "HIDDEN") return st.hidden;
    if (st.hidden) return false;
    if (tab === "RATED") return done;
    if (done) return false; // you already judged it; the tabs hold what the algorithm sorted
    const label = st.corrected || r.label;
    if (tab === "NEW") return A.newIds.has(r.paper.id) && label !== "SKIP";
    return tab === "ALL" || label === tab;
  });
  const shown = new Set(rows.map(r => r.paper.id));
  rows = rows.filter(r => !(r.lead && shown.has(r.lead)));
  if (A.sort === "trust") rows = [...rows].sort((a, b) => (b.score + 0.02 * b.paper._sig.points) - (a.score + 0.02 * a.paper._sig.points));
  else if (A.sort === "newest") rows = [...rows].sort((a, b) => (b.paper.published || "").localeCompare(a.paper.published || ""));
  return { rows, states: s };
}

/** First phase: a fixed list of papers to rate; nothing is labelled yet. */
function starterIds(prof) {
  if (!prof.starter?.length) {
    const fields = new Set(prof.fields.length ? prof.fields : A.catalog.areas.map(a => a.id));
    const st = states(prof);
    prof.starter = A.run.results.filter(r => (fields.has(r.paper.area) || r.paper.area === "imported") && !st.get(r.paper.id)?.hidden)
      .slice(0, PAGE).map(r => r.paper.id);
    save();
  }
  return prof.starter.filter(id => A.byId.has(id));
}

function rateView(prof) {
  const st = states(prof);
  const n = ratingsCount(prof);
  const liked = likedIds(prof).length;
  const ids = starterIds(prof);
  const needLike = isEmptyProfile(prof) && n >= MIN_RATINGS && !liked;
  const desc = prof.focus || prof.description;
  return `<section class="feed rate-phase">
    <header class="feed-head">
      <div>
        <h1>${esc(prof.name)}</h1>
        <p class="interest">${desc ? esc(desc.slice(0, 220)) + (desc.length > 220 ? "…" : "") : esc(fieldNames(prof))}</p>
      </div>
    </header>
    <div class="rate-progress" role="status">
      <div><b>Rate at least ${MIN_RATINGS} papers</b> as relevant or not relevant. Then the algorithm sorts every other paper into Read, Skim and Skip, and keeps re-sorting as you rate more.
        ${needLike ? "<br><span class=\"warn-text\">Mark at least one paper as relevant so it knows what you like.</span>" : ""}</div>
      <div class="rate-meter"><i style="width:${Math.min(100, (n / MIN_RATINGS) * 100)}%"></i></div>
      <span class="rate-count">${Math.min(n, MIN_RATINGS)}/${MIN_RATINGS}</span>
    </div>
    <div class="list rate-list">
      ${ids.map(id => rateCard(A.byId.get(id), st.get(id) || {})).join("")}
      <button class="btn more" data-act="load-more">${icon("globe")}Load ${PAGE} more papers <span>fresh from the web</span></button>
    </div>
  </section>`;
}

function fieldNames(prof) {
  const ids = prof.fields.length ? prof.fields : A.catalog.areas.map(a => a.id);
  return ids.length === A.catalog.areas.length ? "Fields: all" : "Fields: " + ids.map(id => SHORT_AREA[id] || id).join(" · ");
}

function rateCard(p, st) {
  const open = A.expanded.has(p.id);
  const url = safeUrl(p.url);
  const area = areaLabel(p);
  const meta = [authors(p.authors), p.venue, date(p.published || String(p.year || ""))].filter(Boolean).map(esc).join(" · ");
  const abs = p.abstract || "No abstract available.";
  const b = badges(p._sig, p);
  return `<article class="paper unrated ${st.vote === "useful" ? "liked" : st.vote === "not_useful" ? "disliked" : ""} ${A.focusId === p.id ? "focused" : ""}" data-id="${esc(p.id)}" tabindex="-1">
    <div class="paper-top"><span class="area">${esc(area)}</span>${st.vote ? `<span class="voted ${st.vote}">${st.vote === "useful" ? "You: relevant" : "You: not relevant"}</span>` : ""}</div>
    <h2>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener" data-act="open-paper">${esc(p.title)}</a>` : esc(p.title)}</h2>
    <p class="meta">${meta}</p>
    <p class="abstract short">${esc(open || abs.length <= 420 ? abs : abs.slice(0, 420).replace(/\s+\S*$/, "") + "…")}${abs.length > 420 ? ` <button class="link" data-act="expand">${open ? "less" : "more"}</button>` : ""}</p>
    ${b.length ? `<ul class="signals" aria-label="Worth-it signals">${b.map(x => `<li class="sig ${x.tone}" title="${esc(x.tip)}">${esc(x.text)}</li>`).join("")}</ul>` : ""}
    <div class="actions rate-actions">
      <button class="btn rate up ${st.vote === "useful" ? "on" : ""}" data-act="vote" data-v="useful" aria-pressed="${st.vote === "useful"}" title="Relevant (U)">${icon("up")}Relevant</button>
      <button class="btn rate down ${st.vote === "not_useful" ? "on" : ""}" data-act="vote" data-v="not_useful" aria-pressed="${st.vote === "not_useful"}" title="Not relevant (N)">${icon("down")}Not relevant</button>
    </div>
  </article>`;
}

function feedView(prof) {
  if (!hasEnoughSignal(prof) || A.run.discover) return rateView(prof);
  const all = rowsFor(prof, { ignoreTab: true }).rows;
  const s = states(prof);
  const count = lab => all.filter(r => (s.get(r.paper.id)?.corrected || r.label) === lab).length;
  const counts = { READ: count("READ"), SKIM: count("SKIM"), SKIP: count("SKIP"), ALL: all.length, NEW: all.filter(r => A.newIds.has(r.paper.id) && (s.get(r.paper.id)?.corrected || r.label) !== "SKIP").length };
  const hiddenN = [...s.values()].filter(x => x.hidden).length;
  counts.RATED = decidedIds(prof).size;
  const { rows } = rowsFor(prof);
  const page = rows.slice(0, A.shown);
  const tabs = [["READ", "Read"], ["SKIM", "Skim"], ["SKIP", "Skip"], ["ALL", "All"], ["RATED", "Reviewed"]];
  if (counts.NEW) tabs.unshift(["NEW", "New"]);
  if ((A.tab === "NEW" && !counts.NEW) || (A.tab === "HIDDEN" && !hiddenN)) A.tab = "READ"; // that tab no longer exists
  const minutes = counts.READ * MINUTES_PER_READ;
  const budget = prof.hours * 60;
  const m = A.run.model;
  const teachers = counts.RATED + Object.keys(prof.labels).length; // ratings and hand labels both teach the model
  const engineLine = [
    A.run.space === A.dense ? "MiniLM semantic ranking" : "Keyword (TF-IDF) ranking",
    m.info.learned ? `learned from ${teachers} ratings and labels (weight ${pct(m.learnedWeight)}${A.run.blend.validated ? ", validated" : ""})` : `${teachers} ratings so far`,
    ...(A.run.space !== A.dense && A.embedder.status === "loading" ? ["switches to semantic ranking when the model loads"] : []),
  ].join(" · ");
  const kws = [...prof.keywords.slice(0, 5)];
  return `<section class="feed">
    <header class="feed-head">
      <div>
        <h1>${esc(prof.name)}</h1>
        <p class="interest">${(prof.focus || prof.description) ? esc((prof.focus || prof.description).slice(0, 220)) + ((prof.focus || prof.description).length > 220 ? "…" : "") : esc(fieldNames(prof))}</p>
        ${kws.length ? `<div class="kw">${kws.map(k => `<span>${esc(k)}</span>`).join("")}</div>` : ""}
      </div>
      <div class="budget" title="Read papers take about ${MINUTES_PER_READ} minutes each">
        <div class="budget-num"><b>${counts.READ}</b> to read this week</div>
        <div class="meter" role="img" aria-label="${Math.round(minutes / 60 * 10) / 10} of ${prof.hours} hours"><i style="width:${Math.min(100, (minutes / Math.max(1, budget)) * 100)}%"></i></div>
        <div class="budget-sub">≈ ${fmtHours(minutes)} of your ${prof.hours} h a week${prof.budget ? ` · Read capped at ${readBudget(prof.hours)}` : ""}</div>
      </div>
    </header>
    <p class="engine">${icon("spark")} ${engineLine}</p>
    <div class="feed-grid">
      <aside class="filters" aria-label="Filters">
        <details class="filters-box" ${(A.filtersOpen ?? matchMedia("(min-width: 900px)").matches) ? "open" : ""}>
          <summary>${icon("filter")} Filters ${A.signalFilters.size || A.query ? `<span class="count">${A.signalFilters.size + (A.query ? 1 : 0)}</span>` : ""}</summary>
          <label class="search">${icon("search")}<input type="search" id="search" placeholder="Search titles, abstracts, authors" value="${esc(A.query)}" aria-label="Search"></label>
          <h3>Sort</h3>
          <div class="radios">${Object.entries(SORTS).map(([k, t]) => `<label><input type="radio" name="sort" value="${k}" ${A.sort === k ? "checked" : ""}> ${t}</label>`).join("")}</div>
          <h3>Worth-it signals</h3>
          <div class="chips">${Object.entries(SIGNAL_FILTERS).map(([k, f]) => `<label class="chip-toggle small"><input type="checkbox" data-act="signal-filter" value="${k}" ${A.signalFilters.has(k) ? "checked" : ""}><span>${f.text}</span></label>`).join("")}</div>
          <h3>Fields</h3>
          <div class="chips"><label class="chip-toggle"><input type="checkbox" name="feed-fields-all" ${prof.fields.length ? "" : "checked"}><span>All fields</span></label>${fieldChips(prof.fields, "feed-fields", { noneMeansAll: false })}</div>
                  </details>
      </aside>
      <div class="list">
        <div class="tabs" role="tablist">${tabs.map(([k, t]) => `<button role="tab" aria-selected="${A.tab === k}" class="tab ${k}" data-act="tab" data-tab="${k}">${t}<span>${counts[k].toLocaleString()}</span></button>`).join("")}
          ${hiddenN ? `<button class="tab link ${A.tab === "HIDDEN" ? "on" : ""}" data-act="tab" data-tab="HIDDEN">Hidden ${hiddenN}</button>` : ""}</div>
        ${page.length ? page.map(r => card(r, s.get(r.paper.id) || {})).join("") : emptyTab(prof)}
        ${rows.length > A.shown ? `<button class="btn more" data-act="more">Show ${Math.min(PAGE, rows.length - A.shown)} more <span>${(rows.length - A.shown).toLocaleString()} left</span></button>`
          : A.tab !== "RATED" && A.tab !== "HIDDEN" ? `<button class="btn more" data-act="load-more" title="Fetches the past year's papers from your enabled sources (Settings → Sources). Only your keywords and fields are sent.">${icon("globe")}Load ${PAGE} more papers <span>fresh from the web</span></button>` : ""}
      </div>
    </div>
  </section>`;
}

const fmtHours = min => (min < 60 ? `${min} min` : `${Math.round((min / 60) * 10) / 10} h`);

function emptyTab(prof) {
  if (A.query || A.signalFilters.size) return `<div class="empty small"><p>No papers match these filters.</p><button class="btn small" data-act="clear-filters">Clear filters</button></div>`;
  if (A.tab === "READ") return `<div class="empty small"><p>Nothing clears the Read bar yet. Check <b>Skim</b>, rate a few more papers, or <button class="link" data-act="load-more">load more papers from the web</button>.</p></div>`;
  return `<div class="empty small"><p>No papers here.</p></div>`;
}

function card(r, st) {
  const p = r.paper;
  const label = st.corrected || r.label;
  const { reason: why, ev } = explain({ ...r, label });
  const sig = p._sig;
  const open = A.expanded.has(p.id);
  const area = areaLabel(p);
  const url = safeUrl(p.url);
  const meta = [authors(p.authors), p.venue, date(p.published || String(p.year || ""))].filter(Boolean).map(esc).join(" · ");
  const b = badges(sig, p);
  const similar = r.similar.map(id => A.run.byId.get(id)).filter(Boolean);
  return `<article class="paper ${label} ${A.focusId === p.id ? "focused" : ""}" data-id="${esc(p.id)}" tabindex="-1">
    <div class="paper-top">
      ${st.vote || st.saved ? `<span class="voted ${st.vote || "useful"}">${st.vote === "not_useful" ? "You: not relevant" : "You: relevant"}</span>` : profile().labels[p.id] ? `<span class="voted ${profile().labels[p.id].label === "SKIP" ? "not_useful" : "useful"}">You labelled: ${LABEL_TEXT[profile().labels[p.id].label ?? profile().labels[p.id]]}</span>` : `<span class="label-chip ${label}" title="${esc(LABEL_HELP[label])}: decided by the algorithm">${LABEL_TEXT[label]}</span>`}
      ${A.newIds.has(p.id) ? '<span class="new-chip">New</span>' : ""}
      ${area ? `<span class="area">${esc(area)}</span>` : ""}
      <span class="match" title="Relevance score ${r.score.toFixed(2)} (Read ≥ ${(A.run.cutoffs || profile().cutoffs).read.toFixed(2)}, Skim ≥ ${(A.run.cutoffs || profile().cutoffs).skim.toFixed(2)})"><i style="--v:${Math.round(r.score * 100)}%"></i>${Math.round(r.score * 100)}</span>
    </div>
    <h2>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener" data-act="open-paper">${esc(p.title)}</a>` : esc(p.title)}</h2>
    <p class="meta">${meta}</p>
    <p class="why">${esc(why)}${r.note && !st.corrected ? ` <span class="note">${esc(r.note)}</span>` : ""}</p>
    ${b.length ? `<ul class="signals" aria-label="Worth-it signals">${b.map(x => `<li class="sig ${x.tone}" title="${esc(x.tip)}">${esc(x.text)}</li>`).join("")}<li class="level ${sig.level}" title="${sig.points} signal points">${sig.level[0].toUpperCase() + sig.level.slice(1)} signals</li></ul>` : ""}
    <div class="actions">
      <div class="vote">
        <button class="icon-btn ${st.vote === "useful" ? "on up" : ""}" data-act="vote" data-v="useful" aria-pressed="${st.vote === "useful"}" title="Relevant (U)">${icon("up")}<span>Relevant</span></button>
        <button class="icon-btn ${st.vote === "not_useful" ? "on down" : ""}" data-act="vote" data-v="not_useful" aria-pressed="${st.vote === "not_useful"}" title="Not relevant (N)">${icon("down")}<span>Not relevant</span></button>
      </div>
      <button class="icon-btn ${st.saved ? "on saved" : ""}" data-act="save" aria-pressed="${!!st.saved}" title="Save to your reading list (S)">${icon("bookmark")}<span>${st.saved ? "Saved" : "Save"}</span></button>
      <button class="icon-btn" data-act="${st.hidden ? "unhide" : "hide"}" title="${st.hidden ? "Restore" : "Hide this paper (X)"}">${icon(st.hidden ? "undo" : "hide")}<span>${st.hidden ? "Restore" : "Hide"}</span></button>
      <button class="icon-btn more-btn" data-act="expand" aria-expanded="${open}" title="Details (Enter)">${icon("chevron", open ? "flip" : "")}<span>${open ? "Less" : "Details"}</span></button>
    </div>
    ${open ? details(r, ev, similar) : ""}
  </article>`;
}

function details(r, ev, similar) {
  const p = r.paper;
  const terms = highlightTerms(ev);
  const contrib = FEATURES.map((f, j) => ({ f, v: r.f[j], c: PRIOR_WEIGHTS[f] * r.f[j] }));
  const learnedShift = r.learned != null && A.run.model.learnedWeight ? r.score - r.prior : 0;
  const links = [
    safeUrl(p.url) && `<a href="${esc(p.url)}" target="_blank" rel="noopener">${icon("external")}Paper page</a>`,
    safeUrl(p.pdf_url) && `<a href="${esc(p.pdf_url)}" target="_blank" rel="noopener">${icon("download")}PDF</a>`,
    `<a href="https://scholar.google.com/scholar?q=${encodeURIComponent(p.title)}" target="_blank" rel="noopener">${icon("search")}Google Scholar</a>`,
  ].filter(Boolean).join("");
  const sig = p._sig;
  const facts = [
    ["Peer review", sig.peer_reviewed === true ? `Yes · ${p.venue_type || "journal"}` : sig.peer_reviewed === false ? "Not yet (preprint or repository)" : "Unknown"],
    ["Venue", p.venue ? `${p.venue}${p.venue_core ? " (established)" : ""}` : "—"],
    ["Type", p.work_type || "—"],
    ["Open access", sig.open_access ? (p.oa_status || "yes") : "No"],
    ["Citations", p.citation_count != null ? `${p.citation_count}${p.fwci != null ? ` · FWCI ${p.fwci.toFixed(2)}` : ""}` : "—"],
    ["Study design cues", sig.evidence.length ? sig.evidence.join(", ") : "None found in abstract"],
  ];
  return `<div class="details">
    ${ev.bestSentence ? `<blockquote class="best"><span>Most relevant sentence</span>${highlight(ev.bestSentence, terms)}</blockquote>` : ""}
    <p class="abstract">${highlight(p.abstract || "No abstract available.", terms)}</p>
    <div class="links">${links}</div>
    <div class="detail-grid">
      <div><h3>Why this score</h3>
        <table class="bars">${contrib.map(({ f, v, c }) => `<tr><th>${FEATURE_NAMES[f]}</th><td><span class="bar ${c < 0 ? "neg" : ""}" style="--w:${Math.min(100, Math.abs(v) * 100)}%"></span></td><td class="num">${c >= 0 ? "+" : "−"}${Math.abs(c).toFixed(2)}</td></tr>`).join("")}
        <tr class="total"><th>Profile score</th><td></td><td class="num">${r.prior.toFixed(2)}</td></tr>
        ${learnedShift ? `<tr class="total"><th>Learned from your ratings</th><td></td><td class="num">${learnedShift >= 0 ? "+" : "−"}${Math.abs(learnedShift).toFixed(2)}</td></tr>` : ""}
        <tr class="total"><th>Final relevance</th><td></td><td class="num"><b>${r.score.toFixed(2)}</b></td></tr></table>
      </div>
      <div><h3>Publication facts</h3><dl class="facts">${facts.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl></div>
    </div>
    ${similar.length ? `<h3>Near-identical papers (${similar.length})</h3><ul class="similar">${similar.map(s => `<li><span class="label-chip ${s.label}">${LABEL_TEXT[s.label]}</span>${safeUrl(s.paper.url) ? `<a href="${esc(s.paper.url)}" target="_blank" rel="noopener">${esc(s.paper.title)}</a>` : esc(s.paper.title)}</li>`).join("")}</ul>` : ""}
  </div>`;
}

// -------------------------------------------------------------------- saved

function savedView(prof) {
  const s = states(prof);
  const saved = [...s.entries()].filter(([, x]) => x.saved).map(([id]) => A.run.byId.get(id) || (A.byId.get(id) && { paper: A.byId.get(id), label: "READ", score: 0, prior: 0, f: FEATURES.map(() => 0), similar: [], note: "" })).filter(Boolean);
  return `<section class="saved">
    <header class="page-head"><h1>Saved papers</h1><p>Your reading list. Export it to Zotero, Mendeley or EndNote.</p>
      ${saved.length ? `<div class="export-row">
        <button class="btn small" data-act="export" data-fmt="bib">${icon("download")}BibTeX</button>
        <button class="btn small" data-act="export" data-fmt="ris">${icon("download")}RIS</button>
        <button class="btn small" data-act="export" data-fmt="csv">${icon("download")}CSV</button></div>` : ""}</header>
    ${saved.length ? saved.map(r => card(r, s.get(r.paper.id) || {})).join("") : `<div class="empty"><p>Nothing saved yet. Use ${icon("bookmark")} <b>Save</b> on any paper.</p><a class="btn" href="#/feed">Back to your papers</a></div>`}
  </section>`;
}

// -------------------------------------------------------------------- label

function labelQueue(prof) {
  const set = reviewSet(prof);
  const strategy = set && A.labelStrategy === "review" ? "review" : A.labelStrategy === "review" ? "balanced" : A.labelStrategy;
  const key = `${prof.id}|${strategy}`;
  if (A.labelQueue?.key === key) return A.labelQueue.ids.filter(id => !prof.labels[id]);
  if (strategy === "review") {
    A.labelQueue = { key, ids: set.items.map(i => i.paper_id).filter(id => A.byId.has(id) && !prof.labels[id]) };
    return A.labelQueue.ids;
  }
  const fields = new Set(prof.fields.length ? prof.fields : A.catalog.areas.map(a => a.id));
  const cands = A.run.results.filter(r => !prof.labels[r.paper.id] && (fields.has(r.paper.area) || r.paper.area === "imported"));
  let rng = 1234567;
  const rand = () => ((rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648);
  const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  let ids;
  if (strategy === "random") ids = shuffle(cands.map(r => r.paper.id));
  else if (strategy === "cutoffs") {
    const c = prof.cutoffs;
    ids = [...cands].sort((a, b) => Math.min(Math.abs(a.prior - c.read), Math.abs(a.prior - c.skim)) - Math.min(Math.abs(b.prior - c.read), Math.abs(b.prior - c.skim))).map(r => r.paper.id);
  } else { // balanced: five bands by profile score, interleaved from the top band
    const sorted = [...cands].sort((a, b) => b.prior - a.prior);
    const size = Math.max(1, Math.ceil(sorted.length / 5));
    const bands = [0, 1, 2, 3, 4].map(i => shuffle(sorted.slice(i * size, (i + 1) * size).map(r => r.paper.id)));
    ids = [];
    while (bands.some(b => b.length)) for (const b of bands) if (b.length) ids.push(b.pop());
  }
  A.labelQueue = { key, ids };
  return ids;
}

/** The suggestion set this profile is reviewing, if any. */
function reviewSet(prof) {
  return A.proposals?.sets.find(s => s.slug === prof.reviewSet) || null;
}

function suggestionFor(prof, pid) {
  return reviewSet(prof)?.items.find(i => i.paper_id === pid) || null;
}

function reviewPanel(prof) {
  const sets = A.proposals?.sets || [];
  if (!sets.length) return "";
  const rows = sets.map(set => {
    const owner = Object.values(A.state.profiles).find(p => p.reviewSet === set.slug);
    const done = owner ? Object.values(owner.labels).filter(l => l.proposal === set.slug).length : 0;
    const changed = owner ? Object.values(owner.labels).filter(l => l.proposal === set.slug && l.label !== l.proposed).length : 0;
    const active = !!prof && owner?.id === prof.id && A.labelStrategy === "review";
    return `<li class="${active ? "active" : ""}"><div><b>${esc(set.name)}</b><span>${done} of ${set.items.length} reviewed${done ? ` · you changed ${changed}` : ""}</span></div>
      <button class="btn small ${active ? "" : "primary"}" data-act="review-set" data-slug="${esc(set.slug)}" ${active ? "disabled" : ""}>${active ? "Reviewing" : done ? "Continue" : "Review"}</button></li>`;
  }).join("");
  return `<section class="review-sets card-surface">
    <h2>${icon("check")} Review suggested labels</h2>
    <p>Each paper comes with a suggested label proposed by an AI model from the title and abstract (it never sees the ranker’s scores). Check every suggestion: <kbd>Enter</kbd> accepts, <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> give your own answer. Both the suggestion and your answer are saved, so the dataset records them as manual labels.</p>
    <ul>${rows}</ul>
  </section>`;
}

function labelView(prof) {
  const n = Object.keys(prof.labels).length;
  const counts = Object.fromEntries(LABELS.map(l => [l, Object.values(prof.labels).filter(x => x.label === l).length]));
  const queue = labelQueue(prof);
  const p = A.byId.get(queue[0]);
  const reviewing = A.labelStrategy === "review" && !!reviewSet(prof);
  const sug = p && reviewing ? suggestionFor(prof, p.id) : null;
  return `<section class="label-view">
    <header class="page-head"><h1>Label papers</h1>
      <p>Hand labels are the ground truth for <a href="#/insights">Insights</a> and for the published dataset. The ranker’s prediction is hidden while you label, so it can’t sway you.</p></header>
    ${reviewing ? "" : reviewPanel(prof)}
    <div class="label-stats">
      <div class="progress"><div class="progress-num"><b>${n}</b> labelled <span>· goal ${DATASET_GOAL}</span></div><div class="meter"><i style="width:${Math.min(100, (n / DATASET_GOAL) * 100)}%"></i></div>
      <div class="split">${LABELS.map(l => `<span class="t-${l.toLowerCase()}">${LABEL_TEXT[l]} ${counts[l]}</span>`).join("")}</div></div>
      <div class="label-controls">
        <label>Sampling <select id="label-strategy">
          ${reviewSet(prof) ? `<option value="review" ${A.labelStrategy === "review" ? "selected" : ""}>Review suggestions</option>` : ""}
          <option value="balanced" ${A.labelStrategy === "balanced" ? "selected" : ""}>Balanced across relevance</option>
          <option value="random" ${A.labelStrategy === "random" ? "selected" : ""}>Random</option>
          <option value="cutoffs" ${A.labelStrategy === "cutoffs" ? "selected" : ""}>Closest to the cutoffs</option></select></label>
      </div>
    </div>
    ${p ? `<article class="label-card" data-id="${esc(p.id)}">
      <p class="meta">${esc([A.catalog.areas.find(a => a.id === p.area)?.label, p.venue, date(p.published)].filter(Boolean).join(" · "))}</p>
      <h2>${safeUrl(p.url) ? `<a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a>` : esc(p.title)}</h2>
      <p class="meta">${esc(authors(p.authors, 6))}</p>
      <p class="abstract">${esc(p.abstract || "No abstract.")}</p>
      <p class="label-q">For <b>${esc(prof.name)}</b>, this paper is…</p>
      ${sug ? `<div class="suggest"><span class="label-chip ${sug.proposed_label}">Suggested: ${LABEL_TEXT[sug.proposed_label]}</span><span>${esc(sug.reason)}</span>
        <button class="btn small primary" data-act="hand-label" data-v="ACCEPT"><kbd>Enter</kbd> Accept</button></div>` : ""}
      <div class="label-buttons">${LABELS.map((l, i) => `<button class="lbtn ${l}" data-act="hand-label" data-v="${l}"><kbd>${i + 1}</kbd>${LABEL_TEXT[l]}<small>${LABEL_HELP[l].split(":")[0]}</small></button>`).join("")}
        <button class="lbtn unsure" data-act="hand-label" data-v=""><kbd>0</kbd>Not sure<small>Show me later</small></button></div>
      <div class="label-foot">${A.labelUndo.length ? `<button class="link" data-act="label-undo">${icon("undo")}Undo last (Backspace)</button>` : "<span></span>"}<span>${queue.length.toLocaleString()} papers left in this queue</span></div>
    </article>` : `<div class="empty"><p>You’ve labelled every paper in your fields.</p></div>`}
    ${reviewing ? reviewPanel(prof) : ""}
    <section class="dataset card-surface">
      <h2>${icon("layers")} Your labels as a dataset</h2>
      <p>One row per paper: the research profile, the paper’s id, your label, when you set it, and that it is a manual label. Paper details stay in the catalog. Merge a teammate’s file with <b>Import labels</b>.</p>
      <div class="export-row">
        <button class="btn small" data-act="labels-export" data-fmt="csv" ${n ? "" : "disabled"}>${icon("download")}Labels CSV</button>
        <button class="btn small" data-act="labels-export" data-fmt="jsonl" ${n ? "" : "disabled"}>${icon("download")}Labels JSONL</button>
        <label class="btn small ghost file">${icon("upload")}Import labels<input type="file" accept=".json,.jsonl,.csv" data-act="labels-import" hidden></label>
      </div>
    </section>
  </section>`;
}

/** One row per label, in the shape of data/labels/labels.jsonl: no paper details, those live in the catalog. */
function labelRows(prof) {
  return Object.entries(prof.labels).map(([id, l]) => ({
    profile: slug(prof.name), paper_id: id, label: l.label, labeled_at: l.at, origin: "manual",
  }));
}

// ----------------------------------------------------------------- insights

function computeReport() {
  const prof = profile();
  if (!prof || !A.run?.ctx) return;
  try { A.report = evaluate(A.run.ctx, prof.good || "read"); } catch (e) { console.error(e); A.report = { ok: false, message: e.message }; }
  if (A.view === "insights") { render(); setTimeout(computeCurve, 30); }
}

function computeCurve() {
  const prof = profile();
  if (!prof || !A.run?.ctx || A.curve) return;
  const seq = rankSeq;
  let curve;
  try { curve = learningCurve(A.run.ctx, { good: prof.good || "read" }); } catch (e) { console.error(e); curve = { ok: false, message: e.message }; }
  if (seq !== rankSeq) return; // ranking changed while computing: the next report recomputes
  A.curve = curve;
  if (A.view === "insights") render();
}

function learningSection() {
  const head = `<h2>Do the recommendations get better as you label?</h2>`;
  if (!A.curve) return `<div class="card-surface">${head}<div class="empty"><div class="spinner"></div><p>Rebuilding the recommender from fewer and fewer of your labels…</p></div></div>`;
  if (!A.curve.ok) return `<div class="card-surface">${head}<p class="hint">${esc(A.curve.message)}</p></div>`;
  const c = A.curve;
  const first = c.points[0], last = c.points[c.points.length - 1];
  const p = x => `${Math.round(x * 100)}%`;
  const lines = [
    c.gain >= 0.02
      ? `After learning from ${last.k} of your labels it ranks your relevant papers at <b>${p(last.rec)}</b>, against ${p(last.prior)} from your hand-written profile alone (a random order scores ${p(c.random)}).${c.ahead ? ` It pulls clearly ahead at about <b>${c.ahead} labels</b>.` : ""}`
      : `Your profile already ranks your relevant papers about as well as your labels can teach it (<b>${p(last.prior)}</b> from the profile alone, ${p(last.rec)} after learning from ${last.k} labels; random scores ${p(c.random)}). More labels haven’t improved it, because what you wrote matches what you pick. Learning pays off when your taste differs from your description.`,
    `The recommender decides by cross-validation how far to trust your labels: ${p(first.weight)} with ${first.k} labels, ${p(last.weight)} with ${last.k}.`,
    ...(c.skimCounted ? [`Skim counts as relevant here, because only ${c.counts.READ} of your labels are Read.`] : []),
  ];
  const series = [
    { name: "Your profile alone", key: "prior", cls: "s2", band: ["priorLo", "priorHi"] },
    { name: "After learning from your labels", key: "rec", cls: "s1", band: ["recLo", "recHi"] },
    { name: "Random order", key: "random", cls: "ref" },
  ];
  const points = c.points.map(x => ({ ...x, random: c.random }));
  return `<div class="card-surface">${head}
    ${learningChart(points, series, { title: "How high up the list your relevant papers rank, by labels learned from", yLabel: "Higher is better: 100% means every relevant paper is ranked above every other", note: `Each point rebuilds the recommender from only that many of your labels, then ranks papers it never saw (your other ${c.n} labelled papers, the same 5-fold split as the report). Shaded bands show the spread over ${c.repeats} random draws of the training labels.` })}
    <ul class="curve-notes">${lines.map(l => `<li>${l}</li>`).join("")}</ul>
    ${c.note ? `<p class="hint warn-text">${esc(c.note)}</p>` : ""}</div>`;
}

function insightsView(prof) {
  const n = Object.keys(prof.labels).length;
  const m = A.run.model;
  const model = `<section class="card-surface how-model">
    <h2>${icon("layers")} How your ranking works</h2>
    <ol class="pipeline">
      <li><b>Off-the-shelf model</b><span>${A.run.space === A.dense ? "MiniLM (all-MiniLM-L6-v2) turns your description and every paper into 384-number meaning vectors, in your browser." : "TF-IDF keyword vectors (MiniLM loads in the background)."}</span></li>
      <li><b>Seven readable signals</b><span>Similarity to your research, focus and keywords; exact keyword hits; recency; excluded topics; similarity to papers you rated.</span></li>
      <li><b>Profile score</b><span>A fixed, hand-set weighting of those signals. It works from your first visit.</span></li>
      <li><b>Model trained from scratch</b><span>A ridge regression on your ratings and labels (${m.info.n_examples} so far). ${m.info.learned ? `It counts for ${pct(m.learnedWeight)} of the final score${A.run.blend.validated ? ", a weight chosen by cross-validation on your hand labels" : " (capped until your hand labels can validate it)"}.` : esc(m.info.reason || "")}</span></li>
    </ol>
    <table class="weights"><thead><tr><th>Signal</th><th>Profile weight</th><th>Learned weight</th></tr></thead><tbody>
      ${FEATURES.map(f => `<tr><td>${FEATURE_NAMES[f]}</td><td class="num">${PRIOR_WEIGHTS[f].toFixed(2)}</td><td class="num">${m.info.coefficients ? fix(m.info.coefficients[f]) : "—"}</td></tr>`).join("")}
    </tbody></table>
  </section>`;
  if (n < MIN_LABELS) {
    return `<section class="insights"><header class="page-head"><h1>Is it working?</h1><p>Insights compare the ranking with your own hand labels, using cross-validation so no paper is scored by a model that saw its label.</p></header>
      <div class="empty"><p>Label at least <b>${MIN_LABELS}</b> papers (you have ${n}) to see how well the ranking finds what you’d pick yourself. 50+ labels give stable numbers.</p><a class="btn primary" href="#/label">Start labelling</a></div>${model}</section>`;
  }
  const rep = A.report;
  if (!rep) return `<section class="insights"><header class="page-head"><h1>Is it working?</h1></header><div class="empty"><div class="spinner"></div><p>Cross-validating on ${n} labels…</p></div>${model}</section>`;
  const good = prof.good || "read";
  const goodToggle = `<div class="seg inline" role="group" aria-label="Good papers are">${[["read", "Good = Read"], ["read_skim", "Good = Read or Skim"]].map(([k, t]) => `<button class="${good === k ? "on" : ""}" data-act="good-mode" data-v="${k}">${t}</button>`).join("")}</div>`;
  if (!rep.ok) return `<section class="insights"><header class="page-head"><h1>Is it working?</h1>${rep.systems ? goodToggle : ""}</header><div class="empty"><p>${esc(rep.message)}</p><a class="btn" href="#/label">Label more papers</a></div>${model}</section>`;
  const sys = rep.systems[0];
  const prior = rep.systems[1];
  const chart = discoveryChart([
    { name: sys.name.split(" (")[0], values: sys.disc.found, cls: "s1" },
    { name: "Profile only", values: prior.disc.found, cls: "s2" },
    { name: "Random order", values: rep.random, cls: "ref" },
  ]);
  const tile = (label, value, sub) => `<div class="tile"><div class="tl">${label}</div><div class="tv">${value}</div><div class="ts">${sub}</div></div>`;
  const cm = rep.current.confusion;
  return `<section class="insights">
    <header class="page-head"><h1>Is it working?</h1>
      <p>Measured on your ${rep.n} hand labels (${rep.nGood} count as good) with 5-fold cross-validation.</p>
      ${goodToggle}</header>
    <div class="tiles">
      ${tile("Good papers in the top 10", pct(sys.disc.hit10), `random order: ${pct(rep.nGood / rep.n)}`)}
      ${tile("Papers to screen for 80% of good ones", sys.disc.p80 ?? "—", `random order: ${rep.random80}`)}
      ${tile("Screening time saved", rep.minutesSaved != null ? fmtHours(rep.minutesSaved) : "—", `at ${SCREEN_MINUTES} min per abstract`)}
      ${tile("Learning weight", pct(rep.blend.weight ?? 0), rep.blend.weight ? "beats profile-only in validation" : "off: learning didn’t beat profile-only yet")}
    </div>
    <div class="card-surface">${chart}</div>
    ${learningSection()}
    <div class="two">
      <div class="card-surface"><h2>Ranking quality</h2>
        <table class="metrics"><thead><tr><th>Method</th><th title="Ranking quality of the top 10 (1 = perfect)">NDCG@10</th><th title="Average precision for good papers">AP</th><th title="Rank correlation with your labels">Spearman ρ</th></tr></thead>
        <tbody>${rep.systems.map(s => `<tr><td>${esc(s.name)}</td><td class="num">${fix(s.ndcg)}</td><td class="num">${fix(s.ap)}</td><td class="num">${fix(s.rho)}</td></tr>`).join("")}
        <tr class="muted"><td>Random order (expected)</td><td class="num">—</td><td class="num">${fix(rep.randomAp)}</td><td class="num">0.00</td></tr></tbody></table>
      </div>
      <div class="card-surface"><h2>Labels at your cutoffs</h2>
        <table class="confusion"><thead><tr><th></th>${LABELS.map(l => `<th>Predicted ${LABEL_TEXT[l]}</th>`).join("")}</tr></thead>
        <tbody>${LABELS.map((l, i) => `<tr><th>You said ${LABEL_TEXT[l]}</th>${cm[i].map((v, j) => `<td class="${i === j ? "diag" : ""}">${v}</td>`).join("")}</tr>`).join("")}</tbody></table>
        <p class="hint">Macro-F1 ${fix(rep.current.macroF1)} · accuracy ${pct(rep.current.accuracy)} · ${rep.current.severe} Read↔Skip mix-ups.</p>
        <p class="hint">Suggested cutoffs: Read ≥ ${rep.tuned.read.toFixed(2)}, Skim ≥ ${rep.tuned.skim.toFixed(2)} (macro-F1 ${fix(rep.tuned.m.macroF1)}). Tuned on the same labels, so the gain is optimistic.
        <button class="btn small" data-act="apply-cutoffs" data-read="${rep.tuned.read}" data-skim="${rep.tuned.skim}">Apply</button></p>
      </div>
    </div>
    ${model}
  </section>`;
}

// ------------------------------------------------------------------ dialogs

function openDialog(id, html) {
  const d = $(id);
  d.querySelector(".dialog-body").innerHTML = html;
  if (!d.open) d.showModal();
  return d;
}

function profileDialog(prof, isNew = false) {
  const p = prof || newProfile();
  const seeds = p.seeds.map(id => A.byId.get(id)).filter(Boolean);
  openDialog("#profile-dialog", `<form method="dialog" id="profile-form" data-new="${isNew}">
    <h2>${isNew ? "New profile" : "Research interests"}</h2>
    <label class="field"><span>Profile name</span><input name="name" required maxlength="80" value="${esc(isNew ? "" : p.name)}" placeholder="e.g. Thesis: RAG evaluation"></label>
    <label class="field"><span>Research description</span><textarea name="description" rows="4" placeholder="A few sentences about your current research.">${esc(p.description)}</textarea></label>
    <label class="field"><span>Keywords <small>comma-separated · matched exactly and used for live search</small></span><input name="keywords" value="${esc(p.keywords.join(", "))}"></label>
    <label class="field"><span>Current focus <small>weighted most heavily</small></span><input name="focus" value="${esc(p.focus)}" placeholder="What you’re working on this month"></label>
    <label class="field"><span>Exclude topics <small>comma-separated</small></span><input name="avoid" value="${esc(p.avoid.join(", "))}"></label>
    <label class="field narrow"><span>Reading time per week (hours)</span><input name="hours" type="number" min="0.5" max="40" step="0.5" value="${p.hours}"></label>
    <fieldset class="field"><legend>Fields</legend><div class="chips">${fieldChips(p.fields, "profile-fields")}</div></fieldset>
    <fieldset class="field"><legend>Papers you already love <small>DOIs or arXiv IDs · looked up on OpenAlex · pull the ranking towards them</small></legend>
      <div class="seed-row"><textarea name="seeds" rows="2" placeholder="10.18653/v1/2020.emnlp-main.550&#10;2005.11401"></textarea><button type="button" class="btn small" data-act="lookup-seeds">Look up</button></div>
      <ul class="seed-list">${seeds.map(s => `<li data-id="${esc(s.id)}">${esc(s.title)} <button type="button" class="link" data-act="remove-seed">${icon("x")}</button></li>`).join("")}</ul>
    </fieldset>
    <p class="form-error" hidden></p>
    <div class="dialog-actions">${!isNew && Object.keys(A.state.profiles).length ? `<button type="button" class="btn danger ghost" data-act="delete-profile">Delete profile</button>` : "<span></span>"}
      <div><button type="button" class="btn ghost" data-act="close-dialog">Cancel</button><button type="submit" class="btn primary">Save and rank</button></div></div>
  </form>`);
  A.pendingSeeds = [...p.seeds];
}

// ------------------------------------------------------------------ account

function accountDialog(message = "") {
  const { user } = account.acct;
  if (!user) {
    openDialog("#account-dialog", `<form method="dialog" id="account-form">
      <h2>Your account</h2>
      <p class="hint">Sign in to keep your profiles, ratings and labels on the server, so the ranking is personal to you on any device. Your profiles in this browser are added to the account the first time. <b>Sign in</b> uses an existing account; <b>Create account</b> makes a new one.</p>
      <label class="field"><span>Username</span><input name="username" autocomplete="username" required minlength="3" maxlength="40" autofocus></label>
      <label class="field"><span>Password <small>new accounts need at least 8 characters</small></span><input name="password" type="password" autocomplete="current-password" required maxlength="200"></label>
      <p class="form-error" ${message ? "" : "hidden"}>${esc(message)}</p>
      <div class="dialog-actions"><button type="button" class="btn ghost" data-act="close-dialog">Cancel</button>
        <div><button type="submit" class="btn" value="register">Create account</button><button type="submit" class="btn primary" value="login">Sign in</button></div></div>
    </form>`);
    return;
  }
  openDialog("#account-dialog", `<div id="account-panel">
    <h2>${icon("user")}${esc(user.username)}</h2>
    <p class="hint">Your profiles, ratings and labels sync to your account${account.acct.conflict ? ". <b>Syncing is paused: your account changed on another device. Reload to continue.</b>" : "."}</p>
    <h3>Starter labels <small>from this project’s labelled data</small></h3>
    <p class="hint">Add a ready-made profile with its labels. Your own labels are saved separately once you add them.</p>
    <ul class="seed-sets" id="seed-sets"><li class="hint">Loading…</li></ul>
    <div class="dialog-actions"><button type="button" class="btn danger ghost" data-act="sign-out">Sign out</button>
      <button type="button" class="btn primary" data-act="close-dialog">Done</button></div>
  </div>`);
  account.seedSets().then(sets => {
    const ul = $("#seed-sets");
    if (!ul) return;
    ul.innerHTML = sets.map(x => `<li><div><b>${esc(x.name)}</b> <span class="pill ${x.origin === "manual" ? "ok" : ""}">${x.origin === "manual" ? "manual" : x.origin === "synthetic" ? "synthetic" : "mixed"}</span><br><small>${x.n} labels</small></div>
      <button type="button" class="btn small" data-act="import-seed" data-slug="${esc(x.slug)}">Add</button></li>`).join("") || `<li class="hint">No starter labels on this server.</li>`;
  }).catch(e => { const ul = $("#seed-sets"); if (ul) ul.innerHTML = `<li class="hint">${esc(e.message)}</li>`; });
}

/** Adopt the account's saved state; a browser that was anonymous until now contributes its own profiles once. */
function mergeAccountState(remote) {
  const owner = account.acct.user.username;
  // Data left in this browser by a different account must never flow into this one.
  if (A.state.owner && A.state.owner !== owner) A.state = { ...defaultState(), settings: A.state.settings };
  if (remote?.version === 1 && Object.keys(remote.profiles || {}).length) {
    // The account is the source of truth. Only an anonymous browser's own profiles are merged in once;
    // after that this browser is just a cache, so a profile deleted elsewhere can't come back.
    if (!A.state.owner) for (const [id, p] of Object.entries(A.state.profiles)) if (!(id in remote.profiles)) remote.profiles[id] = p;
    remote.active = remote.profiles[remote.active] ? remote.active : Object.keys(remote.profiles)[0];
    remote.settings = { semantic: true, ...(A.state.settings || {}), ...(remote.settings || {}) };
    A.state = remote;
  }
  A.state.owner = owner;
}

async function adoptAccountState() {
  mergeAccountState(await account.pullState());
  save(); // uploads the merge (or this browser's state, for a new account)
  A.run = null; A.report = null; A.curve = null; A.labelQueue = null;
  applyView(profile());
  render();
  if (profile()) { await rerank(); setVisitBaseline(profile()); }
  render();
}

async function importSeedSet(slug, button) {
  button.disabled = true;
  try {
    const set = await account.seedSet(slug);
    const taken = new Set(Object.values(A.state.profiles).map(p => p.name));
    let name = set.profile.name, i = 2;
    while (taken.has(name)) name = `${set.profile.name} (${i++})`;
    const labels = Object.fromEntries(set.labels.filter(l => A.byId.has(l.paper_id)).map(l => [l.paper_id, { label: l.label, at: l.labeled_at || now() }]));
    const p = newProfile({ ...set.profile, name, labels, starter: [], seededFrom: slug });
    A.state.profiles[p.id] = p;
    A.state.active = p.id;
    save();
    $$("dialog[open]").forEach(d => d.close());
    A.view = "insights"; location.hash = "#/insights";
    A.run = null; A.report = null; A.curve = null;
    render();
    await rerank();
    setVisitBaseline(p);
    toast(`Added “${name}” with ${Object.keys(labels).length} labels.`);
  } catch (e) { toast(e.message); button.disabled = false; }
}

function settingsDialog() {
  const prof = profile();
  const live = A.run?.cutoffs && prof.adaptive !== false ? `Now Read ≥ ${A.run.cutoffs.read.toFixed(2)} · Skim ≥ ${A.run.cutoffs.skim.toFixed(2)}` : "Lowers cutoffs when scores run low";
  const row = (title, hint, control) => `<div class="srow"><div class="stext"><b>${title}</b><span>${hint}</span></div><div class="sctl">${control}</div></div>`;
  const sw = (name, on) => `<label class="switch"><input type="checkbox" name="${name}" ${on ? "checked" : ""}><i></i></label>`;
  const range = (name, v) => `<output id="${name}-v">${v.toFixed(2)}</output><input type="range" name="${name}" min="0" max="1" step="0.01" value="${v}">`;
  openDialog("#settings-dialog", `<form method="dialog" id="settings-form" class="settings">
    <header class="shead"><h2>Settings</h2><span class="muted">${esc(prof.name)}</span></header>
    <section class="sgroup"><h3>Triage</h3>
      ${row("Read cutoff", "Minimum relevance for Read", range("read", prof.cutoffs.read))}
      ${row("Skim cutoff", "Minimum relevance for Skim", range("skim", prof.cutoffs.skim))}
      ${row("Adaptive cutoffs", live, sw("adaptive", prof.adaptive !== false))}
      ${row("Weekly reading cap", `${readBudget(prof.hours)} papers in Read`, sw("budget", prof.budget))}
      ${row("Group similar papers", "One card for near-duplicates", sw("group", prof.group))}
    </section>
    <section class="sgroup"><h3>Sources <small>for Load more · only keywords are sent</small></h3>
      ${Object.entries(SOURCES).map(([k, v]) => row(v.label, v.hint, sw(`src-${k}`, enabledSources()[k]))).join("")}
    </section>
    <section class="sgroup"><h3>Model</h3>
      ${row("Semantic ranking", "MiniLM, about 23 MB, downloaded once", sw("semantic", A.state.settings.semantic !== false))}
    </section>
    <section class="sgroup"><h3>Data</h3>
      ${row("Backup", account.acct.user ? "Synced to your account" : "Saved in this browser", `<button type="button" class="btn small" data-act="backup">${icon("download")}Download</button><label class="btn small ghost file">${icon("upload")}Restore<input type="file" accept=".json,application/json" data-act="restore" hidden></label>`)}
      ${row("Export", "Ranked list or reading digest", `<button type="button" class="btn small ghost" data-act="export-ranked">${icon("download")}CSV</button><button type="button" class="btn small ghost" data-act="digest">${icon("download")}Digest</button>`)}
      ${row("Import papers", "Your own collection, as JSON", `<label class="btn small ghost file">${icon("upload")}Import<input type="file" accept=".json,application/json" data-act="import-papers" hidden></label>`)}
    </section>
    <div class="dialog-actions"><button type="button" class="btn danger ghost" data-act="erase">Erase all data</button><button type="submit" class="btn primary">Done</button></div>
  </form>`);
}

function helpDialog() {
  const keys = [["J / K", "Next / previous paper"], ["Enter", "Show or hide details"], ["U / N", "Relevant / not relevant (press again to undo)"], ["S", "Save"], ["X", "Hide"], ["O", "Open the paper"], ["/", "Search"], ["?", "This help"]];
  openDialog("#help-dialog", `<h2>How Paper Triage works</h2>
    <p><b>Relevance</b> says whether a paper is about your research. <b>Worth-it signals</b> say whether it’s worth your time: peer review, released code or data, study-design cues, open access and citation impact, plus cautions like very short abstracts or promotional wording. Signals are cues you can check, not verdicts, and they never change the relevance score. Use them to filter or sort.</p>
    <p>Rate at least ${MIN_RATINGS} papers and the algorithm sorts the rest into Read, Skim and Skip; every further rating re-sorts the lists. Your ratings train a small model in your browser, which gets a bigger say only when cross-validation on your own labels shows it helps.</p>
    <h3>Keyboard</h3><dl class="keys">${keys.map(([k, t]) => `<dt><kbd>${k}</kbd></dt><dd>${t}</dd>`).join("")}</dl>
    <h3>Labelling</h3><p><kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> label Read / Skim / Skip, <kbd>0</kbd> skips, <kbd>Backspace</kbd> undoes.</p>
    <p class="hint">Catalog: ${A.catalog.papers.length.toLocaleString()} papers from OpenAlex, updated ${esc(date(A.catalog.manifest.updated_at))}. <a href="https://github.com/srivathsanb14/research-paper-triage" target="_blank" rel="noopener">Source code</a></p>
    <div class="dialog-actions"><span></span><button class="btn primary" data-act="close-dialog">Got it</button></div>`);
}

// ------------------------------------------------------------------ actions

function readProfileForm(form) {
  const fd = new FormData(form);
  const fields = fd.getAll("profile-fields");
  return {
    name: String(fd.get("name") || "").trim(),
    description: String(fd.get("description") || "").trim(),
    keywords: parseList(fd.get("keywords")),
    focus: String(fd.get("focus") || "").trim(),
    avoid: parseList(fd.get("avoid")),
    hours: Math.min(40, Math.max(0.5, Number(fd.get("hours")) || 3)),
    fields: fields.length === A.catalog.areas.length ? [] : fields,
  };
}

async function submitProfile(form) {
  const values = readProfileForm(form);
  const err = form.querySelector(".form-error");
  const isNew = form.dataset.new === "true";
  const fail = msg => { err.textContent = msg; err.hidden = false; };
  if (!values.name) return fail("Give the profile a name.");
  if (Object.values(A.state.profiles).some(p => p.name === values.name && (isNew || p.id !== A.state.active))) return fail("A profile with that name already exists.");
  if (isNew) {
    const p = newProfile({ ...values, seeds: A.pendingSeeds });
    A.state.profiles[p.id] = p;
    A.state.active = p.id;
  } else Object.assign(profile(), values, { seeds: A.pendingSeeds, starter: [] });
  save();
  $("#profile-dialog").close();
  A.labelQueue = null;
  A.run = null;
  A.view = "feed";
  location.hash = "#/feed";
  render();
  await rerank();
}

async function startFromWelcome(form) {
  const fd = new FormData(form);
  const description = String(fd.get("description") || "").trim();
  const keywords = parseList(fd.get("keywords"));
  const fields = fd.getAll("fields");
  if (!description && !keywords.length && !fields.length) { toast("Pick at least one field, or describe your research."); return; }
  const example = EXAMPLES.find(e => e.description === description);
  // No field chosen means every field.
  const name = example?.name || (description || keywords.length ? "My research" : fields.length && fields.length < A.catalog.areas.length ? fields.map(f => SHORT_AREA[f]).join(" + ").slice(0, 70) : "My papers");
  const p = newProfile({ name, description, keywords, fields: fields.length === A.catalog.areas.length ? [] : fields });
  A.state.profiles[p.id] = p;
  A.state.active = p.id;
  save();
  A.view = "feed";
  A.tab = "READ";
  location.hash = "#/feed";
  A.run = null;
  render();
  await rerank();
  setVisitBaseline(p);
}

function setVisitBaseline(prof) {
  // "New" = papers that were not in the catalog at your previous visit.
  const ids = A.all.map(p => p.id);
  const known = new Set(prof.seen || []);
  A.newIds = known.size ? new Set(ids.filter(id => !known.has(id))) : new Set();
  prof.seen = ids;
  prof.lastVisit = now();
  save();
}

/** OpenAlex field ids for a profile's areas ("ai" is part of Computer Science, field 17). */
function openalexFields(prof) {
  if (!prof.fields.length) return [];
  const ids = new Set();
  for (const a of A.catalog.manifest.areas) if (prof.fields.includes(a.id)) for (const f of a.fields) ids.add(f >= 1000 ? Math.floor(f / 100) : f);
  return [...ids];
}

/** Catalog area, or where an outside paper came from. */
function areaLabel(p) {
  const known = A.catalog.areas.find(a => a.id === p.area)?.label;
  if (known) return known;
  return SOURCES[p.source] ? `From ${SOURCES[p.source].label}` : p.area === "imported" ? "Added by you" : "";
}

const enabledSources = () => ({ ...DEFAULT_SOURCES, ...(A.state.settings.sources || {}) });

function areaFor(p) {
  if (p._subfield === 1702 || p._subfield === 1707) return "ai";
  const a = A.catalog.manifest.areas.find(x => x.id !== "ai" && x.fields.includes(p._field));
  return a?.id || "imported";
}

/** Search words: keywords, else the description, else frequent words in titles you liked. */
function searchTerms(prof) {
  if (prof.keywords.length) return prof.keywords.slice(0, 4).join(" ");
  const text = prof.focus || prof.description;
  if (text) return text.split(/\s+/).slice(0, 12).join(" ");
  const counts = new Map();
  for (const id of likedIds(prof)) for (const t of tokenize(A.byId.get(id)?.title || "")) if (t.length > 3) counts.set(t, (counts.get(t) || 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([t]) => t).join(" ");
}

let loadingMore = false;
/** "Load more": fresh papers from the enabled sources (OpenAlex, Europe PMC, Crossref); falls back to the built-in catalog. */
async function loadMore() {
  const prof = profile();
  if (!prof || loadingMore) return;
  loadingMore = true;
  $$('[data-act="load-more"]').forEach(b => { b.disabled = true; b.classList.add("busy"); });
  const rating = !hasEnoughSignal(prof) || A.run?.discover;
  let found = [], failed = false, status = {};
  try {
    prof.livePage = (prof.livePage || 0) + 1;
    // Europe PMC covers life sciences, so skip it when the profile's fields exclude them.
    const bio = !prof.fields.length || prof.fields.some(f => ["life", "environment"].includes(f));
    ({ papers: found, status } = await searchAll(searchTerms(prof), { sources: enabledSources(), fields: openalexFields(prof), limit: 40, page: prof.livePage, bio }));
    failed = Object.values(status).length > 0 && Object.values(status).every(x => x.error);
    for (const p of found) p.area = areaFor(p);
    await addVisitorPapers(found, prof);
    await rerank();
  } catch (e) {
    console.warn(e);
    failed = true;
  }
  if (rating) {
    const st = states(prof);
    const have = new Set(prof.starter);
    const fresh = found.map(p => p.id).filter(id => A.byId.has(id) && !have.has(id) && !st.get(id)?.vote);
    const fields = new Set(prof.fields.length ? prof.fields : A.catalog.areas.map(a => a.id));
    const local = A.run.results.map(r => r.paper).filter(p => !have.has(p.id) && (fields.has(p.area) || p.area === "imported")).map(p => p.id);
    prof.starter.push(...[...new Set([...fresh, ...local])].slice(0, PAGE));
    save();
  } else A.shown += PAGE;
  loadingMore = false;
  render();
  const parts = Object.entries(status).map(([k, v]) => (v.error ? `${SOURCES[k].label} unavailable` : `${v.n} from ${SOURCES[k].label}`));
  const some = Object.values(status).some(v => v.error);
  toast(failed ? "Couldn’t reach any source, so here are more papers from the built-in catalog." : found.length ? `Loaded ${found.length} papers (${parts.join(", ")}).` : `Nothing new from your sources${some ? ` (${parts.filter(x => x.includes("unavailable")).join(", ")})` : ""}, so here are more from the built-in catalog.`, { ms: some ? 7000 : 4200 });
}

function backupJson(prof) {
  const ids = new Set([...prof.extra, ...prof.seeds, ...Object.keys(prof.labels), ...prof.feedback.map(f => f.pid)]);
  const papers = A.visitor.filter(p => ids.has(p.id)).map(p => { const r = toRecord(p); delete r.dense; return r; });
  const { seen, ...rest } = prof;
  return JSON.stringify({ format: "paper-triage-web-profile", version: 1, exported_at: now(), profile: rest, papers }, null, 1);
}

async function restoreBackup(text) {
  let obj;
  try { obj = JSON.parse(text); } catch { throw new Error("This file is not valid JSON."); }
  let prof, papers = [];
  if (obj?.format === "paper-triage-web-profile" && obj.version === 1 && obj.profile) {
    const r = obj.profile;
    prof = newProfile({
      name: r.name, description: r.description, keywords: r.keywords, focus: r.focus, avoid: r.avoid, hours: r.hours, fields: r.fields,
      cutoffs: r.cutoffs, budget: r.budget, group: r.group, seeds: r.seeds, labels: r.labels, feedback: r.feedback, extra: r.extra, good: r.good, reviewSet: r.reviewSet,
    });
    papers = obj.papers?.length ? parseImport(JSON.stringify(obj.papers)) : [];
  } else throw new Error("Choose a Paper Triage backup file.");
  sanitizeProfile(prof);
  if (!prof.name) throw new Error("The backup has no profile name.");
  let name = prof.name, i = 1;
  while (Object.values(A.state.profiles).some(p => p.name === prof.name)) prof.name = `${name} (restored ${i++})`;
  await addVisitorPapers(papers, prof);
  A.state.profiles[prof.id] = prof;
  A.state.active = prof.id;
  save();
  A.run = null;
  A.view = "feed";
  location.hash = "#/feed";
  render();
  await rerank();
  toast(`Restored “${prof.name}”.`);
}

/** Coerce a restored profile into the expected shape; drops anything malformed. */
function sanitizeProfile(p) {
  const str = (v, max = 5000) => (typeof v === "string" ? v.slice(0, max) : "");
  const strs = v => (Array.isArray(v) ? v.filter(x => typeof x === "string").map(x => x.slice(0, 300)) : []);
  p.name = str(p.name, 80).trim();
  p.description = str(p.description);
  p.focus = str(p.focus, 500);
  for (const k of ["keywords", "avoid", "seeds", "extra", "fields"]) p[k] = strs(p[k]);
  p.fields = p.fields.filter(f => A.catalog.areas.some(a => a.id === f));
  p.hours = Math.min(40, Math.max(0.5, Number(p.hours) || 3));
  const c = p.cutoffs || {};
  p.cutoffs = Number.isFinite(c.read) && Number.isFinite(c.skim) && 0 <= c.skim && c.skim <= c.read && c.read <= 1 ? { read: c.read, skim: c.skim } : { ...DEFAULT_CUTOFFS };
  p.budget = p.budget !== false;
  p.adaptive = p.adaptive !== false;
  p.group = p.group !== false;
  p.good = p.good === "read_skim" ? "read_skim" : "read";
  p.reviewSet = typeof p.reviewSet === "string" ? p.reviewSet.slice(0, 80) : undefined;
  p.starter = Array.isArray(p.starter) ? p.starter.filter(x => typeof x === "string").slice(0, 2000) : [];
  const labels = {};
  for (const [pid, l] of Object.entries(p.labels && typeof p.labels === "object" ? p.labels : {})) {
    const label = typeof l === "string" ? l : l?.label;
    if (LABELS.includes(label)) labels[pid] = { label, at: str(l?.at, 40) || now(),
      ...(LABELS.includes(l?.proposed) ? { proposed: l.proposed, proposal: str(l.proposal, 80) } : {}) };
  }
  p.labels = labels;
  const actions = new Set(["useful", "not_useful", "clear_vote", "correct", "dismiss", "undismiss", "save", "unsave", "open", "explanation_ok", "explanation_bad"]);
  p.feedback = (Array.isArray(p.feedback) ? p.feedback : []).filter(f => f && typeof f.pid === "string" && actions.has(f.action) && (f.action !== "correct" || LABELS.includes(f.value)))
    .map(f => ({ pid: f.pid, action: f.action, value: f.value ?? null, at: str(f.at, 40) || now(), predicted: LABELS.includes(f.predicted) ? f.predicted : null, score: Number.isFinite(f.score) ? f.score : null }));
  return p;
}

function importLabels(text, name) {
  const prof = profile();
  let rows;
  if (name.endsWith(".csv")) {
    const lines = text.split(/\r?\n/).filter(Boolean);
    const head = lines.shift().split(",");
    const iId = head.indexOf("paper_id"), iL = head.indexOf("label"), iT = head.indexOf("labeled_at");
    if (iId < 0 || iL < 0) throw new Error("The CSV needs paper_id and label columns.");
    rows = lines.map(l => { const c = parseCsvLine(l); return { paper_id: c[iId], label: c[iL], labeled_at: c[iT] }; });
  } else if (name.endsWith(".jsonl")) rows = text.split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l));
  else {
    const obj = JSON.parse(text);
    rows = Array.isArray(obj) ? obj : Object.entries(obj.labels || obj).map(([paper_id, label]) => ({ paper_id, label }));
  }
  let n = 0, skipped = 0;
  for (const r of rows) {
    if (LABELS.includes(r.label) && A.byId.has(r.paper_id)) { prof.labels[r.paper_id] = { label: r.label, at: r.labeled_at || now() }; n++; }
    else skipped++;
  }
  save();
  A.labelQueue = null;
  toast(`Imported ${n} labels${skipped ? `; ${skipped} skipped (unknown papers or labels)` : ""}.`);
  scheduleRerank();
  render();
}

function parseCsvLine(line) {
  const out = [];
  let cur = "", q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) { if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; } else if (c === '"') q = false; else cur += c; }
    else if (c === '"') q = true;
    else if (c === ",") { out.push(cur); cur = ""; }
    else cur += c;
  }
  out.push(cur);
  return out;
}

function paperAction(act, pid, v) {
  const prof = profile();
  const st = states(prof).get(pid) || {};
  const r = A.run.byId.get(pid);
  A.focusId = pid;
  if (act === "vote") {
    const undoLen = prof.feedback.length;
    const before = ratingsCount(prof);
    logFeedback(pid, st.vote === v ? "clear_vote" : v);
    const after = ratingsCount(prof);
    if (before < MIN_RATINGS && after >= MIN_RATINGS) {
      A.tab = "READ";
      A.shown = PAGE;
      toast(`That’s ${MIN_RATINGS}! The algorithm has sorted your papers into Read, Skim and Skip.`, { ms: 6000 });
      render();
      window.scrollTo(0, 0);
      return rerank();
    }
    if (st.vote === v) toast("Rating removed");
    else if (before < MIN_RATINGS) toast(`${after} of ${MIN_RATINGS} rated`, { action: { label: "Undo", run: () => undoTo(undoLen) } });
    else toast(v === "useful" ? "Marked relevant · re-sorting" : "Marked not relevant · re-sorting", { action: { label: "Undo", run: () => undoTo(undoLen) } });
  } else if (act === "save") logFeedback(pid, st.saved ? "unsave" : "save");
  else if (act === "hide") {
    const undoLen = prof.feedback.length;
    logFeedback(pid, "dismiss");
    toast("Hidden", { action: { label: "Undo", run: () => undoTo(undoLen) } });
  } else if (act === "unhide") logFeedback(pid, "undismiss");
  else if (act === "open-paper") { logFeedback(pid, "open"); return; }
  keepAnchor(pid, render);
  scheduleRerank();
}

function undoTo(len) {
  profile().feedback.length = len;
  save();
  render();
  scheduleRerank();
}

/** Re-render without the clicked card jumping on screen. */
function keepAnchor(pid, fn) {
  const sel = `[data-id="${CSS.escape(pid)}"]`;
  const before = $(sel)?.getBoundingClientRect().top;
  fn();
  const el = $(sel);
  if (el && before != null) window.scrollBy(0, el.getBoundingClientRect().top - before);
}

function handLabel(v) {
  const prof = profile();
  const queue = labelQueue(prof);
  const pid = queue[0];
  if (!pid) return;
  const sug = A.labelStrategy === "review" ? suggestionFor(prof, pid) : null;
  if (v === "ACCEPT") {
    if (!sug) return;
    v = sug.proposed_label;
  }
  if (v) {
    prof.labels[pid] = { label: v, at: now(), ...(sug ? { proposed: sug.proposed_label, proposal: reviewSet(prof).slug } : {}) };
    A.labelUndo.push(pid);
    save();
    scheduleRerank();
  } else A.labelQueue.ids.push(A.labelQueue.ids.splice(A.labelQueue.ids.indexOf(pid), 1)[0]);
  render();
}

// ------------------------------------------------------------------- events

document.addEventListener("click", async e => {
  const el = e.target.closest("[data-act]");
  if (!el || el.tagName === "INPUT") return;
  const act = el.dataset.act;
  const pid = el.closest("[data-id]")?.dataset.id;
  const prof = profile();
  switch (act) {
    case "vote": case "save": case "hide": case "unhide": return paperAction(act, pid, el.dataset.v);
    case "open-paper": return paperAction(act, pid);
    case "expand":
      A.expanded.has(pid) ? A.expanded.delete(pid) : A.expanded.add(pid);
      A.focusId = pid;
      return keepAnchor(pid, render);
    case "tab": A.tab = el.dataset.tab; A.shown = PAGE; saveView(); return render();
    case "more": A.shown += PAGE; return render();
    case "clear-filters": A.query = ""; A.signalFilters.clear(); saveView(); return render();
    case "example": {
      const ex = EXAMPLES[+el.dataset.i];
      const f = $("#start-form");
      f.description.value = ex.description;
      f.keywords.value = ex.keywords.join(", ");
      for (const box of f.querySelectorAll('input[name="fields"]')) box.checked = ex.fields.includes(box.value);
      return;
    }
    case "edit-profile": closeMenu(); return profileDialog(prof);
    case "new-profile": closeMenu(); return profileDialog(null, true);
    case "switch-profile":
      closeMenu();
      A.state.active = el.dataset.id; applyView(profile()); save(); A.run = null; A.labelQueue = null; A.expanded.clear(); render(); await rerank(); return setVisitBaseline(profile());
    case "close-dialog": return el.closest("dialog").close();
    case "delete-profile":
      if (!confirm(`Delete “${prof.name}” with its ratings and labels? This can’t be undone.`)) return;
      delete A.state.profiles[prof.id];
      A.state.active = Object.keys(A.state.profiles)[0] || null;
      save(); el.closest("dialog").close(); A.run = null; render(); return rerank();
    case "lookup-seeds": {
      const form = el.closest("form");
      const ids = parseIdentifiers(form.seeds.value);
      if (!ids.length) return toast("Paste DOIs or arXiv IDs first.");
      el.disabled = true;
      try {
        const found = await resolveIdentifiers(ids);
        if (!found.length) toast("None of those were found on OpenAlex.");
        await addVisitorPapers(found, prof || newProfile(), { asSeeds: false });
        for (const p of found) if (!A.pendingSeeds.includes(p.id)) A.pendingSeeds.push(p.id);
        form.querySelector(".seed-list").innerHTML = A.pendingSeeds.map(id => A.byId.get(id)).filter(Boolean).map(s => `<li data-id="${esc(s.id)}">${esc(s.title)} <button type="button" class="link" data-act="remove-seed">${icon("x")}</button></li>`).join("");
        form.seeds.value = "";
        if (found.length < ids.length) toast(`Found ${found.length} of ${ids.length}.`);
      } catch (err) { toast(err.message); } finally { el.disabled = false; }
      return;
    }
    case "remove-seed": {
      const li = el.closest("li");
      A.pendingSeeds = A.pendingSeeds.filter(id => id !== li.dataset.id);
      return li.remove();
    }
    case "account": return accountDialog();
    case "gate-mode": return showLoginGate("", el.dataset.v);
    case "guest":
      setGuest(true);
      location.hash = "";
      location.reload();
      return;
    case "import-seed": return importSeedSet(el.dataset.slug, el);
    case "sign-out":
      await account.signOut();
      await store.clearAll(); // don't leave this account's data behind in the browser
      location.hash = "";
      location.reload();
      return;
    case "settings": return settingsDialog();
    case "help": return helpDialog();
    case "load-more": return loadMore();
    case "digest": {
      const rows = rowsFor(prof, { ignoreTab: true }).rows;
      const reasons = Object.fromEntries(rows.filter(r => r.label === "READ").map(r => [r.paper.id, explain(r).reason]));
      return download(`digest-${slug(prof.name)}.md`, digestMarkdown(prof.name, rows, reasons), "text/markdown");
    }
    case "export": {
      const s = states(prof);
      const papers = [...s.entries()].filter(([, x]) => x.saved).map(([id]) => A.byId.get(id)).filter(Boolean);
      const f = el.dataset.fmt;
      if (f === "bib") return download(`${slug(prof.name)}-saved.bib`, toBibtex(papers), "application/x-bibtex");
      if (f === "ris") return download(`${slug(prof.name)}-saved.ris`, toRis(papers), "application/x-research-info-systems");
      return download(`${slug(prof.name)}-saved.csv`, toCsv(papers, ["title", "authors", "venue", "published", "url", "id"]), "text/csv");
    }
    case "export-ranked": {
      const rows = A.run.results.map(r => ({ rank: r.rank, label: r.label, score: r.score.toFixed(4), title: r.paper.title, authors: r.paper.authors, venue: r.paper.venue, published: r.paper.published, signals: r.paper._sig.level, url: r.paper.url }));
      return download(`${slug(prof.name)}-ranked.csv`, toCsv(rows, Object.keys(rows[0] || { rank: 0 })), "text/csv");
    }
    case "hand-label": return handLabel(el.dataset.v);
    case "review-set": {
      const set = A.proposals.sets.find(x => x.slug === el.dataset.slug);
      let owner = Object.values(A.state.profiles).find(p => p.reviewSet === set.slug);
      if (!owner) {
        owner = newProfile({ name: set.name, description: set.description, keywords: set.keywords, focus: set.focus || "", avoid: set.avoid || [], fields: set.fields || [], reviewSet: set.slug });
        let i = 2;
        while (Object.values(A.state.profiles).some(p => p.name === owner.name)) owner.name = `${set.name} (${i++})`;
        A.state.profiles[owner.id] = owner;
      }
      const switching = A.state.active !== owner.id;
      A.state.active = owner.id;
      A.labelStrategy = "review";
      A.labelQueue = null;
      A.labelUndo = [];
      save();
      if (switching) { A.run = null; render(); await rerank(); return setVisitBaseline(owner); }
      return render();
    }
    case "label-undo": {
      const pid2 = A.labelUndo.pop();
      if (pid2) { delete prof.labels[pid2]; A.labelQueue?.ids.unshift(pid2); save(); scheduleRerank(); }
      return render();
    }
    case "labels-export": {
      const rows = labelRows(prof);
      const stamp = new Date().toISOString().slice(0, 10);
      if (el.dataset.fmt === "jsonl") return download(`labels-${slug(prof.name)}-${stamp}.jsonl`, rows.map(r => JSON.stringify(r)).join("\n") + "\n", "application/jsonl");
      return download(`labels-${slug(prof.name)}-${stamp}.csv`, toCsv(rows, Object.keys(rows[0])), "text/csv");
    }
    case "good-mode": prof.good = el.dataset.v; save(); A.report = null; A.curve = null; return render();
    case "apply-cutoffs":
      prof.cutoffs = { read: +el.dataset.read, skim: +el.dataset.skim }; save(); toast("Cutoffs applied."); return rerank();
    case "backup": return download(`paper-triage-${slug(prof.name)}.json`, backupJson(prof), "application/json");
    case "erase":
      if (!confirm("Erase all profiles, ratings, labels and added papers from this browser?")) return;
      await store.clearAll();
      location.hash = "";
      location.reload();
      return;
  }
});

document.addEventListener("change", async e => {
  const t = e.target;
  const prof = profile();
  if (t.dataset.act === "signal-filter") { t.checked ? A.signalFilters.add(t.value) : A.signalFilters.delete(t.value); A.shown = PAGE; saveView(); return render(); }
  if (t.name === "sort") { A.sort = t.value; saveView(); return render(); }
  if (t.name === "feed-fields" || t.name === "feed-fields-all") {
    // "All fields" is the default (no narrowing); picking specific fields narrows the feed.
    const picked = t.name === "feed-fields-all" ? [] : $$('input[name="feed-fields"]').filter(x => x.checked).map(x => x.value);
    prof.fields = picked.length === A.catalog.areas.length ? [] : picked;
    save(); A.labelQueue = null; A.shown = PAGE; return render();
  }
  if (t.id === "label-strategy") { A.labelStrategy = t.value; return render(); }
  if (t.form?.id === "settings-form" && t.type !== "file") {
    const f = t.form;
    if (t.name === "read" || t.name === "skim") {
      let read = +f.read.value, skim = +f.skim.value;
      if (skim > read) { if (t.name === "read") skim = read; else read = skim; f.read.value = read; f.skim.value = skim; }
      prof.cutoffs = { read, skim };
    }
    if (t.name === "budget") prof.budget = t.checked;
    if (t.name === "adaptive") prof.adaptive = t.checked;
    if (t.name === "group") prof.group = t.checked;
    if (t.name.startsWith("src-")) {
      A.state.settings.sources = { ...enabledSources(), [t.name.slice(4)]: t.checked };
      if (!Object.values(A.state.settings.sources).some(Boolean)) { A.state.settings.sources.openalex = true; t.form["src-openalex"].checked = true; toast("At least one source stays on."); }
      return save();
    }
    if (t.name === "semantic") {
      A.state.settings.semantic = t.checked;
      if (t.checked) startEmbedder();
      else toast("Semantic model off. Reload to free its memory.");
    }
    save();
    return rerank();
  }
  if (t.type === "file" && t.files?.[0]) {
    const file = t.files[0];
    const text = await file.text();
    t.value = "";
    try {
      if (t.dataset.act === "restore") { $$("dialog[open]").forEach(d => d.close()); await restoreBackup(text); }
      else if (t.dataset.act === "import-papers") {
        $$("dialog[open]").forEach(d => d.close());
        const papers = parseImport(text);
        toast(`Importing ${papers.length} papers…`);
        const added = await addVisitorPapers(papers, prof);
        await rerank();
        toast(`Imported ${added} new papers.`);
      } else if (t.dataset.act === "labels-import") importLabels(text, file.name.toLowerCase());
    } catch (err) { toast(err.message || "Couldn’t read that file."); }
  }
});

document.addEventListener("input", e => {
  const t = e.target;
  if (t.id === "search") {
    A.query = t.value;
    A.shown = PAGE;
    clearTimeout(A.searchTimer);
    A.searchTimer = setTimeout(() => {
      if (A.view !== "feed") return;
      render();
      const s = $("#search");
      if (s) { s.focus(); s.setSelectionRange(s.value.length, s.value.length); }
    }, 200);
  }
  if (t.form?.id === "settings-form" && (t.name === "read" || t.name === "skim")) $(`#${t.name}-v`).textContent = (+t.value).toFixed(2);
});

document.addEventListener("submit", e => {
  if (e.target.id === "start-form") { e.preventDefault(); startFromWelcome(e.target); }
  if (e.target.id === "profile-form") { e.preventDefault(); submitProfile(e.target); }
  if (e.target.id === "settings-form") { e.preventDefault(); e.target.closest("dialog").close(); }
  if (e.target.id === "gate-form") { e.preventDefault(); submitGate(e.target); }
  if (e.target.id === "account-form") { e.preventDefault(); submitAccount(e.target, e.submitter?.value); }
});

async function submitAccount(form, mode) {
  const username = form.username.value.trim(), password = form.password.value;
  if (mode === "register" && password.length < 8) { // sign-in accepts any length, so existing short passwords still work
    accountDialog("A new account needs a password of at least 8 characters.");
    $("#account-form").username.value = username;
    return;
  }
  const buttons = $$("button", form);
  buttons.forEach(b => { b.disabled = true; });
  try {
    await (mode === "register" ? account.register : account.signIn)(username, password);
    setGuest(false);
    $$("dialog[open]").forEach(d => d.close());
    await adoptAccountState();
    toast(mode === "register" ? "Account created. Your work now syncs." : `Signed in as ${username}.`);
  } catch (e) {
    accountDialog(e.message);
    $("#account-form").username.value = username;
  }
}

document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey || $$("dialog[open]").length) return;
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if (typing) { if (e.key === "Escape") document.activeElement.blur(); return; }
  const prof = profile();
  if (!prof || !A.run || A.run.error) return;
  if (e.key === "?") { e.preventDefault(); return helpDialog(); }
  if (A.view === "label") {
    const map = { 1: "READ", 2: "SKIM", 3: "SKIP", 0: "", Enter: "ACCEPT" };
    if (e.key === "Enter" && ["BUTTON", "A", "SUMMARY"].includes(document.activeElement?.tagName)) return; // let the focused control handle it
    if (e.key in map) { e.preventDefault(); return handLabel(map[e.key]); }
    if (e.key === "Backspace") { e.preventDefault(); return $('[data-act="label-undo"]')?.click(); }
    return;
  }
  if (A.view !== "feed" && A.view !== "saved") return;
  if (e.key === "/") { e.preventDefault(); return $("#search")?.focus(); }
  const cards = $$("article.paper");
  if (!cards.length) return;
  let i = cards.findIndex(c => c.dataset.id === A.focusId);
  const focus = idx => {
    const c = cards[Math.max(0, Math.min(cards.length - 1, idx))];
    cards.forEach(x => x.classList.toggle("focused", x === c));
    A.focusId = c.dataset.id;
    c.focus({ preventScroll: true });
    c.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };
  const k = e.key.toLowerCase();
  if (k === "j" || e.key === "ArrowDown" && e.shiftKey) { e.preventDefault(); return focus(i + 1); }
  if (k === "k" || e.key === "ArrowUp" && e.shiftKey) { e.preventDefault(); return focus(i < 0 ? 0 : i - 1); }
  if (i < 0) return;
  const pid = A.focusId;
  const acts = { u: ["vote", "useful"], n: ["vote", "not_useful"], s: ["save"], x: ["hide"] };
  if (acts[k]) { e.preventDefault(); return paperAction(acts[k][0], pid, acts[k][1]); }
  if (e.key === "Enter") { e.preventDefault(); A.expanded.has(pid) ? A.expanded.delete(pid) : A.expanded.add(pid); return keepAnchor(pid, render); }
  if (k === "o") { const a = cards[i].querySelector("h2 a"); if (a) { a.click(); } }
});

document.addEventListener("toggle", e => { if (e.target.classList?.contains("filters-box")) A.filtersOpen = e.target.open; }, true);

function closeMenu() { $("#profile-menu").removeAttribute("open"); }
document.addEventListener("click", e => { const m = $("#profile-menu"); if (m?.open && !m.contains(e.target)) closeMenu(); });

function route() {
  const v = (location.hash.match(/^#\/(\w+)/) || [])[1];
  A.view = ["feed", "saved", "label", "insights"].includes(v) ? v : "feed";
  render();
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);
window.addEventListener("pagehide", () => { store.flush(); account.flush(true); });
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") { store.flush(); account.flush(true); } });

// --------------------------------------------------------------------- boot

function startEmbedder() {
  A.embedder.start().then(async () => {
    try { await embedMissing(); } catch (e) { console.warn(e); }
    if (profile()) { const wasDense = A.run?.space === A.dense; await rerank(); if (!wasDense) toast("Semantic model ready · ranking by meaning now"); }
  }).catch(e => console.warn("Semantic model unavailable:", e));
}

account.onSyncConflict(() => toast("Your account was changed on another device. Reload to get the latest; syncing is paused until you do.", { ms: 9000 }));

async function adoptServerStateAtBoot() {
  try { mergeAccountState(await account.pullState()); account.queue(A.state); } catch (e) { console.warn("Could not load your account state", e); }
}

// "Continue without an account": this browser keeps everything locally, as on the static site.
const GUEST_KEY = "paper-triage:guest";
const isGuest = () => { try { return localStorage.getItem(GUEST_KEY) === "1"; } catch { return false; } };
const setGuest = on => { try { on ? localStorage.setItem(GUEST_KEY, "1") : localStorage.removeItem(GUEST_KEY); } catch { /* storage blocked */ } };

/** With the account backend present, visitors sign in, create an account, or continue as a guest. */
function showLoginGate(message = "", mode = "login") {
  document.body.classList.add("gated");
  const create = mode === "register";
  $("#main").innerHTML = `<section class="gate"><form id="gate-form" class="card-surface" data-mode="${mode}">
    <div class="mark" aria-hidden="true"><i></i><i></i><i></i></div>
    <h1>Paper Triage</h1>
    <div class="seg gate-mode" role="tablist">
      <button type="button" role="tab" aria-selected="${!create}" class="${create ? "" : "on"}" data-act="gate-mode" data-v="login">Sign in</button>
      <button type="button" role="tab" aria-selected="${create}" class="${create ? "on" : ""}" data-act="gate-mode" data-v="register">Create account</button>
    </div>
    <p class="hint">${create ? "New here? Pick a username and password. Your profiles, ratings and labels will sync to this account on any device." : "Already have an account? Sign in to get your papers back."}</p>
    <label class="field"><span>Username</span><input name="username" autocomplete="username" required minlength="${create ? 3 : 1}" maxlength="40" autofocus></label>
    <label class="field"><span>Password${create ? " <small>at least 8 characters</small>" : ""}</span><input name="password" type="password" autocomplete="${create ? "new-password" : "current-password"}" required minlength="${create ? 8 : 1}" maxlength="200"></label>
    ${create ? `<label class="field"><span>Confirm password</span><input name="confirm" type="password" autocomplete="new-password" required minlength="8" maxlength="200"></label>` : ""}
    <p class="form-error" ${message ? "" : "hidden"}>${esc(message)}</p>
    <button type="submit" class="btn primary">${create ? "Create account" : "Sign in"}</button>
    <div class="gate-or"><span>or</span></div>
    <button type="button" class="btn ghost" data-act="guest">Continue without an account</button>
    <p class="hint small">Without an account everything stays in this browser. You can sign in later from the top bar.</p>
  </form></section>`;
}

async function submitGate(form) {
  const mode = form.dataset.mode;
  const username = form.username.value.trim(), password = form.password.value;
  if (mode === "register" && password !== form.confirm.value) {
    showLoginGate("The passwords don’t match.", mode);
    $("#gate-form").username.value = username;
    return;
  }
  $$("button", form).forEach(b => { b.disabled = true; });
  try {
    await (mode === "register" ? account.register : account.signIn)(username, password);
    setGuest(false);
    location.hash = "#/feed"; // land on For you
    location.reload();
  } catch (e) {
    showLoginGate(e.message, mode);
    $("#gate-form").username.value = username;
  }
}

async function boot() {
  const status = $("#boot-status");
  await account.init();
  if (account.acct.available && !account.acct.user && !isGuest()) return showLoginGate();
  try {
    const [rules, state, visitor, catalog, proposals] = await Promise.all([
      loadRules(new URL("quality-rules.json", ROOT)),
      store.loadState(),
      store.loadPapers(),
      loadCatalog(new URL("catalog/", ROOT), f => { status.textContent = `Loading papers… ${Math.round(f * 100)}%`; }),
      fetch(new URL("catalog/proposals.json", ROOT)).then(r => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    A.proposals = proposals?.sets ? proposals : { sets: [] };
    void rules;
    A.state = state && state.version === 1 ? state : defaultState();
    A.state.settings ||= { semantic: true };
    // A guest never sees what a signed-in account left cached in this browser.
    if (!account.acct.user && A.state.owner) A.state = { ...defaultState(), settings: A.state.settings };
    if (account.acct.user) await adoptServerStateAtBoot();
    A.visitor = visitor.map(fromRecord);
    A.catalog = catalog;
    buildCorpus();
  } catch (e) {
    console.error(e);
    status.textContent = "Couldn’t load the paper catalog. Check your connection and reload.";
    $("#boot-retry").hidden = false;
    return;
  }
  A.embedder.addEventListener("change", renderModelPill);
  if (A.state.settings.semantic !== false) startEmbedder();
  applyView(profile());
  A.ready = true;
  document.body.classList.add("ready");
  $("#boot").remove();
  const v = (location.hash.match(/^#\/(\w+)/) || [])[1];
  A.view = ["feed", "saved", "label", "insights"].includes(v) ? v : "feed";
  render();
  if (profile()) { await rerank(); setVisitBaseline(profile()); render(); }
  if (!(await store.persistent())) toast("Storage is blocked in this window, so your ratings won’t be kept after you close it.", { ms: 8000 });
}

$("#boot-retry")?.addEventListener("click", () => location.reload());
boot();

// Exposed for the automated browser check (scripts/check_pages.cjs).
window.__triage = A;
