// Ranking engine for the browser, ported from the Python package (triage/).
//
//   interest profile ─┐
//   paper vectors ────┼─> 7 interpretable features ─> prior (hand-set weights)
//   your feedback ────┘                            └> ridge regression (learned)
//   final = (1 − w)·prior + w·learned, w chosen by cross-validation on your labels
//
// Two interchangeable vector spaces: MiniLM sentence embeddings (an off-the-shelf
// model, precomputed for the catalog) and TF-IDF (used until MiniLM is ready).
import { phraseInText, rawTokens, splitSentences, stemTokens, textCache, tokenize } from "./text.js";

export const FEATURES = ["semantic", "focus", "keyword_sem", "keyword_lex", "recency", "avoid", "feedback"];
export const FEATURE_NAMES = {
  semantic: "Similar to your research",
  focus: "Similar to your current focus",
  keyword_sem: "Closest keyword (meaning)",
  keyword_lex: "Keywords found in the paper",
  recency: "Recently published",
  avoid: "Excluded topic",
  feedback: "Like papers you rated",
};
export const PRIOR_WEIGHTS = { semantic: 0.40, focus: 0.15, keyword_sem: 0.15, keyword_lex: 0.25, recency: 0.05, avoid: -0.40, feedback: 0.25 };
export const LABELS = ["READ", "SKIM", "SKIP"];
export const LABEL_VALUE = { READ: 1, SKIM: 0.5, SKIP: 0 };
export const DEFAULT_CUTOFFS = { read: 0.62, skim: 0.40 };
export const MINUTES_PER_READ = 30;
export const MINUTES_PER_SKIM = 5;

const BLEND_K = 30;
const MAX_UNVALIDATED_WEIGHT = 0.3;
export const BLEND_GRID = [0, 0.15, 0.3, 0.5, 0.7];
const MIN_TRAIN_EXAMPLES = 8;
export const MIN_LABELS = 6;
const RIDGE_ALPHA = 1.0;
const FEEDBACK_TARGETS = { useful: [1, 1], not_useful: [0, 1], save: [1, 1], open: [0.75, 0.3], dismiss: [0.15, 0.5] };
const CORRECTION_WEIGHT = 1.5;
const SEED_WEIGHT = 2.0;
const W = { description: 1.0, keywords: 1.0, focus: 1.5, seeds: 1.5 };

const clip01 = x => (x < 0 ? 0 : x > 1 ? 1 : x);

// ------------------------------------------------------------------ vectors

function normalize(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n);
  if (n > 0) for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

/** MiniLM (all-MiniLM-L6-v2) sentence embeddings, unit length. */
export class DenseSpace {
  name = "minilm";
  label = "MiniLM sentence embeddings";
  cal = { doc: [0.15, 0.50], short: [0.12, 0.45] }; // same ranges as the Python sbert backend
  groupThreshold = 0.72;
  vec = p => p._dense;
  dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
  combine(parts) {
    const out = new Float32Array(parts[0][0].length);
    for (const [v, w] of parts) for (let i = 0; i < out.length; i++) out[i] += w * v[i];
    return normalize(out);
  }
}

/** TF-IDF over word stems; the title counts twice, as in the Python engine. */
export class SparseSpace {
  name = "tfidf";
  label = "TF-IDF keywords";
  cal = { doc: [0.02, 0.30], short: [0.01, 0.25] };
  groupThreshold = 0.5;
  constructor(papers) {
    const df = new Map();
    for (const p of papers) for (const t of new Set(stemTokens(`${p.title} ${p.abstract || ""}`))) df.set(t, (df.get(t) || 0) + 1);
    const n = papers.length;
    this.idf = new Map([...df].map(([t, c]) => [t, Math.log((1 + n) / (1 + c)) + 1]));
    this.cache = new WeakMap();
  }
  vectorize(text) {
    const tf = new Map();
    for (const t of stemTokens(text)) tf.set(t, (tf.get(t) || 0) + 1);
    const v = new Map();
    let norm = 0;
    for (const [t, c] of tf) {
      const w = (1 + Math.log(c)) * (this.idf.get(t) ?? Math.log(1 + 1e6));
      v.set(t, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [t, w] of v) v.set(t, w / norm);
    return v;
  }
  vec = p => {
    let v = this.cache.get(p);
    if (!v) { v = this.vectorize(`${p.title}. ${p.title}. ${p.abstract || ""}`); this.cache.set(p, v); }
    return v;
  };
  dot(a, b) {
    if (a.size > b.size) [a, b] = [b, a];
    let s = 0;
    for (const [t, w] of a) { const x = b.get(t); if (x) s += w * x; }
    return s;
  }
  combine(parts) {
    const out = new Map();
    for (const [v, w] of parts) for (const [t, x] of v) out.set(t, (out.get(t) || 0) + w * x);
    let n = 0;
    for (const x of out.values()) n += x * x;
    n = Math.sqrt(n) || 1;
    for (const [t, x] of out) out.set(t, x / n);
    return out;
  }
}

export function calibrate(space, cos, kind = "doc") {
  const [lo, hi] = space.cal[kind];
  return clip01((cos - lo) / (hi - lo));
}

/** Tokenised text, cached on the paper object for keyword / phrase matching. */
export function prepare(p) {
  if (!p._title) {
    p._title = textCache(p.title);
    p._body = textCache(p.abstract || "");
    p._all = textCache(`${p.title} ${p.abstract || ""}`);
  }
  return p;
}

// ------------------------------------------------------------------ profile

export function profileTexts(profile) {
  const t = [];
  if (profile.description?.trim()) t.push(["desc", profile.description]);
  if (profile.focus?.trim()) t.push(["focus", profile.focus]);
  for (const k of profile.keywords || []) t.push(["kw", k]);
  for (const a of profile.avoid || []) t.push(["avoid", a]);
  return t;
}

export function isEmptyProfile(profile) {
  return !(profile.description?.trim() || profile.focus?.trim() || profile.keywords?.length);
}

/**
 * `encoded` holds one vector per profileTexts() entry, in order (from the
 * embedding worker or SparseSpace.vectorize). Seeds are papers in the space.
 */
export function buildProfileVectors(profile, space, encoded, seedPapers = []) {
  const slots = { desc: [], focus: [], kw: [], avoid: [] };
  profileTexts(profile).forEach(([slot], i) => slots[slot].push(encoded[i]));
  const parts = [];
  if (slots.desc.length) parts.push([slots.desc[0], W.description]);
  if (slots.kw.length) parts.push([space.combine(slots.kw.map(v => [v, 1])), W.keywords]);
  if (slots.focus.length) parts.push([slots.focus[0], W.focus]);
  const seedVecs = seedPapers.map(space.vec).filter(Boolean);
  if (seedVecs.length) parts.push([space.combine(seedVecs.map(v => [v, 1])), W.seeds]);
  if (!parts.length) throw new Error("Add a project description, keywords, or a current focus.");
  return {
    main: space.combine(parts),
    focus: slots.focus[0] || null,
    keywords: slots.kw,
    avoid: slots.avoid,
    keywordNames: [...(profile.keywords || [])],
    avoidNames: [...(profile.avoid || [])],
  };
}

// ----------------------------------------------------------------- training

/** One example per paper. Priority: seed > hand label > correction > explicit > implicit. */
export function buildExamples(labels, feedback, { exclude = new Set(), seeds = [], useLabels = true } = {}) {
  const out = new Map();
  for (const pid of seeds) if (!exclude.has(pid)) out.set(pid, { pid, target: 1, weight: SEED_WEIGHT, source: "seed" });
  if (useLabels) {
    for (const [pid, lab] of Object.entries(labels)) {
      const label = typeof lab === "string" ? lab : lab.label;
      if (!exclude.has(pid) && !out.has(pid)) out.set(pid, { pid, target: LABEL_VALUE[label], weight: 1, source: "label" });
    }
  }
  const explicit = new Map();
  const implicit = new Map();
  for (const f of feedback) { // chronological: later events override earlier ones
    const { pid, action } = f;
    if (exclude.has(pid) || out.has(pid)) continue;
    if (action === "correct" && f.value in LABEL_VALUE) explicit.set(pid, { pid, target: LABEL_VALUE[f.value], weight: CORRECTION_WEIGHT, source: "correct" });
    else if (action === "useful" || action === "not_useful" || action === "save") {
      const [t, w] = FEEDBACK_TARGETS[action];
      explicit.set(pid, { pid, target: t, weight: w, source: action });
    } else if (action === "unsave" && explicit.get(pid)?.source === "save") explicit.delete(pid);
    else if (action === "open" || action === "dismiss") {
      const [t, w] = FEEDBACK_TARGETS[action];
      const prev = implicit.get(pid);
      implicit.set(pid, { pid, target: t, weight: prev ? Math.max(w, prev.weight) : w, source: action });
    } else if (action === "undismiss" && implicit.get(pid)?.source === "dismiss") implicit.delete(pid);
  }
  for (const [pid, ex] of implicit) if (!explicit.has(pid)) explicit.set(pid, ex);
  for (const [pid, ex] of explicit) if (!out.has(pid)) out.set(pid, ex);
  return [...out.values()];
}

/** Weighted ridge regression with an unpenalised intercept (scikit-learn semantics). */
export function ridgeFit(X, y, w, alpha = RIDGE_ALPHA) {
  const n = X.length, d = X[0].length;
  const sw = w.reduce((a, b) => a + b, 0);
  const xm = new Array(d).fill(0);
  let ym = 0;
  for (let i = 0; i < n; i++) { ym += w[i] * y[i]; for (let j = 0; j < d; j++) xm[j] += w[i] * X[i][j]; }
  ym /= sw;
  for (let j = 0; j < d; j++) xm[j] /= sw;
  const A = Array.from({ length: d }, (_, j) => Array.from({ length: d }, (_, k) => (j === k ? alpha : 0)));
  const b = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    const xc = X[i].map((v, j) => v - xm[j]);
    const yc = y[i] - ym;
    for (let j = 0; j < d; j++) {
      b[j] += w[i] * xc[j] * yc;
      for (let k = 0; k < d; k++) A[j][k] += w[i] * xc[j] * xc[k];
    }
  }
  const coef = solve(A, b);
  return { coef, intercept: ym - coef.reduce((s, c, j) => s + c * xm[j], 0) };
}

function solve(A, b) { // Gaussian elimination with partial pivoting (A is SPD, small)
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

// ---------------------------------------------------------------- the model

function recency(p, today) {
  const d = p.published ? Date.parse(p.published.slice(0, 10)) : p.year ? Date.UTC(p.year, 6, 1) : NaN;
  if (Number.isNaN(d)) return 0.5;
  return Math.exp(-Math.max(0, (today - d) / 86400000) / 365);
}

export function keywordMatches(p, keywords) {
  prepare(p);
  const inTitle = keywords.filter(k => phraseInText(k, p._title));
  const inBody = keywords.filter(k => !inTitle.includes(k) && phraseInText(k, p._body));
  return { inTitle, inBody };
}

function keywordLex(p, keywords) {
  if (!keywords.length) return 0;
  const { inTitle, inBody } = keywordMatches(p, keywords);
  return Math.min(1, (inTitle.length + 0.6 * inBody.length) / Math.min(keywords.length, 3));
}

function topkMean(sims, k = 3) {
  if (!sims.length) return 0;
  const top = [...sims].sort((a, b) => b - a).slice(0, k);
  return top.reduce((a, b) => a + b, 0) / top.length;
}

export class RelevanceModel {
  constructor(space, pv, { weightOverride = null, today = Date.now() } = {}) {
    Object.assign(this, { space, pv, weightOverride, today });
    this.coef = null;
    this.intercept = 0;
    this.learnedWeight = 0;
    this.pos = [];
    this.neg = [];
    this.info = { n_examples: 0, learned: false };
  }

  features(papers) {
    const { space, pv } = this;
    const cal = (x, kind) => calibrate(space, x, kind);
    return papers.map(p => {
      prepare(p);
      const v = space.vec(p);
      const sem = cal(space.dot(v, pv.main));
      const focus = pv.focus ? cal(space.dot(v, pv.focus), "short") : sem;
      const kwSem = pv.keywords.length ? cal(Math.max(...pv.keywords.map(k => space.dot(v, k))), "short") : sem;
      let avoid = 0;
      if (pv.avoid.length) {
        const avoidSem = cal(Math.max(...pv.avoid.map(a => space.dot(v, a))), "short");
        const avoidLex = pv.avoidNames.some(a => phraseInText(a, p._all)) ? 1 : 0;
        avoid = Math.max(clip01((avoidSem - 0.5) * 2), avoidLex * 0.8); // only when clearly present
      }
      return [sem, focus, kwSem, keywordLex(p, pv.keywordNames), recency(p, this.today), avoid, this.feedbackAffinity(p, v)];
    });
  }

  feedbackAffinity(p, v) {
    let out = 0;
    for (const [refs, sign] of [[this.pos, 1], [this.neg, -1]]) {
      if (!refs.length) continue;
      // Leave-self-out: a paper must not match its own feedback.
      const sims = refs.map(r => (r.id === p.id ? -1 : this.space.dot(v, this.space.vec(r))));
      out += sign * calibrate(this.space, topkMean(sims));
    }
    return out;
  }

  static prior(f) {
    return clip01(FEATURES.reduce((s, name, j) => s + PRIOR_WEIGHTS[name] * f[j], 0));
  }

  learned(f) {
    return this.coef ? clip01(f.reduce((s, x, j) => s + x * this.coef[j], this.intercept)) : null;
  }

  /** [{final, prior, learned, f}] for each paper. */
  score(papers) {
    return this.features(papers).map(f => {
      const prior = RelevanceModel.prior(f);
      const learned = this.learned(f);
      const final = learned == null || !this.learnedWeight ? prior : (1 - this.learnedWeight) * prior + this.learnedWeight * learned;
      return { final, prior, learned, f };
    });
  }

  fit(examples, byId) {
    const ex = examples.filter(e => byId.has(e.pid) && this.space.vec(byId.get(e.pid)));
    this.pos = ex.filter(e => e.target >= 0.75).map(e => byId.get(e.pid));
    this.neg = ex.filter(e => e.target <= 0.25).map(e => byId.get(e.pid));
    this.coef = null;
    this.intercept = 0;
    this.learnedWeight = 0;
    const targets = ex.map(e => e.target);
    this.info = { n_examples: ex.length, n_pos: this.pos.length, n_neg: this.neg.length, learned: false };
    if (ex.length < MIN_TRAIN_EXAMPLES || Math.max(...targets) === Math.min(...targets)) {
      this.info.reason = `Needs ${MIN_TRAIN_EXAMPLES}+ ratings with at least two different answers (have ${ex.length}).`;
      return this;
    }
    const F = this.features(ex.map(e => byId.get(e.pid)));
    const { coef, intercept } = ridgeFit(F, targets, ex.map(e => e.weight));
    this.coef = coef;
    this.intercept = intercept;
    this.learnedWeight = this.weightOverride ?? Math.min(MAX_UNVALIDATED_WEIGHT, ex.length / (ex.length + BLEND_K));
    Object.assign(this.info, {
      learned: true,
      learned_weight: this.learnedWeight,
      validated: this.weightOverride != null,
      coefficients: Object.fromEntries(FEATURES.map((name, j) => [name, coef[j]])),
      intercept,
    });
    return this;
  }
}

// ----------------------------------------------------------- validation

export function averagePrecision(scores, good) {
  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a] || a - b);
  const total = good.filter(Boolean).length;
  if (!total) return NaN;
  let hits = 0, sum = 0;
  order.forEach((i, rank) => { if (good[i]) { hits++; sum += hits / (rank + 1); } });
  return sum / total;
}

function hash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

/** Stratified, deterministic folds over labelled paper ids. */
export function folds(ids, labelOf, k) {
  const byLabel = {};
  for (const id of [...ids].sort((a, b) => hash(a) - hash(b))) (byLabel[labelOf(id)] ||= []).push(id);
  const out = Array.from({ length: k }, () => []);
  let i = 0;
  for (const lab of LABELS) for (const id of byLabel[lab] || []) out[i++ % k].push(id);
  return out.filter(f => f.length);
}

/**
 * Cross-validated prior / learned scores for every hand-labelled paper: each
 * paper is scored by a model that never saw its label or any feedback on it.
 */
export function crossValidate(ctx, k = 5) {
  const { labels, byId } = ctx;
  const ids = Object.keys(labels).filter(id => byId.has(id) && ctx.space.vec(byId.get(id)));
  const labelOf = id => labels[id].label ?? labels[id];
  const out = new Map();
  for (const fold of folds(ids, labelOf, Math.min(k, ids.length))) {
    const exclude = new Set(fold);
    const model = new RelevanceModel(ctx.space, ctx.pv, { today: ctx.today }).fit(
      buildExamples(labels, ctx.feedback, { exclude, seeds: ctx.seeds }), byId);
    model.learnedWeight = 1; // report the learned score on its own; blending happens below
    const rows = model.score(fold.map(id => byId.get(id)));
    fold.forEach((id, i) => out.set(id, { ...rows[i], learned: rows[i].learned ?? rows[i].prior, truth: labelOf(id) }));
  }
  return { ids: [...out.keys()], rows: [...out.values()] };
}

/** How much the learned model counts, chosen by cross-validated average precision. */
export function selectBlend(ctx, margin = 0.01) {
  const labelled = Object.keys(ctx.labels).filter(id => ctx.byId.has(id));
  const classes = new Set(labelled.map(id => ctx.labels[id].label ?? ctx.labels[id]));
  if (labelled.length < MIN_LABELS || classes.size < 2) return { weight: null, validated: false, scores: {} };
  const cv = crossValidate(ctx);
  let goodSet = new Set(["READ"]);
  if (!cv.rows.some(r => r.truth === "READ")) goodSet = new Set(["READ", "SKIM"]);
  const good = cv.rows.map(r => goodSet.has(r.truth));
  const scores = {};
  for (const w of BLEND_GRID) scores[w] = averagePrecision(cv.rows.map(r => (1 - w) * r.prior + w * r.learned), good);
  if (Object.values(scores).some(Number.isNaN)) return { weight: 0, validated: true, scores, cv };
  let best = 0;
  for (const w of BLEND_GRID.slice(1)) if (scores[w] > Math.max(scores[0] + margin, scores[best] + 1e-9)) best = w;
  return { weight: best, validated: true, scores, cv };
}

// ------------------------------------------------------------------ triage

export const labelFor = (s, cut) => (s >= cut.read ? "READ" : s >= cut.skim ? "SKIM" : "SKIP");
export const readBudget = hours => Math.max(1, Math.floor((Math.max(0, hours) * 60) / MINUTES_PER_READ));

/** Latest correction per paper, falling back to hand labels: the user already told us. */
export function userLabels(labels, feedback) {
  const out = Object.fromEntries(Object.entries(labels).map(([pid, l]) => [pid, l.label ?? l]));
  for (const f of feedback) if (f.action === "correct" && LABELS.includes(f.value)) out[f.pid] = f.value;
  return out;
}

export function triage(papers, scored, cut, { maxRead = null, overrides = {} } = {}) {
  const order = papers.map((_, i) => i).sort((a, b) => scored[b].final - scored[a].final || a - b);
  let nRead = 0;
  return order.map((i, r) => {
    const p = papers[i];
    const s = scored[i];
    let label = labelFor(s.final, cut);
    let note = "";
    if (overrides[p.id]) {
      if (overrides[p.id] !== label) note = `Model said ${label.toLowerCase()}; your label is applied.`;
      label = overrides[p.id];
    } else if (label === "READ" && maxRead != null && nRead >= maxRead) {
      label = "SKIM";
      note = "Moved to Skim: your weekly reading time is already full.";
    }
    if (label === "READ") nRead++;
    return { paper: p, score: s.final, prior: s.prior, learned: s.learned, f: s.f, label, rank: r + 1, note, lead: "", similar: [] };
  });
}

/**
 * Group near-identical papers under the highest-ranked one (scores and labels
 * unchanged). Only the top `limit` results are compared: duplicates matter where
 * people read, and all-pairs over the whole pool is quadratic.
 */
export function groupSimilar(results, space, limit = 400) {
  const leads = [];
  const byId = new Map(results.map(r => [r.paper.id, r]));
  for (const r of results.slice(0, limit)) {
    const v = space.vec(r.paper);
    let best = null, bestSim = -1;
    for (const l of leads) {
      const s = space.dot(v, space.vec(l.paper));
      if (s > bestSim) { bestSim = s; best = l; }
    }
    if (best && bestSim >= space.groupThreshold) {
      r.lead = best.paper.id;
      byId.get(best.paper.id).similar.push(r.paper.id);
    } else leads.push(r);
  }
  return results;
}

/**
 * Full ranking for a profile. ctx: {papers, byId, space, pv, labels, feedback,
 * seeds, cutoffs, hours, budget, group, today}. Returns {results, model, blend}.
 */
export function rank(ctx) {
  const blend = selectBlend(ctx);
  const model = new RelevanceModel(ctx.space, ctx.pv, { weightOverride: blend.weight, today: ctx.today })
    .fit(buildExamples(ctx.labels, ctx.feedback, { seeds: ctx.seeds }), ctx.byId);
  const seedSet = new Set(ctx.seeds);
  const papers = ctx.papers.filter(p => !seedSet.has(p.id) && ctx.space.vec(p));
  const results = triage(papers, model.score(papers), ctx.cutoffs, {
    maxRead: ctx.budget ? readBudget(ctx.hours) : null,
    overrides: userLabels(ctx.labels, ctx.feedback),
  });
  if (ctx.group) groupSimilar(results, ctx.space);
  return { results, model, blend };
}

// ------------------------------------------------------------ explanations

export function documentFrequencies(papers) {
  const df = new Map();
  for (const p of papers) for (const t of new Set(stemTokens(`${p.title} ${p.abstract || ""}`))) df.set(t, (df.get(t) || 0) + 1);
  return { df, n: Math.max(1, papers.length) };
}

/** Deterministic evidence: only what can be pointed to in the paper text. */
export function evidence(p, profile, { df, n }) {
  prepare(p);
  const { inTitle, inBody } = keywordMatches(p, profile.keywords || []);
  const avoidHits = (profile.avoid || []).filter(a => phraseInText(a, p._all));
  const profileWords = new Map();
  for (const w of tokenize(`${profile.description || ""} ${profile.focus || ""}`)) {
    const s = stemTokens(w)[0] || w;
    if (!profileWords.has(s)) profileWords.set(s, w);
  }
  const paperStems = new Set(stemTokens(`${p.title} ${p.abstract || ""}`));
  const kwStems = new Set((profile.keywords || []).flatMap(k => stemTokens(k)));
  const idf = s => Math.log((1 + n) / (1 + (df.get(s) || 0))) + 1;
  const shared = [...profileWords].filter(([s]) => paperStems.has(s) && !kwStems.has(s))
    .map(([s, w]) => [idf(s), w]).sort((a, b) => b[0] - a[0]).slice(0, 4).map(([, w]) => w);
  const want = new Set(stemTokens([profile.description, profile.focus, ...(profile.keywords || [])].join(" ")));
  let best = "", bestScore = 0;
  for (const s of splitSentences(p.abstract)) {
    const toks = stemTokens(s);
    if (!toks.length) continue;
    const sc = toks.filter(t => want.has(t)).length / Math.sqrt(toks.length);
    if (sc > bestScore) { best = s; bestScore = sc; }
  }
  return { keywordsInTitle: inTitle, keywordsInAbstract: inBody, sharedTerms: shared, avoidHits, bestSentence: best };
}

const q = (items, n = 2) => items.slice(0, n).map(i => `“${i}”`).join(", ");

/** One short sentence that only states verified evidence (ported from explain.template_reason). */
export function reason(label, ev, f, focus) {
  if (ev.avoidHits.length) return `Covers ${q(ev.avoidHits, 1)}, which you excluded.`;
  const fv = Object.fromEntries(FEATURES.map((name, j) => [name, f?.[j] ?? 0]));
  let bits = [];
  if (ev.keywordsInTitle.length) bits.push(`${q(ev.keywordsInTitle)} in the title`);
  else if (ev.keywordsInAbstract.length) bits.push(`mentions ${q(ev.keywordsInAbstract)}`);
  else if (ev.sharedTerms.length && label !== "SKIP") bits.push(`shares ${q(ev.sharedTerms)} with your project`);
  if (focus && fv.focus >= 0.6 && label !== "SKIP") bits.push("close to your current focus");
  if (fv.feedback >= 0.3) bits.push("like papers you found relevant");
  else if (fv.feedback <= -0.3) bits.push("like papers you found not relevant");
  bits = bits.slice(0, 2);
  if (!bits.length) bits = [{ READ: "strong match to your research description", SKIM: "related to your research, but not central", SKIP: "little overlap with your interests" }[label]];
  else if (label === "SKIP" && !bits.some(b => b.startsWith("like papers"))) bits = ["little overlap beyond " + bits[0].replace("mentions ", "").replace(" in the title", "")];
  const text = bits.join("; ") + ".";
  return text[0].toUpperCase() + text.slice(1);
}

/** Words worth highlighting in the abstract: matched keywords and shared terms. */
export function highlightTerms(ev) {
  return [...ev.keywordsInTitle, ...ev.keywordsInAbstract, ...ev.sharedTerms];
}

export { rawTokens };
