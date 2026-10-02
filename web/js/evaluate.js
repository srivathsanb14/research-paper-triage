// Evaluation against your hand labels (ported from triage/evaluate.py).
// Every number comes from cross-validation: no paper is scored by a model that
// saw its label. Baselines are reported next to the personalised model.
import { LABELS, MIN_LABELS, MIN_TRAIN_EXAMPLES, RelevanceModel, averagePrecision, buildExamples, labelFor, selectBlend } from "./engine.js";

const GAIN = { READ: 2, SKIM: 1, SKIP: 0 };
export const SCREEN_MINUTES = 2; // reading a title + abstract and deciding

function order(scores) {
  return scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b);
}

export function ndcgAt(scores, truth, k = 10) {
  const o = order(scores).slice(0, k);
  const disc = i => 1 / Math.log2(i + 2);
  const dcg = o.reduce((s, i, r) => s + (2 ** GAIN[truth[i]] - 1) * disc(r), 0);
  const ideal = truth.map(t => GAIN[t]).sort((a, b) => b - a).slice(0, k);
  const idcg = ideal.reduce((s, g, r) => s + (2 ** g - 1) * disc(r), 0);
  return idcg > 0 ? dcg / idcg : NaN;
}

function ranks(xs) {
  const o = xs.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(xs.length);
  for (let i = 0; i < o.length;) {
    let j = i;
    while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++;
    for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}

export function spearman(scores, truth) {
  const a = ranks(scores), b = ranks(truth.map(t => GAIN[t]));
  const n = a.length, ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return da && db ? num / Math.sqrt(da * db) : NaN;
}

export function discovery(scores, good) {
  const total = good.filter(Boolean).length;
  let hits = 0;
  const found = order(scores).map(i => (hits += good[i] ? 1 : 0) / Math.max(1, total));
  const reach = share => { const i = found.findIndex(v => v >= share - 1e-9); return i < 0 ? null : i + 1; };
  const top = order(scores).slice(0, 10).filter(i => good[i]).length;
  return { found, first: reach(1 / Math.max(1, total)), half: reach(0.5), p80: reach(0.8), hit10: top / Math.max(1, Math.min(10, total)) };
}

/** Expected share of good papers found after k random picks is k/n. */
export function randomCurve(n) {
  return Array.from({ length: n }, (_, k) => (k + 1) / n);
}

export function labelMetrics(truth, pred) {
  const confusion = LABELS.map(t => LABELS.map(p => truth.filter((x, i) => x === t && pred[i] === p).length));
  const f1s = LABELS.map((lab, i) => {
    const tp = confusion[i][i];
    const fp = confusion.reduce((s, row, j) => s + (j !== i ? row[i] : 0), 0);
    const fn = confusion[i].reduce((s, v, j) => s + (j !== i ? v : 0), 0);
    return tp ? (2 * tp) / (2 * tp + fp + fn) : 0;
  });
  const present = LABELS.map(l => truth.includes(l));
  const macro = f1s.filter((_, i) => present[i]).reduce((a, b) => a + b, 0) / Math.max(1, present.filter(Boolean).length);
  const accuracy = truth.filter((t, i) => t === pred[i]).length / Math.max(1, truth.length);
  const severe = truth.filter((t, i) => (t === "READ" && pred[i] === "SKIP") || (t === "SKIP" && pred[i] === "READ")).length;
  return { confusion, macroF1: macro, accuracy, severe };
}

export function bestCutoffs(scores, truth) {
  const grid = [...new Set([...Array(51).keys()].map(i => +(i / 50).toFixed(2)))];
  let best = null;
  for (const skim of grid) for (const read of grid) {
    if (read < skim) continue;
    const m = labelMetrics(truth, scores.map(s => labelFor(s, { read, skim })));
    if (!best || m.macroF1 > best.m.macroF1 + 1e-9 || (Math.abs(m.macroF1 - best.m.macroF1) < 1e-9 && m.severe < best.m.severe)) best = { read, skim, m };
  }
  return best;
}

/**
 * Evaluation report for a profile context (same ctx as engine.rank).
 * `good` is "read" (READ counts as good) or "read_skim".
 */
export function evaluate(ctx, good = "read") {
  const ids = Object.keys(ctx.labels).filter(id => ctx.byId.has(id));
  const truthAll = ids.map(id => ctx.labels[id].label ?? ctx.labels[id]);
  const counts = Object.fromEntries(LABELS.map(l => [l, truthAll.filter(t => t === l).length]));
  if (ids.length < MIN_LABELS || Object.values(counts).filter(Boolean).length < 2) {
    return { ok: false, n: ids.length, counts, message: `Label at least ${MIN_LABELS} papers with at least two different labels.` };
  }
  const blend = selectBlend(ctx);
  const { rows } = blend.cv;
  const truth = rows.map(r => r.truth);
  const goodSet = good === "read_skim" ? new Set(["READ", "SKIM"]) : new Set(["READ"]);
  const isGood = truth.map(t => goodSet.has(t));
  const w = blend.weight ?? 0;
  const systems = [
    { key: "model", name: `Personalised (learning ${Math.round(w * 100)}%)`, scores: rows.map(r => (1 - w) * r.prior + w * r.learned) },
    { key: "prior", name: "Profile only (no learning)", scores: rows.map(r => r.prior) },
    { key: "learned", name: "Learned from feedback only", scores: rows.map(r => r.learned) },
    { key: "semantic", name: "Semantic similarity only", scores: rows.map(r => r.f[0]) },
    { key: "keyword", name: "Keyword matches only", scores: rows.map(r => r.f[3] + 1e-6 * r.f[0]) },
  ].map(s => ({ ...s, ndcg: ndcgAt(s.scores, truth), ap: averagePrecision(s.scores, isGood), rho: spearman(s.scores, truth), disc: discovery(s.scores, isGood) }));
  const nGood = isGood.filter(Boolean).length;
  const randomAp = nGood / truth.length;
  const model = systems[0];
  const current = labelMetrics(truth, model.scores.map(s => labelFor(s, ctx.cutoffs)));
  const tuned = bestCutoffs(model.scores, truth);
  const rand = randomCurve(truth.length);
  const random80 = rand.findIndex(v => v >= 0.8 - 1e-9) + 1;
  return {
    ok: nGood > 0 && nGood < truth.length,
    message: nGood === 0 ? "None of your labelled papers count as good yet." : nGood === truth.length ? "Every labelled paper counts as good: label some Skip papers too." : "",
    n: truth.length, counts, nGood, blend, systems, randomAp, current, tuned, random: rand, random80,
    minutesSaved: model.disc.p80 ? Math.max(0, (random80 - model.disc.p80) * SCREEN_MINUTES) : null,
    truth,
  };
}

function hashOf(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

/** Training sizes to try: every 5 up to 20, then roughly +50% each step, up to `pool` labels. */
export function curveSizes(pool) {
  const sizes = [];
  for (let k = MIN_TRAIN_EXAMPLES; k <= pool; k = k < 20 ? k + 5 : Math.round((k * 1.5) / 5) * 5) sizes.push(k);
  if (sizes.length > 1 && pool - sizes[sizes.length - 1] >= 5 && pool > 20) sizes.push(pool);
  return sizes;
}

/** Share of labels held out for scoring: enough for a stable NDCG@10, never fewer than 10. */
export const curveTestSize = n => Math.min(n - MIN_TRAIN_EXAMPLES, Math.max(10, Math.round(n * 0.3)));

/**
 * Learning curve: how ranking quality on papers the model has NOT seen grows
 * with the number of labels it learned from. Each repeat holds out a fixed test
 * set (about 30% of your labels, so every point is scored on the same papers),
 * then trains on the first k of the remaining labels in a random order (labels
 * only: no votes, no seeds). It reports NDCG@10 for the model as the app would
 * run it with k ratings, for the hand-set profile alone, and for a random order.
 * Averaged over `repeats` random splits because one split is noisy.
 */
export function learningCurve(ctx, { repeats = 4 } = {}) {
  const { labels, byId } = ctx;
  const ids = Object.keys(labels).filter(id => byId.has(id) && ctx.space.vec(byId.get(id)));
  const labelOf = id => labels[id].label ?? labels[id];
  const testN = curveTestSize(ids.length);
  const sizes = curveSizes(ids.length - testN);
  if (ids.length < MIN_TRAIN_EXAMPLES + 10 || new Set(ids.map(labelOf)).size < 2 || sizes.length < 2) {
    return { ok: false, n: ids.length, message: `Label at least ${MIN_TRAIN_EXAMPLES + 10} papers, with at least two different labels, to see a learning curve.` };
  }
  const acc = sizes.map(() => ({ model: [], prior: [], random: [] }));
  for (let r = 0; r < repeats; r++) {
    const order = [...ids].sort((a, b) => hashOf(`${r}|${a}`) - hashOf(`${r}|${b}`));
    sizes.forEach((k, si) => {
      const test = order.slice(0, testN);
      const train = order.slice(testN, testN + k);
      const trainLabels = Object.fromEntries(train.map(id => [id, labels[id]]));
      const model = new RelevanceModel(ctx.space, ctx.pv, { today: ctx.today }).fit(buildExamples(trainLabels, []), byId);
      const rows = model.score(test.map(id => byId.get(id)));
      const truth = test.map(labelOf);
      const m = ndcgAt(rows.map(x => x.final), truth);
      const pr = ndcgAt(rows.map(x => x.prior), truth);
      const rnd = [0, 1, 2, 3, 4, 5, 6, 7].reduce((a, d) => a + ndcgAt(test.map(id => hashOf(`${r}|${k}|${d}|${id}`)), truth), 0) / 8; // a steadier baseline than one draw
      if (![m, pr, rnd].some(Number.isNaN)) { acc[si].model.push(m); acc[si].prior.push(pr); acc[si].random.push(rnd); }
    });
  }
  const mean = xs => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  const points = sizes.map((k, i) => ({ k, model: mean(acc[i].model), prior: mean(acc[i].prior), random: mean(acc[i].random) })).filter(p => !Number.isNaN(p.model));
  if (points.length < 2) return { ok: false, n: ids.length, message: "Not enough variety in your labels yet for a learning curve." };
  const first = points[0], last = points[points.length - 1];
  return { ok: true, n: ids.length, testN, repeats, points, gain: last.model - last.prior, learned: last.model - first.model };
}
