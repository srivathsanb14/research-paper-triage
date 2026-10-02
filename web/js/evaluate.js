// Evaluation against your hand labels (ported from triage/evaluate.py).
// Every number comes from cross-validation: no paper is scored by a model that
// saw its label. Baselines are reported next to the personalised model.
import { LABELS, MIN_LABELS, MIN_TRAIN_EXAMPLES, RelevanceModel, averagePrecision, buildExamples, folds, labelFor, selectBlend } from "./engine.js";

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

/** Training sizes to try: every 5 up to 20, then roughly +50% each step, up to `pool` labels (the last point is the full pool). */
export function curveSizes(pool) {
  const sizes = [];
  for (let k = MIN_TRAIN_EXAMPLES; k <= pool; k = k < 20 ? k + 5 : Math.round((k * 1.5) / 5) * 5) sizes.push(k);
  if (sizes.length > 1 && pool - sizes[sizes.length - 1] >= 5 && pool > 20) sizes.push(pool);
  return sizes;
}

/** `k` ids that keep the pool's label mix, with at least one of every label that exists (so early points can learn at all). */
export function stratifiedSubset(ids, k, labelOf, seed) {
  const groups = Object.fromEntries(LABELS.map(l => [l, []]));
  for (const id of [...ids].sort((a, b) => hashOf(`${seed}|${a}`) - hashOf(`${seed}|${b}`))) groups[labelOf(id)].push(id);
  const present = LABELS.filter(l => groups[l].length);
  const take = Object.fromEntries(present.map(l => [l, Math.max(1, Math.round((k * groups[l].length) / ids.length))]));
  const total = () => present.reduce((n, l) => n + take[l], 0);
  while (total() > k) { const l = present.reduce((a, b) => (take[a] >= take[b] ? a : b)); if (take[l] <= 1) break; take[l]--; }
  while (total() < k) {
    const room = present.filter(l => take[l] < groups[l].length);
    if (!room.length) break;
    const l = room.reduce((a, b) => (groups[a].length - take[a] >= groups[b].length - take[b] ? a : b));
    take[l]++;
  }
  return present.flatMap(l => groups[l].slice(0, take[l]));
}

const mean = xs => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);

/**
 * Learning curve: how well the model ranks papers it has NOT seen as it learns from more of your labels.
 *
 * It uses the same stratified 5-fold split as the cross-validated report above it. For each training
 * size k, the model learns from k labels drawn (with your label mix) from each fold's training part and
 * then scores that fold's held-out papers. Every labelled paper is therefore scored by a model that never
 * saw it, and each point is NDCG@10 over all of them. This repeats `repeats` times with different draws,
 * and the band is the spread of those draws. At the largest k the model has all the training labels,
 * which is exactly the report's own cross-validation.
 *
 * Lines: the hand-set profile alone (it still uses the k labels as reference papers, so it moves a little),
 * the model learning from labels only, the app's blend at `weight` (the cross-validated learning weight),
 * and a random order. At the largest k these match the "Profile only", "Learned from feedback only" and
 * "Personalised" rows of the report.
 */
export function learningCurve(ctx, { weight = 0, repeats = 5, k: folds5 = 5 } = {}) {
  const { labels, byId } = ctx;
  const ids = Object.keys(labels).filter(id => byId.has(id) && ctx.space.vec(byId.get(id)));
  const labelOf = id => labels[id].label ?? labels[id];
  const counts = Object.fromEntries(LABELS.map(l => [l, ids.filter(id => labelOf(id) === l).length]));
  const present = LABELS.filter(l => counts[l]).length;
  const parts = folds(ids, labelOf, Math.min(folds5, ids.length));
  const pool = Math.min(...parts.map(f => ids.length - f.length));
  const sizes = curveSizes(pool);
  if (ids.length < 20 || present < 2 || sizes.length < 2) {
    return { ok: false, n: ids.length, message: "Label at least 20 papers, with at least two different labels, to see a learning curve." };
  }
  const index = new Map(ids.map((id, i) => [id, i]));
  const truth = ids.map(labelOf);
  const w = Math.min(1, Math.max(0, weight || 0));
  // learned/blend/prior [si][r][i]: score of paper i by a model trained on sizes[si] labels in repeat r
  const grid = () => sizes.map(() => Array.from({ length: repeats }, () => new Array(ids.length)));
  const learned = grid(), blend = grid(), prior = grid();
  parts.forEach((test, fi) => {
    const trainPool = ids.filter(id => !test.includes(id));
    const testPapers = test.map(id => byId.get(id));
    sizes.forEach((k, si) => {
      for (let r = 0; r < repeats; r++) {
        const train = stratifiedSubset(trainPool, Math.min(k, trainPool.length), labelOf, `${fi}|${r}`);
        const trainLabels = Object.fromEntries(train.map(id => [id, labels[id]]));
        const model = new RelevanceModel(ctx.space, ctx.pv, { today: ctx.today }).fit(buildExamples(trainLabels, []), byId);
        model.learnedWeight = 1;
        const rows = model.score(testPapers);
        test.forEach((id, j) => {
          const i = index.get(id), row = rows[j];
          prior[si][r][i] = row.prior;
          const l = row.learned ?? row.prior; // too few labels or one answer so far: nothing learned yet
          learned[si][r][i] = l;
          blend[si][r][i] = (1 - w) * row.prior + w * l;
        });
      }
    });
  });
  const ndcg = scores => ndcgAt(scores, truth);
  const spread = vals => { const m = mean(vals); return { m, lo: Math.min(...vals), hi: Math.max(...vals) }; };
  const points = sizes.map((k, si) => {
    const l = spread(learned[si].map(ndcg)), b = spread(blend[si].map(ndcg)), p = spread(prior[si].map(ndcg));
    return { k, learned: l.m, learnedLo: l.lo, learnedHi: l.hi, blend: b.m, blendLo: b.lo, blendHi: b.hi, prior: p.m, priorLo: p.lo, priorHi: p.hi };
  });
  // A random order's expected NDCG@10, from many deterministic shuffles of the same labels.
  const random = mean(Array.from({ length: 200 }, (_, d) => ndcg(ids.map(id => hashOf(`random|${d}|${id}`)))));
  const last = points[points.length - 1];
  const caught = points.find(p => p.learned >= p.prior - 1e-9);
  const lowSignal = counts.READ < 10;
  return {
    ok: true, n: ids.length, counts, repeats, weight: w, random, points,
    // How many labels until learning from labels alone matches the hand-set profile (null: not within your labels).
    catchUp: caught ? caught.k : null,
    gain: (w > 0 ? last.blend : last.learned) - last.prior,
    learnedGain: last.learned - last.prior,
    reliability: lowSignal || ids.length < 60 ? "low" : "ok",
    note: lowSignal ? `Only ${counts.READ} Read label${counts.READ === 1 ? "" : "s"}: one paper moves this curve a lot, so read it as a rough guide.` : ids.length < 60 ? "Under 60 labels: the curve is rough." : "",
  };
}
