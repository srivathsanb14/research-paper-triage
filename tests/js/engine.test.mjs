// Behaviour of the browser ranking engine: npm run test:js
import assert from "node:assert/strict";
import test from "node:test";
import { parseImport } from "../../web/js/data.js";
import {
  DEFAULT_CUTOFFS, SparseSpace, adaptiveCutoffs, buildExamples, buildProfileVectors, groupSimilar, profileTexts, rank, readBudget,
} from "../../web/js/engine.js";
import { curveSizes, evaluate, learningCurve, stratifiedSubset } from "../../web/js/evaluate.js";
import { toBibtex, toCsv } from "../../web/js/export.js";
import { paperFromCrossref, paperFromEuropePmc, parseIdentifiers, paperFromWork, searchAll } from "../../web/js/live.js";

const RAG = [
  ["Retrieval-Augmented Generation for Scientific Question Answering", "We present a retrieval-augmented generation pipeline that answers questions over scientific papers using dense retrieval and a reranker."],
  ["Evaluating Retrieval Quality in RAG Pipelines", "We study how retrieval quality affects the faithfulness of answers in retrieval augmented generation systems and propose evaluation metrics."],
  ["Dense Passage Retrieval with Hard Negatives for Open-Domain QA", "Dense passage retrieval improves open-domain question answering when trained with hard negatives mined from BM25."],
  ["LLM Agents that Search: Tool Use for Multi-hop Question Answering", "Large language model agents call search tools iteratively to answer multi-hop questions and cite retrieved evidence."],
  ["Reranking Retrieved Passages with Cross-Encoders", "Cross-encoder rerankers improve precision of retrieved passages for question answering and retrieval augmented generation."],
  ["Faithfulness Metrics for Retrieval Augmented Generation", "We propose metrics that check whether generated answers are supported by retrieved passages in retrieval augmented generation."],
];
const OFF = [
  ["Quadrotor Control with Model Predictive Control", "We design a model predictive controller for agile quadrotor flight in cluttered outdoor environments."],
  ["Legged Robot Locomotion via Reinforcement Learning", "A quadruped robot learns to walk over rough terrain with deep reinforcement learning in simulation."],
  ["Coral Reef Bleaching under Marine Heatwaves", "Marine heatwaves drive coral bleaching; we analyse two decades of reef surveys across the Pacific."],
  ["Medieval Trade Routes in the Baltic", "Archival evidence on medieval trade routes and merchant networks in the Baltic Sea region."],
  ["Soil Carbon in Agricultural Landscapes", "Soil organic carbon responds to tillage and crop rotation in agricultural landscapes over ten years."],
  ["Robotic Grasping with Tactile Sensors", "Tactile sensing enables robust robotic grasping of deformable objects in manipulation tasks."],
];
const papers = () => [...RAG, ...OFF].map(([title, abstract], i) => ({ id: `p${i}`, title, abstract, authors: ["A. Author"], published: "2026-09-01" }));
const profile = { description: "I build retrieval-augmented generation systems for question answering", keywords: ["retrieval augmented generation", "question answering"], focus: "faithfulness of RAG answers", avoid: ["robotics"] };

function ctx(ps, extra = {}) {
  const space = new SparseSpace(ps);
  const pv = buildProfileVectors(profile, space, profileTexts(profile).map(t => space.vectorize(t[1])));
  return { papers: ps, byId: new Map(ps.map(p => [p.id, p])), space, pv, labels: {}, feedback: [], seeds: [], cutoffs: DEFAULT_CUTOFFS, hours: 3, budget: false, group: false, today: Date.parse("2026-10-01"), ...extra };
}

test("on-topic papers outrank off-topic ones from the profile alone", () => {
  const { results } = rank(ctx(papers()));
  const top = results.slice(0, RAG.length).map(r => r.paper.id);
  assert.deepEqual(new Set(top), new Set(RAG.map((_, i) => `p${i}`)));
  assert.equal(results.at(-1).label, "SKIP");
});

test("excluded topics are penalised", () => {
  const { results } = rank(ctx(papers()));
  const robot = results.find(r => r.paper.title.startsWith("Robotic Grasping"));
  assert.ok(robot.f[5] > 0, "avoid feature fires");
  assert.equal(robot.label, "SKIP");
});

test("a correction overrides the model label, and the reading budget demotes overflow", () => {
  const ps = papers();
  const c = ctx(ps, { cutoffs: { read: 0.05, skim: 0.01 }, budget: true, hours: 1, feedback: [{ pid: "p10", action: "correct", value: "READ" }] });
  const { results } = rank(c);
  assert.equal(results.find(r => r.paper.id === "p10").label, "READ");
  const modelReads = results.filter(r => r.label === "READ" && r.paper.id !== "p10");
  assert.ok(modelReads.length <= readBudget(1), "the model never puts more than the budget in Read");
  assert.ok(results.some(r => r.note.startsWith("Moved to Skim")));
});

test("learning switches on with enough ratings and only counts when validated", () => {
  const ps = papers();
  const labels = Object.fromEntries(ps.map((p, i) => [p.id, { label: i < RAG.length ? (i % 2 ? "SKIM" : "READ") : "SKIP" }]));
  const out = rank(ctx(ps, { labels }));
  assert.equal(out.model.info.learned, true);
  assert.equal(out.blend.validated, true);
  assert.ok([0, 0.15, 0.3, 0.5, 0.7].includes(out.blend.weight));
  const rep = evaluate(ctx(ps, { labels }));
  assert.ok(rep.ok);
  assert.ok(rep.systems[0].ap > rep.randomAp, "beats random order");
  assert.equal(rep.truth.length, ps.length);
});

test("feedback examples follow priority: seed > label > correction > vote > implicit", () => {
  const ex = buildExamples({ a: { label: "SKIP" } }, [
    { pid: "a", action: "useful" }, { pid: "b", action: "dismiss" }, { pid: "b", action: "useful" },
    { pid: "c", action: "save" }, { pid: "c", action: "unsave" }, { pid: "d", action: "dismiss" }, { pid: "d", action: "undismiss" },
  ], { seeds: ["s"] });
  const by = Object.fromEntries(ex.map(e => [e.pid, e]));
  assert.equal(by.s.source, "seed");
  assert.equal(by.a.target, 0);
  assert.equal(by.b.target, 1);
  assert.equal(by.c, undefined);
  assert.equal(by.d, undefined);
});

test("near-identical papers are grouped under the higher-ranked one", () => {
  const ps = papers();
  ps.push({ ...ps[0], id: "dup", title: ps[0].title + " (extended)" });
  const c = ctx(ps);
  const { results } = rank(c);
  groupSimilar(results, c.space);
  const lead = results.find(r => r.similar.includes("dup") || r.similar.includes("p0"));
  assert.ok(lead, "duplicate grouped");
});

test("imports are validated like the Python importer", () => {
  assert.throws(() => parseImport('[{"id":"a","title":"T","url":"javascript:alert(1)"}]'), /http/);
  assert.throws(() => parseImport('[{"id":"a"}]'), /title/);
  assert.throws(() => parseImport('[{"id":"a","title":"T"},{"id":"a","title":"U"}]'), /Duplicate/);
  assert.throws(() => parseImport('[{"id":"a","title":"T","year":"2026"}]'), /integer/);
  assert.equal(parseImport('[{"id":"a","title":"T"}]')[0].area, "imported");
});

test("OpenAlex works and identifiers are parsed safely", () => {
  const words = "We evaluate retrieval augmented generation with a benchmark of many tasks and release code for the community to reuse in future work on faithfulness and grounding today across many open domains".split(" ");
  const index = {};
  words.forEach((w, i) => (index[w] ||= []).push(i));
  const p = paperFromWork({ id: "https://openalex.org/W1", doi: "https://doi.org/10.1/ABC", title: "T", abstract_inverted_index: index, primary_location: { source: { type: "journal" }, landing_page_url: "javascript:x" } });
  assert.equal(p.id, "doi:10.1/abc");
  assert.equal(p.abstract, words.join(" "));
  assert.equal(p.venue_type, "journal");
  assert.equal(p.url, "https://doi.org/10.1/abc");
  assert.deepEqual(parseIdentifiers("10.18653/v1/2020.emnlp-main.550, https://arxiv.org/abs/2005.11401v2 W123456789"),
    ["doi:10.18653/v1/2020.emnlp-main.550", "doi:10.48550/arxiv.2005.11401", "openalex:W123456789"]);
});

test("exports escape BibTeX and CSV specials", () => {
  const bib = toBibtex([{ id: "doi:10.1/x", title: "Costs & {braces} at 50%", authors: ["Ada Lovelace"], year: 2026, venue: "J", venue_type: "journal", url: "https://doi.org/10.1/x" }]);
  assert.match(bib, /^@article\{lovelace2026costs,/);
  assert.match(bib, /Costs \\& \\\{braces\\\} at 50\\%/);
  assert.equal(toCsv([{ a: 'x,"y"', b: ["p", "q"] }], ["a", "b"]), 'a,b\n"x,""y""",p; q\n');
});

test("adaptive cutoffs fill Read when scores run low, and never raise the fixed cutoffs", () => {
  const low = Array.from({ length: 100 }, (_, i) => 0.5 - i * 0.004);
  const c = adaptiveCutoffs(low, DEFAULT_CUTOFFS, { read: 6 });
  assert.equal(low.filter(x => x >= c.read).length, 6);
  assert.ok(c.skim <= DEFAULT_CUTOFFS.skim && c.skim >= 0.2);
  const high = Array.from({ length: 100 }, () => 0.9);
  assert.deepEqual(adaptiveCutoffs(high, DEFAULT_CUTOFFS), DEFAULT_CUTOFFS);
});

test("pressing a rating again clears it", () => {
  const ex = buildExamples({}, [{ pid: "a", action: "useful" }, { pid: "a", action: "clear_vote" }, { pid: "b", action: "not_useful" }]);
  assert.deepEqual(ex.map(e => e.pid), ["b"]);
});

test("stratified subsets keep the label mix and at least one of every label", () => {
  const ids = Array.from({ length: 100 }, (_, i) => `p${i}`);
  const labelOf = id => (+id.slice(1) < 3 ? "READ" : +id.slice(1) < 20 ? "SKIM" : "SKIP");
  for (const k of [5, 10, 40]) {
    const sub = stratifiedSubset(ids, k, labelOf, "s");
    assert.equal(sub.length, k);
    assert.equal(new Set(sub).size, k);
    assert.deepEqual([...new Set(sub.map(labelOf))].sort(), ["READ", "SKIM", "SKIP"]);
  }
  assert.deepEqual(stratifiedSubset(ids, 10, labelOf, "s"), stratifiedSubset(ids, 10, labelOf, "s"), "deterministic");
});

test("learning curve rebuilds the recommender from k labels and ranks unseen labelled papers", () => {
  assert.deepEqual(curveSizes(8), [5]);
  assert.deepEqual(curveSizes(42), [5, 10, 15, 20, 30, 45].filter(k => k <= 42).concat(42));
  const ps = Array.from({ length: 5 }, (_, j) => papers().map(p => ({ ...p, id: `${p.id}-${j}`, title: `${p.title} (${j})` }))).flat();
  const labels = Object.fromEntries(ps.map(p => [p.id, { label: RAG.some(([t]) => p.title.startsWith(t)) ? "READ" : "SKIP" }]));
  const c = ctx(ps, { labels });
  const curve = learningCurve(c, { repeats: 2 });
  assert.ok(curve.ok);
  assert.equal(curve.n, 60);
  assert.equal(curve.points[0].k, 5);
  assert.ok(curve.points.every((p, i) => i === 0 || p.k > curve.points[i - 1].k));
  for (const pt of curve.points) {
    for (const key of ["rec", "prior"]) {
      assert.ok(pt[key] >= 0 && pt[key] <= 1);
      assert.ok(pt[`${key}Lo`] <= pt[key] + 1e-12 && pt[key] <= pt[`${key}Hi`] + 1e-12, `${key} sits inside its band`);
    }
    assert.ok(pt.weight >= 0 && pt.weight <= 1);
  }
  // The profile-alone line at the largest size is the report's own cross-validated "profile only" score.
  const rep = evaluate(c, "read");
  assert.ok(Math.abs(curve.points.at(-1).prior - rep.systems.find(s => s.key === "prior").ap) < 1e-9, "profile-only matches the report");
  assert.ok(curve.random < curve.points.at(-1).rec, "the recommender beats a random order");
  assert.equal(curve.goodLabel, "Read");
  const sparse = Object.fromEntries(Object.entries(labels).map(([id, l], i) => [id, { label: l.label === "READ" && i % 6 ? "SKIM" : l.label }]));
  assert.equal(learningCurve(ctx(ps, { labels: sparse }), { repeats: 1 }).goodLabel, "Read or Skim", "with under 10 Read labels, Skim counts as relevant too");
  assert.equal(learningCurve(ctx(papers(), { labels: { p0: { label: "READ" }, p6: { label: "SKIP" } } })).ok, false);
});

const LONG = Array.from({ length: 40 }, (_, i) => `word${i}`).join(" ");

test("Europe PMC and Crossref records map to the shared paper shape", () => {
  const e = paperFromEuropePmc({
    title: "A cohort study.", doi: "10.1000/ABC", abstractText: `<h4>Background</h4><p>${LONG} &amp; more</p>`, pubType: "research-article; journal article",
    authorList: { author: [{ fullName: "Ada L" }] }, journalInfo: { journal: { title: "J Health" } }, firstPublicationDate: "2026-09-03", pubYear: "2026", citedByCount: 4, isOpenAccess: "Y", source: "MED",
  });
  assert.equal(e.id, "doi:10.1000/abc");
  assert.equal(e.title, "A cohort study");
  assert.equal(e.venue_type, "journal");
  assert.equal(e.source, "europepmc");
  assert.ok(!/[<>]/.test(e.abstract) && e.abstract.includes("& more"));
  assert.equal(paperFromEuropePmc({ title: "x", doi: "10.1/x", abstractText: "too short" }), null);
  assert.equal(paperFromEuropePmc({ title: "x", abstractText: LONG }), null); // no DOI, so it can't be de-duplicated
  assert.equal(paperFromEuropePmc({ title: "P", doi: "10.1/p", abstractText: LONG, source: "PPR" }).venue_type, "repository");

  const c = paperFromCrossref({
    DOI: "10.1002/AIDI.1", title: ["From Data to Discovery"], abstract: `<jats:p>Abstract ${LONG}</jats:p>`, type: "journal-article",
    author: [{ given: "Zhi", family: "Cao" }], "container-title": ["Adv Discovery"], issued: { "date-parts": [[2026, 6, 28]] }, "is-referenced-by-count": 2,
  });
  assert.equal(c.id, "doi:10.1002/aidi.1");
  assert.equal(c.published, "2026-06-28");
  assert.deepEqual(c.authors, ["Zhi Cao"]);
  assert.ok(c.abstract.startsWith("word0"));
  assert.equal(c.work_type, "article");
});

test("searching several sources merges, de-duplicates and survives a failing source", async () => {
  const real = globalThis.fetch;
  const work = doi => ({ id: `https://openalex.org/W${doi.length}`, doi: `https://doi.org/${doi}`, title: `T ${doi}`, abstract_inverted_index: Object.fromEntries(LONG.split(" ").map((w, i) => [w, [i]])), publication_date: "2026-09-01", publication_year: 2026, primary_location: {} });
  globalThis.fetch = async url => {
    const u = String(url);
    if (u.includes("openalex")) return { ok: true, status: 200, json: async () => ({ results: [work("10.1/a"), work("10.1/b")] }) };
    if (u.includes("europepmc")) return { ok: true, status: 200, json: async () => ({ resultList: { result: [{ title: "Dup", doi: "10.1/a", abstractText: LONG, journalInfo: { journal: { title: "J" } } }, { title: "New", doi: "10.1/c", abstractText: LONG }] } }) };
    return { ok: false, status: 503, json: async () => ({}) }; // Crossref down
  };
  try {
    const { papers, status } = await searchAll("rag", { limit: 10 });
    assert.deepEqual(papers.map(p => p.id).sort(), ["doi:10.1/a", "doi:10.1/b", "doi:10.1/c"]);
    assert.equal(status.openalex.n, 2);
    assert.ok(status.crossref.error);
    const only = await searchAll("rag", { sources: { openalex: true }, limit: 10 });
    assert.deepEqual(Object.keys(only.status), ["openalex"]);
    const skipBio = await searchAll("rag", { sources: { europepmc: true }, bio: false });
    assert.deepEqual(skipBio, { papers: [], status: {} }); // not asked, not reported
  } finally { globalThis.fetch = real; }
});
