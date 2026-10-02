// Offline evaluation of the browser ranking engine on the shared label dataset.
//
//   node scripts/evaluate_web.mjs [--out docs/evaluation]
//
// Reads data/labels/profiles.json and data/labels/labels.jsonl (one row per profile and paper,
// joined to data/catalog/ by paper id). For every profile, runs the same 5-fold cross-validated
// evaluation as the Insights page, with MiniLM (off-the-shelf) and TF-IDF vectors, and writes
// report.md + report.json. Profiles are tagged by label origin (manual / reviewed / rule).
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "@huggingface/transformers";
import { decodeVectors, cleanPaper } from "../web/js/data.js";
import { DEFAULT_CUTOFFS, DenseSpace, SparseSpace, buildProfileVectors, profileTexts } from "../web/js/engine.js";
import { evaluate } from "../web/js/evaluate.js";
import { MODEL, DTYPE, paperText } from "./embed_catalog.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CAT = path.join(ROOT, "data", "catalog");

export async function readLabels() {
  const dir = path.join(ROOT, "data", "labels");
  const profiles = new Map(JSON.parse(await readFile(path.join(dir, "profiles.json"), "utf8")).profiles.map(p => [p.slug, p]));
  const rows = (await readFile(path.join(dir, "labels.jsonl"), "utf8")).split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l));
  return { profiles, rows };
}

async function catalogPapers() {
  const man = JSON.parse(await readFile(path.join(CAT, "manifest.json"), "utf8"));
  const emb = JSON.parse(await readFile(path.join(CAT, "embeddings.json"), "utf8"));
  const out = new Map();
  for (const a of man.areas) {
    const rows = JSON.parse(await readFile(path.join(CAT, a.file), "utf8"));
    const e = emb.areas[a.id];
    const b = await readFile(path.join(CAT, e.file));
    const vecs = decodeVectors(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), rows.length, emb.dim);
    rows.forEach((r, i) => out.set(r.id, cleanPaper(r, { area: a.id, _dense: vecs[i] })));
  }
  return out;
}

const f2 = x => (x == null || Number.isNaN(x) ? "—" : x.toFixed(2));

async function main() {
  const args = process.argv.slice(2);
  const outDir = args.includes("--out") ? args[args.indexOf("--out") + 1] : path.join(ROOT, "docs", "evaluation");
  const { profiles, rows } = await readLabels();
  const catalog = await catalogPapers();
  const extractor = await pipeline("feature-extraction", MODEL, { dtype: DTYPE });
  const embed = async texts => { const o = await extractor(texts, { pooling: "mean", normalize: true }); return Array.from({ length: o.dims[0] }, (_, i) => o.data.slice(i * o.dims[1], (i + 1) * o.dims[1])); };

  const groups = new Map();
  for (const r of rows) {
    if (!groups.has(r.profile)) groups.set(r.profile, []);
    groups.get(r.profile).push(r);
  }
  const results = [];
  for (const [slug, rs] of groups) {
    const p0 = profiles.get(slug);
    const origin = [...new Set(rs.map(r => r.origin))].sort().join("/");
    const profile = { name: p0.name, description: p0.description || "", keywords: p0.keywords || [], focus: p0.focus || "", avoid: p0.avoid || [] };
    const byId = catalog;
    const labels = Object.fromEntries(rs.filter(r => byId.has(r.paper_id)).map(r => [r.paper_id, { label: r.label }]));
    const papers = [...byId.values()];
    for (const [spaceName, space] of [["MiniLM", new DenseSpace()], ["TF-IDF", new SparseSpace(papers)]]) {
      const texts = profileTexts(profile).map(t => t[1]);
      const enc = spaceName === "MiniLM" ? await embed(texts) : texts.map(t => space.vectorize(t));
      const pv = buildProfileVectors(profile, space, enc);
      const ctx = { papers, byId, space, pv, labels, feedback: [], seeds: [], cutoffs: DEFAULT_CUTOFFS, hours: 3, budget: false, group: false, today: Date.now() };
      for (const good of ["read", "read_skim"]) {
        const rep = evaluate(ctx, good);
        results.push({ origin, profile: profile.name, space: spaceName, good, n: rep.n, counts: rep.counts, ok: rep.ok, message: rep.message,
          blend: rep.blend ? { weight: rep.blend.weight, scores: rep.blend.scores } : null, randomAp: rep.randomAp,
          systems: rep.systems?.map(s => ({ name: s.name, ndcg10: s.ndcg, ap: s.ap, spearman: s.rho, hit10: s.disc.hit10, to80: s.disc.p80 })),
          random80: rep.random80, current: rep.current && { macroF1: rep.current.macroF1, accuracy: rep.current.accuracy, severe: rep.current.severe } });
      }
    }
    console.log(`${origin} · ${profile.name}: ${rs.length} labels`);
  }

  const lines = ["# Evaluation report", "", `Generated ${new Date().toISOString().slice(0, 10)} by \`node scripts/evaluate_web.mjs\` from \`data/labels/labels.jsonl\` and \`data/labels/profiles.json\`.`,
    "Every number is 5-fold cross-validated: no paper is scored by a model trained on its own label. “Good” = Read (or Read + Skim where noted).",
    "Origin: `reviewed` = a model proposed the label and a person accepted or changed it; `rule` = a transparent keyword rule (demo and pipeline check, not evidence of quality).", ""];
  for (const group of [["reviewed", "Labels reviewed by a person"], ["rule", "Rule-based labels (pipeline check)"]]) {
    const rs = results.filter(r => r.origin === group[0]);
    if (!rs.length) continue;
    lines.push(`## ${group[1]}`, "");
    for (const r of rs) {
      lines.push(`### ${r.profile} · ${r.space} · good = ${r.good === "read" ? "Read" : "Read or Skim"}`, "");
      lines.push(`${r.n} labels (Read ${r.counts.READ}, Skim ${r.counts.SKIM}, Skip ${r.counts.SKIP}).`, "");
      if (!r.ok) { lines.push(`_${r.message}_`, ""); continue; }
      lines.push(`Learning weight chosen by cross-validation: **${Math.round((r.blend.weight ?? 0) * 100)}%**. Random-order AP: ${f2(r.randomAp)}; papers to screen for 80% of good ones in random order: ${r.random80}.`, "");
      lines.push("| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |", "|---|---|---|---|---|---|");
      for (const s of r.systems) lines.push(`| ${s.name} | ${f2(s.ndcg10)} | ${f2(s.ap)} | ${f2(s.spearman)} | ${Math.round(s.hit10 * 100)}% | ${s.to80 ?? "—"} |`);
      lines.push("", `At the default cutoffs (Read ≥ ${DEFAULT_CUTOFFS.read}, Skim ≥ ${DEFAULT_CUTOFFS.skim}): macro-F1 ${f2(r.current.macroF1)}, accuracy ${f2(r.current.accuracy)}, ${r.current.severe} Read↔Skip errors.`, "");
    }
  }
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "report.md"), lines.join("\n") + "\n");
  await writeFile(path.join(outDir, "report.json"), JSON.stringify(results, null, 2) + "\n");
  console.log(`Wrote ${path.relative(ROOT, outDir)}/report.md`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(e => { console.error(e); process.exitCode = 1; });
