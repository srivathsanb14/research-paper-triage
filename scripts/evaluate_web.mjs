// Offline evaluation of the browser ranking engine on exported label files.
//
//   node scripts/evaluate_web.mjs data/labels/manual/*.jsonl [--out docs/evaluation]
//
// Label files are the CSV/JSONL exports from the app's Label tab; each row
// carries the paper, the label and the research profile it was judged against.
// For every profile, runs the same 5-fold cross-validated evaluation as the
// Insights page, with MiniLM (off-the-shelf) and TF-IDF vectors, and writes
// report.md + report.json. Synthetic files (label_source != "manual") are
// reported separately and never mixed with manual labels.
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

function parseCsv(text) {
  const rows = [];
  let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { cur += '"'; i++; } else if (c === '"') q = false; else cur += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(cur); cur = ""; }
    else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cur); rows.push(row); row = []; cur = ""; }
    else cur += c;
  }
  if (cur || row.length) { row.push(cur); rows.push(row); }
  const [head, ...body] = rows.filter(r => r.length > 1);
  return body.map(r => Object.fromEntries(head.map((h, i) => [h, r[i] ?? ""])));
}

export async function readLabels(files) {
  const rows = [];
  for (const f of files) {
    const text = await readFile(f, "utf8");
    const parsed = f.endsWith(".csv") ? parseCsv(text) : text.split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l));
    for (const r of parsed) {
      if (typeof r.profile_keywords === "string") r.profile_keywords = r.profile_keywords ? r.profile_keywords.split(/;\s*/) : [];
      if (typeof r.authors === "string") r.authors = r.authors ? r.authors.split(/;\s*/) : [];
      rows.push({ ...r, file: path.basename(f) });
    }
  }
  return rows;
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
  const files = args.filter((a, i) => !a.startsWith("--") && args[i - 1] !== "--out");
  if (!files.length) { console.error("Usage: node scripts/evaluate_web.mjs <labels.jsonl|csv>… [--out dir]"); process.exit(2); }
  const rows = await readLabels(files);
  const catalog = await catalogPapers();
  const extractor = await pipeline("feature-extraction", MODEL, { dtype: DTYPE });
  const embed = async texts => { const o = await extractor(texts, { pooling: "mean", normalize: true }); return Array.from({ length: o.dims[0] }, (_, i) => o.data.slice(i * o.dims[1], (i + 1) * o.dims[1])); };

  const groups = new Map();
  for (const r of rows) {
    const human = r.label_source === "manual" || r.label_source === "human-verified";
    const key = `${human ? "manual" : "synthetic"}|${r.profile_name}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  const results = [];
  for (const [key, rs] of groups) {
    const [source, name] = key.split("|");
    const r0 = rs[0];
    const profile = { name, description: r0.profile_description || "", keywords: r0.profile_keywords || [], focus: r0.profile_focus || "", avoid: [] };
    // Pool: the whole catalog plus any labelled paper the visitor had added.
    const byId = new Map(catalog);
    const missing = [];
    for (const r of rs) if (!byId.has(r.paper_id) && r.title) { const p = cleanPaper({ id: r.paper_id, title: r.title, abstract: r.abstract || "", authors: r.authors || [], published: r.published || "" }); byId.set(p.id, p); missing.push(p); }
    for (let i = 0; i < missing.length; i += 32) { const v = await embed(missing.slice(i, i + 32).map(paperText)); missing.slice(i, i + 32).forEach((p, j) => (p._dense = v[j])); }
    const labels = Object.fromEntries(rs.filter(r => byId.has(r.paper_id)).map(r => [r.paper_id, { label: r.label }]));
    const papers = [...byId.values()];
    const labelers = new Set(rs.map(r => r.labeler).filter(Boolean));
    const verified = rs.filter(r => r.proposed_label);
    const changed = verified.filter(r => r.proposed_label !== r.label).length;
    for (const [spaceName, space] of [["MiniLM", new DenseSpace()], ["TF-IDF", new SparseSpace(papers)]]) {
      const texts = profileTexts(profile).map(t => t[1]);
      const enc = spaceName === "MiniLM" ? await embed(texts) : texts.map(t => space.vectorize(t));
      const pv = buildProfileVectors(profile, space, enc);
      const ctx = { papers, byId, space, pv, labels, feedback: [], seeds: [], cutoffs: DEFAULT_CUTOFFS, hours: 3, budget: false, group: false, today: Date.now() };
      for (const good of ["read", "read_skim"]) {
        const rep = evaluate(ctx, good);
        results.push({ source, profile: name, space: spaceName, good, n: rep.n, counts: rep.counts, labelers: [...labelers], verified: verified.length, changed, ok: rep.ok, message: rep.message,
          blend: rep.blend ? { weight: rep.blend.weight, scores: rep.blend.scores } : null, randomAp: rep.randomAp,
          systems: rep.systems?.map(s => ({ name: s.name, ndcg10: s.ndcg, ap: s.ap, spearman: s.rho, hit10: s.disc.hit10, to80: s.disc.p80 })),
          random80: rep.random80, current: rep.current && { macroF1: rep.current.macroF1, accuracy: rep.current.accuracy, severe: rep.current.severe } });
      }
    }
    console.log(`${source} · ${name}: ${rs.length} labels`);
  }

  const lines = ["# Evaluation report", "", `Generated ${new Date().toISOString().slice(0, 10)} by \`node scripts/evaluate_web.mjs\` from ${files.map(f => `\`${path.relative(ROOT, f)}\``).join(", ")}.`,
    "Every number is 5-fold cross-validated: no paper is scored by a model trained on its own label. “Good” = Read (or Read + Skim where noted).", ""];
  for (const source of ["manual", "synthetic"]) {
    const rs = results.filter(r => r.source === source);
    if (!rs.length) continue;
    lines.push(`## ${source === "manual" ? "Human labels (manual, and model-proposed labels checked by a person)" : "Synthetic labels (pipeline check only, not evidence of quality)"}`, "");
    for (const r of rs) {
      lines.push(`### ${r.profile} · ${r.space} · good = ${r.good === "read" ? "Read" : "Read or Skim"}`, "");
      lines.push(`${r.n} labels (Read ${r.counts.READ}, Skim ${r.counts.SKIM}, Skip ${r.counts.SKIP})${r.labelers.length ? `, labellers: ${r.labelers.join(", ")}` : ""}.${r.verified ? ` ${r.verified} were model-proposed and checked by a person, who changed ${r.changed}.` : ""}`, "");
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
