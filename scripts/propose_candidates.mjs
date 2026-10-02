// Pick papers to label for a few research profiles: balanced across five bands
// of the profile score so the set is not almost all Skip.
//   node scripts/propose_candidates.mjs  → data/labels/proposals/candidates/<slug>.jsonl
// Rows hold only what a labeller needs (profile + paper); the score is not included,
// so proposals made from these files cannot just echo the ranker.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "@huggingface/transformers";
import { cleanPaper, decodeVectors } from "../web/js/data.js";
import { DenseSpace, RelevanceModel, buildProfileVectors, profileTexts } from "../web/js/engine.js";
import { MODEL, DTYPE } from "./embed_catalog.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CAT = path.join(ROOT, "data/catalog");
export const PROFILES = JSON.parse(await readFile(path.join(ROOT, "data/labels/proposals/profiles.json"), "utf8")).profiles;
const PER_PROFILE = 200;

function rng(seed) { let s = seed; return () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648); }

async function main() {
  const man = JSON.parse(await readFile(path.join(CAT, "manifest.json"), "utf8"));
  const emb = JSON.parse(await readFile(path.join(CAT, "embeddings.json"), "utf8"));
  const papers = [];
  for (const a of man.areas) {
    const rows = JSON.parse(await readFile(path.join(CAT, a.file), "utf8"));
    const b = await readFile(path.join(CAT, emb.areas[a.id].file));
    const vecs = decodeVectors(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), rows.length, emb.dim);
    rows.forEach((r, i) => papers.push(cleanPaper(r, { area: a.id, _dense: vecs[i] })));
  }
  const ex = await pipeline("feature-extraction", MODEL, { dtype: DTYPE });
  const space = new DenseSpace();
  await mkdir(path.join(ROOT, "data/labels/proposals/candidates"), { recursive: true });
  for (const prof of PROFILES) {
    const texts = profileTexts(prof).map(t => t[1]);
    const o = await ex(texts, { pooling: "mean", normalize: true });
    const enc = texts.map((_, i) => o.data.slice(i * o.dims[1], (i + 1) * o.dims[1]));
    const model = new RelevanceModel(space, buildProfileVectors(prof, space, enc));
    const pool = papers.filter(p => prof.fields.includes(p.area));
    const scored = model.score(pool).map((s, i) => [s.prior, pool[i]]).sort((a, b) => b[0] - a[0]);
    const rand = rng(prof.slug.length * 7919);
    const size = Math.ceil(scored.length / 5);
    const bands = [0, 1, 2, 3, 4].map(i => scored.slice(i * size, (i + 1) * size).map(x => x[1]));
    // The top band is small but where Read/Skim live, so it is taken in full.
    const take = [Math.min(bands[0].length, 70), 45, 35, 25, 25];
    let chosen = bands.flatMap((b, i) => [...b].sort(() => rand() - 0.5).slice(0, take[i]));
    chosen = chosen.slice(0, PER_PROFILE).sort(() => rand() - 0.5);
    const lines = chosen.map(p => JSON.stringify({ paper_id: p.id, title: p.title, abstract: p.abstract, venue: p.venue, published: p.published }));
    await writeFile(path.join(ROOT, `data/labels/proposals/candidates/${prof.slug}.jsonl`), lines.join("\n") + "\n");
    console.log(`${prof.name}: ${chosen.length} candidates from ${pool.length}`);
  }
  process.exit(0);
}
main();
