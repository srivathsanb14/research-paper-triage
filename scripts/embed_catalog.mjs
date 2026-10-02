// Precompute MiniLM sentence embeddings for the public catalog.
//
//   node scripts/embed_catalog.mjs            (after scripts/build_catalog.py)
//
// The browser app embeds the visitor's interests with the *same* model and
// quantization (transformers.js, Xenova/all-MiniLM-L6-v2, q8), so paper and
// profile vectors share one space. Shards whose papers have not changed keep
// their vectors. Output: data/catalog/embeddings.json + one .bin per area.
//
// .bin layout (little endian): n float32 scales, then n × dim int8 values.
// A vector is int8 × scale, re-normalised to unit length by the reader.
import { createHash } from "node:crypto";
import { readFile, writeFile, readdir, unlink } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "@huggingface/transformers";

export const MODEL = "Xenova/all-MiniLM-L6-v2";
export const DTYPE = "q8";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DIR = path.join(ROOT, "data", "catalog");

// Must match paperText() in web/js/engine.js.
export const paperText = p => `${p.title}. ${p.abstract || ""}`.slice(0, 2000);

const sha256 = buf => createHash("sha256").update(buf).digest("hex");

function quantize(data, n, dim) {
  const out = Buffer.alloc(n * 4 + n * dim);
  for (let i = 0; i < n; i++) {
    let max = 0;
    for (let j = 0; j < dim; j++) max = Math.max(max, Math.abs(data[i * dim + j]));
    const scale = max / 127 || 1;
    out.writeFloatLE(scale, i * 4);
    for (let j = 0; j < dim; j++) out.writeInt8(Math.round(data[i * dim + j] / scale), n * 4 + i * dim + j);
  }
  return out;
}

async function main() {
  const manifest = JSON.parse(await readFile(path.join(DIR, "manifest.json"), "utf8"));
  let previous = {};
  try { previous = JSON.parse(await readFile(path.join(DIR, "embeddings.json"), "utf8")); } catch { /* first run */ }
  const sameModel = previous.model === MODEL && previous.dtype === DTYPE;
  let extractor;
  const info = { version: 1, model: MODEL, dtype: DTYPE, dim: 384, text: "title. abstract (first 2,000 characters)", areas: {} };
  for (const area of manifest.areas) {
    const old = sameModel ? previous.areas?.[area.id] : null;
    if (old && old.papers_sha256 === area.sha256) {
      try {
        if (sha256(await readFile(path.join(DIR, old.file))) === old.sha256) {
          info.areas[area.id] = old;
          console.log(`${area.label}: unchanged`);
          continue;
        }
      } catch { /* re-embed */ }
    }
    extractor ??= await pipeline("feature-extraction", MODEL, { dtype: DTYPE });
    const papers = JSON.parse(await readFile(path.join(DIR, area.file), "utf8"));
    const dim = info.dim;
    const all = new Float32Array(papers.length * dim);
    for (let i = 0; i < papers.length; i += 32) {
      const batch = papers.slice(i, i + 32).map(paperText);
      const out = await extractor(batch, { pooling: "mean", normalize: true });
      all.set(out.data, i * dim);
    }
    const bin = quantize(all, papers.length, dim);
    const digest = sha256(bin);
    const file = `${area.id}-${digest}.bin`;
    await writeFile(path.join(DIR, file), bin);
    info.areas[area.id] = { file, sha256: digest, papers_sha256: area.sha256, count: papers.length };
    console.log(`${area.label}: embedded ${papers.length} papers`);
  }
  await writeFile(path.join(DIR, "embeddings.json"), JSON.stringify(info, null, 2) + "\n");
  const keep = new Set(Object.values(info.areas).map(a => a.file));
  for (const name of await readdir(DIR)) {
    if (name.endsWith(".bin") && !keep.has(name)) await unlink(path.join(DIR, name));
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
