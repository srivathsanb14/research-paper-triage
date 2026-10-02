// Loads the public catalog (same-site static files; nothing about the visitor is sent).
import { titleKey } from "./text.js";

export const PAPER_FIELDS = ["id", "title", "abstract", "authors", "venue", "year", "published", "url", "pdf_url", "categories",
  "source", "citation_count", "work_type", "venue_type", "venue_core", "version", "oa_status", "fwci", "references_count"];

async function sha256(buf) {
  if (!crypto?.subtle) return null; // insecure origins (plain http) cannot verify; files are same-site
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function fetchVerified(url, expected) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  const digest = await sha256(buf);
  if (digest && expected && digest !== expected) throw new Error("checksum mismatch");
  return buf;
}

/** int8 vectors with per-vector scales → unit-length Float32Arrays. */
export function decodeVectors(buf, n, dim) {
  const view = new DataView(buf);
  const ints = new Int8Array(buf, n * 4);
  const out = [];
  for (let i = 0; i < n; i++) {
    const scale = view.getFloat32(i * 4, true);
    const v = new Float32Array(dim);
    let norm = 0;
    for (let j = 0; j < dim; j++) { v[j] = ints[i * dim + j] * scale; norm += v[j] * v[j]; }
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < dim; j++) v[j] /= norm;
    out.push(v);
  }
  return out;
}

export function cleanPaper(raw, extra = {}) {
  const p = {};
  for (const k of PAPER_FIELDS) if (raw[k] !== undefined) p[k] = raw[k];
  p.authors ||= [];
  p.categories ||= [];
  p.abstract ||= "";
  return Object.assign(p, extra);
}

/**
 * {manifest, embeddings, areas: [{id,label,count}], papers} — papers carry
 * `area` and, when embeddings are available, `_dense`.
 */
export async function loadCatalog(base, onProgress = () => {}) {
  const manifest = await (await fetch(new URL("manifest.json", base))).json();
  let embeddings = null;
  try { embeddings = await (await fetch(new URL("embeddings.json", base))).json(); } catch { /* semantic vectors optional */ }
  let done = 0;
  const shards = await Promise.all(manifest.areas.map(async area => {
    const buf = await fetchVerified(new URL(area.file, base), area.sha256);
    const rows = JSON.parse(new TextDecoder().decode(buf));
    let vecs = null;
    const e = embeddings?.areas?.[area.id];
    if (e && e.papers_sha256 === area.sha256 && e.count === rows.length) {
      try { vecs = decodeVectors(await fetchVerified(new URL(e.file, base), e.sha256), e.count, embeddings.dim); } catch (err) { console.warn("Embeddings unavailable for", area.id, err); }
    }
    onProgress(++done / manifest.areas.length);
    return rows.map((r, i) => cleanPaper(r, { area: area.id, ...(vecs ? { _dense: vecs[i] } : {}) }));
  }));
  const papers = [];
  const seen = new Set();
  for (const p of shards.flat()) {
    const key = titleKey(p.title);
    if (seen.has(p.id) || (key && seen.has(key))) continue;
    seen.add(p.id);
    if (key) seen.add(key);
    papers.push(p);
  }
  const areas = manifest.areas.map(a => ({ id: a.id, label: a.label, count: papers.filter(p => p.area === a.id).length }));
  return { manifest, embeddings, areas, papers };
}

const MAX_IMPORT = 2000;
/** Validates an imported JSON collection (same rules as triage/transfer.parse_papers). */
export function parseImport(text) {
  if (text.length > 10 * 1024 * 1024) throw new Error("Choose a JSON file smaller than 10 MB.");
  let rows;
  try { rows = JSON.parse(text); } catch { throw new Error("This file is not valid JSON."); }
  if (!Array.isArray(rows) || rows.length < 1 || rows.length > MAX_IMPORT) throw new Error(`A collection must contain 1–${MAX_IMPORT.toLocaleString()} papers.`);
  const ids = new Set();
  return rows.map(r => {
    if (!r || typeof r !== "object" || Array.isArray(r)) throw new Error("Each paper must be a JSON object.");
    if (typeof r.id !== "string" || !r.id.trim() || typeof r.title !== "string" || !r.title.trim()) throw new Error("Every paper needs a non-empty id and title.");
    if (ids.has(r.id)) throw new Error(`Duplicate paper id: ${r.id}`);
    ids.add(r.id);
    for (const k of ["authors", "categories"]) if (r[k] !== undefined && (!Array.isArray(r[k]) || r[k].some(x => typeof x !== "string"))) throw new Error(`${k} must be a list of text values.`);
    for (const k of ["year", "citation_count", "references_count"]) if (r[k] != null && (!Number.isInteger(r[k]) || r[k] < 0)) throw new Error(`${k} must be a non-negative integer or null.`);
    for (const k of ["abstract", "venue", "published", "url", "pdf_url", "source", "work_type", "venue_type", "version", "oa_status"]) if (r[k] !== undefined && typeof r[k] !== "string") throw new Error(`${k} must be text.`);
    for (const k of ["url", "pdf_url"]) if (r[k] && !/^https?:\/\/[^\s/]+/i.test(r[k])) throw new Error(`${k} must be an http or https URL.`);
    return cleanPaper(r, { area: "imported", source: r.source || "import" });
  });
}

/** What is stored for a visitor-added paper (no runtime caches). */
export function toRecord(p) {
  const r = cleanPaper(p, { area: p.area || "imported", added_at: p.added_at || new Date().toISOString() });
  if (p._dense) r.dense = Array.from(p._dense);
  return r;
}

export function fromRecord(r) {
  const p = cleanPaper(r, { area: r.area || "imported", added_at: r.added_at });
  if (r.dense) p._dense = Float32Array.from(r.dense);
  return p;
}
