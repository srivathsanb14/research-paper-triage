// Optional live lookups against the public OpenAlex API (free, no key, CORS-enabled).
// Only search terms or identifiers are sent; never ratings, labels or profiles.
import { cleanPaper } from "./data.js";

const API = "https://api.openalex.org/works";
const SELECT = "id,doi,title,type,abstract_inverted_index,authorships,primary_location,best_oa_location,open_access," +
  "publication_date,publication_year,primary_topic,cited_by_count,fwci,referenced_works_count";

export function abstractText(index) {
  if (!index || typeof index !== "object") return "";
  const words = [];
  for (const [w, positions] of Object.entries(index)) {
    if (!Array.isArray(positions)) continue;
    for (const i of positions) if (Number.isInteger(i) && i >= 0 && i < 10000) words[i] = w;
  }
  return words.filter(Boolean).join(" ");
}

/** Same mapping as scripts/build_catalog.paper_from_work. */
export function paperFromWork(work, minWords = 30) {
  const abstract = abstractText(work.abstract_inverted_index).replace(/\s+/g, " ").trim();
  if (!work.title || abstract.split(" ").length < minWords) return null;
  const loc = work.primary_location || {};
  const src = loc.source || {};
  const oa = work.best_oa_location || {};
  const topic = work.primary_topic || {};
  const doi = (work.doi || "").trim().toLowerCase();
  const safe = u => (typeof u === "string" && /^https?:\/\//i.test(u) ? u : "");
  return cleanPaper({
    id: doi ? "doi:" + doi.replace("https://doi.org/", "") : "openalex:" + String(work.id).split("/").pop(),
    title: String(work.title).replace(/<[^>]+>/g, ""),
    abstract,
    authors: (work.authorships || []).map(a => a.author?.display_name).filter(Boolean),
    venue: src.display_name || "",
    published: work.publication_date || "",
    year: work.publication_year ?? null,
    url: safe(doi) || safe(loc.landing_page_url) || safe(work.id),
    pdf_url: safe(oa.pdf_url),
    categories: [topic.field?.display_name, topic.display_name].filter(Boolean),
    source: "openalex",
    citation_count: work.cited_by_count ?? null,
    work_type: work.type || "",
    venue_type: src.type || "",
    venue_core: src.is_core === true,
    version: loc.version || "",
    oa_status: work.open_access?.oa_status || "",
    fwci: typeof work.fwci === "number" ? work.fwci : null,
    references_count: work.referenced_works_count ?? null,
  }, {
    // OpenAlex topic ids, so the app can file the paper under one of its areas.
    _field: Number(String(topic.field?.id || "").split("/").pop()) || null,
    _subfield: Number(String(topic.subfield?.id || "").split("/").pop()) || null,
  });
}

async function get(params) {
  const url = new URL(API);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  let res;
  for (let attempt = 0; ; attempt++) { // one dropped connection shouldn't fail the search
    try { res = await fetch(url); break; } catch (e) { if (attempt >= 2) throw e; await new Promise(r => setTimeout(r, 800 * (attempt + 1))); }
  }
  if (res.status === 429) throw new Error("OpenAlex is busy right now. Try again in a minute.");
  if (!res.ok) throw new Error(`OpenAlex returned HTTP ${res.status}.`);
  return res.json();
}

/**
 * Recent papers matching the visitor's own search terms and/or OpenAlex field
 * ids. Without a query, returns the most-cited and newest papers in the fields.
 */
export async function searchRecent(query, { days = 365, limit = 100, fields = [], page = 1 } = {}) {
  const since = new Date(Date.now() - days * 86400000).toISOString().slice(0, 10);
  const fieldFilter = fields.length ? `,primary_topic.field.id:${fields.join("|")}` : "";
  const params = {
    filter: `from_publication_date:${since},has_abstract:true,is_retracted:false,language:en,type:article|review|preprint${fieldFilter}`,
    per_page: String(Math.min(100, limit)),
    page: String(page),
    select: SELECT,
  };
  if (query.trim()) params.search = query.trim().slice(0, 300);
  const sorts = query.trim() ? ["relevance_score:desc", "cited_by_count:desc"] : ["cited_by_count:desc", "publication_date:desc"];
  const out = [];
  const seen = new Set();
  for (const sort of sorts) {
    const data = await get({ ...params, sort, per_page: String(Math.ceil(limit / 2)) });
    for (const w of data.results || []) {
      const p = paperFromWork(w);
      if (p && !seen.has(p.id)) { seen.add(p.id); out.push(p); }
    }
  }
  return out;
}

/** DOIs, doi.org links, arXiv ids/links, or OpenAlex W-ids → papers (abstract optional). */
export function parseIdentifiers(text) {
  const out = [];
  for (const raw of String(text).split(/[\s,;]+/).map(s => s.trim()).filter(Boolean)) {
    const doi = raw.match(/10\.\d{4,9}\/[^\s"<>]+/i);
    const arxiv = raw.match(/(?:arxiv\.org\/(?:abs|pdf)\/|arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?/i);
    const oa = raw.match(/\b(W\d{6,})\b/);
    if (doi) out.push(`doi:${doi[0].replace(/[.)]+$/, "").toLowerCase()}`);
    else if (arxiv) out.push(`doi:10.48550/arxiv.${arxiv[1]}`);
    else if (oa) out.push(`openalex:${oa[1]}`);
  }
  return [...new Set(out)].slice(0, 50);
}

export async function resolveIdentifiers(ids) {
  const dois = ids.filter(i => i.startsWith("doi:")).map(i => i.slice(4));
  const works = ids.filter(i => i.startsWith("openalex:")).map(i => i.slice(9));
  const found = [];
  if (dois.length) found.push(...(await get({ filter: `doi:${dois.join("|")}`, per_page: "50", select: SELECT })).results);
  if (works.length) found.push(...(await get({ filter: `openalex:${works.join("|")}`, per_page: "50", select: SELECT })).results);
  return found.map(w => paperFromWork(w, 0)).filter(Boolean);
}
