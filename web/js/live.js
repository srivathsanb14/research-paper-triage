// Optional live lookups against public, keyless, CORS-enabled scholarly APIs: OpenAlex
// (default), Europe PMC and Crossref. Only search terms or identifiers are sent; never
// ratings, labels or profiles. Papers from every source share one shape and are
// de-duplicated by DOI, so the same paper from two sources appears once.
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

// ------------------------------------------------------------ Europe PMC and Crossref

const EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search";
const CROSSREF = "https://api.crossref.org/works";
const safeUrl = u => (typeof u === "string" && /^https?:\/\//i.test(u) ? u : "");
const stripTags = t => String(t || "").replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/\s+/g, " ").trim();
const doiId = doi => `doi:${String(doi).trim().toLowerCase().replace(/^https?:\/\/(dx\.)?doi\.org\//, "")}`;

async function getJson(url, name) {
  let res;
  for (let attempt = 0; ; attempt++) {
    try { res = await fetch(url); break; } catch (e) { if (attempt >= 2) throw e; await new Promise(r => setTimeout(r, 800 * (attempt + 1))); }
  }
  if (res.status === 429) throw new Error(`${name} is busy right now.`);
  if (!res.ok) throw new Error(`${name} returned HTTP ${res.status}.`);
  return res.json();
}

/** Europe PMC "core" record → paper (null without a DOI or a real abstract). */
export function paperFromEuropePmc(r, minWords = 30) {
  const abstract = stripTags(r.abstractText);
  if (!r.title || !r.doi || abstract.split(" ").length < minWords) return null;
  const preprint = r.source === "PPR" || /preprint/i.test(r.pubType || "");
  const review = /review/i.test(r.pubType || "");
  const journal = r.journalInfo?.journal?.title || (preprint ? r.bookOrReportDetails?.publisher || "" : "");
  const pdf = (r.fullTextUrlList?.fullTextUrl || []).find(u => u.documentStyle === "pdf" && u.availability !== "Subscription required");
  return cleanPaper({
    id: doiId(r.doi),
    title: stripTags(r.title).replace(/\.$/, ""),
    abstract,
    authors: (r.authorList?.author || []).map(a => a.fullName).filter(Boolean),
    venue: journal,
    published: r.firstPublicationDate || "",
    year: Number(r.pubYear) || null,
    url: safeUrl(`https://doi.org/${String(r.doi).toLowerCase()}`),
    pdf_url: safeUrl(pdf?.url),
    categories: ["Life sciences"],
    source: "europepmc",
    citation_count: Number.isFinite(r.citedByCount) ? r.citedByCount : null,
    work_type: preprint ? "preprint" : review ? "review" : "article",
    venue_type: preprint ? "repository" : journal ? "journal" : "",
    oa_status: r.isOpenAccess === "Y" ? "open" : "",
  });
}

/** Crossref work → paper (null without a title or a real abstract; Crossref abstracts are JATS XML). */
export function paperFromCrossref(w, minWords = 30) {
  const title = Array.isArray(w.title) ? w.title[0] : w.title;
  const abstract = stripTags(w.abstract).replace(/^abstract\s*/i, "");
  if (!title || !w.DOI || abstract.split(" ").length < minWords) return null;
  const parts = w.issued?.["date-parts"]?.[0] || [];
  const published = parts.length ? [parts[0], String(parts[1] || 1).padStart(2, "0"), String(parts[2] || 1).padStart(2, "0")].join("-") : "";
  const preprint = w.type === "posted-content";
  const venue = Array.isArray(w["container-title"]) ? w["container-title"][0] || "" : "";
  return cleanPaper({
    id: doiId(w.DOI),
    title: stripTags(title),
    abstract,
    authors: (w.author || []).map(a => [a.given, a.family].filter(Boolean).join(" ") || a.name).filter(Boolean),
    venue: venue || (preprint ? w.institution?.[0]?.name || "" : ""),
    published,
    year: parts[0] ?? null,
    url: safeUrl(`https://doi.org/${String(w.DOI).toLowerCase()}`),
    categories: [],
    source: "crossref",
    citation_count: Number.isFinite(w["is-referenced-by-count"]) ? w["is-referenced-by-count"] : null,
    work_type: preprint ? "preprint" : "article",
    venue_type: preprint ? "repository" : w.type === "proceedings-article" ? "conference" : venue ? "journal" : "",
  });
}

const isoDay = daysAgo => new Date(Date.now() - daysAgo * 86400000).toISOString().slice(0, 10);

export async function searchEuropePmc(query, { days = 365, limit = 40, page = 1 } = {}) {
  const q = query.trim();
  if (!q) return [];
  const url = new URL(EPMC);
  url.search = new URLSearchParams({
    query: `(${q.slice(0, 300)}) AND (FIRST_PDATE:[${isoDay(days)} TO ${isoDay(0)}]) AND (HAS_ABSTRACT:y) AND (LANG:eng)`,
    format: "json", resultType: "core", pageSize: String(limit), page: String(page), sort: "P_PDATE_D desc",
  });
  const data = await getJson(url, "Europe PMC");
  return (data.resultList?.result || []).map(r => paperFromEuropePmc(r)).filter(Boolean);
}

export async function searchCrossref(query, { days = 365, limit = 40, page = 1 } = {}) {
  const q = query.trim();
  if (!q) return [];
  const url = new URL(CROSSREF);
  url.search = new URLSearchParams({
    "query.bibliographic": q.slice(0, 300),
    filter: `from-pub-date:${isoDay(days)},has-abstract:true,type:journal-article`,
    rows: String(limit), offset: String((page - 1) * limit),
    select: "DOI,title,abstract,author,container-title,issued,is-referenced-by-count,type",
  });
  const data = await getJson(url, "Crossref");
  return (data.message?.items || []).map(w => paperFromCrossref(w)).filter(Boolean);
}

export const SOURCES = {
  openalex: { label: "OpenAlex", hint: "All fields" },
  europepmc: { label: "Europe PMC", hint: "Biomedicine and life sciences, including preprints" },
  crossref: { label: "Crossref", hint: "Journal articles from publishers" },
};
export const DEFAULT_SOURCES = { openalex: true, europepmc: true, crossref: true };

/**
 * Search every enabled source at once. One failing source never blocks the others.
 * Europe PMC and Crossref need search terms; OpenAlex also works from fields alone.
 * Returns {papers (de-duplicated), status: {source: {n} | {error}}}.
 */
export async function searchAll(query, { sources = DEFAULT_SOURCES, fields = [], limit = 40, page = 1, bio = true } = {}) {
  const jobs = {
    openalex: () => searchRecent(query, { fields, limit, page }),
    europepmc: () => searchEuropePmc(query, { limit, page }),
    crossref: () => searchCrossref(query, { limit, page }),
  };
  const names = Object.keys(jobs).filter(n => sources[n] && (n !== "europepmc" || bio)); // Europe PMC only makes sense for life-science profiles
  const settled = await Promise.allSettled(names.map(n => jobs[n]()));
  const papers = [], seen = new Set(), status = {};
  settled.forEach((r, i) => {
    if (r.status === "rejected") { status[names[i]] = { error: r.reason?.message || "failed" }; return; }
    status[names[i]] = { n: r.value.length };
    for (const p of r.value) if (!seen.has(p.id)) { seen.add(p.id); papers.push(p); }
  });
  return { papers, status };
}
