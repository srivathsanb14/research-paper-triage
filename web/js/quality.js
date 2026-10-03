// Worth-it signals; mirrors triage/quality.py (which builds the dataset) and reads the same rules file.
// Signals never change the relevance score. They are cues a reader can check.

let R = null;

export async function loadRules(url) {
  const raw = await (await fetch(url)).json();
  const rx = p => new RegExp(p, "i");
  R = {
    ...raw,
    codeRe: raw.code.map(rx),
    dataRe: raw.data.map(rx),
    evidenceRe: Object.entries(raw.evidence).map(([name, p]) => [name, rx(p)]),
    hypeRe: raw.hype.map(rx),
    reviewRe: rx(raw.review_title),
  };
  return R;
}

export function peerReviewed(p) {
  if (p.work_type === "preprint" || p.venue_type === "repository" || p.source === "arxiv" || p.version === "submittedVersion") return false;
  if (R.peer_reviewed_venues.includes(p.venue_type)) return true;
  return null;
}

export function signals(p) {
  const text = `${p.title}. ${p.abstract || ""}`;
  const evidence = R.evidenceRe.filter(([, rx]) => rx.test(text)).map(([name]) => name);
  const words = (p.abstract || "").split(/\s+/).filter(Boolean).length;
  const s = {
    peer_reviewed: peerReviewed(p),
    established_venue: !!p.venue_core,
    review: p.work_type === "review" || R.reviewRe.test(p.title),
    code: R.codeRe.some(rx => rx.test(text)),
    data: R.dataRe.some(rx => rx.test(text)),
    evidence,
    open_access: !!p.pdf_url || !["", "closed", undefined, null].includes(p.oa_status),
    impact: p.fwci != null && p.fwci >= R.impact_fwci,
    thin_abstract: words < R.thin_abstract_words,
    promotional: R.hypeRe.filter(rx => rx.test(text)).length >= 2,
    few_references: p.references_count != null && p.references_count > 0 && p.references_count < R.few_references,
  };
  const pts = R.points;
  s.points =
    pts.peer_reviewed * !!s.peer_reviewed + pts.established_venue * s.established_venue +
    pts.code * s.code + pts.data * s.data + Math.min(pts.evidence_max, pts.evidence_each * evidence.length) +
    pts.open_access * s.open_access + pts.impact * s.impact + pts.thin_abstract * s.thin_abstract +
    pts.promotional * s.promotional + pts.few_references * s.few_references;
  s.level = s.points >= R.levels.strong ? "strong" : s.points >= R.levels.moderate ? "moderate" : "limited";
  return s;
}

/** Badges for display: [{key, text, tone: "good"|"info"|"warn", tip}] */
export function badges(s, p) {
  const out = [];
  if (s.peer_reviewed === true) out.push({ key: "peer", text: "Peer-reviewed", tone: "good", tip: `Published in a ${p.venue_type || "journal"}${p.venue ? `: ${p.venue}` : ""}` });
  else if (s.peer_reviewed === false) out.push({ key: "preprint", text: "Preprint", tone: "info", tip: "Not yet peer-reviewed (repository or submitted version)" });
  if (s.established_venue) out.push({ key: "core", text: "Established venue", tone: "good", tip: "Venue is in the CWTS core list of indexed journals" });
  if (s.review) out.push({ key: "review", text: "Review", tone: "info", tip: "Survey, review or overview — good for getting oriented" });
  if (s.code) out.push({ key: "code", text: "Code", tone: "good", tip: "Abstract links to or mentions released code" });
  if (s.data) out.push({ key: "data", text: "Data", tone: "good", tip: "Abstract mentions released or public data" });
  for (const e of s.evidence.slice(0, 2)) out.push({ key: "ev", text: e, tone: "good", tip: "Study-design cue found in the abstract" });
  if (s.open_access) out.push({ key: "oa", text: "Open access", tone: "info", tip: "Free full text is available" });
  if (s.impact) out.push({ key: "impact", text: "Cited above average", tone: "good", tip: `Field-weighted citation impact ${p.fwci?.toFixed(1)} (1.0 = field average)` });
  if (s.thin_abstract) out.push({ key: "thin", text: "Short abstract", tone: "warn", tip: `Fewer than ${R.thin_abstract_words} words: hard to judge from the abstract` });
  if (s.promotional) out.push({ key: "hype", text: "Promotional wording", tone: "warn", tip: "Two or more hype terms (e.g. “groundbreaking”, “unprecedented”)" });
  if (s.few_references) out.push({ key: "refs", text: "Few references", tone: "warn", tip: `Cites fewer than ${R.few_references} works` });
  return out;
}
