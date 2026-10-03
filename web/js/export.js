// Reference-manager exports (BibTeX, RIS, CSV) and file downloads.

const bibEscape = s => String(s || "").replace(/\\/g, "\\textbackslash{}").replace(/[{}%&]/g, m => "\\" + m);

function citeKey(p, used) {
  const last = (p.authors?.[0]?.split(/\s+/).pop() || "anon").normalize("NFKD").replace(/[^\x00-\x7f]/g, "").toLowerCase() || "anon";
  const word = (p.title.toLowerCase().match(/[a-z]+/g) || []).find(w => w.length > 3) || "paper";
  const base = `${last}${p.year || ""}${word}`.replace(/[^a-z0-9]/g, "") || "paper";
  let key = base, n = 2;
  while (used.has(key)) key = `${base}${n++}`;
  used.add(key);
  return key;
}

const isPreprint = p => !p.venue || p.venue === "arXiv preprint" || p.venue_type === "repository";

export function toBibtex(papers) {
  const used = new Set();
  return papers.map(p => {
    const f = { title: `{${bibEscape(p.title)}}`, author: bibEscape((p.authors || []).join(" and ")), year: p.year || "", url: p.url, abstract: bibEscape(p.abstract) };
    if (p.id.startsWith("doi:")) f.doi = p.id.slice(4);
    let type = "misc";
    if (!isPreprint(p)) { type = "article"; f.journal = bibEscape(p.venue); }
    const body = Object.entries(f).filter(([, v]) => v).map(([k, v]) => `  ${k} = {${v}}`).join(",\n");
    return `@${type}{${citeKey(p, used)},\n${body}\n}`;
  }).join("\n\n") + (papers.length ? "\n" : "");
}

export function toRis(papers) {
  return papers.map(p => {
    const l = [isPreprint(p) ? "TY  - GEN" : "TY  - JOUR", `TI  - ${p.title}`, ...(p.authors || []).map(a => `AU  - ${a}`)];
    if (p.year) l.push(`PY  - ${p.year}`);
    if (p.venue) l.push(`JO  - ${p.venue}`);
    if (p.abstract) l.push(`AB  - ${p.abstract}`);
    if (p.url) l.push(`UR  - ${p.url}`);
    if (p.id.startsWith("doi:")) l.push(`DO  - ${p.id.slice(4)}`);
    l.push("ER  - ");
    return l.join("\n");
  }).join("\n\n") + (papers.length ? "\n" : "");
}

const csvCell = v => {
  const s = Array.isArray(v) ? v.join("; ") : v == null ? "" : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function toCsv(rows, columns) {
  return [columns.join(","), ...rows.map(r => columns.map(c => csvCell(r[c])).join(","))].join("\n") + "\n";
}

export function digestMarkdown(profileName, results, reasons) {
  const reads = results.filter(r => r.label === "READ" && !r.lead);
  const skims = results.filter(r => r.label === "SKIM" && !r.lead);
  const when = new Date().toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
  const lines = [`# Reading digest — ${profileName}`, `_${when} · ${reads.length} to read · ${skims.length} to skim_`, ""];
  if (reads.length) {
    lines.push("## Read");
    for (const r of reads) {
      lines.push(`- **[${r.paper.title}](${r.paper.url})** — ${(r.paper.authors || []).slice(0, 3).join(", ")}`);
      if (reasons[r.paper.id]) lines.push(`  ${reasons[r.paper.id]}`);
    }
    lines.push("");
  }
  if (skims.length) {
    lines.push("## Skim");
    for (const r of skims.slice(0, 15)) lines.push(`- [${r.paper.title}](${r.paper.url})`);
    if (skims.length > 15) lines.push(`- …and ${skims.length - 15} more in the app`);
    lines.push("");
  }
  if (!reads.length && !skims.length) lines.push("Nothing worth your time in this batch.");
  return lines.join("\n");
}

export function download(name, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const slug = s => String(s || "papers").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "papers";
