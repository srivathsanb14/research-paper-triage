"""Exports (BibTeX, RIS — both import into Zotero, Mendeley, EndNote) and the reading digest.

The digest can be downloaded, posted to Slack (incoming webhook) or emailed
(SMTP). Delivery credentials come only from environment variables:

    SLACK_WEBHOOK_URL
    SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, DIGEST_FROM, DIGEST_TO
"""

from __future__ import annotations

import os
import re
import smtplib
import unicodedata
from datetime import datetime
from email.message import EmailMessage

import requests

from . import config
from .models import Paper, TriageResult

LABEL_TEXT = {"READ": "Read", "SKIM": "Skim", "SKIP": "Skip"}


def _bib_escape(s: str) -> str:
    return (s or "").replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}").replace("%", "\\%").replace("&", "\\&")


def _cite_key(p: Paper, used: set[str]) -> str:
    last = (p.authors[0].split()[-1] if p.authors else "anon").lower()
    last = unicodedata.normalize("NFKD", last).encode("ascii", "ignore").decode() or "anon"
    word = next((w for w in re.findall(r"[a-z]+", p.title.lower()) if len(w) > 3), "paper")
    base = re.sub(r"[^a-z0-9]", "", f"{last}{p.year or ''}{word}") or "paper"
    key, n = base, 2
    while key in used:
        key, n = f"{base}{n}", n + 1
    used.add(key)
    return key


def to_bibtex(papers: list[Paper]) -> str:
    used: set[str] = set()
    out = []
    for p in papers:
        fields = {
            "title": "{" + _bib_escape(p.title) + "}",
            "author": _bib_escape(" and ".join(p.authors)),
            "year": str(p.year or ""),
            "url": p.url,
            "abstract": _bib_escape(p.abstract),
        }
        entry = "misc"
        if p.id.startswith("arxiv:"):
            fields.update(eprint=p.id.split(":", 1)[1], archiveprefix="arXiv")
            if p.categories:
                fields["primaryclass"] = p.categories[0]
        if p.id.startswith("doi:"):
            fields["doi"] = p.id.split(":", 1)[1]
        if p.venue and p.venue != "arXiv preprint":
            entry = "article"
            fields["journal"] = _bib_escape(p.venue)
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
        out.append(f"@{entry}{{{_cite_key(p, used)},\n{body}\n}}")
    return "\n\n".join(out) + ("\n" if out else "")


def to_ris(papers: list[Paper]) -> str:
    out = []
    for p in papers:
        lines = ["TY  - JOUR" if p.venue and p.venue != "arXiv preprint" else "TY  - GEN", f"TI  - {p.title}"]
        lines += [f"AU  - {a}" for a in p.authors]
        if p.year:
            lines.append(f"PY  - {p.year}")
        if p.venue:
            lines.append(f"JO  - {p.venue}")
        if p.abstract:
            lines.append(f"AB  - {p.abstract}")
        if p.url:
            lines.append(f"UR  - {p.url}")
        if p.id.startswith("doi:"):
            lines.append(f"DO  - {p.id.split(':', 1)[1]}")
        lines.append("ER  - ")
        out.append("\n".join(lines))
    return "\n\n".join(out) + ("\n" if out else "")


def digest_markdown(
    profile_name: str,
    results: list[TriageResult],
    reasons: dict[str, str],
    new_ids: set[str] | None = None,
    max_skim: int = 10,
) -> str:
    """A short, skimmable digest: the Read list with reasons, then top Skim titles."""
    pool = [r for r in results if new_ids is None or r.paper.id in new_ids]
    reads = [r for r in pool if r.label == "READ" and not r.group_lead]
    skims = [r for r in pool if r.label == "SKIM" and not r.group_lead]
    when = datetime.now().strftime("%d %B %Y")
    scope = f"{len(pool)} new papers" if new_ids is not None else f"{len(pool)} papers"
    lines = [f"# Reading digest — {profile_name}", f"_{when} · {scope} · {len(reads)} to read · {len(skims)} to skim_", ""]
    if reads:
        lines.append("## Read")
        for r in reads:
            lines.append(f"- **[{r.paper.title}]({r.paper.url})** — {r.paper.author_str}")
            if reasons.get(r.paper.id):
                lines.append(f"  {reasons[r.paper.id]}")
        lines.append("")
    if skims:
        lines.append("## Skim")
        for r in skims[:max_skim]:
            lines.append(f"- [{r.paper.title}]({r.paper.url})")
        if len(skims) > max_skim:
            lines.append(f"- …and {len(skims) - max_skim} more in the app")
        lines.append("")
    if not reads and not skims:
        lines.append("Nothing new worth your time this period.")
    return "\n".join(lines)


def delivery_channels() -> dict[str, bool]:
    return {
        "slack": bool(os.environ.get("SLACK_WEBHOOK_URL")),
        "email": all(os.environ.get(k) for k in ("SMTP_HOST", "DIGEST_FROM", "DIGEST_TO")),
    }


def send_slack(markdown: str, webhook: str | None = None) -> None:
    url = webhook or os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        raise RuntimeError("SLACK_WEBHOOK_URL is not set")
    # Slack mrkdwn: **bold** → *bold*, [t](u) → <u|t>, headings → bold lines.
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", markdown)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    text = re.sub(r"^#+\s*(.+)$", r"*\1*", text, flags=re.M)
    r = requests.post(url, json={"text": text}, timeout=config.HTTP_TIMEOUT_S)
    if r.status_code >= 300:
        raise RuntimeError(f"Slack returned HTTP {r.status_code}: {r.text[:200]}")


def send_email(markdown: str, subject: str) -> None:
    host = os.environ.get("SMTP_HOST")
    sender, to = os.environ.get("DIGEST_FROM"), os.environ.get("DIGEST_TO")
    if not (host and sender and to):
        raise RuntimeError("SMTP_HOST, DIGEST_FROM and DIGEST_TO must be set")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    msg.set_content(markdown)
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=30) as s:
        s.starttls()
        if os.environ.get("SMTP_USER"):
            s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
        s.send_message(msg)
