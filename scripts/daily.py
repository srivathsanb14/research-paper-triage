"""Daily job: fetch new arXiv listings for every profile, then optionally send a digest.

Schedule it with cron / launchd, e.g. every weekday at 7:30:

    30 7 * * 1-5  cd /path/to/project && .venv/bin/python scripts/daily.py --digest slack

Options:
    --profile NAME        only this profile (default: all non-demo profiles)
    --force               fetch even if the last fetch was < 20 h ago
    --digest {none,print,slack,email}
    --out FILE            also write the digest (Markdown) to FILE
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import export  # noqa: E402
from triage.explain import explain_many  # noqa: E402
from triage.pipeline import TriageEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--digest", choices=["none", "print", "slack", "email"], default="none")
    ap.add_argument("--out")
    args = ap.parse_args()

    engine = TriageEngine()
    store = engine.store
    profiles = [p for p in store.list_profiles() if not store.get_setting(p.id, "demo", False)]
    if args.profile:
        profiles = [p for p in profiles if p.name == args.profile]
    if not profiles:
        print("No matching profiles.")
        return 1

    failures = 0
    for prof in profiles:
        started = datetime.now(timezone.utc).isoformat()
        res = engine.refresh_if_due(prof, force=args.force)
        if res is None:
            print(f"[{prof.name}] nothing to do (no categories set, auto-fetch off, or fetched recently)")
        elif not res["ok"]:
            failures += 1
            print(f"[{prof.name}] fetch failed: {res['message']}")
        else:
            print(f"[{prof.name}] {res['added']} new papers ({res['fetched']} fetched), full text for {res['full_text']}")
        if args.digest == "none" and not args.out:
            continue
        since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat() if res and res.get("ok") else started
        new_ids = engine.new_since(prof, since)
        run = engine.run(prof)
        shown = [r for r in run.results if r.paper.id in new_ids and r.label in ("READ", "SKIM")]
        reasons = {k: v["reason"] for k, v in explain_many(shown, prof, run.papers).items()}
        md = export.digest_markdown(prof.name, run.results, reasons, new_ids)
        if args.out:
            Path(args.out).write_text(md)
        if args.digest == "print":
            print(md)
        elif args.digest == "slack":
            export.send_slack(md)
        elif args.digest == "email":
            export.send_email(md, f"Reading digest — {prof.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
