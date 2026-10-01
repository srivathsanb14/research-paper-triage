"""Build the GitHub Pages artifact with only explicitly selected public app files.

    python scripts/build_pages.py
    python -m http.server 8000 --directory _site

No application dependencies are needed at build time. Never copy the working
directory wholesale: local databases, secrets and caches must stay local.
"""

from __future__ import annotations

import json
import hashlib
import re
import shutil
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "_site"


def build(output: Path = OUTPUT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "loader.js", "site.css"):
        shutil.copyfile(ROOT / "web" / name, output / name)
    (output / ".nojekyll").touch()
    paths = [ROOT / "app.py", *sorted((ROOT / "triage").glob("*.py")), ROOT / "data/sample_papers.json"]
    catalog_dir = ROOT / "data/catalog"
    manifest_path = catalog_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths.append(manifest_path)
    public_catalog = output / "catalog"
    public_catalog.mkdir(exist_ok=True)
    for area in manifest["areas"]:
        filename = area["file"]
        if not re.fullmatch(r"[a-z]+-[0-9a-f]{64}\.json", filename):
            raise ValueError("Invalid catalog filename")
        raw = (catalog_dir / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != area["sha256"]:
            raise ValueError("Catalog checksum mismatch")
        (public_catalog / filename).write_bytes(raw)
    shutil.copyfile(manifest_path, public_catalog / "manifest.json")
    with ZipFile(output / "app.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    theme = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())["theme"]
    settings = {f"theme.{key}": value for key, value in theme.items() if not isinstance(value, dict)}
    settings.update({f"theme.sidebar.{key}": value for key, value in theme.get("sidebar", {}).items()})
    # Stlite 1.9.2's custom web-font loader crashes outside a Helmet provider.
    # Built-in fonts also avoid another external dependency on first launch.
    settings.update({"theme.font": "sans-serif", "theme.headingFont": "serif"})
    settings.update({"client.toolbarMode": "minimal", "browser.gatherUsageStats": False})
    (output / "app-config.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Built {output} ({len(paths)} app files; no local databases or secrets).")


if __name__ == "__main__":
    build()
