"""Copy a Kaggle artifacts zip into the repo and regenerate docs/METRICS.md.

The Kaggle notebooks end by writing `deeptrace_*_artifacts.zip` containing `ml/reports/`,
`docs/plots/` and (for the final notebook) `docs/METRICS.md`. This script unpacks those into the
working tree so the committed numbers always come from a real run.

    python scripts/import_kaggle_artifacts.py --zip ~/Downloads/deeptrace_final_artifacts.zip

Nothing outside ml/reports and docs/plots is written, and existing files are replaced only by files
of the same name present in the zip (use --dry-run to see the plan first).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from deeptrace_ml.utils.logging import get_logger

log = get_logger("import_kaggle_artifacts")
ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent
ALLOWED_PREFIXES = ("ml/reports/", "docs/plots/", "docs/METRICS.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", type=Path, required=True, help="deeptrace_*_artifacts.zip downloaded from Kaggle")
    p.add_argument("--dry-run", action="store_true", help="List what would be copied and exit")
    p.add_argument("--skip-render", action="store_true", help="Don't regenerate docs/METRICS.md afterwards")
    return p.parse_args()


def safe_members(archive: zipfile.ZipFile) -> list[str]:
    """Only files under the expected prefixes, and no path traversal."""
    members = []
    for name in archive.namelist():
        if name.endswith("/"):
            continue
        normalised = name.replace("\\", "/").lstrip("./")
        if ".." in Path(normalised).parts:
            log.warning("Skipping suspicious path in zip: %s", name)
            continue
        if normalised.startswith(ALLOWED_PREFIXES):
            members.append(name)
        else:
            log.debug("Ignoring %s (outside ml/reports, docs/plots)", name)
    return members


def main() -> None:
    args = parse_args()
    if not args.zip.is_file():
        raise SystemExit(f"Zip not found: {args.zip}")

    with zipfile.ZipFile(args.zip) as archive:
        members = safe_members(archive)
        if not members:
            raise SystemExit(
                "Nothing to import: the zip has no ml/reports or docs/plots entries. "
                "Download the artifacts zip produced by the last cell of the Kaggle notebook."
            )
        log.info("%d files to import from %s", len(members), args.zip.name)
        for name in sorted(members)[:10]:
            log.info("  %s", name)
        if len(members) > 10:
            log.info("  ... and %d more", len(members) - 10)
        if args.dry_run:
            return

        with tempfile.TemporaryDirectory() as tmp:
            archive.extractall(tmp, members=members)
            for name in members:
                relative = Path(name.replace("\\", "/").lstrip("./"))
                destination = REPO_DIR / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(tmp) / name, destination)

    reports = sorted(p.name for p in (ML_DIR / "reports").glob("*.json"))
    plots = len(list((REPO_DIR / "docs/plots").rglob("*.png")))
    log.info("Imported %d report JSONs and %d plots", len(reports), plots)
    log.info("Reports: %s", ", ".join(reports) if reports else "(none)")

    if not args.skip_render:
        subprocess.run([sys.executable, str(ML_DIR / "scripts/render_metrics_md.py")], check=True)
        log.info("Regenerated docs/METRICS.md from the imported reports")

    log.info("Next: review `git status`, check docs/METRICS.md, then commit.")


if __name__ == "__main__":
    main()
