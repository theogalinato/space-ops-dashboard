"""
v1.1: capture the fixed demo data snapshot under data/demo_snapshot/.

This app is deployed as a DEMO build (see the module comment at the top
of satellite_data.py for the full reasoning): live requests to
celestrak.org fail from both Streamlit Community Cloud and Render at the
TCP connect stage, while the identical request from a home network
succeeds every time. Rather than build an app that quietly degrades as
an unmaintained cache goes stale, the deployed app always reads from one
fixed snapshot captured here, with an honest "captured on [date]" banner
in the UI instead of a fallback that pretends to be temporary.

Run this ONCE, from a machine that can actually reach celestrak.org --
i.e. your computer, not a cloud host. There's no expectation you'll run
it again; if you ever do want a fresher snapshot, running it again and
committing the result is all that's needed -- nothing else in the app
depends on how often (or whether) this gets re-run.

Usage:
    python capture_demo_snapshot.py

Then commit what it wrote:
    git add data/demo_snapshot/
    git commit -m "Add demo data snapshot"
    git push
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

CELESTRAK_GROUP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
CELESTRAK_CATNR_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=tle"
CATALOGUE_CSV_PATH = "data/canadian_assets.csv"

_REQUEST_HEADERS = {
    "User-Agent": (
        "space-ops-dashboard/1.0 (educational SDA project; "
        "https://github.com/theogalinato/space-ops-dashboard)"
    )
}
_REQUEST_TIMEOUT_SECONDS = 15

SNAPSHOT_DIR = "data/demo_snapshot"
MANIFEST_PATH = os.path.join(SNAPSHOT_DIR, "manifest.json")


def _count_tles(text: str) -> int:
    """Quick sanity count: a TLE is 2 lines starting '1 ' / '2 ', each 69+ chars."""
    lines = text.splitlines()
    count = 0
    for i in range(len(lines) - 1):
        if (
            lines[i].startswith("1 ")
            and len(lines[i]) >= 69
            and lines[i + 1].startswith("2 ")
            and len(lines[i + 1]) >= 69
        ):
            count += 1
    return count


def _fetch(url: str) -> str:
    response = requests.get(url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def capture_group(group: str) -> str:
    """Fetch one named CelesTrak group, return its raw TLE text."""
    url = CELESTRAK_GROUP_URL.format(group=group)
    print(f"Fetching group '{group}' from {url} ...")
    text = _fetch(url)
    count = _count_tles(text)
    if count == 0:
        raise RuntimeError(f"CelesTrak responded, but no parseable TLEs were in '{group}'.")
    print(f"  -> {count} satellites")
    return text


def capture_catalogue() -> str:
    """
    Fetch every Canadian catalogue satellite directly by CATNR, so the
    catalogue tab never depends on whether a satellite happens to be a
    member of 'visual' or 'active' (see get_satellite_by_catnr's
    docstring in satellite_data.py -- RADARSAT-2 is a real example of a
    catalogue satellite NOT in 'visual' despite being trackable there).
    """
    catalogue_df = pd.read_csv(CATALOGUE_CSV_PATH)
    catnrs = catalogue_df["catnr"].tolist()
    print(f"Fetching {len(catnrs)} catalogue satellites individually by CATNR ...")

    blocks = []
    missing = []
    for catnr in catnrs:
        url = CELESTRAK_CATNR_URL.format(catnr=catnr)
        try:
            text = _fetch(url)
        except requests.exceptions.RequestException as exc:
            print(f"  -> CATNR {catnr}: FAILED ({exc})")
            missing.append(catnr)
            continue
        if _count_tles(text) == 0:
            print(f"  -> CATNR {catnr}: no TLE returned")
            missing.append(catnr)
            continue
        print(f"  -> CATNR {catnr}: OK")
        blocks.append(text.strip())

    if missing:
        print(
            f"\nWARNING: {len(missing)} catalogue satellite(s) could not be "
            f"fetched: {missing}. The demo will raise a clear error for "
            f"these specific satellites rather than silently omitting them "
            f"-- re-run this script later if you want another attempt."
        )
    if not blocks:
        raise RuntimeError("No catalogue satellites could be fetched at all.")

    return "\n".join(blocks) + "\n"


def write_group(group: str, text: str, manifest: dict) -> None:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, f"{group}.tle")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    manifest[group] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "satellite_count": _count_tles(text),
    }
    print(f"  -> wrote {path} ({manifest[group]['satellite_count']} satellites)")


def main() -> None:
    manifest: dict = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    steps = [
        ("visual", lambda: capture_group("visual")),
        ("active", lambda: capture_group("active")),
        ("catalogue", capture_catalogue),
    ]

    succeeded, failed = [], []
    for group, fn in steps:
        try:
            text = fn()
            write_group(group, text, manifest)
            succeeded.append(group)
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            print(f"FAILED capturing '{group}': {exc}\n")
            failed.append(group)

    if succeeded:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"\nWrote {MANIFEST_PATH} for: {', '.join(succeeded)}")

    if failed:
        print(f"Groups that failed (left untouched, old snapshot if any still stands): {', '.join(failed)}")
    if not succeeded:
        print("Nothing captured -- no changes to commit.")
        raise SystemExit(1)

    print("\nNext steps:")
    print("  git add data/demo_snapshot/")
    print('  git commit -m "Add demo data snapshot"')
    print("  git push")


if __name__ == "__main__":
    main()