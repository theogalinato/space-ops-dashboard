"""
catalogue.py

Canadian asset catalogue integration for the Space Operations Dashboard.

Two jobs:
  1. Load the curated catalogue CSV (Day 7) for the filterable info table
     -- mission/operator/orbit/purpose/launch date, static curated data,
     no network involved.
  2. Fetch LIVE TLEs for those same satellites so they're selectable and
     trackable on the map/passes tool alongside the 'visual' group --
     this is the "known gap" flagged back on Day 9.

All catalogue satellites are confirmed absent from the 'visual' group
(checked directly, Day 12). Optimization: try the bulk 'active' group
first (one network request) instead of one CATNR fetch per satellite.
get_satellite_by_catnr() already does per-satellite CATNR fallback, so
anything NOT in 'active' still resolves correctly -- it just costs an
extra individual fetch for that one satellite instead of the whole set.
"""

import pandas as pd
from satellite_data import load_tle_group, get_satellite_by_catnr

CATALOGUE_CSV_PATH = "data/canadian_assets.csv"


def load_catalogue() -> pd.DataFrame:
    """Load the curated Canadian asset catalogue CSV (static data, no network)."""
    return pd.read_csv(CATALOGUE_CSV_PATH)


def get_catalogue_satellites(catalogue_df: pd.DataFrame) -> list:
    """
    Fetch live EarthSatellite objects for every satellite in the catalogue.

    Loads the 'active' group once, then checks each catalogue CATNR against
    it before falling back to an individual fetch. Best case (all members
    of 'active'): 1 network request total instead of len(catalogue_df).
    Worst case (none are): same as before -- 1 + len(catalogue_df) requests,
    no worse than the naive approach.
    """
    active_sats = load_tle_group("active")

    catalogue_sats = []
    for catnr in catalogue_df["catnr"]:
        sat = get_satellite_by_catnr(int(catnr), group="active", satellites=active_sats)
        catalogue_sats.append(sat)
    return catalogue_sats


def merge_satellite_lists(primary: list, catalogue: list) -> list:
    """
    Combine the 'visual' group list with the catalogue list, de-duplicated
    by NORAD catalog number.

    Known state as of Day 12: none of the catalogue satellites are in
    'visual', so no de-dup should actually trigger today. Still guarding
    against it rather than assuming -- CelesTrak's group membership isn't
    something this app controls, and silently double-listing a satellite
    in the dropdown would be a confusing bug to chase later if it ever
    changed.
    """
    seen_catnrs = {sat.model.satnum for sat in primary}
    merged = list(primary)
    for sat in catalogue:
        if sat.model.satnum not in seen_catnrs:
            merged.append(sat)
            seen_catnrs.add(sat.model.satnum)
    return merged