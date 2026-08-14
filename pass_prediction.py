"""
Day 10 — Passes part 1: rise/culminate/set for RADARSAT-2 over Ottawa.

Topocentric geometry: azimuth/elevation/range relative to a ground observer,
as opposed to the geocentric subpoint work from Days 1-9.

RADARSAT-2 is NOT in the CelesTrak 'visual' group (confirmed Day 8-9), so
get_satellite_by_catnr() must fall back to a direct CATNR query.
"""

from skyfield.api import wgs84
from satellite_data import get_satellite_by_catnr, ts

RADARSAT2_CATNR = 32382

# Minimum elevation to count as a usable "pass". 10 deg is the common default
# for optical/visual passes (below that, atmosphere + horizon obstructions
# usually kill visibility/link quality anyway). For a real ground station,
# this threshold is a design choice tied to antenna mask, not a law of
# physics -- worth a one-line callout in the report.
MIN_ELEVATION_DEG = 10.0

# Ottawa, ON. Elevation is a minor correction to range/elevation calcs,
# not critical here, but wgs84.latlon wants it.
OTTAWA = wgs84.latlon(45.4215, -75.6972, elevation_m=70)

SEARCH_HOURS = 48  # how far ahead to search for passes

EVENT_NAMES = {0: "rise", 1: "culminate", 2: "set"}


def get_passes(sat, ts, observer, hours_ahead=SEARCH_HOURS, min_elevation=MIN_ELEVATION_DEG):
    """
    Return a list of dicts, one per pass, each with rise/culminate/set
    times, duration, and max elevation.

    find_events returns a flat stream of rise/culminate/set events across
    the whole window -- we group them in triples. If the search window
    starts or ends mid-pass, the first or last triple may be incomplete;
    we just drop incomplete groups rather than guessing.
    """
    t0 = ts.now()
    t1 = ts.utc(t0.utc_datetime().year, t0.utc_datetime().month,
                t0.utc_datetime().day, t0.utc_datetime().hour + hours_ahead)

    times, events = sat.find_events(observer, t0, t1, altitude_degrees=min_elevation)

    passes = []
    current = {}
    for ti, event in zip(times, events):
        label = EVENT_NAMES[event]
        current[label] = ti

        if label == "set":
            if "rise" in current and "culminate" in current:
                diff = sat - observer
                alt, az, distance = diff.at(current["culminate"]).altaz()
                duration_s = (current["set"] - current["rise"]) * 86400.0
                passes.append({
                    "rise_utc": current["rise"].utc_strftime("%Y-%m-%d %H:%M:%S"),
                    "set_utc": current["set"].utc_strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_min": round(duration_s / 60.0, 1),
                    "max_elevation_deg": round(alt.degrees, 1),
                    "culminate_azimuth_deg": round(az.degrees, 1),
                })
            current = {}  # reset for next pass, whether or not this one was complete

    return passes


def main():
    """Print RADARSAT-2's next passes over Ottawa to the console. Entry
    point for running this script standalone (`python pass_prediction.py`);
    superseded by `passes.py`'s `get_next_n_passes()`, which this module's
    `get_passes()` duplicates almost exactly -- see the README's "Open
    Items" section for the open question of whether to trim it."""
    sat = get_satellite_by_catnr(RADARSAT2_CATNR)

    print(f"Computing passes for {sat.name} over Ottawa, "
          f"next {SEARCH_HOURS}h, min elevation {MIN_ELEVATION_DEG} deg\n")

    passes = get_passes(sat, ts, OTTAWA)

    if not passes:
        print("No passes above threshold in this window. "
              "(RADARSAT-2 is sun-synchronous LEO -- if this looks wrong, "
              "check TLE epoch age before assuming the code is broken.)")
        return

    for i, p in enumerate(passes, start=1):
        print(f"Pass {i}: rise {p['rise_utc']} UTC  ->  set {p['set_utc']} UTC")
        print(f"   duration: {p['duration_min']} min   "
              f"max elevation: {p['max_elevation_deg']} deg   "
              f"az @ culmination: {p['culminate_azimuth_deg']} deg\n")


if __name__ == "__main__":
    main()