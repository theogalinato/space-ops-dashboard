"""
passes.py

Pass-prediction logic for the Space Operations Dashboard.

Pulled out of Day 10's pass_prediction.py script into an importable module
(same reasoning as satellite_data.py on Day 6): app.py needs to call this
for a user-selected satellite and city, not run it as a standalone script.

Site coordinates are the operationally relevant reference points -- roughly
the metro-area city center for each. Elevation is a minor correction to
range/elevation, not load-bearing here.
"""

from skyfield.api import wgs84
from satellite_data import ts  # shared timescale, same object as satellite_data.py

# Minimum elevation to count as a usable pass. See Day 10 note: this is a
# ground-station design choice (antenna mask), not a universal constant --
# flag as a tunable, not a hardcoded truth, if this shows up in the report.
MIN_ELEVATION_DEG = 10.0

SEARCH_HOURS = 48  # how far ahead to search for passes

EVENT_NAMES = {0: "rise", 1: "culminate", 2: "set"}

# (lat_deg, lon_deg, elevation_m). Elevation is rough (nearest ~10m), fine
# for this purpose.
CITIES = {
    "Ottawa":     (45.4215, -75.6972, 70),
    "Halifax":    (44.6488, -63.5752, 10),
    "Edmonton":   (53.5461, -113.4938, 645),
    "Vancouver":  (49.2827, -123.1207, 5),
    "Yellowknife": (62.4540, -114.3718, 190),
}


def get_observer(city: str):
    """Return a Skyfield wgs84 topos object for a city name in CITIES."""
    if city not in CITIES:
        raise KeyError(f"Unknown city '{city}'. Known cities: {list(CITIES)}")
    lat, lon, elev_m = CITIES[city]
    return wgs84.latlon(lat, lon, elevation_m=elev_m)


def get_passes(sat, observer, hours_ahead: int = SEARCH_HOURS,
               min_elevation: float = MIN_ELEVATION_DEG) -> list[dict]:
    """
    Return a list of dicts, one per complete pass, each with rise/set times,
    duration, max elevation, and azimuth at culmination.

    Unchanged from Day 10 other than dropping the module-level `ts` import
    dependency issue (now takes ts from satellite_data directly) and
    accepting any observer, not just Ottawa.
    """
    t0 = ts.now()
    t0_dt = t0.utc_datetime()
    t1 = ts.utc(t0_dt.year, t0_dt.month, t0_dt.day, t0_dt.hour + hours_ahead)

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
            current = {}

    return passes


def get_next_n_passes(sat, city: str, n: int = 5, hours_ahead: int = SEARCH_HOURS,
                       min_elevation: float = MIN_ELEVATION_DEG) -> list[dict]:
    """
    Convenience wrapper for the UI: passes for one satellite over one named
    city, capped to the next N. Does NOT re-search with a longer window if
    fewer than N passes are found in hours_ahead -- it just returns what's
    there. A satellite in a low-inclination orbit relative to a
    high-latitude site (Yellowknife, 62.5N) may legitimately have zero
    passes in 48h; that's real geometry, not a bug, and the caller (app.py)
    should be able to show "no passes in window" rather than the code
    silently expanding the search and hiding that fact.
    """
    observer = get_observer(city)
    all_passes = get_passes(sat, observer, hours_ahead, min_elevation)
    return all_passes[:n]


if __name__ == "__main__":
    # Quick manual check across all five cities without touching Streamlit.
    from satellite_data import get_satellite_by_catnr

    RADARSAT2_CATNR = 32382
    sat = get_satellite_by_catnr(RADARSAT2_CATNR)

    for city in CITIES:
        print(f"\n=== {city} ===")
        city_passes = get_next_n_passes(sat, city, n=3)
        if not city_passes:
            print("  No passes above threshold in window.")
            continue
        for p in city_passes:
            print(f"  {p['rise_utc']} -> {p['set_utc']}  "
                  f"({p['duration_min']} min, max el {p['max_elevation_deg']} deg)")