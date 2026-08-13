"""
satellite_data.py

Shared TLE loading and subpoint/orbital-parameter computation for the
Space Operations Dashboard project.

Consolidates logic that was duplicated across plot_map.py (Day 4) and
ground_track.py (Day 5):
  - fetching/caching TLE groups from CelesTrak
  - looking up a specific satellite by NORAD catalog number
  - computing subpoints (lat/lon/altitude) for many satellites at one instant
  - computing a ground track for one satellite over a time window
  - computing orbital parameters (inclination, period, instantaneous speed)

Every later day (8+, Streamlit) should import from here instead of
re-fetching or re-deriving any of this.
"""

from __future__ import annotations

from datetime import timedelta
from math import pi

import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84

ts = load.timescale()

CELESTRAK_GROUP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
CELESTRAK_CATNR_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=tle"


def load_tle_group(group: str = "visual", reload: bool = False) -> list[EarthSatellite]:
    """
    Load a TLE group from CelesTrak. Cached locally on disk by Skyfield's
    `load.tle_file` as f"{group}.tle".

    Parameters
    ----------
    group : CelesTrak group name, e.g. "visual", "active", "stations".
    reload : force a fresh download even if a local cache file exists.
        Day 4 lesson: an interrupted download can leave a truncated cache
        that loads "successfully" with the wrong satellite count and no
        error. If a count looks off, reload=True first before debugging
        anything else.
    """
    filename = f"{group}.tle"
    url = CELESTRAK_GROUP_URL.format(group=group)
    return load.tle_file(url, filename=filename, reload=reload)


def get_satellite_by_catnr(
    catnr: int,
    group: str = "visual",
    satellites: list[EarthSatellite] | None = None,
    reload: bool = False,
) -> EarthSatellite:
    """
    Look up one satellite by NORAD catalog number.

    Pass in an already-loaded `satellites` list to search it first and
    avoid a second network fetch when possible (e.g. plot_map.py's own
    satellite happens to be a member of the group it already loaded for
    the map). If not found there -- or if `satellites` is None -- falls
    back to a direct CATNR-specific CelesTrak query.

    Note: not every satellite of interest belongs to every named group.
    RADARSAT-2, for example, is NOT in CelesTrak's "visual" group, even
    though it's visually trackable in the "visual" map's context -- that's
    a fact about how CelesTrak curates that particular group, not a bug.
    The CATNR fallback exists specifically for cases like this.
    """
    if satellites is not None:
        by_catnr = {sat.model.satnum: sat for sat in satellites}
        if catnr in by_catnr:
            return by_catnr[catnr]

    # Not found in the provided group (or none was provided) -- fetch
    # this specific satellite directly by its NORAD catalog number.
    url = CELESTRAK_CATNR_URL.format(catnr=catnr)
    filename = f"catnr_{catnr}.tle"
    direct_result = load.tle_file(url, filename=filename, reload=reload)
    if not direct_result:
        raise KeyError(
            f"NORAD catalog number {catnr} not found -- not in group "
            f"'{group}', and the direct CATNR query returned nothing. "
            f"Check the catalog number is correct."
        )
    return direct_result[0]


def compute_subpoints(satellites: list[EarthSatellite], t) -> pd.DataFrame:
    """
    Snapshot: subpoint lat/lon/altitude for MANY satellites at ONE instant t.
    This is the shape plot_map.py needs.

    t must be a scalar Skyfield Time (e.g. ts.now()), not a time array --
    use compute_ground_track for the one-satellite/many-times case instead.

    Returns a DataFrame with columns: name, catnr, latitude_deg,
    longitude_deg, altitude_km -- one row per satellite.
    """
    rows = []
    for sat in satellites:
        geocentric = sat.at(t)
        subpoint = wgs84.subpoint(geocentric)
        rows.append(
            {
                "name": sat.name,
                "catnr": sat.model.satnum,
                "latitude_deg": subpoint.latitude.degrees,
                "longitude_deg": subpoint.longitude.degrees,
                "altitude_km": subpoint.elevation.km,
            }
        )
    return pd.DataFrame(rows)


def compute_ground_track(
    sat: EarthSatellite, t0, minutes: int = 100, step_seconds: int = 60
) -> pd.DataFrame:
    """
    Track: lat/lon/altitude for ONE satellite over a time WINDOW.
    This is the shape ground_track.py needs.

    Builds the time array with ts.from_datetimes() over a list of Python
    datetimes -- NOT ts.utc() with a generator (Day 5 lesson: that raises
    a TypeError). Uses one vectorized sat.at(times) call rather than
    looping per-timestep, which matters once you're doing this for
    multiple satellites on Day 20's real-time view.

    Returns a DataFrame with columns: time_utc, latitude_deg,
    longitude_deg, altitude_km -- one row per timestep.
    """
    n_steps = int((minutes * 60) / step_seconds) + 1
    t0_dt = t0.utc_datetime()
    datetimes = [t0_dt + timedelta(seconds=step_seconds * i) for i in range(n_steps)]
    times = ts.from_datetimes(datetimes)

    geocentric = sat.at(times)
    subpoint = wgs84.subpoint(geocentric)

    return pd.DataFrame(
        {
            "time_utc": datetimes,
            "latitude_deg": subpoint.latitude.degrees,
            "longitude_deg": subpoint.longitude.degrees,
            "altitude_km": subpoint.elevation.km,
        }
    )


def compute_orbital_params(sat: EarthSatellite, t) -> dict:
    """
    Inclination, orbital period, and instantaneous speed for one satellite.

    inclination_deg and period_min come from the TLE's mean elements, so
    they're effectively constant regardless of t (period does drift very
    slowly as mean motion updates between TLE epochs, but not within one
    TLE). speed_km_s is genuinely instantaneous -- computed from the
    velocity vector at t -- so it WILL vary with position (faster at
    perigee for an eccentric orbit) even though period/inclination don't.
    Don't be surprised if speed changes slightly between two calls at
    different times for the same satellite; that's real orbital mechanics,
    not a bug.
    """
    geocentric = sat.at(t)
    vx, vy, vz = geocentric.velocity.km_per_s
    speed_km_s = (vx**2 + vy**2 + vz**2) ** 0.5

    inclination_deg = sat.model.inclo * 180.0 / pi
    mean_motion_rev_per_day = sat.model.no_kozai * 1440.0 / (2 * pi)
    period_min = 1440.0 / mean_motion_rev_per_day

    return {
        "inclination_deg": inclination_deg,
        "period_min": period_min,
        "speed_km_s": speed_km_s,
    }