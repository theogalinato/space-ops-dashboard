from __future__ import annotations

from datetime import timedelta
from math import pi

import pandas as pd
import requests
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs
from skyfield.iokit import parse_tle_file

ts = load.timescale()

CELESTRAK_GROUP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
CELESTRAK_CATNR_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=tle"

# --- Deploy fix (post-v1.0): fetch TLEs with `requests` instead of
# Skyfield's built-in `load.tle_file()` downloader. -----------------------
#
# On Streamlit Community Cloud, `get_satellite_pool()` crashed at startup
# with a bare, Streamlit-redacted OSError inside Skyfield's `download()`.
# Tracing it (Skyfield 1.55's `skyfield/iokit.py`) showed the failure was
# specifically the `urlopen(...)` call to CelesTrak failing and getting
# wrapped as `IOError('cannot download {url} because {e}')` -- a NETWORK
# failure, not (as first suspected) a read-only-working-directory problem.
# Streamlit deliberately redacts OSError/IOError text in the deployed UI
# (they can leak local filesystem details), which is exactly why the app
# showed a generic "error redacted" banner instead of the real reason --
# the real message only ever reached the Cloud "Manage app" logs.
#
# Two independent things about Skyfield's built-in downloader are worth
# fixing regardless of which one actually caused this specific crash:
#   1. It sends Python's generic default User-Agent header on every
#      request. Some public data hosts (CelesTrak included, at times)
#      quietly reject or rate-limit generic scripted User-Agents.
#   2. It sets no request timeout at all -- a slow/unresponsive CelesTrak
#      response would hang the whole Streamlit worker rather than fail.
#
# It also caches every download to a file written next to the app's own
# code, which assumes a writable working directory -- not guaranteed on
# every hosting platform, and not needed here anyway: `get_satellite_pool()`
# in app.py is already `@st.cache_resource(ttl=TLE_TTL_SECONDS)`, so
# per-rerun freshness is already handled at the Streamlit layer. Fetching
# with `requests` (already a project dependency) and handing the raw bytes
# to Skyfield's own `parse_tle_file()` keeps the exact same EarthSatellite
# output, adds a real User-Agent and timeout, and never touches disk.
_REQUEST_HEADERS = {
    "User-Agent": (
        "space-ops-dashboard/1.0 (educational SDA project; "
        "https://github.com/theogalinato/space-ops-dashboard)"
    )
}
_REQUEST_TIMEOUT_SECONDS = 15


def _fetch_tle_lines(url: str) -> list[EarthSatellite]:
    """
    Fetch a CelesTrak TLE URL over HTTP and parse it into EarthSatellite
    objects, bypassing Skyfield's own disk-caching downloader entirely.

    Raises a plain RuntimeError (never an OSError/IOError) on failure.
    That's deliberate: Streamlit Cloud redacts OSError/IOError details
    from the deployed app's UI by design, which is exactly why a network
    hiccup here used to surface as an opaque "error redacted" banner
    instead of a real message. RuntimeError isn't special-cased, so the
    actual reason now shows up directly in the app, not just in logs.
    """
    try:
        response = requests.get(
            url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not fetch TLE data from CelesTrak ({url}): {exc}. "
            f"CelesTrak may be temporarily unavailable -- try again in a "
            f"moment; if it persists, check https://celestrak.org directly."
        ) from exc

    satellites = list(parse_tle_file(response.iter_lines(), ts))
    if not satellites:
        raise RuntimeError(
            f"CelesTrak returned no parseable TLE data for {url}. "
            f"The group name or catalog number may be wrong, or "
            f"CelesTrak's response format may have changed."
        )
    return satellites


def load_tle_group(group: str = "visual", reload: bool = False) -> list[EarthSatellite]:
    """
    Load a TLE group from CelesTrak.

    Parameters
    ----------
    group : CelesTrak group name, e.g. "visual", "active", "stations".
    reload : accepted for backward compatibility with existing call sites,
        but no longer changes behavior. It used to force past Skyfield's
        on-disk TLE cache (the original Day 4 lesson lived here: an
        interrupted download could leave a truncated cache file that
        loaded "successfully" with the wrong satellite count and no
        error). That on-disk cache is gone -- see the module-level
        comment above `_fetch_tle_lines` -- so every call already fetches
        fresh from CelesTrak; there's nothing left for `reload` to force.
    """
    url = CELESTRAK_GROUP_URL.format(group=group)
    return _fetch_tle_lines(url)


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

    `reload` is accepted for backward compatibility but no longer changes
    behavior -- see `load_tle_group`'s docstring for why.
    """
    if satellites is not None:
        by_catnr = {sat.model.satnum: sat for sat in satellites}
        if catnr in by_catnr:
            return by_catnr[catnr]

    # Not found in the provided group (or none was provided) -- fetch
    # this specific satellite directly by its NORAD catalog number.
    url = CELESTRAK_CATNR_URL.format(catnr=catnr)
    direct_result = _fetch_tle_lines(url)
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


def build_time_array(t0, minutes: float, step_seconds: int):
    """
    Shared by compute_ground_track, Day 19's compute_orbit_arc, and Day 23's
    conjunction.py screening functions: build a Skyfield time array spanning
    `minutes` starting at t0, one point every step_seconds.

    Builds it with ts.from_datetimes() over a list of Python datetimes --
    NOT ts.utc() with a generator (Day 5 lesson: that raises a TypeError).
    Pulled out on Day 19 so compute_ground_track and compute_orbit_arc could
    share one implementation instead of the same loop (and the same Day-5
    lesson comment) living in two places. Renamed on Day 23, dropping the
    leading underscore, when conjunction.py became a second MODULE that
    needed it, not just a second function in this one -- a leading
    underscore signals "private to this file," and importing one across a
    module boundary anyway is worse than admitting this helper is now a
    shared utility and naming it like one.
    """
    n_steps = int((minutes * 60) / step_seconds) + 1
    t0_dt = t0.utc_datetime()
    datetimes = [t0_dt + timedelta(seconds=step_seconds * i) for i in range(n_steps)]
    return ts.from_datetimes(datetimes), datetimes


def compute_ground_track(
    sat: EarthSatellite, t0, minutes: int = 100, step_seconds: int = 60
) -> pd.DataFrame:
    """
    Track: lat/lon/altitude for ONE satellite over a time WINDOW.
    This is the shape ground_track.py needs.

    Uses one vectorized sat.at(times) call rather than looping per-timestep,
    which matters once you're doing this for multiple satellites on Day
    20's real-time view.

    Returns a DataFrame with columns: time_utc, latitude_deg,
    longitude_deg, altitude_km -- one row per timestep.
    """
    times, datetimes = build_time_array(t0, minutes, step_seconds)

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


def compute_ecef_positions(satellites: list[EarthSatellite], t) -> pd.DataFrame:
    """
    Snapshot: Earth-Centered Earth-Fixed (ECEF) x/y/z position in km for
    MANY satellites at ONE instant t. Day 18's ECI-to-ECEF conversion.

    sat.at(t) (used everywhere else in this module) returns a geocentric
    position in Skyfield's GCRS frame -- an INERTIAL frame (fixed relative
    to the stars, not the ground) sometimes loosely called ECI. That's the
    correct frame for orbital mechanics, but it is the WRONG frame to plot
    next to a solid Earth model whose continents sit at fixed lat/lon:
    GCRS doesn't rotate with the planet, so raw GCRS x/y/z plotted against
    a fixed-continent sphere would show the continents visibly drifting
    under the satellites as the Earth turns beneath the inertial frame.

    geocentric.frame_xyz(itrs) reprojects the same physical position into
    ITRS (International Terrestrial Reference System) -- the Earth-fixed
    frame that rotates with the planet, i.e. ECEF. This is exactly the
    frame wgs84.subpoint() already uses internally to derive lat/lon/
    altitude elsewhere in this module; a satellite's ITRS x/y/z and its
    wgs84 subpoint describe the same point two different ways (Cartesian
    vs. geodetic). test_globe.py round-trips through both and checks they
    agree, including accounting for the small, expected geodetic-vs-
    geocentric latitude gap that WGS84's ellipsoid (not a sphere) causes.

    Returns a DataFrame with columns: name, catnr, x_km, y_km, z_km.
    """
    rows = []
    for sat in satellites:
        geocentric = sat.at(t)
        x, y, z = geocentric.frame_xyz(itrs).km
        rows.append(
            {
                "name": sat.name,
                "catnr": sat.model.satnum,
                "x_km": x,
                "y_km": y,
                "z_km": z,
            }
        )
    return pd.DataFrame(rows)


def compute_orbit_arc(
    sat: EarthSatellite, t0, minutes: float | None = None, step_seconds: int = 30
) -> pd.DataFrame:
    """
    Day 19: ECEF x/y/z for ONE satellite traced over one orbital period --
    the "orbit arc" for the 3D globe. The 3D-space counterpart to
    compute_ground_track: same time-window idea (reuses
    build_time_array), but returns the satellite's own ECEF path through
    space rather than its ground-projected subpoint.

    minutes defaults to None, meaning "use this satellite's own orbital
    period" (via compute_orbital_params) rather than a fixed window --
    unlike compute_ground_track's fixed 100-minute default, one arc should
    mean one full lap, and LEO (~90-100 min) and GEO (~1436 min) laps are
    very different lengths. Pass an explicit minutes to override.

    IMPORTANT frame note, easy to get backwards: this arc is computed in
    ECEF (Earth-fixed), the SAME frame compute_ecef_positions() uses, on
    purpose -- so it's consistent with the satellite dot and Earth mesh
    already drawn in that frame in globe.py. That has a real, visible
    consequence: in an INERTIAL frame, one orbital period traces a closed
    ellipse (ignoring precession). In ECEF, it does NOT close, because the
    Earth keeps rotating underneath the satellite for the whole period --
    the same underlying effect that makes ground tracks drift westward
    orbit over orbit (Day 5), just applied to the satellite's actual 3D
    position instead of its ground-projected subpoint. For a LEO satellite
    (~90 min, Earth rotates ~22.5 deg during one lap) this is a visibly
    open spiral, not a closed loop -- that's correct, not a bug. For a GEO
    satellite (period matches Earth's rotation almost exactly, by design)
    the arc stays close to a single point instead of sweeping a big loop --
    which is the same physical fact Day 13 already found the hard way
    (GEO doesn't have discrete passes because it's roughly fixed relative
    to the ground), just visible here as geometry instead of an empty pass
    table. test_globe.py checks both shapes explicitly.

    Returns a DataFrame with columns: time_utc, x_km, y_km, z_km.
    """
    if minutes is None:
        minutes = compute_orbital_params(sat, t0)["period_min"]

    times, datetimes = build_time_array(t0, minutes, step_seconds)

    geocentric = sat.at(times)
    x, y, z = geocentric.frame_xyz(itrs).km

    return pd.DataFrame(
        {
            "time_utc": datetimes,
            "x_km": x,
            "y_km": y,
            "z_km": z,
        }
    )


def classify_orbit_regime(altitude_km: float) -> str:
    """
    LEO / MEO / GEO classification by altitude. Moved here on Day 19 from
    a copy that lived only in app.py -- globe.py now needs the exact same
    classification (to color-code satellites by regime, since LEO and GEO
    sit at wildly different scales on one 3D plot), and duplicating the
    thresholds in two places risked them silently drifting apart. app.py
    now imports this instead of defining its own.
    """
    if altitude_km < 2000:
        return "LEO"
    elif altitude_km < 35000:
        return "MEO"
    else:
        return "GEO"


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