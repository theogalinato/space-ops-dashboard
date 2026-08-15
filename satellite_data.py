from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from math import pi

import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs
from skyfield.iokit import parse_tle_file

ts = load.timescale()

# --- v1.1: demo build, fixed data snapshot, no live network calls. -------
#
# Earlier attempts (round 1: fetch via `requests` with a real User-Agent
# and timeout instead of Skyfield's own downloader; round 2: fall back to
# a periodically-refreshed local cache when the live fetch fails) both
# ran into the same underlying wall: deployed on two independent hosts
# (Streamlit Community Cloud on GCP, Render on AWS), the live request to
# celestrak.org fails at the TCP connect stage -- a 15s timeout, no
# response -- while the identical request from a home network succeeds
# every time. That's consistent with celestrak.org (a small,
# volunteer/nonprofit-run service) being unreliably reachable from
# cloud/datacenter-originated network paths specifically, not a bug in
# how this app asks for data.
#
# Decision: rather than ship an app that silently degrades over time as
# an unmaintained cache goes stale, this build is explicitly a DEMO --
# it always loads from a single fixed data snapshot captured once (see
# `capture_demo_snapshot.py`) and bundled into the repo under
# `data/demo_snapshot/`. There is no live-fetch attempt anywhere in this
# module, on purpose: identical behavior locally and deployed, zero
# network dependency for TLE data, and an honest, permanent "this is a
# snapshot from [date], not live-tracked" banner in the UI (see app.py)
# instead of a fallback that pretends to be temporary. If celestrak.org
# ever becomes reliably reachable from these hosts, or the project moves
# to a host where it is, re-enabling a live path is straightforward --
# see the git history for the round-1/round-2 versions of this file.
_SNAPSHOT_DIR = "data/demo_snapshot"
_SNAPSHOT_MANIFEST_PATH = os.path.join(_SNAPSHOT_DIR, "manifest.json")


def _read_snapshot_manifest() -> dict:
    if not os.path.exists(_SNAPSHOT_MANIFEST_PATH):
        return {}
    with open(_SNAPSHOT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_snapshot_group(group: str) -> list[EarthSatellite]:
    """
    Load a TLE group from the bundled demo snapshot at
    data/demo_snapshot/{group}.tle. Raises RuntimeError with a clear,
    actionable message if the snapshot is missing -- that's a real setup
    problem (repo cloned without running capture_demo_snapshot.py, or
    data/demo_snapshot/ excluded by .gitignore), not something to fall
    back from silently.
    """
    manifest = _read_snapshot_manifest()
    entry = manifest.get(group)
    snapshot_file = os.path.join(_SNAPSHOT_DIR, f"{group}.tle")

    if entry is None or not os.path.exists(snapshot_file):
        raise RuntimeError(
            f"No demo snapshot found for group '{group}' at {snapshot_file}. "
            f"Run `python capture_demo_snapshot.py` once from a machine "
            f"that can reach CelesTrak, then commit data/demo_snapshot/."
        )

    with open(snapshot_file, "rb") as f:
        satellites = list(parse_tle_file(f, ts))

    if not satellites:
        raise RuntimeError(
            f"{snapshot_file} exists but contains no parseable TLEs. "
            f"Re-run capture_demo_snapshot.py to regenerate it."
        )
    return satellites


def get_snapshot_info(group: str) -> dict | None:
    """
    Metadata about the bundled snapshot for `group`, for app.py's demo
    banner: {"captured_at": ISO string, "age_days": float}. Returns None
    if this group was never captured (manifest has no entry for it).
    """
    manifest = _read_snapshot_manifest()
    entry = manifest.get(group)
    if entry is None:
        return None
    captured_at = datetime.fromisoformat(entry["fetched_at"])
    age_days = (datetime.now(timezone.utc) - captured_at).total_seconds() / 86400.0
    return {"captured_at": entry["fetched_at"], "age_days": age_days}


def load_tle_group(group: str = "visual", reload: bool = False) -> list[EarthSatellite]:
    """
    Load a TLE group. This build always reads from the bundled demo
    snapshot -- see the module comment above for why there's no live
    fetch here. `reload` is accepted for backward compatibility with
    existing call sites but has no effect (there's no live source to
    reload from).
    """
    return _load_snapshot_group(group)


def get_satellite_by_catnr(
    catnr: int,
    group: str = "visual",
    satellites: list[EarthSatellite] | None = None,
    reload: bool = False,
) -> EarthSatellite:
    """
    Look up one satellite by NORAD catalog number.

    Pass in an already-loaded `satellites` list to search it first (e.g.
    plot_map.py's own satellite happens to be a member of the group it
    already loaded for the map). If not found there -- or if `satellites`
    is None -- falls back to the dedicated `catalogue` demo snapshot,
    which capture_demo_snapshot.py builds by fetching every Canadian
    catalogue satellite's CATNR directly, specifically so this lookup
    never depends on whether a given satellite happens to be a member of
    'visual' or 'active' (RADARSAT-2, for example, is not a member of
    CelesTrak's 'visual' group even though it's visually trackable in
    this app's map -- that's a fact about CelesTrak's own group curation,
    not a bug, and the dedicated catalogue snapshot exists to not depend
    on it).

    `reload` is accepted for backward compatibility but has no effect --
    see `load_tle_group`'s docstring for why.
    """
    if satellites is not None:
        by_catnr = {sat.model.satnum: sat for sat in satellites}
        if catnr in by_catnr:
            return by_catnr[catnr]

    catalogue_sats = _load_snapshot_group("catalogue")
    by_catnr = {sat.model.satnum: sat for sat in catalogue_sats}
    if catnr in by_catnr:
        return by_catnr[catnr]

    raise KeyError(
        f"NORAD catalog number {catnr} not found -- not in group "
        f"'{group}', and not in the dedicated catalogue demo snapshot "
        f"either. If this is a newly-added catalogue satellite, re-run "
        f"capture_demo_snapshot.py to pick it up."
    )


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