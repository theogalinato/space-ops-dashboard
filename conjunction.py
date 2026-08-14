"""
conjunction.py
Day 22: Week 4 opens with reduced conjunction screening -- the project's
honesty constraint on this feature matters from the first line of code, not
just the disclaimer text that goes on the UI later (Day 24): public TLE data
carries no usable covariance, so a real collision-probability (Pc) number is
not something this project can honestly compute. What IS honest and useful is
a SCREENING step -- narrowing a large population down to "these objects are
worth a closer look" -- which is exactly what real conjunction-assessment
pipelines do first too, before the expensive high-precision propagation work.

Today's piece is the first, cheapest stage of that screening: given a PRIMARY
satellite (a Canadian asset worth protecting) and a POPULATION of candidate
objects, throw out everything whose orbit could not possibly bring it near
the primary's altitude, using nothing but each satellite's own TLE mean
elements. No propagation happens in this module at all -- that is deliberately
Day 23's job, once the population here has already been cut down to a much
smaller candidate set worth the cost of propagating.

This filter is a real, standard first-pass technique (often called a
perigee/apogee test or altitude-band filter in conjunction-screening
literature), not an invented simplification -- it just isn't the WHOLE
screening pipeline a real SDA system would run, which would go on to check
orbital plane geometry (RAAN/inclination) and, eventually, propagated
time-of-closest-approach. Being a coarse, necessary-but-not-sufficient filter
is the honest description of what this stage does.

Day 23 is that "eventually": propagate the (now much smaller) survivor set
from bound_population() and compute actual minimum separation over a look-
ahead window, classified into LOW/MODERATE/HIGH. This is where the honesty
constraint on this feature shows up most concretely, not just in a
docstring -- see DISCLAIMER below and the "grid sampling, not root-finding"
limitation explained on screen_conjunctions().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skyfield.api import EarthSatellite

from satellite_data import build_time_array

# Deliberately generous default. This is a coarse PRE-filter, not the final
# screening result -- Day 23 does the real narrowing by propagating survivors
# and computing actual minimum separation. A false negative here (wrongly
# excluding an object that Day 23 would have flagged) is a worse failure than
# a false positive (an object that survives this stage but turns out fine on
# Day 23), because a false negative can never be caught downstream -- the
# object is simply gone from the candidate set. So this margin errs wide on
# purpose rather than trying to be a tight, "precise-looking" number that
# public TLE data (no covariance, no maneuver plans, mean elements only,
# already imprecise by kilometers after a few days -- see Day 20's TLE-age
# warning) can't actually back up.
DEFAULT_MARGIN_KM = 50.0


def compute_altitude_band(sat: EarthSatellite) -> tuple[float, float]:
    """
    Perigee and apogee altitude (km), straight from the TLE's own mean
    orbital elements -- NO propagation, and notably NOT the same computation
    compute_subpoints() does. compute_subpoints() evaluates SGP4 at a
    specific instant t and asks "how high is the satellite RIGHT NOW,"
    which changes continuously around the orbit. This function instead asks
    "what is the full altitude RANGE this orbit ever reaches," which is
    fixed for a given TLE and does not depend on time at all. That's exactly
    the property a population-bounding filter needs: it has to be evaluable
    once per satellite, not once per satellite per instant.

    Uses sat.model.altp / .alta -- perigee/apogee altitude in Earth radii,
    computed directly during SGP4 initialization from the TLE's mean
    eccentricity and semi-major axis, multiplied by sat.model.radiusearthkm
    (6378.135 km, the SGP4/WGS72 reference radius). This is a DIFFERENT
    reference radius than earth_mesh.py's WGS84 equatorial radius
    (6378.137 km) -- a 2 m difference, invisible at the km-scale altitude
    banding this function exists for, but worth naming rather than silently
    mixing two "Earth radius" constants and hoping nobody notices. Confirmed
    self-consistent by cross-checking against compute_subpoints(): a
    satellite's instantaneous altitude at its own TLE epoch falls inside the
    [perigee, apogee] band this function returns, as it must.

    Returns (perigee_altitude_km, apogee_altitude_km).
    """
    model = sat.model
    perigee_km = model.altp * model.radiusearthkm
    apogee_km = model.alta * model.radiusearthkm
    return perigee_km, apogee_km


def bound_population(
    primary: EarthSatellite,
    population: list[EarthSatellite],
    margin_km: float = DEFAULT_MARGIN_KM,
) -> pd.DataFrame:
    """
    Stage 1 of reduced conjunction screening: cut a full tracked population
    down to the candidates whose orbit's altitude RANGE overlaps the
    primary's altitude range (padded by margin_km on each side).

    Deliberately a RANGE-overlap test, not a comparison of single altitude
    numbers (e.g. "primary's current altitude" vs "candidate's current
    altitude"). The reason is physical, not stylistic: an eccentric orbit's
    instantaneous altitude sweeps its entire [perigee, apogee] band over one
    orbit, so an object whose average or "current" altitude looks nothing
    like the primary's can still pass directly through the primary's
    altitude shell at some point in its orbit -- and that crossing is
    exactly the kind of geometry this screening stage exists to catch. A
    single-number comparison would silently miss that object. See
    test_conjunction.py's CAND-ECC-600-800 fixture, which does exactly this:
    its perigee/apogee straddle the primary's band even though neither of
    its own perigee or apogee equals the primary's altitude.

    The primary itself is excluded from its own candidate population by
    NORAD catalog number (a satellite cannot conjunct with itself).

    Returns a DataFrame (name, catnr, perigee_km, apogee_km) for every
    surviving candidate, sorted by perigee_km ascending. Empty (not an
    error) if nothing survives -- a legitimate result, not a failure, the
    same way Day 13 treated an empty pass list as a fact about geometry
    rather than a bug to work around.
    """
    primary_perigee_km, primary_apogee_km = compute_altitude_band(primary)
    band_lo_km = primary_perigee_km - margin_km
    band_hi_km = primary_apogee_km + margin_km
    primary_catnr = primary.model.satnum

    rows = []
    for sat in population:
        if sat.model.satnum == primary_catnr:
            continue

        perigee_km, apogee_km = compute_altitude_band(sat)

        # Two ranges [band_lo, band_hi] and [perigee_km, apogee_km] overlap
        # iff each one's low end is <= the other's high end. This is the
        # standard interval-overlap test -- it also correctly handles a
        # candidate's band being entirely CONTAINED within the primary's
        # padded band, or vice versa, not just partial overlap at one edge.
        if apogee_km >= band_lo_km and perigee_km <= band_hi_km:
            rows.append(
                {
                    "name": sat.name,
                    "catnr": sat.model.satnum,
                    "perigee_km": perigee_km,
                    "apogee_km": apogee_km,
                }
            )

    df = pd.DataFrame(rows, columns=["name", "catnr", "perigee_km", "apogee_km"])
    return df.sort_values("perigee_km").reset_index(drop=True)


# ============================================================
# Day 23: stage 2 -- propagate survivors, compute minimum separation,
# classify. Everything below DOES propagate (unlike Day 22's altitude-band
# filter above), which is exactly why it only ever runs against the small
# survivor set bound_population() already narrowed things down to.
# ============================================================

# Look-ahead window and sampling step for the minimum-separation search.
# 24 hours, not the multi-day windows a real SDA system screens (typically
# ~3-7 days): this project's own TLE-age warning (Day 20, app.py) already
# flags a TLE as degraded past 3 days old, so screening much further ahead
# than 24h would mean feeding an already-honest-about-being-imprecise input
# into an even less trustworthy output. 30 s matches compute_orbit_arc's
# (Day 19) existing default step, for no reason beyond consistency -- there
# is no physical argument that 30 s is uniquely correct, see the grid-
# sampling limitation below.
DEFAULT_WINDOW_HOURS = 24.0
DEFAULT_STEP_SECONDS = 30

# Distance-only risk bands. Deliberately NOT calibrated against any
# published probability-of-collision standard -- public TLE data supplies
# no covariance, so no such calibration is honestly possible here (see
# DISCLAIMER). These thresholds are a coarse, order-of-magnitude proxy for
# "worth a closer look," loosely in the neighborhood of the few-km screening
# volumes real conjunction-assessment pipelines use to decide whether to
# generate a conjunction data message in the first place -- NOT the tighter,
# covariance-based thresholds those pipelines then use to decide whether an
# operator should maneuver.
HIGH_RISK_MAX_KM = 1.0
MODERATE_RISK_MAX_KM = 5.0


def classify_conjunction_risk(min_separation_km: float) -> str:
    """
    LOW / MODERATE / HIGH from a predicted minimum separation distance
    alone. See the module-level threshold comments above and DISCLAIMER
    below for why these are a coarse proxy, not a calibrated probability
    band -- worth re-reading before wiring this into the UI on Day 24,
    since a label like "HIGH" reads as far more authoritative than a plain
    distance number does, and that gap is exactly what DISCLAIMER exists to
    close.

    Boundaries are inclusive on the safer side: exactly 1.0 km is MODERATE,
    not HIGH; exactly 5.0 km is LOW, not MODERATE. Arbitrary but consistent
    with how classify_orbit_regime (Day 19) and classify_geomagnetic_status
    (Day 16) both already draw their own inclusive-on-the-lower-band-edge
    boundaries.
    """
    if min_separation_km < HIGH_RISK_MAX_KM:
        return "HIGH"
    elif min_separation_km < MODERATE_RISK_MAX_KM:
        return "MODERATE"
    else:
        return "LOW"


def compute_min_separation(primary: EarthSatellite, candidate: EarthSatellite, times):
    """
    Minimum Euclidean separation between primary and candidate across a
    shared array of instants `times` (typically built by
    satellite_data.build_time_array), plus the specific instant it occurs
    at.

    Uses sat.at(times).position.km -- Skyfield's native GCRS/ECI frame --
    NOT the ECEF conversion Day 18's compute_ecef_positions() uses for the
    3D globe. That's a deliberate difference worth naming, not an
    inconsistency: the globe needs ECEF because it plots satellites against
    a fixed-continent Earth model that would otherwise drift. This function
    only ever computes the DIFFERENCE between two positions at the SAME
    instant, and a rigid rotation (which is all ECEF-vs-ECI amounts to at
    one instant) preserves distances between points -- so the separation
    comes out identical in either frame, and GCRS is simply the one
    Skyfield hands back with no extra conversion step.

    THE central limitation of this whole screening feature lives here, not
    in the classification thresholds: this is a GRID search, evaluating
    separation only at the sampled instants in `times` and taking the
    smallest, NOT a true continuous time-of-closest-approach found by
    root-finding on relative range-rate (that upgrade is explicitly out of
    scope for this project's core 4 weeks -- see the README's Week 5
    extension notes). A grid search can only ever report a separation
    GREATER THAN OR EQUAL TO the true minimum -- taking the smallest of a
    finite sample can't do better than the true continuous minimum, only
    miss it -- so this method can UNDERSTATE risk (miss or soften a real
    close approach that happened between two samples) but can never
    OVERSTATE it from grid coarseness alone. At a worst-case LEO relative
    velocity around 15 km/s (near-opposite orbital planes), the default
    30 s step means two objects could be up to ~450 km apart in-track
    between consecutive samples -- a real, honestly-stated blind spot, not
    a hypothetical one. test_conjunction.py proves the DIRECTION of this
    effect (a finer grid's minimum is never larger than a coarser grid's,
    for the same window) even though this project's test fixtures don't
    happen to manufacture a dramatic real-world gap.

    Returns (min_separation_km, time_of_min), where time_of_min is a
    Python datetime (UTC).
    """
    primary_km = primary.at(times).position.km  # shape (3, N)
    candidate_km = candidate.at(times).position.km  # shape (3, N)
    separations_km = np.linalg.norm(primary_km - candidate_km, axis=0)
    min_idx = int(np.argmin(separations_km))
    return float(separations_km[min_idx]), times[min_idx].utc_datetime()


def screen_conjunctions(
    primary: EarthSatellite,
    candidates: list[EarthSatellite],
    t0,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
) -> pd.DataFrame:
    """
    Stage 2 of reduced conjunction screening: propagate the primary and
    every candidate over a shared time grid starting at t0, and report each
    candidate's minimum separation, when it occurs, and a LOW/MODERATE/HIGH
    classification.

    Intended input is bound_population()'s survivor set (candidate
    EarthSatellite objects matching the catnrs it returned), not the full
    tracked population -- this function propagates every candidate over
    potentially thousands of timesteps, which is exactly the cost Day 22's
    cheap altitude-band filter exists to avoid paying for hundreds of
    satellites that could never plausibly conjunct with the primary in the
    first place. Nothing here enforces that composition -- see
    test_conjunction.py's pipeline test for a worked example of the
    intended Day 22 -> Day 23 handoff.

    Propagates the primary ONCE and reuses it for every candidate, rather
    than calling compute_min_separation() once per candidate (which would
    silently re-propagate the primary N times over). That duplication is a
    deliberate efficiency choice worth naming, not a missed opportunity to
    reuse code: compute_min_separation() stays the simple, independently
    testable single-pair primitive; this function is the batched version
    that shares the one expensive-ish computation (primary propagation)
    across every candidate.

    The primary is excluded from its own results by NORAD catalog number,
    same defensive pattern as bound_population() -- a satellite can't
    conjunct with itself, and this way a caller who accidentally includes
    the primary in `candidates` still gets a correct answer instead of a
    silent zero-separation row.

    Returns a DataFrame (name, catnr, min_separation_km,
    time_of_closest_approach_utc, risk_level), sorted by min_separation_km
    ascending -- the most concerning candidate first. Empty (not an error)
    if `candidates` is empty, same "empty is a legitimate result" principle
    as bound_population() and, before that, Day 13's pass tables.
    """
    times, _ = build_time_array(t0, window_hours * 60, step_seconds)
    primary_km = primary.at(times).position.km  # shape (3, N), propagated ONCE
    primary_catnr = primary.model.satnum

    rows = []
    for sat in candidates:
        if sat.model.satnum == primary_catnr:
            continue

        candidate_km = sat.at(times).position.km
        separations_km = np.linalg.norm(primary_km - candidate_km, axis=0)
        min_idx = int(np.argmin(separations_km))
        min_separation_km = float(separations_km[min_idx])

        rows.append(
            {
                "name": sat.name,
                "catnr": sat.model.satnum,
                "min_separation_km": min_separation_km,
                "time_of_closest_approach_utc": times[min_idx].utc_datetime(),
                "risk_level": classify_conjunction_risk(min_separation_km),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "name",
            "catnr",
            "min_separation_km",
            "time_of_closest_approach_utc",
            "risk_level",
        ],
    )
    return df.sort_values("min_separation_km").reset_index(drop=True)


DISCLAIMER = (
    "Simplified public-data conjunction SCREENING, not a certified "
    "collision-probability (Pc) assessment. Public TLEs carry no position/"
    "velocity covariance, so a real Pc cannot be honestly computed here -- "
    "these LOW/MODERATE/HIGH labels reflect predicted minimum separation "
    "over a grid-sampled look-ahead window only, not a probability of "
    "collision. Grid sampling can miss or understate a fast, brief close "
    "approach that falls between two samples; it can never invent one that "
    "isn't there. For actual spacecraft operations, consult 18th Space "
    "Defense Squadron conjunction data messages (space-track.org) or your "
    "operator's own flight dynamics team."
)