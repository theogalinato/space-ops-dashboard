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
"""

from __future__ import annotations

import pandas as pd
from skyfield.api import EarthSatellite

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