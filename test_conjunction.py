"""
Offline sanity test for Day 22's conjunction.py -- compute_altitude_band()
and bound_population(). Same hardcoded-TLE approach as every other test_*.py
in this project, since this sandbox can't reach celestrak.org. No propagation
happens in conjunction.py yet (that's Day 23), so unlike test_globe.py this
test doesn't need to worry about epoch-vs-now propagation error at all --
altp/alta come straight from each TLE's mean elements, evaluated once,
independent of time.

Six synthetic fixtures, all clearly labeled "(TEST FIXTURE)" and none
resembling a real NORAD catalog number, generated to hit specific known
perigee/apogee values (see the comment above each block for the math). SGP4's
Kozai-to-Brouwer mean-motion conversion shifts the actual perigee/apogee a
few km from the naive Keplerian target used to generate the mean motion --
expected physics (J2 secular correction), not an error -- so the values
asserted below are the REAL sat.model.altp/.alta output, confirmed once by
hand and hardcoded here, not the original targets.
"""

import pandas as pd
from skyfield.api import EarthSatellite, load

from conjunction import compute_altitude_band, bound_population, DEFAULT_MARGIN_KM

ts = load.timescale()

# --- Real ISS fixture, same as every other test_*.py in this project ------
iss_line1 = "1 25544U 98067A   24079.51782528  .00016717  00000-0  30187-3 0  9994"
iss_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49560343440557"
iss = EarthSatellite(iss_line1, iss_line2, "ISS (ZARYA)", ts)

# --- Same synthetic GEO fixture as test_globe.py, redefined locally per
# this project's per-file fixture convention -----------------------------
geo_line1 = "1 99999U 98067A   24079.50000000  .00016717  00000-0  30187-3 0  9996"
geo_line2 = "2 99999  0.0500  180.0000 0002000  90.0000   0.0000  1.00270000 10003"
geo = EarthSatellite(geo_line1, geo_line2, "SYNTHETIC-GEO (TEST FIXTURE)", ts)

# --- New Day 22 fixtures: six circular/eccentric orbits at known altitude
# bands, generated from Kepler's third law (mu=398600.4418, Re=6378.135 km,
# matching sat.model.radiusearthkm) then hand-verified against the actual
# sat.model.altp/.alta SGP4 returns -- see module docstring above for why
# those differ slightly (by a few km) from the naive Keplerian target.
#
#   fixture              target perigee/apogee   ACTUAL (asserted below)
#   PRIMARY-700-CIRC      700 / 700 km            697.10 / 697.10 km
#   CAND-SAME-700         700 / 700 km            697.10 / 697.10 km  (identical orbit to primary)
#   CAND-IN-660-CIRC      660 / 660 km            657.08 / 657.08 km  (inside primary's +-50km band)
#   CAND-OUT-645-CIRC     645 / 645 km            642.08 / 642.08 km  (just OUTSIDE the band)
#   CAND-FAR-900-CIRC     900 / 900 km            897.18 / 897.18 km  (far outside)
#   CAND-ECC-600-800      600 / 800 km            597.14 / 797.06 km  (eccentric, straddles the band)
primary = EarthSatellite(
    "1 99001U 98067A   24079.51782528  .00000000  00000-0  00000-0 0  9999",
    "2 99001  98.6000 100.0000 0000000   0.0000   0.0000 14.57889136    18",
    "PRIMARY-700-CIRC (TEST FIXTURE)", ts,
)
cand_same = EarthSatellite(
    "1 99002U 98067A   24079.51782528  .00000000  00000-0  00000-0 0  9990",
    "2 99002  98.6000 100.0000 0000000   0.0000   0.0000 14.57889136    19",
    "CAND-SAME-700 (TEST FIXTURE)", ts,
)
cand_in = EarthSatellite(
    "1 99003U 98067A   24079.51782528  .00000000  00000-0  00000-0 0  9991",
    "2 99003  98.6000 100.0000 0000000   0.0000   0.0000 14.70335262    11",
    "CAND-IN-660-CIRC (TEST FIXTURE)", ts,
)
cand_out = EarthSatellite(
    "1 99004U 98067A   24079.51782528  .00000000  00000-0  00000-0 0  9992",
    "2 99004  98.6000 100.0000 0000000   0.0000   0.0000 14.75048285    13",
    "CAND-OUT-645-CIRC (TEST FIXTURE)", ts,
)
cand_far = EarthSatellite(
    "1 99005U 98067A   24079.51782528  .00000000  00000-0  00000-0 0  9993",
    "2 99005  98.6000 100.0000 0000000   0.0000   0.0000 13.98210637    10",
    "CAND-FAR-900-CIRC (TEST FIXTURE)", ts,
)
cand_ecc = EarthSatellite(
    "1 99006U 98067A   24079.51782528  .00000000  00000-0  00000-0 0  9994",
    "2 99006  98.6000 100.0000 0141280   0.0000   0.0000 14.57889136    19",
    "CAND-ECC-600-800 (TEST FIXTURE)", ts,
)

print("=== compute_altitude_band: cross-check against a known-good source ===")
# ISS's TLE-derived band should bracket the SAME altitude compute_subpoints()
# (Day 3, trusted since) reports at that TLE's own epoch -- two independent
# computations describing the same physical orbit, from two different
# functions in this codebase, should agree.
from satellite_data import compute_subpoints
iss_perigee, iss_apogee = compute_altitude_band(iss)
iss_epoch_altitude = compute_subpoints([iss], iss.epoch)["altitude_km"].iloc[0]
print(f"  ISS band: [{iss_perigee:.2f}, {iss_apogee:.2f}] km, "
      f"compute_subpoints() at epoch: {iss_epoch_altitude:.2f} km")
assert iss_perigee <= iss_epoch_altitude <= iss_apogee, (
    "compute_subpoints()'s epoch altitude should fall inside conjunction.py's "
    "own perigee/apogee band for the identical satellite and instant"
)
print("  PASS: independently-computed instantaneous altitude falls inside the TLE-derived band\n")

print("=== compute_altitude_band: new Day 22 fixtures match their known values ===")
for sat, expected_perigee, expected_apogee in [
    (primary, 697.10, 697.10),
    (cand_same, 697.10, 697.10),
    (cand_in, 657.08, 657.08),
    (cand_out, 642.08, 642.08),
    (cand_far, 897.18, 897.18),
    (cand_ecc, 597.14, 797.06),
]:
    perigee_km, apogee_km = compute_altitude_band(sat)
    assert abs(perigee_km - expected_perigee) < 0.05, f"{sat.name}: perigee {perigee_km} != {expected_perigee}"
    assert abs(apogee_km - expected_apogee) < 0.05, f"{sat.name}: apogee {apogee_km} != {expected_apogee}"
print(f"  PASS: all {6} fixtures land within 0.05 km of their pre-verified perigee/apogee\n")

print("=== bound_population: default margin (50 km) against the Day 22 fixture set ===")
population = [primary, cand_same, cand_in, cand_out, cand_far, cand_ecc]
survivors = bound_population(primary, population, margin_km=DEFAULT_MARGIN_KM)
survivor_names = set(survivors["name"])

assert primary.name not in survivor_names, "primary must never appear in its own survivor list"
assert cand_same.name in survivor_names, "identical orbit to primary should survive"
assert cand_in.name in survivor_names, "orbit inside the padded band should survive"
assert cand_ecc.name in survivor_names, (
    "eccentric orbit straddling the primary's band should survive -- this is "
    "the range-overlap test, not a point comparison; see conjunction.py docstring"
)
assert cand_out.name not in survivor_names, "orbit just outside the padded band should be excluded"
assert cand_far.name not in survivor_names, "orbit far outside the band should be excluded"
assert len(survivors) == 3, f"expected exactly 3 survivors, got {len(survivors)}: {list(survivor_names)}"
print(f"  PASS: {len(survivors)} survivors ({sorted(survivor_names)}), "
      f"correctly including the eccentric crosser and excluding both out-of-band circular orbits\n")

print("=== bound_population: margin_km changes the outcome, as designed ===")
# With NO margin, CAND-IN-660 (band [657.08, 657.08]) no longer overlaps the
# primary's un-padded band [697.10, 697.10] -- demonstrates the margin is
# doing real work, not a cosmetic parameter.
survivors_no_margin = bound_population(primary, population, margin_km=0.0)
assert cand_in.name not in set(survivors_no_margin["name"]), (
    "with zero margin, CAND-IN-660 should no longer overlap the primary's exact band"
)
assert cand_same.name in set(survivors_no_margin["name"]), (
    "an EXACTLY identical orbit should always survive regardless of margin"
)
print(f"  PASS: margin_km=0 drops CAND-IN-660 ({len(survivors_no_margin)} survivors vs "
      f"{len(survivors)} at the default {DEFAULT_MARGIN_KM} km margin)\n")

print("=== bound_population: cross-regime sanity check (real fixtures, not synthetic) ===")
# A LEO primary (ISS, ~420 km) screened against a population that includes a
# GEO object (~35,786 km) should exclude the GEO object outright -- the
# ~35,000 km gap dwarfs any reasonable margin. This is the same LEO-vs-GEO
# scale fact Day 19's globe surfaced visually, now showing up as a filter
# correctly excluding rather than a camera struggling to frame both.
mixed_population = [iss, geo, cand_in]  # cand_in is LEO-ish, should still survive against ISS
iss_survivors = bound_population(iss, mixed_population)
assert geo.name not in set(iss_survivors["name"]), "GEO object must not survive an LEO primary's screening"
print(f"  PASS: SYNTHETIC-GEO correctly excluded when screening against ISS as primary "
      f"(band gap ~{compute_altitude_band(geo)[0] - iss_apogee:,.0f} km, dwarfing the "
      f"{DEFAULT_MARGIN_KM:.0f} km default margin)\n")

print("=== bound_population: empty result is a valid outcome, not a crash ===")
lonely_primary_population = [geo]  # nothing else in range of a GEO-altitude primary
empty_survivors = bound_population(geo, lonely_primary_population)
assert isinstance(empty_survivors, pd.DataFrame)
assert list(empty_survivors.columns) == ["name", "catnr", "perigee_km", "apogee_km"]
assert len(empty_survivors) == 0
print("  PASS: an empty candidate population returns an empty, correctly-shaped DataFrame\n")

print("ALL CHECKS PASSED")