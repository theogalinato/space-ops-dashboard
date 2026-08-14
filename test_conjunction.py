"""
Offline sanity test for conjunction.py, covering both days' work.

Day 22 section: compute_altitude_band() and bound_population() -- no
propagation involved at all, so unlike test_globe.py this doesn't need to
worry about epoch-vs-now propagation error. altp/alta come straight from
each TLE's mean elements, evaluated once, independent of time.

Day 23 section: compute_min_separation(), classify_conjunction_risk(), and
screen_conjunctions() -- these DO propagate, over a hardcoded time window
starting at each fixture's own TLE epoch (never "now"), same reasoning
test_globe.py already established for why epoch, not now: SGP4 accuracy
degrades the further you propagate past epoch, so pinning to epoch keeps
this test's numbers exact and reproducible regardless of when it's run.

Same hardcoded/synthetic-TLE approach as every other test_*.py in this
project, since this sandbox can't reach celestrak.org.

Six Day 22 fixtures, all clearly labeled "(TEST FIXTURE)" and none
resembling a real NORAD catalog number, generated to hit specific known
perigee/apogee values (see the comment above each block for the math). SGP4's
Kozai-to-Brouwer mean-motion conversion shifts the actual perigee/apogee a
few km from the naive Keplerian target used to generate the mean motion --
expected physics (J2 secular correction), not an error -- so the values
asserted below are the REAL sat.model.altp/.alta output, confirmed once by
hand and hardcoded here, not the original targets.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from skyfield.api import EarthSatellite, load

from conjunction import (
    compute_altitude_band,
    bound_population,
    compute_min_separation,
    classify_conjunction_risk,
    screen_conjunctions,
    DEFAULT_MARGIN_KM,
    HIGH_RISK_MAX_KM,
    MODERATE_RISK_MAX_KM,
)
from satellite_data import build_time_array

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


# ============================================================
# Day 23 fixtures: PRIMARY-700-CIRC (reused exactly as above) plus three new
# "phase-offset" satellites -- IDENTICAL orbit to the primary (same a, e, i,
# RAAN, argument of perigee) but shifted by a small mean anomaly. Two
# satellites on the same physical orbit, just offset in where they sit along
# it, stay NEAR-constant chord distance apart at all times (approximately
# 2r*sin(dM/2), r the orbital radius, dM the mean-anomaly offset in radians).
#
# "Near", not exactly, constant turned out to matter and is worth recording
# rather than glossing over: a first pass at this test compared the grid-
# search minimum against the separation measured at a single instant (TLE
# epoch) and found a real ~0.02 km gap, not floating-point noise. The cause:
# the mean-anomaly offset keeps the two satellites' MEAN position constantly
# offset by construction, but SGP4's short-period J2 correction terms depend
# on where each satellite actually is along its orbit at a given moment --
# and since the two are always at DIFFERENT points along an identical orbit
# shape, those short-period corrections don't cancel between them the way
# the mean-motion terms do. The result is a small, genuine periodic wobble
# in true separation (confirmed stable across step=5s/60s/300s below, so
# it's real signal, not sampling noise) rather than a perfectly flat line.
# The values asserted below are therefore the actual grid-search minimum
# over the test's own 24h/60s window, not the single-instant approximation
# -- what the function under test actually and correctly returns, not what
# a simpler mental model predicts.
# ============================================================
t0 = primary.epoch
pos0_km = primary.at(t0).position.km

def _make_phase_offset(catnr, ma_deg, label):
    def checksum(line68):
        total = 0
        for ch in line68:
            if ch.isdigit():
                total += int(ch)
            elif ch == '-':
                total += 1
        return total % 10
    l1 = f"1 {catnr:05d}U 98067A   24079.51782528  .00000000  00000-0  00000-0 0  999"
    l1 = l1 + str(checksum(l1))
    l2_body = f"2 {catnr:05d}  98.6000 100.0000 0000000   0.0000 {ma_deg:8.4f} 14.57889136    1"
    l2 = l2_body + str(checksum(l2_body))
    return EarthSatellite(l1, l2, label, ts)

# Mean-anomaly offsets chosen to land one grid-search minimum in each risk
# band (values below are the actual 24h/60s grid-search result, confirmed
# stable across step sizes in the next block -- see fixture comment above
# for why this is the grid minimum, not the naive single-instant estimate).
phase_high = _make_phase_offset(99011, 0.0041, "PHASE-HIGH (TEST FIXTURE)")      # 0.5059 km -> HIGH
phase_moderate = _make_phase_offset(99012, 0.0200, "PHASE-MODERATE (TEST FIXTURE)")  # 2.4676 km -> MODERATE
phase_low = _make_phase_offset(99013, 0.1000, "PHASE-LOW (TEST FIXTURE)")        # 12.3380 km -> LOW

print("=== compute_min_separation: near-constant-separation co-orbital fixtures ===")
window_hours = 24.0
times, _ = build_time_array(t0, window_hours * 60, 60)
for cand, expected_km, label in [
    (phase_high, 0.5059, "HIGH-band"),
    (phase_moderate, 2.4676, "MODERATE-band"),
    (phase_low, 12.3380, "LOW-band"),
]:
    min_km, t_at = compute_min_separation(primary, cand, times)
    assert abs(min_km - expected_km) < 0.005, (
        f"{cand.name}: measured {min_km:.4f} km, expected ~{expected_km} km"
    )
    print(f"  {label:14s} {cand.name:36s} min_separation={min_km:.4f} km at {t_at}")
print("  PASS: all three co-orbital fixtures land within 0.005 km of their pre-verified grid-search minimum\n")

print("=== compute_min_separation: this near-constant-separation case is grid-independent ===")
# A useful cross-check precisely BECAUSE these fixtures have near-constant,
# not sharply time-varying, separation: the reported minimum should come out
# the same regardless of step size, since the small periodic wobble
# described above is slow relative to any of these step sizes -- there's no
# "in between the samples" that a coarser grid could miss here. This is a
# property of THESE fixtures, not a general claim that grid size never
# matters -- see the dedicated grid-resolution section further down for a
# case with genuinely time-varying separation.
times_coarse, _ = build_time_array(t0, window_hours * 60, 300)
times_fine, _ = build_time_array(t0, window_hours * 60, 5)
min_coarse, _ = compute_min_separation(primary, phase_moderate, times_coarse)
min_fine, _ = compute_min_separation(primary, phase_moderate, times_fine)
assert abs(min_coarse - min_fine) < 0.005, (
    f"near-constant-separation fixture should agree regardless of step size: "
    f"coarse={min_coarse:.4f} km, fine={min_fine:.4f} km"
)
print(f"  PASS: 300s-step ({min_coarse:.4f} km) and 5s-step ({min_fine:.4f} km) agree for a near-constant-separation pair\n")

print("=== classify_conjunction_risk: bands and their exact boundaries ===")
assert classify_conjunction_risk(0.5) == "HIGH"
assert classify_conjunction_risk(0.999) == "HIGH"
assert classify_conjunction_risk(HIGH_RISK_MAX_KM) == "MODERATE"  # 1.0 km exactly -- inclusive on the safer side
assert classify_conjunction_risk(1.001) == "MODERATE"
assert classify_conjunction_risk(3.0) == "MODERATE"
assert classify_conjunction_risk(4.999) == "MODERATE"
assert classify_conjunction_risk(MODERATE_RISK_MAX_KM) == "LOW"  # 5.0 km exactly -- inclusive on the safer side
assert classify_conjunction_risk(5.001) == "LOW"
assert classify_conjunction_risk(500.0) == "LOW"
print("  PASS: HIGH < 1.0 km, MODERATE [1.0, 5.0) km, LOW >= 5.0 km, boundaries land on the safer band\n")

print("=== screen_conjunctions: batched result matches per-pair compute_min_separation ===")
candidates = [phase_high, phase_moderate, phase_low, cand_far]  # cand_far reused from Day 22 (far outside)
results = screen_conjunctions(primary, candidates, t0, window_hours=24.0, step_seconds=60)

assert list(results.columns) == [
    "name", "catnr", "min_separation_km", "time_of_closest_approach_utc", "risk_level"
]
assert len(results) == len(candidates), "one row per candidate expected"
assert primary.name not in set(results["name"]), "primary must never appear in its own results"

# Self-consistency: every row's risk_level must match classify_conjunction_risk
# applied independently to that same row's min_separation_km.
for _, row in results.iterrows():
    assert row["risk_level"] == classify_conjunction_risk(row["min_separation_km"]), (
        f"{row['name']}: risk_level {row['risk_level']} doesn't match its own "
        f"min_separation_km {row['min_separation_km']:.4f}"
    )

# Sorted ascending by min_separation_km -- most concerning candidate first.
assert list(results["min_separation_km"]) == sorted(results["min_separation_km"]), (
    "results should be sorted by min_separation_km ascending"
)
assert results.iloc[0]["name"] == phase_high.name, "PHASE-HIGH should be the top (most concerning) row"

risk_by_name = dict(zip(results["name"], results["risk_level"]))
assert risk_by_name[phase_high.name] == "HIGH"
assert risk_by_name[phase_moderate.name] == "MODERATE"
assert risk_by_name[phase_low.name] == "LOW"
assert risk_by_name[cand_far.name] == "LOW"
print(f"  PASS: {len(results)} candidates screened, correctly sorted, self-consistent risk labels, "
      f"top row is {results.iloc[0]['name']} at {results.iloc[0]['min_separation_km']:.4f} km\n")

print("=== screen_conjunctions: empty candidate list is a valid outcome ===")
empty_results = screen_conjunctions(primary, [], t0)
assert isinstance(empty_results, pd.DataFrame)
assert len(empty_results) == 0
assert list(empty_results.columns) == [
    "name", "catnr", "min_separation_km", "time_of_closest_approach_utc", "risk_level"
]
print("  PASS: an empty candidate list returns an empty, correctly-shaped DataFrame\n")

print("=== grid resolution: a finer grid's minimum is never LARGER than a coarser grid's ===")
# Proof of the property compute_min_separation()'s docstring claims, using a
# pair with REAL time-varying separation (not the constant-separation phase
# fixtures above) -- two coplanar circular orbits at different altitudes
# (Day 22's PRIMARY-700-CIRC and CAND-IN-660-CIRC), whose separation
# genuinely drifts over the window as their different periods carry them in
# and out of alignment. step=60s and step=6s are constructed so every
# coarse-grid sample time is ALSO a fine-grid sample time (6 divides 60,
# same t0) -- so the fine grid evaluates a strict SUPERSET of the instants
# the coarse grid does, and its minimum over that superset can only be
# less-than-or-equal-to the coarse grid's, never greater, by construction.
times_60s, _ = build_time_array(t0, window_hours * 60, 60)
times_6s, _ = build_time_array(t0, window_hours * 60, 6)
min_60s, _ = compute_min_separation(primary, cand_in, times_60s)
min_6s, _ = compute_min_separation(primary, cand_in, times_6s)
assert min_6s <= min_60s, (
    f"a finer grid (superset of the coarser grid's sample times) must find a "
    f"minimum <= the coarser grid's: got 6s={min_6s:.4f} km, 60s={min_60s:.4f} km"
)
print(f"  PASS: 60s-step found {min_60s:.4f} km, 6s-step found {min_6s:.4f} km ({min_60s - min_6s:.4f} km "
      f"tighter) -- confirms the guaranteed direction of the effect. NOTE: these particular fixtures happen "
      f"to be near-coplanar with slow relative drift, so the gap here is small; DISCLAIMER and the "
      f"compute_min_separation() docstring's ~450 km worst-case figure (15 km/s relative velocity x 30s "
      f"step) describe the real-world stakes for a genuinely fast crossing, which this test does not "
      f"attempt to manufacture.\n")

print("=== pipeline: Day 22's bound_population() output feeds Day 23's screen_conjunctions() directly ===")
# The intended real usage: bound_population() tells you WHICH catnrs are
# worth propagating; screen_conjunctions() does the propagating. This test
# performs that handoff explicitly against the Day 22 fixture population
# plus the three new phase-offset satellites, confirming the two stages
# compose without any glue code beyond a catnr lookup.
full_population = [primary, cand_same, cand_in, cand_out, cand_far, cand_ecc,
                    phase_high, phase_moderate, phase_low]
survivors_df = bound_population(primary, full_population, margin_km=DEFAULT_MARGIN_KM)
survivor_catnrs = set(survivors_df["catnr"])
survivor_sats = [sat for sat in full_population if sat.model.satnum in survivor_catnrs]

assert len(survivor_sats) == len(survivors_df), "every survivor catnr should resolve back to exactly one satellite"

screened = screen_conjunctions(primary, survivor_sats, t0)
assert set(screened["catnr"]) == survivor_catnrs, "screen_conjunctions should cover exactly what bound_population handed it"
# cand_far and cand_out were excluded at Day 22's stage -- confirm they never even reach Day 23's stage.
assert cand_far.name not in set(screened["name"])
assert cand_out.name not in set(screened["name"])
print(f"  PASS: {len(survivors_df)} Day-22 survivors piped directly into Day 23's screening, "
      f"producing {len(screened)} classified results with no manual glue code\n")

print("ALL CHECKS PASSED")