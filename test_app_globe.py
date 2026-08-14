"""
Day 21: offline sanity test for the DataFrame wiring inside app.py's new
live_globe() fragment -- the merge of compute_ecef_positions() (Day 18) with
compute_subpoints()'s altitude column, the regime filter driven by the Map
tab's new multiselect, and the "is the selected satellite in this filtered
set" lookup that decides whether an orbit arc gets drawn.

This is NOT a re-test of the underlying orbital math -- compute_ecef_positions,
compute_orbit_arc, and classify_orbit_regime already have direct coverage in
test_globe.py using these exact two fixtures (a real ISS TLE for LEO, a
hand-built synthetic TLE for GEO, both explained there). What Day 18/19 never
exercised is two satellites of DIFFERENT regimes going through the SAME
merge/filter path together, which is exactly the new code live_globe() adds
on top of that math. live_globe() itself is a Streamlit fragment and can't be
called directly outside a Streamlit runtime -- Day 20 established the pattern
for testing runtime-specific behavior (test_refresh_mechanics.py, a headless
Streamlit app); this test instead replicates live_globe()'s plain-Python
DataFrame logic line-for-line against the same fixtures, which is the part
that can actually break silently (a column-name typo, an empty filter result)
without needing a Streamlit process at all.

Same network caveat as every other test_*.py in this project: no live
celestrak.org call, hardcoded TLEs only.
"""

from skyfield.api import EarthSatellite, load

from satellite_data import (
    compute_ecef_positions,
    compute_subpoints,
    compute_orbit_arc,
    classify_orbit_regime,
)

ts = load.timescale()

# Same ISS fixture as test_globe.py / test_satellite_data.py.
iss_line1 = "1 25544U 98067A   24079.51782528  .00016717  00000-0  30187-3 0  9994"
iss_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49560343440557"
iss = EarthSatellite(iss_line1, iss_line2, "ISS (ZARYA)", ts)

# Same synthetic GEO fixture as test_globe.py -- hand-built, not a real
# satellite, epoch chosen to match the ISS fixture's day so both can be
# evaluated at one shared timestamp the way live_globe() does with t_now.
geo_line1 = "1 99999U 98067A   24079.50000000  .00016717  00000-0  30187-3 0  9996"
geo_line2 = "2 99999  0.0500  180.0000 0002000  90.0000   0.0000  1.00270000 10003"
geo = EarthSatellite(geo_line1, geo_line2, "SYNTHETIC-GEO (TEST FIXTURE)", ts)

all_sats = [iss, geo]
t = iss.epoch  # one shared timestamp, same as live_globe()'s t_now for every satellite

print("=== merge: compute_ecef_positions + compute_subpoints altitude ===")
ecef_df = compute_ecef_positions(all_sats, t)
subpoint_df = compute_subpoints(all_sats, t)
df = ecef_df.merge(subpoint_df[["catnr", "altitude_km"]], on="catnr")

assert len(df) == 2, f"expected both satellites to survive the merge, got {len(df)} rows"
assert "altitude_km" in df.columns, "merge should have attached altitude_km from compute_subpoints"
assert "x_km" in df.columns and "y_km" in df.columns and "z_km" in df.columns
print(f"  PASS: merge kept both satellites, columns = {list(df.columns)}\n")

print("=== regime classification on the merged frame ===")
df["regime"] = df["altitude_km"].apply(classify_orbit_regime)
iss_regime = df.loc[df["name"] == "ISS (ZARYA)", "regime"].iloc[0]
geo_regime = df.loc[df["name"] == "SYNTHETIC-GEO (TEST FIXTURE)", "regime"].iloc[0]
assert iss_regime == "LEO", f"expected ISS to classify as LEO, got {iss_regime}"
assert geo_regime == "GEO", f"expected synthetic satellite to classify as GEO, got {geo_regime}"
print(f"  PASS: ISS -> {iss_regime}, synthetic satellite -> {geo_regime}\n")

print("=== regime_filter: the Map tab's multiselect, applied as .isin() ===")
leo_only = df[df["regime"].isin(("LEO",))]
geo_only = df[df["regime"].isin(("GEO",))]
both = df[df["regime"].isin(("LEO", "GEO"))]
assert list(leo_only["name"]) == ["ISS (ZARYA)"]
assert list(geo_only["name"]) == ["SYNTHETIC-GEO (TEST FIXTURE)"]
assert len(both) == 2
print("  PASS: filtering to one regime excludes the other; filtering to both keeps both\n")

print("=== selected-satellite lookup used to decide whether to draw an orbit arc ===")
# Case 1: selected satellite is inside the filtered set -- arc should draw.
selected_row = leo_only[leo_only["name"] == "ISS (ZARYA)"]
assert not selected_row.empty, "ISS should be found when the filter includes LEO"
arc_df = compute_orbit_arc(iss, t)
assert len(arc_df) > 1, "orbit arc should have more than one point to draw a line"
print("  PASS: selected satellite present in filter -> lookup succeeds, arc has real points\n")

# Case 2: selected satellite is filtered OUT (e.g. user picked ISS but only
# ticked the GEO checkbox) -- this is the st.info() branch in live_globe(),
# and the lookup must come back empty rather than raising, since live_globe()
# branches on `.empty` to decide whether to call compute_orbit_arc() at all.
selected_row_filtered_out = geo_only[geo_only["name"] == "ISS (ZARYA)"]
assert selected_row_filtered_out.empty, "ISS should NOT be found when the filter excludes LEO"
print("  PASS: selected satellite excluded by filter -> lookup correctly comes back empty\n")

print("All app.py live_globe() wiring checks passed.")