"""
Quick sanity test for satellite_data.py using a hardcoded TLE, since this
sandbox can't reach celestrak.org. This only validates compute_subpoints,
compute_ground_track, and compute_orbital_params (the pure-computation
functions) -- load_tle_group and get_satellite_by_catnr still need a live
network test on your end (they should just work, since they're the same
load.tle_file() call you already validated on Day 2/4, now with reload
exposed as a real parameter).
"""

from skyfield.api import EarthSatellite, load
from satellite_data import compute_subpoints, compute_ground_track, compute_orbital_params

ts = load.timescale()

# ISS TLE (won't match live position exactly since it's not fresh, but
# valid enough to sanity-check the math/shape of every function)
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   24079.51782528  .00016717  00000-0  30187-3 0  9994"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49560343440557"

# Evaluate near the TLE's own epoch, not "now" -- SGP4 accuracy degrades
# fast the further you propagate past epoch (877 days past epoch gave a
# ~200 km altitude error when this was first tried with ts.now()). This
# validates the module's math, not the freshness of this hardcoded TLE.
sat = EarthSatellite(line1, line2, name, ts)
t = sat.epoch

print("=== compute_orbital_params ===")
params = compute_orbital_params(sat, t)
for k, v in params.items():
    print(f"  {k}: {v:.4f}")
assert 0 < params["inclination_deg"] < 180
assert 80 < params["period_min"] < 100  # ISS orbit is ~92-93 min
assert 6 < params["speed_km_s"] < 9  # LEO orbital speed sanity range
print("  PASS: values in expected physical ranges for ISS\n")

print("=== compute_subpoints (snapshot, list of 1) ===")
df_snap = compute_subpoints([sat], t)
print(df_snap.to_string(index=False))
assert list(df_snap.columns) == ["name", "catnr", "latitude_deg", "longitude_deg", "altitude_km"]
assert len(df_snap) == 1
assert -90 <= df_snap.loc[0, "latitude_deg"] <= 90
assert 300 < df_snap.loc[0, "altitude_km"] < 500  # ISS altitude range
print("  PASS: one row, correct columns, altitude in ISS range\n")

print("=== compute_ground_track (window, one satellite) ===")
df_track = compute_ground_track(sat, t, minutes=100, step_seconds=60)
print(df_track.head(3).to_string(index=False))
print("  ...")
print(df_track.tail(2).to_string(index=False))
assert len(df_track) == 101  # 100 min / 60s step + 1
assert list(df_track.columns) == ["time_utc", "latitude_deg", "longitude_deg", "altitude_km"]
# ground track should show real movement, not a static point
lat_range = df_track["latitude_deg"].max() - df_track["latitude_deg"].min()
assert lat_range > 10, f"expected real ground-track movement, got lat_range={lat_range}"
print(f"  PASS: 101 rows as expected, latitude swept {lat_range:.1f} deg over 100 min\n")

print("ALL CHECKS PASSED")