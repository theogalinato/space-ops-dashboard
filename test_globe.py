"""
Offline sanity test for the 3D globe work: Day 18's compute_ecef_positions()
and build_earth_sphere(), plus Day 19's compute_orbit_arc() and
classify_orbit_regime() (all in satellite_data.py except build_earth_sphere,
in earth_mesh.py). Same hardcoded-TLE approach as test_satellite_data.py,
since this sandbox can't reach celestrak.org -- evaluated at each TLE's own
epoch, not "now", for the same reason (SGP4 accuracy degrades the further
you propagate past epoch).

Day 18's section checks that the ECI-to-ECEF conversion is self-consistent
with wgs84.subpoint(), which this project has trusted since Day 6. If a
satellite's ECEF x/y/z, converted back to spherical lat/lon/radius by hand,
doesn't agree with what wgs84.subpoint() already reports for the exact same
geocentric position, one of the two frame conversions is wrong. Longitude
should match exactly (geodetic and geocentric longitude are the same thing
-- longitude doesn't care about the ellipsoid). Latitude and radius should
be CLOSE but not identical, because WGS84 is an ellipsoid, not a sphere --
see the module docstrings in satellite_data.py and earth_mesh.py for why.

Day 19's section checks compute_orbit_arc() produces the RIGHT KIND of
shape for a LEO vs. a GEO satellite -- a big, non-closing spiral for LEO
(Earth rotates under the satellite during one ~90 min lap) vs. a tiny,
near-stationary loop for GEO (period matches Earth's rotation almost
exactly, by design) -- using a second, SYNTHETIC/hand-constructed TLE for a
made-up geostationary satellite, clearly labeled as such rather than
presented as a real one. It exists purely to give this test a GEO-regime
orbit to check against, the same way test_satellite_data.py's ISS TLE
exists to sanity-check math, not to represent live position.
"""

import math

from skyfield.api import EarthSatellite, load, wgs84

from satellite_data import compute_ecef_positions, compute_orbit_arc, compute_orbital_params, classify_orbit_regime
from earth_mesh import build_earth_sphere, EARTH_RADIUS_KM

ts = load.timescale()

name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   24079.51782528  .00016717  00000-0  30187-3 0  9994"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49560343440557"
sat = EarthSatellite(line1, line2, name, ts)
t = sat.epoch

print("=== compute_ecef_positions ===")
df = compute_ecef_positions([sat], t)
print(df.to_string(index=False))
assert list(df.columns) == ["name", "catnr", "x_km", "y_km", "z_km"]
assert len(df) == 1
x, y, z = df.loc[0, "x_km"], df.loc[0, "y_km"], df.loc[0, "z_km"]

# ISS is ~400-450 km up; geocentric radius should be Earth-radius-ish plus
# that, generously bounded since "Earth radius" itself varies with latitude
# on the real ellipsoid (see below).
radius_km = math.sqrt(x**2 + y**2 + z**2)
assert 6700 < radius_km < 6900, f"geocentric radius {radius_km} outside ISS-altitude range"
print(f"  PASS: geocentric radius {radius_km:.1f} km in expected ISS-altitude range\n")

print("=== round-trip vs. wgs84.subpoint() ===")
geocentric = sat.at(t)
subpoint = wgs84.subpoint(geocentric)

geocentric_lat_deg = math.degrees(math.asin(z / radius_km))
geocentric_lon_deg = math.degrees(math.atan2(y, x))

lon_diff = abs(geocentric_lon_deg - subpoint.longitude.degrees)
lat_diff = subpoint.latitude.degrees - geocentric_lat_deg  # geodetic - geocentric

print(f"  ECEF-derived (geocentric) lat/lon: {geocentric_lat_deg:.4f}, {geocentric_lon_deg:.4f}")
print(f"  wgs84.subpoint (geodetic) lat/lon:  {subpoint.latitude.degrees:.4f}, {subpoint.longitude.degrees:.4f}")
print(f"  lat diff (geodetic - geocentric): {lat_diff:.4f} deg")
print(f"  lon diff: {lon_diff:.2e} deg")

# Longitude: no ellipsoid effect, should match to floating-point precision.
assert lon_diff < 1e-9, f"longitude should match exactly, got diff={lon_diff}"

# Latitude: WGS84's ~0.34% flattening produces a real geodetic-vs-geocentric
# gap, maximal (~11.5 arcmin, ~0.19 deg) around 45 deg latitude and zero at
# the equator/poles. ISS is at ~51.5 deg inclination, so a gap in that
# neighborhood is EXPECTED, not a bug -- assert it's present and bounded,
# not zero.
assert 0.0 < lat_diff < 0.25, f"expected a small nonzero geodetic/geocentric gap, got {lat_diff}"
print(f"  PASS: longitude matches exactly, latitude gap ({lat_diff:.4f} deg) matches expected WGS84 ellipsoid effect\n")

print("=== build_earth_sphere ===")
ex, ey, ez = build_earth_sphere(resolution=10)
assert ex.shape == ey.shape == ez.shape == (10, 10)

mesh_radii = (ex**2 + ey**2 + ez**2) ** 0.5
assert (abs(mesh_radii - EARTH_RADIUS_KM) < 1e-6).all(), "every mesh point should sit exactly on the sphere"
print(f"  PASS: mesh shape {ex.shape}, every point at radius {EARTH_RADIUS_KM} km\n")

# Spot-check known reference points instead of trusting the grid blindly.
# Equator/Greenwich (lat=0, lon=0) -> (R, 0, 0). North pole (lat=90) -> (0, 0, R).
res = 25
import numpy as np
lat_deg = np.linspace(-90, 90, res)
lon_deg = np.linspace(-180, 180, res)
eq_idx = np.argmin(np.abs(lat_deg - 0))
greenwich_idx = np.argmin(np.abs(lon_deg - 0))
pole_idx = np.argmin(np.abs(lat_deg - 90))

fx, fy, fz = build_earth_sphere(resolution=res)
eq_point = (fx[eq_idx, greenwich_idx], fy[eq_idx, greenwich_idx], fz[eq_idx, greenwich_idx])
pole_point = (fx[pole_idx, 0], fy[pole_idx, 0], fz[pole_idx, 0])

print(f"  equator/Greenwich point: {eq_point}")
print(f"  north pole point: {pole_point}")
assert abs(eq_point[0] - EARTH_RADIUS_KM) < 5 and abs(eq_point[1]) < 5 and abs(eq_point[2]) < 5
assert abs(pole_point[2] - EARTH_RADIUS_KM) < 5 and abs(pole_point[0]) < 5 and abs(pole_point[1]) < 5
print("  PASS: equator/Greenwich lands on +x axis, north pole lands on +z axis, as ECEF convention requires\n")

print("=== classify_orbit_regime ===")
assert classify_orbit_regime(420) == "LEO"
assert classify_orbit_regime(20200) == "MEO"  # GPS altitude, for reference
assert classify_orbit_regime(35786) == "GEO"
assert classify_orbit_regime(1999) == "LEO" and classify_orbit_regime(2000) == "MEO"  # boundary
print("  PASS: LEO/MEO/GEO thresholds correct, including the 2000 km boundary\n")

print("=== compute_orbit_arc: LEO (ISS) should NOT close, should span a large extent ===")
arc = compute_orbit_arc(sat, t)  # sat/t = the ISS fixture from above; minutes=None -> uses its own period
period_min = compute_orbital_params(sat, t)["period_min"]
expected_rows = int((period_min * 60) / 30) + 1  # default step_seconds=30
assert list(arc.columns) == ["time_utc", "x_km", "y_km", "z_km"]
assert len(arc) == expected_rows, f"expected {expected_rows} rows for a {period_min:.1f} min period at 30s steps, got {len(arc)}"

leo_extent_km = max(
    arc.x_km.max() - arc.x_km.min(),
    arc.y_km.max() - arc.y_km.min(),
    arc.z_km.max() - arc.z_km.min(),
)
start = arc.iloc[0]
end = arc.iloc[-1]
close_gap_km = math.sqrt(
    (start.x_km - end.x_km) ** 2 + (start.y_km - end.y_km) ** 2 + (start.z_km - end.z_km) ** 2
)
print(f"  {len(arc)} rows over a {period_min:.1f} min period")
print(f"  spatial extent (largest axis range): {leo_extent_km:.0f} km")
print(f"  start-to-end gap after one period: {close_gap_km:.0f} km")

# A LEO orbit's own diameter is roughly 2*(Earth radius + altitude) -- for
# the ISS that's over 13,000 km, so the traced arc should span thousands
# of km, not sit near a point.
assert leo_extent_km > 5000, f"expected a large LEO orbit extent, got {leo_extent_km:.0f} km"
# And it should NOT close on itself -- Earth rotates ~22.5 deg under a
# ~93 min ISS orbit, which is hundreds to low-thousands of km at this
# altitude. A near-zero gap here would mean the ECEF rotation isn't being
# applied (i.e. this had accidentally become an ECI/inertial arc instead).
assert close_gap_km > 500, f"expected the arc to NOT close (ECEF, not ECI) -- gap only {close_gap_km:.0f} km"
print("  PASS: large extent, arc does not close -- consistent with ECEF, not an inertial frame\n")

print("=== compute_orbit_arc: GEO (synthetic) should stay near-stationary ===")
# Hand-built geostationary-regime TLE: near-zero inclination, near-zero
# eccentricity, mean motion set to Earth's sidereal rotation rate
# (1.00270 rev/day, period ~1436.1 min) -- NOT a real satellite, a test
# fixture, following the same TLE checksum format as the ISS line above.
geo_line1 = "1 99999U 98067A   24079.50000000  .00016717  00000-0  30187-3 0  9996"
geo_line2 = "2 99999  0.0500  180.0000 0002000  90.0000   0.0000  1.00270000 10003"
geo_sat = EarthSatellite(geo_line1, geo_line2, "SYNTHETIC-GEO (TEST FIXTURE)", ts)
geo_t = geo_sat.epoch

geo_params = compute_orbital_params(geo_sat, geo_t)
geo_altitude_km = wgs84.subpoint(geo_sat.at(geo_t)).elevation.km
print(f"  period: {geo_params['period_min']:.1f} min, altitude: {geo_altitude_km:.0f} km, regime: {classify_orbit_regime(geo_altitude_km)}")
assert 1430 < geo_params["period_min"] < 1445, "expected a ~sidereal-day period for a GEO fixture"
assert classify_orbit_regime(geo_altitude_km) == "GEO"

geo_arc = compute_orbit_arc(geo_sat, geo_t)
geo_extent_km = max(
    geo_arc.x_km.max() - geo_arc.x_km.min(),
    geo_arc.y_km.max() - geo_arc.y_km.min(),
    geo_arc.z_km.max() - geo_arc.z_km.min(),
)
print(f"  {len(geo_arc)} rows over one period; spatial extent (largest axis range): {geo_extent_km:.1f} km")

# This is the Day 13 GEO lesson again, now visible as geometry: a
# geostationary satellite's period matches Earth's rotation almost
# exactly, so in ECEF it barely moves at all over one "orbit" -- a tiny
# loop, nowhere close to the LEO satellite's extent above.
assert geo_extent_km < 500, f"expected a small, near-stationary GEO arc, got {geo_extent_km:.1f} km"
assert geo_extent_km < leo_extent_km / 10, "GEO arc should be dramatically smaller than the LEO arc"
print("  PASS: GEO arc stays near-stationary in ECEF, consistent with Day 13's 'GEO has no discrete passes' finding\n")

print("ALL CHECKS PASSED")