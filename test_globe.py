"""
Offline sanity test for Day 18: compute_ecef_positions() (satellite_data.py)
and build_earth_sphere() (earth_mesh.py). Same hardcoded-ISS-TLE approach as
test_satellite_data.py, since this sandbox can't reach celestrak.org --
evaluated at the TLE's own epoch, not "now", for the same reason (SGP4
accuracy degrades the further you propagate past epoch).

What this test is actually checking: that the ECI-to-ECEF conversion is
self-consistent with wgs84.subpoint(), which this project has trusted since
Day 6. If a satellite's ECEF x/y/z, converted back to spherical lat/lon/
radius by hand, doesn't agree with what wgs84.subpoint() already reports
for the exact same geocentric position, one of the two frame conversions is
wrong. Longitude should match exactly (geodetic and geocentric longitude
are the same thing -- longitude doesn't care about the ellipsoid). Latitude
and radius should be CLOSE but not identical, because WGS84 is an
ellipsoid, not a sphere -- see the module docstrings in satellite_data.py
and earth_mesh.py for why.
"""

import math

from skyfield.api import EarthSatellite, load, wgs84

from satellite_data import compute_ecef_positions
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

print("ALL CHECKS PASSED")