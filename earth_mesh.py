"""
earth_mesh.py
Day 18: builds a simple sphere mesh approximating Earth, in the same
ECEF (Earth-fixed) x/y/z km coordinates satellite_data.py's
compute_ecef_positions() now produces -- so a globe built from this mesh
and satellites plotted from that function share one consistent frame.

Kept deliberately separate from satellite_data.py: this module has no TLEs,
no Skyfield propagation, no time dependence at all -- it's static geometry,
not orbital mechanics. It doesn't belong in the "shared TLE/orbital math"
module any more than a CAD model of a bracket belongs in a stress-analysis
script; it's a different kind of thing that script happens to use.
"""

from __future__ import annotations

import numpy as np

# WGS84 EQUATORIAL radius. Earth's true shape is an oblate spheroid --
# about 21 km (0.3%) flatter pole-to-pole than equator-to-equator, which is
# exactly why compute_ecef_positions()'s geodetic-vs-geocentric latitude
# check in test_globe.py isn't a bug, it's real ellipsoid geometry. A single
# sphere at the equatorial radius ignores that flattening. That's a
# deliberate simplification for this module: 0.3% is invisible at dashboard
# viewing scale, and wgs84.subpoint() elsewhere in this project already
# does the real ellipsoid math for anything that actually needs geodetic
# precision (altitude, passes). This mesh is for visual context only, not
# a source of any geodetic calculation.
EARTH_RADIUS_KM = 6378.137


def build_earth_sphere(radius_km: float = EARTH_RADIUS_KM, resolution: int = 50):
    """
    Build a lat/lon grid mesh for a sphere, in ECEF x/y/z km, sized for
    Plotly's go.Surface (which expects 2D arrays, not flat lists).

    Follows the standard spherical-to-Cartesian convention that matches
    ECEF/ITRS: x toward (lat=0, lon=0) -- the Greenwich meridian crossing
    the equator -- y toward (lat=0, lon=90E), z toward the north pole.
    This is the same convention Skyfield's itrs frame uses, which is what
    makes this mesh and compute_ecef_positions()'s satellite positions
    plottable together without a second conversion.

    resolution trades visual smoothness for mesh size (resolution**2
    points) -- 50 is a reasonable default for an interactive Plotly figure;
    push it higher only if the sphere looks visibly faceted.

    Returns (x, y, z), each a (resolution, resolution) numpy array of km.
    """
    lat_deg = np.linspace(-90, 90, resolution)
    lon_deg = np.linspace(-180, 180, resolution)
    lon_grid, lat_grid = np.meshgrid(lon_deg, lat_deg)

    lat_rad = np.radians(lat_grid)
    lon_rad = np.radians(lon_grid)

    x = radius_km * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius_km * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius_km * np.sin(lat_rad)

    return x, y, z