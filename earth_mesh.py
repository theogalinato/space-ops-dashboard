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

import json
from pathlib import Path

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


def _latlon_to_ecef(lat_deg: float, lon_deg: float, radius_km: float) -> tuple[float, float, float]:
    """Single-point version of build_earth_sphere()'s formula above --
    same convention, same simplification (simple sphere, not the WGS84
    ellipsoid), used by build_coastlines() below so coastline points and
    the sphere mesh agree on exactly what "the surface" means."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = radius_km * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius_km * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius_km * np.sin(lat_rad)
    return x, y, z


# Day 27: coastline outlines, so the globe reads as a recognizable planet
# instead of a plain sphere. Bundled locally at data/ne_110m_coastline.geojson
# rather than fetched at runtime -- Natural Earth's 110m-resolution
# coastlines (public domain, no attribution required, naturalearthdata.com,
# fetched via a GitHub mirror since naturalearthdata.com itself isn't
# reachable from this project's dev sandbox). 110m is Natural Earth's
# coarsest published resolution, which is the right amount of detail for a
# small globe view where continents just need to be recognizable, not a
# coastline anyone would zoom into. Bundling it locally is a deliberate
# difference from the 2D map's basemap, which fetches its land/country
# boundaries from cdn.plot.ly on every load (see live_map's comments) --
# this view needs no network at all beyond the satellite TLE/catalogue
# data the app already fetches.
COASTLINE_GEOJSON_PATH = Path(__file__).parent / "data" / "ne_110m_coastline.geojson"


def build_coastlines(radius_km: float = EARTH_RADIUS_KM * 1.001, path=COASTLINE_GEOJSON_PATH):
    """
    Load the bundled coastline vector data and convert it to ECEF x/y/z
    km, ready to hand straight to a single Plotly Scatter3d line trace.

    Uses the SAME simple-sphere formula build_earth_sphere() uses above --
    deliberately not Skyfield's true WGS84 ellipsoid conversion, even
    though that's more geodetically correct. This sphere is already a
    simplification (equatorial radius, no flattening -- see the module
    docstring), so coastline points need to land on THIS sphere, not on a
    more accurate one: using the true ellipsoid here would put coastline
    points up to ~21 km off this module's own sphere surface at some
    latitudes, the same geocentric-vs-geodetic gap `test_globe.py`
    documents for `compute_ecef_positions()`. Internal consistency with
    the sphere this draws on top of matters more than geodetic precision
    this module never claimed in the first place.

    radius_km defaults to 0.1% ABOVE the sphere's own radius, not exactly
    on it -- rendering it exactly at the sphere's surface let the line and
    the sphere mesh fight over the same points at render time (a visible
    flicker/dropout artifact, seen and rejected before picking this).

    Returns (x, y, z) flat lists with a None inserted between each
    coastline segment, so Plotly draws every segment as its own line
    within one trace instead of connecting the end of one coastline to
    the start of the next with a stray line straight across the ocean.
    """
    with open(path) as f:
        geojson = json.load(f)

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for feature in geojson["features"]:
        geometry = feature["geometry"]
        # The bundled file is all LineStrings, but a MultiLineString is
        # handled too rather than assumed away -- cheap insurance against
        # a future swap to a differently-structured source file.
        if geometry["type"] == "LineString":
            segments = [geometry["coordinates"]]
        elif geometry["type"] == "MultiLineString":
            segments = geometry["coordinates"]
        else:
            continue

        for segment in segments:
            for lon_deg, lat_deg in segment:
                x, y, z = _latlon_to_ecef(lat_deg, lon_deg, radius_km)
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
            xs.append(None)
            ys.append(None)
            zs.append(None)

    return xs, ys, zs