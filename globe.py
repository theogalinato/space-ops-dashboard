"""
globe.py
Day 18: first 3D globe -- an Earth sphere mesh (earth_mesh.py) plus
satellites plotted in ECI-converted-to-ECEF coordinates
(satellite_data.py's compute_ecef_positions), rendered with Plotly's
go.Scatter3d/go.Surface. Same role in this project's timeline as Day 4's
plot_map.py: a standalone script proving out one day's specific technique
before later days build on it. Day 19 adds true-altitude accuracy checks,
orbit arcs, and camera framing; the fallback if Day 19 runs short is to
ship exactly this, the orthographic globe, and move on -- so this script is
written to already be a legitimate, presentable artifact on its own, not a
throwaway proof of concept.

Network note: like plot_map.py (Day 4) and space_weather.py (Day 15), this
sandboxed dev environment can't reach celestrak.org, so the load_tle_group()
call below is untested end-to-end here. compute_ecef_positions() and
build_earth_sphere(), the two new pieces of math this script depends on,
ARE tested offline in test_globe.py using a hardcoded TLE, the same
approach test_satellite_data.py established on Day 6. Run this script
directly (`python globe.py`) once on a machine with real network access to
verify the live path end to end, then open globe.html in a browser.
"""

from skyfield.api import load

from satellite_data import ts, load_tle_group, get_satellite_by_catnr, compute_ecef_positions
from earth_mesh import build_earth_sphere

import plotly.graph_objects as go

# --- Load data ---------------------------------------------------------
t = ts.now()

print("Starting fetch: visual group...")
visual_sats = load_tle_group("visual", reload=False)
print(f"Got {len(visual_sats)} satellites")

radarsat2 = get_satellite_by_catnr(32382, satellites=visual_sats)

# --- ECI -> ECEF positions ----------------------------------------------
# compute_ecef_positions (satellite_data.py, Day 18) converts each
# satellite's inertial GCRS position into the Earth-fixed ITRS/ECEF frame,
# the same frame the Earth sphere mesh below is built in. Plotting both in
# one consistent frame is the entire point of today's conversion -- do
# this step wrong and satellites and continents disagree about which way
# the planet is facing.
print("Converting to ECEF...")
df = compute_ecef_positions(visual_sats, t)
r2_df = compute_ecef_positions([radarsat2], t)

# --- Earth sphere mesh ----------------------------------------------------
print("Building Earth sphere mesh...")
ex, ey, ez = build_earth_sphere(resolution=50)

# --- Plot -----------------------------------------------------------------
print("Building figure...")
fig = go.Figure()

# Solid-color sphere (flat colorscale, both stops the same color) rather
# than a real texture map -- this is Day 18's "prove the geometry works"
# step. A textured/coastline-accurate Earth is a visual-polish upgrade for
# later, not something the ECI->ECEF conversion or scatter3d mechanics
# depend on.
fig.add_trace(go.Surface(
    x=ex, y=ey, z=ez,
    colorscale=[[0, "rgb(25,55,109)"], [1, "rgb(25,55,109)"]],
    showscale=False,
    opacity=1.0,
    hoverinfo="skip",
    name="Earth",
))

fig.add_trace(go.Scatter3d(
    x=df["x_km"], y=df["y_km"], z=df["z_km"],
    text=df["name"],
    mode="markers",
    marker=dict(size=2.5, color="gray", opacity=0.7),
    name="Tracked objects (visual group)",
))

fig.add_trace(go.Scatter3d(
    x=r2_df["x_km"], y=r2_df["y_km"], z=r2_df["z_km"],
    text=["RADARSAT-2"],
    mode="markers+text",
    textposition="top center",
    marker=dict(size=6, color="red", symbol="diamond"),
    name="RADARSAT-2 (Canadian asset)",
))

# aspectmode="data" is not optional here: Plotly's default 3D scene
# auto-scales each axis to fill the viewport independently, which turns a
# sphere into a squashed ellipsoid the moment the point cloud isn't
# perfectly cubic (it never is). "data" forces x/y/z to share one scale, so
# a sphere actually renders as a sphere.
fig.update_layout(
    title=f"3D Globe -- ECEF positions, {t.utc_strftime('%Y-%m-%d %H:%M UTC')}",
    scene=dict(
        aspectmode="data",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
)

fig.write_html("globe.html")
print("Wrote globe.html -- open it manually in your browser")