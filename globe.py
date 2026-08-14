"""
globe.py
Day 18: first 3D globe -- an Earth sphere mesh (earth_mesh.py) plus
satellites plotted in ECI-converted-to-ECEF coordinates
(satellite_data.py's compute_ecef_positions), rendered with Plotly's
go.Scatter3d/go.Surface. Same role in this project's timeline as Day 4's
plot_map.py: a standalone script proving out one day's specific technique
before later days build on it.

Day 19 (today) adds three things: the Canadian catalogue's satellites are
now merged in alongside the 'visual' group (Day 18 only had 'visual'),
specifically so this globe actually contains a GEO satellite (Anik F2/F3)
and can show the real LEO-vs-GEO scale problem instead of just describing
it; an orbit arc for the highlighted satellite; and explicit camera
framing. The fallback if today ran short was to ship exactly Day 18's
output and move on -- that fallback was NOT needed; this is the full-scope
version.

Network note: like plot_map.py (Day 4) and space_weather.py (Day 15), this
sandboxed dev environment can't reach celestrak.org, so the load_tle_group()
/ catalogue calls below are untested end-to-end here. Every new piece of
math this script depends on (compute_orbit_arc, classify_orbit_regime) IS
tested offline in test_globe.py using hardcoded TLEs, the same approach
test_satellite_data.py established on Day 6. Run this script directly
(`python globe.py`) once on a machine with real network access to verify
the live path end to end, then open globe.html in a browser.
"""

from satellite_data import (
    ts,
    load_tle_group,
    get_satellite_by_catnr,
    compute_ecef_positions,
    compute_subpoints,
    compute_orbit_arc,
    classify_orbit_regime,
)
from earth_mesh import build_earth_sphere
from catalogue import load_catalogue, get_catalogue_satellites, merge_satellite_lists

import plotly.graph_objects as go

# --- Load data ---------------------------------------------------------
t = ts.now()

print("Starting fetch: visual group...")
visual_sats = load_tle_group("visual", reload=False)
print(f"Got {len(visual_sats)} satellites")

# Day 19: merge in the Canadian catalogue, same pattern app.py has used
# since Day 12. Day 18's globe only plotted 'visual' -- an all-LEO/MEO
# population, which meant the scale problem this day is supposed to
# confront (GEO sits ~5.6x farther from Earth's center than a typical LEO
# satellite) never actually showed up on screen. Anik F2/F3, both GEO, are
# what makes that real on this globe instead of a claim in a docstring.
catalogue_df = load_catalogue()
catalogue_sats = get_catalogue_satellites(catalogue_df)
all_sats = merge_satellite_lists(visual_sats, catalogue_sats)

radarsat2 = get_satellite_by_catnr(32382, satellites=all_sats)

# --- ECI -> ECEF positions, plus altitude for regime classification -----
print("Converting to ECEF and classifying regimes...")
ecef_df = compute_ecef_positions(all_sats, t)
subpoint_df = compute_subpoints(all_sats, t)  # same satellites, same t -- gives altitude_km per catnr
df = ecef_df.merge(subpoint_df[["catnr", "altitude_km"]], on="catnr")
df["regime"] = df["altitude_km"].apply(classify_orbit_regime)

r2_df = df[df["catnr"] == 32382]

# --- Orbit arc for the highlighted satellite -----------------------------
# compute_orbit_arc (Day 19) traces RADARSAT-2's own ECEF path over one
# full orbital period (~100 min, sun-synchronous). See its docstring in
# satellite_data.py for why this does NOT come out as a closed loop --
# that's the Earth rotating under the satellite during the lap, same
# effect as ground-track drift (Day 5), just in 3D instead of projected
# onto the surface.
print("Computing RADARSAT-2 orbit arc...")
arc_df = compute_orbit_arc(radarsat2, t)

# --- Earth sphere mesh ----------------------------------------------------
print("Building Earth sphere mesh...")
ex, ey, ez = build_earth_sphere(resolution=50)

# --- Plot -----------------------------------------------------------------
print("Building figure...")
fig = go.Figure()

# Solid-color sphere (flat colorscale, both stops the same color) rather
# than a real texture map -- a textured/coastline-accurate Earth is a
# visual-polish upgrade for later, not something this day's techniques
# depend on.
fig.add_trace(go.Surface(
    x=ex, y=ey, z=ez,
    colorscale=[[0, "rgb(25,55,109)"], [1, "rgb(25,55,109)"]],
    showscale=False,
    opacity=1.0,
    hoverinfo="skip",
    name="Earth",
))

# One trace per orbit regime rather than one gray trace for everything
# (Day 18's approach). Point of today: LEO, MEO, and GEO satellites sit at
# such different distances from Earth that lumping them into one trace
# hides the very scale problem this globe exists to show honestly.
#
# Day 26: MEO/GEO recolored after running the original orange/gold pair
# through a colorblind + contrast validator (Anthropic's dataviz skill) --
# they failed even the plain normal-vision separation check (Delta-E 12.1,
# need >=15), meaning they were genuinely hard to tell apart, not just a
# colorblind-accessibility gap. d95926/199e70 is a validated pair (all
# checks pass, including the harder all-pairs test a scatter/globe view
# needs, not just adjacent-pair). LEO's "silver" was flagged too (reads as
# zero-chroma gray, which is real), but a render-and-look check
# (references/dataviz's own step 7) showed silver-on-navy-sphere is the
# single most legible combination tried for the majority category, and
# LEO can't be mistaken for a muted/de-emphasized "other" here -- it's the
# dominant category by a wide margin (typically ~95% of the tracked
# population) with its own labeled count. Kept deliberately, not an
# oversight.
_REGIME_STYLE = {
    "LEO": dict(color="silver", size=2.5),
    "MEO": dict(color="#d95926", size=4),
    "GEO": dict(color="#199e70", size=5),
}
for regime, style in _REGIME_STYLE.items():
    subset = df[df["regime"] == regime]
    if subset.empty:
        continue
    fig.add_trace(go.Scatter3d(
        x=subset["x_km"], y=subset["y_km"], z=subset["z_km"],
        text=subset["name"],
        mode="markers",
        marker=dict(size=style["size"], color=style["color"], opacity=0.75),
        name=f"{regime} ({len(subset)})",
    ))

# Day 26: highlight color moved from red to white. Red now means HIGH/
# critical risk specifically (Space Weather tab since Day 16, Conjunction
# Screening tab since Day 24) -- reusing it here for "this is the
# satellite you're looking at" was a real semantic collision, not just a
# style choice: a red diamond next to a satellite invites reading it as
# "this one is dangerous" when it just means "selected." White reads as a
# highlight/flag with no status meaning attached, the same role a bright
# neutral plays in conventional ops-map symbology.
fig.add_trace(go.Scatter3d(
    x=r2_df["x_km"], y=r2_df["y_km"], z=r2_df["z_km"],
    text=["RADARSAT-2"],
    mode="markers+text",
    textposition="top center",
    marker=dict(size=7, color="#ffffff", symbol="diamond"),
    name="RADARSAT-2 (Canadian asset)",
))

fig.add_trace(go.Scatter3d(
    x=arc_df["x_km"], y=arc_df["y_km"], z=arc_df["z_km"],
    mode="lines",
    line=dict(color="#ffffff", width=3),
    opacity=0.6,
    name="RADARSAT-2 orbit arc (1 period, ECEF)",
    hoverinfo="skip",
))

# aspectmode="data" is not optional here: Plotly's default 3D scene
# auto-scales each axis to fill the viewport independently, which turns a
# sphere into a squashed ellipsoid the moment the point cloud isn't
# perfectly cubic (it never is). "data" forces x/y/z to share one scale, so
# a sphere actually renders as a sphere -- this matters MORE today than on
# Day 18, since the data now spans Earth's surface all the way out to the
# GEO belt (~42,164 km from Earth's center) rather than just the LEO/MEO
# shell.
#
# Camera: tried several eye vectors and rendered/screenshotted each rather
# than guessing (same approach as Day 18's aspectmode check). Finding worth
# recording: no single static camera position makes a 420 km-altitude
# satellite AND a 35,786 km-altitude satellite both read as more than a
# dot -- that ~85x altitude difference (~5.6x difference in distance from
# Earth's center) is real geometry, not a framing bug to solve away. Plotly
# picks its own default eye (roughly (1.25, 1.25, 1.25) in the data's
# normalized cube) sized to fit the FULL extent including GEO, which
# leaves the LEO shell near the surface barely visible. The eye below is
# pulled in enough that Earth and the LEO/MEO population read at a useful
# size, while GEO satellites still land clearly in frame as a small,
# distant point -- an honest "here's local space, and here's how far out
# GEO really is" framing, not an attempt to make both regimes equally
# detailed at once. Actually inspecting either regime closely is what
# Plotly's built-in interactive rotate (drag) and zoom (scroll) are for --
# this is a starting view, not the only view.
fig.update_layout(
    title=f"3D Globe -- ECEF positions, {t.utc_strftime('%Y-%m-%d %H:%M UTC')}",
    # Day 26: dark chrome to match the app's new dark theme (and the
    # sphere's own dark navy) instead of Plotly's default white page
    # around a dark scene -- this is a standalone script, so it doesn't
    # inherit Streamlit's theme, but there's no reason for it to look
    # inconsistent with the tool it's the proof-of-concept for.
    template="plotly_dark",
    paper_bgcolor="#1a1a19",
    scene=dict(
        aspectmode="data",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        camera=dict(
            eye=dict(x=1.05, y=1.05, z=0.75),
            up=dict(x=0, y=0, z=1),
        ),
    ),
    legend=dict(itemsizing="constant"),
)

fig.write_html("globe.html")
print("Wrote globe.html -- open it manually in your browser")
print(
    f"Population: {len(df)} tracked objects "
    f"({(df['regime'] == 'LEO').sum()} LEO, "
    f"{(df['regime'] == 'MEO').sum()} MEO, "
    f"{(df['regime'] == 'GEO').sum()} GEO)"
)