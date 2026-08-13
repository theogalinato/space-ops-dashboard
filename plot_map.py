"""
plot_map.py
Day 4: Plot subpoints of all satellites in the CelesTrak 'visual' group,
with RADARSAT-2 highlighted as a distinct marker.
"""
from skyfield.api import load
import pandas as pd
import plotly.graph_objects as go
from satellite_data import load_tle_group, get_satellite_by_catnr, compute_subpoints

# --- Load data -------------------------------------------------------
ts = load.timescale()
t = ts.now()

print("Starting fetch: visual group...")
visual_sats = load_tle_group("visual", reload=False)
print(f"Got {len(visual_sats)} satellites")

radarsat2 = get_satellite_by_catnr(32382, satellites=visual_sats)

# --- Build dataframe of subpoints -------------------------------------
print("Computing subpoints...")
df = compute_subpoints(visual_sats, t)

r2_df = compute_subpoints([radarsat2], t)
r2_lat, r2_lon = r2_df.loc[0, "latitude_deg"], r2_df.loc[0, "longitude_deg"]
print("Subpoints done, building figure...")

# --- Plot ---------------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lat=df["latitude_deg"], lon=df["longitude_deg"],
    text=df["name"],
    mode="markers",
    marker=dict(size=4, color="gray", opacity=0.6),
    name="Tracked objects (visual group)"
))

fig.add_trace(go.Scattergeo(
    lat=[r2_lat], lon=[r2_lon],
    text=["RADARSAT-2"],
    mode="markers+text",
    textposition="top center",
    marker=dict(size=12, color="red", symbol="star"),
    name="RADARSAT-2 (Canadian asset)"
))

fig.update_layout(
    title=f"Satellite Subpoints — {t.utc_strftime('%Y-%m-%d %H:%M UTC')}",
    geo=dict(projection_type="natural earth", showland=True, landcolor="rgb(230,230,230)"),
)

fig.write_html("plot_map.html")
print("Wrote plot_map.html — open it manually in your browser")
print("fig.show() called")
