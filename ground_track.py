"""
ground_track.py
Day 5: Plot a satellite's ground track (subpoint path) over the next ~100 minutes.
Day 6: refactored to use shared satellite_data.py module.
"""
from skyfield.api import load
import pandas as pd
import plotly.graph_objects as go
from satellite_data import get_satellite_by_catnr, compute_ground_track

ts = load.timescale()
now = ts.now()

print("Fetching RADARSAT-2 TLE...")
radarsat2 = get_satellite_by_catnr(32382, group="visual")
print("Got RADARSAT-2")

# --- Build ground track: now -> now + 100 minutes, one point per minute ---
print("Propagating ground track...")
df = compute_ground_track(radarsat2, now, minutes=100, step_seconds=60)
print(f"Computed {len(df)} track points")

# --- Plot ---
fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lat=df["latitude_deg"], lon=df["longitude_deg"],
    mode="markers+lines",
    line=dict(width=1, color="red"),
    marker=dict(size=3, color="red"),
    name="RADARSAT-2 ground track (next 100 min)"
))

# Mark the starting point distinctly
fig.add_trace(go.Scattergeo(
    lat=[df["latitude_deg"].iloc[0]], lon=[df["longitude_deg"].iloc[0]],
    mode="markers",
    marker=dict(size=10, color="black", symbol="star"),
    name="Now"
))

fig.update_layout(
    title=f"RADARSAT-2 Ground Track — {now.utc_strftime('%Y-%m-%d %H:%M UTC')} + 100 min",
    geo=dict(projection_type="natural earth", showland=True, landcolor="rgb(230,230,230)"),
)

fig.write_html("ground_track.html")
print("Wrote ground_track.html")
from satellite_data import compute_orbital_params
print(compute_orbital_params(radarsat2, now))