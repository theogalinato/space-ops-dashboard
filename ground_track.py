"""
ground_track.py
Day 5: Plot a satellite's ground track (subpoint path) over the next ~100 minutes.
"""
from skyfield.api import load, wgs84
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

ts = load.timescale()
now = ts.now()

print("Fetching RADARSAT-2 TLE...")
radarsat2 = load.tle_file(
    "https://celestrak.org/NORAD/elements/gp.php?CATNR=32382&FORMAT=tle",
    reload=False
)[0]
print("Got RADARSAT-2")

# --- Build a time series: now -> now + 100 minutes, one point per minute ---
print("Propagating ground track...")
minutes = range(0, 101)  # 0 to 100 inclusive
times_list = [now.utc_datetime() + timedelta(minutes=m) for m in minutes]
times = ts.from_datetimes(times_list)

geocentric = radarsat2.at(times)
subpoints = wgs84.subpoint(geocentric)

df = pd.DataFrame({
    "minute": list(minutes),
    "lat": subpoints.latitude.degrees,
    "lon": subpoints.longitude.degrees,
})
print(f"Computed {len(df)} track points")

# --- Plot ---
fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lat=df["lat"], lon=df["lon"],
    mode="markers+lines",
    line=dict(width=1, color="red"),
    marker=dict(size=3, color="red"),
    name="RADARSAT-2 ground track (next 100 min)"
))

# Mark the starting point distinctly
fig.add_trace(go.Scattergeo(
    lat=[df["lat"][0]], lon=[df["lon"][0]],
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