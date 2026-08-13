"""
plot_map.py
Day 4: Plot subpoints of all satellites in the CelesTrak 'visual' group,
with RADARSAT-2 highlighted as a distinct marker.
"""
from skyfield.api import load, wgs84
import pandas as pd
import plotly.graph_objects as go

# --- Load data -------------------------------------------------------
ts = load.timescale()
t = ts.now()

print("Starting fetch: visual group...")
visual_sats = load.tle_file(
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle",
    reload=False
)
print(f"Got {len(visual_sats)} satellites")

print("Starting fetch: RADARSAT-2...")
radarsat2 = load.tle_file(
    "https://celestrak.org/NORAD/elements/gp.php?CATNR=32382&FORMAT=tle",
    reload=False
)[0]
print("Got RADARSAT-2")

# --- Build dataframe of subpoints -------------------------------------
print("Computing subpoints...")
rows = []
for sat in visual_sats:
    geocentric = sat.at(t)
    subpoint = wgs84.subpoint(geocentric)
    rows.append({
        "name": sat.name,
        "lat": subpoint.latitude.degrees,
        "lon": subpoint.longitude.degrees,
        "alt_km": subpoint.elevation.km,
    })
df = pd.DataFrame(rows)

r2_geo = radarsat2.at(t)
r2_sub = wgs84.subpoint(r2_geo)
r2_lat, r2_lon = r2_sub.latitude.degrees, r2_sub.longitude.degrees
print("Subpoints done, building figure...")

# --- Plot ---------------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lat=df["lat"], lon=df["lon"],
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
