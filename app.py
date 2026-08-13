import streamlit as st
from skyfield.api import load
from satellite_data import load_tle_group, compute_subpoints, compute_orbital_params
import plotly.express as px

st.set_page_config(page_title="Space Operations Dashboard", layout="wide")
st.title("Space Operations Dashboard")

ts = load.timescale()
t = ts.now()  # <-- this is the missing piece: a scalar Skyfield Time

# Load data
satellites = load_tle_group("visual")

# Dropdown of satellite names
names = [sat.name for sat in satellites]
selected_name = st.selectbox("Select a satellite", names)
selected_sat = next(sat for sat in satellites if sat.name == selected_name)

# --- Info panel ---
st.write(f"Tracking: **{selected_sat.name}**")
params = compute_orbital_params(selected_sat, t)
col1, col2, col3 = st.columns(3)
col1.metric("Inclination", f"{params['inclination_deg']:.2f}°")
col2.metric("Period", f"{params['period_min']:.1f} min")
col3.metric("Speed", f"{params['speed_km_s']:.2f} km/s")

# --- Map: subpoints for the WHOLE group, at this one instant t ---
df = compute_subpoints(satellites, t)

fig = px.scatter_geo(
    df,
    lat="latitude_deg",
    lon="longitude_deg",
    hover_name="name",
    projection="natural earth",
)

# Highlight the selected satellite
selected_row = df[df["name"] == selected_name]
fig.add_scattergeo(
    lat=selected_row["latitude_deg"],
    lon=selected_row["longitude_deg"],
    text=selected_row["name"],
    mode="markers",
    marker=dict(size=14, color="red"),
    name="Selected",
)

st.plotly_chart(fig, use_container_width=True)