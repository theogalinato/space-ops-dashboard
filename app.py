import streamlit as st
from skyfield.api import load
from satellite_data import load_tle_group, compute_subpoints, compute_orbital_params
import plotly.express as px

st.set_page_config(page_title="Space Operations Dashboard", layout="wide")
st.title("Space Operations Dashboard")

ts = load.timescale()
t = ts.now()

# Load data
satellites = load_tle_group("visual")

# ============================================================
# ADD #1: the orbit-classification helper function.
# Put it near the top, with your other function-like code —
# right after the imports/setup, before it gets used below.
# ============================================================
def classify_orbit_regime(altitude_km: float) -> str:
    if altitude_km < 2000:
        return "LEO"
    elif altitude_km < 35000:
        return "MEO"
    else:
        return "GEO"


# Dropdown of satellite names
names = [sat.name for sat in satellites]

# ============================================================
# ADD #2: search box. This REPLACES your existing selectbox
# block — don't add it alongside the old one, swap it in.
# ============================================================
search_term = st.text_input("Search satellites", "")
if search_term:
    filtered_names = [n for n in names if search_term.lower() in n.lower()]
else:
    filtered_names = names
if not filtered_names:
    st.warning(f"No satellites in the 'visual' group match '{search_term}'.")
    st.stop()  # halts the script cleanly here instead of crashing further down

selected_name = st.selectbox("Select a satellite", filtered_names)
selected_sat = next(sat for sat in satellites if sat.name == selected_name)

# ============================================================
# ADD #3: this whole block REPLACES your existing info-panel
# code (the st.write + col1/col2/col3 + metric calls from Day 8).
# ============================================================
st.write(f"Tracking: **{selected_sat.name}** (NORAD {selected_sat.model.satnum})")

params = compute_orbital_params(selected_sat, t)
subpoint_df = compute_subpoints([selected_sat], t)
altitude_km = subpoint_df["altitude_km"].iloc[0]
regime = classify_orbit_regime(altitude_km)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Altitude", f"{altitude_km:.0f} km")
col2.metric("Inclination", f"{params['inclination_deg']:.2f}°")
col3.metric("Period", f"{params['period_min']:.1f} min")
col4.metric("Speed", f"{params['speed_km_s']:.2f} km/s")

st.caption(f"Orbit regime: {regime}")

age_days = t - selected_sat.epoch
if age_days > 3:
    st.warning(f"TLE is {age_days:.1f} days old — position accuracy may be degraded.")
else:
    st.caption(f"TLE age: {age_days:.1f} days")

# ============================================================
# UNCHANGED: your map code from Day 8 stays exactly as-is,
# below all of this.
# ============================================================
df = compute_subpoints(satellites, t)

fig = px.scatter_geo(
    df,
    lat="latitude_deg",
    lon="longitude_deg",
    hover_name="name",
    projection="natural earth",
)

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