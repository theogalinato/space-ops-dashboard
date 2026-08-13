import streamlit as st
import pandas as pd
from satellite_data import ts, load_tle_group, compute_subpoints, compute_orbital_params
from passes import get_next_n_passes, CITIES
from catalogue import load_catalogue, get_catalogue_satellites, merge_satellite_lists
import plotly.express as px

st.set_page_config(page_title="Space Operations Dashboard", layout="wide")
st.title("Space Operations Dashboard")

# Day 11 fix: use the SAME Timescale object satellite_data.py and passes.py
# already use, instead of creating a second one with load.timescale() here.
# Two Timescale instances agree numerically (same leap-second data), so this
# wasn't a correctness bug -- but it's two sources of "now" in one app, and
# that's exactly the kind of inconsistency that bites later once Day 20
# adds autorefresh/caching around time-dependent calls. One shared clock.
t = ts.now()

# Load data
satellites = load_tle_group("visual")

# Day 12: merge in the 8 Canadian catalogue satellites so they're
# selectable/trackable too (none are members of 'visual' -- confirmed).
# Bulk-fetch via the 'active' group first (Optimization 1) rather than
# 8 individual CATNR requests -- see catalogue.py for the fallback logic.
catalogue_df = load_catalogue()
catalogue_sats = get_catalogue_satellites(catalogue_df)
satellites = merge_satellite_lists(satellites, catalogue_sats)


def classify_orbit_regime(altitude_km: float) -> str:
    if altitude_km < 2000:
        return "LEO"
    elif altitude_km < 35000:
        return "MEO"
    else:
        return "GEO"


# Dropdown of satellite names
names = [sat.name for sat in satellites]

search_term = st.text_input("Search satellites", "")
if search_term:
    filtered_names = [n for n in names if search_term.lower() in n.lower()]
else:
    filtered_names = names
if not filtered_names:
    st.warning(f"No satellites in the 'visual' group match '{search_term}'.")
    st.stop()

selected_name = st.selectbox("Select a satellite", filtered_names)
selected_sat = next(sat for sat in satellites if sat.name == selected_name)

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

# Map (unchanged from Day 8)
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

# ============================================================
# Day 12: Canadian asset catalogue -- filterable reference table.
# Static curated data (Day 7 CSV), not live-computed. Filter options are
# pulled from the data itself (df["category"].unique()) rather than
# hardcoded, so this doesn't silently break if a category is renamed
# or a satellite is added/removed from the CSV later.
# ============================================================
st.divider()
st.subheader("Canadian Asset Catalogue")

categories = sorted({
    c.strip()
    for entry in catalogue_df["category"].dropna()
    for c in entry.split("/")
})
selected_categories = st.multiselect(
    "Filter by category", categories, default=categories
)

def _row_matches_selected(entry: str, selected: list[str]) -> bool:
    row_categories = {c.strip() for c in entry.split("/")}
    return bool(row_categories & set(selected))

filtered_catalogue = catalogue_df[
    catalogue_df["category"].apply(lambda x: _row_matches_selected(x, selected_categories))
]
st.dataframe(filtered_catalogue, use_container_width=True, hide_index=True)

st.caption(
    "Sapphire is Canada's dedicated space-surveillance satellite -- "
    "purpose-built to track objects in Earth orbit, distinct from the "
    "Earth-observation (RADARSAT-2, RCM) and science (NEOSSat, SCISAT) "
    "assets in this catalogue."
)

# ============================================================
# Day 11: Next-passes section for the currently selected satellite.
# ============================================================
st.divider()
st.subheader("Next Passes")

pass_col1, pass_col2 = st.columns([1, 1])
with pass_col1:
    selected_city = st.selectbox("Site", list(CITIES.keys()))
with pass_col2:
    n_passes = st.slider("Number of passes", min_value=1, max_value=10, value=5)

city_passes = get_next_n_passes(selected_sat, selected_city, n=n_passes)

if not city_passes:
    st.info(
        f"No passes above 10\u00b0 elevation for {selected_sat.name} "
        f"over {selected_city} in the next 48h. This can be real geometry "
        f"(orbit/site alignment), not necessarily a bug -- try a different "
        f"satellite or city to sanity check."
    )
else:
    passes_df = pd.DataFrame(city_passes)
    passes_df = passes_df.rename(columns={
        "rise_utc": "Rise (UTC)",
        "set_utc": "Set (UTC)",
        "duration_min": "Duration (min)",
        "max_elevation_deg": "Max Elevation (deg)",
        "culminate_azimuth_deg": "Azimuth @ Max El (deg)",
    })
    st.dataframe(passes_df, use_container_width=True, hide_index=True)