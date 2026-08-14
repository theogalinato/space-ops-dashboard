import streamlit as st
import pandas as pd
from satellite_data import ts, load_tle_group, compute_subpoints, compute_orbital_params
from passes import get_next_n_passes, get_static_visibility, CITIES, MIN_ELEVATION_DEG
from catalogue import load_catalogue, get_catalogue_satellites, merge_satellite_lists
from space_weather import get_kp_index, get_xray_flux, get_solar_wind
from space_weather_status import classify_geomagnetic_status, classify_radio_blackout_status
from operational_assessment import assess_geomagnetic_impact, assess_radio_blackout_impact, DISCLAIMER
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

# ============================================================
# Day 13: three tabs instead of one long vertical scroll. Map/Catalogue/
# Passes are logically separate operator questions ("where is it," "what
# do we have," "when's it overhead") and don't need to share screen space
# -- the search/select panel and metrics above stay outside the tabs since
# they drive all three.
# ============================================================
map_tab, catalogue_tab, passes_tab, weather_tab = st.tabs(
    ["Map", "Canadian Asset Catalogue", "Next Passes", "Space Weather"]
)

with map_tab:
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
    st.caption(f"{len(satellites)} tracked objects: CelesTrak 'visual' group + the Canadian asset catalogue.")

# ============================================================
# Day 12: Canadian asset catalogue -- filterable reference table.
# Static curated data (Day 7 CSV), not live-computed. Filter options are
# pulled from the data itself (df["category"].unique()) rather than
# hardcoded, so this doesn't silently break if a category is renamed
# or a satellite is added/removed from the CSV later.
# ============================================================
with catalogue_tab:
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
# Day 13: GEO satellites (e.g. Anik F2/F3, merged in on Day 12) don't have
# discrete passes -- they're roughly fixed relative to the ground, so
# find_events() never sees a threshold crossing and get_next_n_passes()
# correctly returns [] whether the satellite is continuously visible or
# never visible from the site. Route GEO through get_static_visibility()
# instead, which answers "which of those two is it" directly.
# ============================================================
with passes_tab:
    pass_col1, pass_col2 = st.columns([1, 1])
    with pass_col1:
        selected_city = st.selectbox("Site", list(CITIES.keys()))
    with pass_col2:
        n_passes = st.slider("Number of passes", min_value=1, max_value=10, value=5)

    if regime == "GEO":
        vis = get_static_visibility(selected_sat, selected_city)
        if vis["visible"]:
            st.success(
                f"{selected_sat.name} is geostationary -- continuously visible from "
                f"{selected_city} at roughly {vis['elevation_deg']}\u00b0 elevation "
                f"(azimuth {vis['azimuth_deg']}\u00b0). GEO satellites don't have "
                f"discrete passes; this is a parked, roughly-constant view, not "
                f"a live sighting."
            )
        else:
            st.warning(
                f"{selected_sat.name} is geostationary and sits at roughly "
                f"{vis['elevation_deg']}\u00b0 elevation from {selected_city} -- below "
                f"the {MIN_ELEVATION_DEG}\u00b0 usable threshold, and it will stay "
                f"there. This isn't 'no passes yet' -- it's not reachable from "
                f"this site at all, by geometry."
            )
    else:
        city_passes = get_next_n_passes(selected_sat, selected_city, n=n_passes)

        if not city_passes:
            st.info(
                f"No passes above {MIN_ELEVATION_DEG:.0f}\u00b0 elevation for "
                f"{selected_sat.name} over {selected_city} in the next 48h. This "
                f"can be real geometry (orbit/site alignment), not necessarily a "
                f"bug -- try a different satellite or city to sanity check."
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

# ============================================================
# Day 15: raw NOAA SWPC ingestion and display.
# Day 16: added the two LOW/MODERATE/HIGH status banners below, from
# space_weather_status.py. Those banners state a classification and its
# factual basis (Kp value vs. threshold, flare class vs. threshold) --
# deliberately not what that means for GNSS, HF radio, or satellite ops.
# Day 17 (protected): added the Operational Impact panel, from
# operational_assessment.py. It imports the same StatusResult objects the
# Day 16 banners already compute rather than re-deriving anything, and
# turns each one into a plain-language "so what" across the three domains
# this project's scope names (GNSS, HF radio, sat-ops). This is the
# differentiator feature -- see DISCLAIMER for why it's still clearly
# labeled a simplified educational assessment, not a real warning system.
# ============================================================
with weather_tab:
    st.caption(
        "Raw NOAA SWPC data, a rule-based status classification, and a "
        "plain-language operational impact assessment. This is a "
        "simplified educational tool, not a real space weather warning "
        "system -- see the disclaimer below the status banners."
    )

    try:
        kp_df = get_kp_index()
        xray_df = get_xray_flux()
        wind_df = get_solar_wind()
    except Exception as exc:
        st.error(
            f"Could not reach NOAA SWPC: {exc}. This tab depends on live "
            f"network access to services.swpc.noaa.gov; if this is a "
            f"connectivity problem rather than a code problem, the rest of "
            f"the app (Map/Catalogue/Next Passes) is unaffected since they "
            f"talk to CelesTrak, a completely separate service."
        )
    else:
        latest_kp = kp_df.iloc[-1]
        latest_xray = xray_df.iloc[-1]
        latest_wind = wind_df.iloc[-1]

        # Day 16: two independent status bands, not one fused "overall"
        # number. Geomagnetic activity and radio blackout risk are
        # different physical hazards driven by different mechanisms
        # (magnetospheric coupling vs. sunlit-ionosphere X-ray exposure),
        # so collapsing them into a single banner would hide which hazard
        # is actually elevated. Solar wind/Bz has no formal NOAA scale
        # this project can reduce (that needs >=10 MeV proton flux data
        # this project doesn't ingest), so it stays informational-only
        # below rather than getting an invented threshold.
        geomag_status = classify_geomagnetic_status(latest_kp["estimated_kp"])
        blackout_status = classify_radio_blackout_status(latest_xray["flare_class"])

        _STATUS_RENDER = {"LOW": st.success, "MODERATE": st.warning, "HIGH": st.error}
        banner_col1, banner_col2 = st.columns(2)
        with banner_col1:
            _STATUS_RENDER[geomag_status.level](
                f"Geomagnetic activity: {geomag_status.level} -- {geomag_status.basis}"
            )
        with banner_col2:
            _STATUS_RENDER[blackout_status.level](
                f"Radio blackout risk: {blackout_status.level} -- {blackout_status.basis}"
            )

        # Day 17 (protected): the "so what" -- plain-language operational
        # impact for each hazard, across the three domains this project's
        # scope names. Shown right after the status banners and before the
        # raw metrics/charts below, so an operator gets the bottom line
        # first and the supporting data second, rather than having to read
        # numbers before getting to what they mean.
        st.subheader('Operational Impact ("so what?")')
        st.caption(DISCLAIMER)
        geomag_impact = assess_geomagnetic_impact(geomag_status)
        blackout_impact = assess_radio_blackout_impact(blackout_status)
        impact_col1, impact_col2 = st.columns(2)
        with impact_col1:
            st.markdown(f"**{geomag_impact.hazard} -- {geomag_impact.level}**")
            st.markdown(f"- **GNSS:** {geomag_impact.gnss}")
            st.markdown(f"- **HF radio:** {geomag_impact.hf_radio}")
            st.markdown(f"- **Sat-ops:** {geomag_impact.sat_ops}")
        with impact_col2:
            st.markdown(f"**{blackout_impact.hazard} -- {blackout_impact.level}**")
            st.markdown(f"- **GNSS:** {blackout_impact.gnss}")
            st.markdown(f"- **HF radio:** {blackout_impact.hf_radio}")
            st.markdown(f"- **Sat-ops:** {blackout_impact.sat_ops}")

        weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)
        weather_col1.metric("Planetary Kp (estimated)", f"{latest_kp['estimated_kp']:.2f}")
        weather_col2.metric("X-ray flare class", latest_xray["flare_class"])
        weather_col3.metric("Solar wind speed", f"{latest_wind['proton_speed_km_s']:.0f} km/s")
        weather_col4.metric("IMF Bz (GSM)", f"{latest_wind['bz_gsm_nt']:.1f} nT")

        st.caption(
            f"Kp as of {latest_kp['time_utc']:%Y-%m-%d %H:%M} UTC. "
            f"X-ray as of {latest_xray['time_utc']:%Y-%m-%d %H:%M} UTC. "
            f"Solar wind as of {latest_wind['time_utc']:%Y-%m-%d %H:%M} UTC."
        )

        st.subheader("Planetary Kp index")
        st.line_chart(kp_df.set_index("time_utc")["estimated_kp"])

        st.subheader("GOES X-ray flux, long channel (0.1-0.8nm)")
        st.line_chart(xray_df.set_index("time_utc")["flux_w_m2"])
        st.caption(
            "Linear scale for Day 15. X-ray flux spans several orders of "
            "magnitude across flare classes, so a log-scale axis will "
            "likely read better once this tab gets more attention later; "
            "flagged here rather than fixed today."
        )

        st.subheader("Solar wind speed")
        st.line_chart(wind_df.set_index("time_utc")["proton_speed_km_s"])

        st.subheader("Solar wind density")
        st.line_chart(wind_df.set_index("time_utc")["proton_density_n_cm3"])

        st.subheader("Interplanetary magnetic field, Bz (GSM)")
        st.line_chart(wind_df.set_index("time_utc")["bz_gsm_nt"])
        st.caption(
            "Sustained negative (southward) Bz is what actually couples "
            "solar wind energy into Earth's magnetosphere and drives "
            "geomagnetic storms. Speed and density alone don't tell you "
            "that; Bz direction does."
        )