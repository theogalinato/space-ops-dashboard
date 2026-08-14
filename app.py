import streamlit as st
import pandas as pd
from satellite_data import ts, load_tle_group, compute_subpoints, compute_orbital_params, classify_orbit_regime
from passes import get_next_n_passes, get_static_visibility, CITIES, MIN_ELEVATION_DEG
from catalogue import load_catalogue, get_catalogue_satellites, merge_satellite_lists
from space_weather import get_kp_index, get_xray_flux, get_solar_wind
from space_weather_status import classify_geomagnetic_status, classify_radio_blackout_status
from operational_assessment import assess_geomagnetic_impact, assess_radio_blackout_impact, DISCLAIMER
import plotly.express as px

st.set_page_config(page_title="Space Operations Dashboard", layout="wide")
st.title("Space Operations Dashboard")

# ============================================================
# Day 20: caching + real-time refresh.
#
# The organizing idea is DATA CADENCE vs. DISPLAY CADENCE -- how often the
# underlying source actually produces new information, versus how often
# the screen redraws. They are not the same number, and treating them as
# if they were is how a dashboard ends up hammering someone else's free
# service for data that hasn't changed since the last request.
#
#   source           new data appears     cached for    screen redraws
#   -------------------------------------------------------------------
#   CelesTrak TLEs   a few times a day    6 hours       n/a (not drawn
#                                                       directly)
#   NOAA SWPC        about once a minute  60 seconds    every 30 s
#   satellite        continuously, but    NOT cached    every 10 s
#     positions      computed locally     at all
#
# Satellite positions are the interesting case. They change continuously,
# but they are not FETCHED from anywhere -- they're SGP4 propagation run
# locally against TLEs already in memory. So the right split is: cache the
# expensive network fetch aggressively (6 h), recompute the cheap local
# math often (10 s). Caching positions would be actively counterproductive
# -- compute_subpoints() takes the timestamp as an argument, so the
# timestamp would land in the cache key, every tick would be a guaranteed
# miss, and all the caching would buy is hashing overhead on top of the
# same work. There is deliberately no cached position function below.
#
# Note also what does NOT refresh: pass predictions and the catalogue
# table. Passes are computed by searching a 48-hour window for elevation
# threshold crossings -- expensive, and the answer barely moves minute to
# minute. Recomputing that every 10 seconds would burn real CPU to
# redisplay the same table. Choosing what not to refresh is part of the
# design, not an omission.
# ============================================================

TLE_TTL_SECONDS = 6 * 60 * 60      # CelesTrak republishes a few times daily
SWPC_TTL_SECONDS = 60              # NOAA SWPC feeds update about once a minute
POSITION_REFRESH_SECONDS = 10      # display cadence: live positions/metrics
SWPC_REFRESH_SECONDS = 30          # display cadence: space weather panel


# ------------------------------------------------------------
# Cached loaders.
#
# These decorators live HERE, in app.py, and deliberately not in
# satellite_data.py / space_weather.py / catalogue.py. Those modules are
# plain Python and know nothing about Streamlit. Keeping @st.cache_* at
# the app layer is what lets globe.py and every test_*.py import the same
# orbital math and run it from a bare `python foo.py` with no Streamlit
# runtime anywhere in the picture.
# ------------------------------------------------------------

@st.cache_data
def get_catalogue_df() -> pd.DataFrame:
    """
    The curated Canadian asset catalogue CSV (Day 7).

    @st.cache_data rather than cache_resource: a DataFrame serializes
    fine, and cache_data hands back a COPY on each call, so if any part of
    the app ever mutates this frame it can't corrupt what the next caller
    sees. No ttl -- this is a static file committed to the repo, not a
    live feed, so it can never go stale within a session.
    """
    return load_catalogue()


@st.cache_resource(ttl=TLE_TTL_SECONDS)
def get_satellite_pool() -> list:
    """
    The full trackable pool: CelesTrak's 'visual' group merged with the
    Canadian asset catalogue (the Day 12 pattern).

    @st.cache_RESOURCE, not cache_data, and the distinction is the whole
    reason this function exists. cache_data serializes (pickles) whatever
    it returns; these are Skyfield EarthSatellite objects wrapping SGP4
    satrec objects from a C extension, which don't pickle cleanly.
    cache_resource stores the object itself with no copy, which is exactly
    right for a large read-only pool that every panel shares.

    ttl of 6 hours matches the DATA cadence: CelesTrak republishes element
    sets a few times a day, so refetching per page load -- never mind per
    refresh tick -- would be pure waste against someone else's free
    service, for element sets that are byte-for-byte identical.
    """
    visual = load_tle_group("visual")
    catalogue_sats = get_catalogue_satellites(get_catalogue_df())
    return merge_satellite_lists(visual, catalogue_sats)


@st.cache_data(ttl=SWPC_TTL_SECONDS)
def get_space_weather_frames():
    """
    All three NOAA SWPC feeds (Day 15) behind one cached call. Plain
    DataFrames, so cache_data is the right decorator.

    ttl is 60 s against a 30 s display cadence, which is the data-vs-
    display split made concrete: roughly every second refresh tick is a
    cache hit that touches no network at all. The panel stays responsive
    without pretending NOAA publishes faster than it does.

    Streamlit does not cache exceptions -- if SWPC is unreachable the
    error is re-raised and retried on the next call rather than being
    pinned for the full ttl. That's the behaviour we want here: a
    momentary network blip shouldn't blank the tab for a full minute.
    """
    return get_kp_index(), get_xray_flux(), get_solar_wind()


# ------------------------------------------------------------
# Live fragments.
#
# st.fragment(run_every=...) reruns ONLY the decorated function on a
# timer, not the whole script. That distinction is the entire reason to
# use it instead of a whole-app autorefresh: a full rerun would re-run TLE
# loading, rebuild all four tabs, and reset widget state on every tick. A
# fragment redraws one panel and leaves the rest of the page -- including
# the user's search box, selectbox, and slider -- untouched.
#
# Each fragment calls ts.now() ITSELF instead of closing over a timestamp
# computed once at page load. This is subtle and load-bearing: when a
# fragment auto-reruns, the surrounding script does NOT re-execute, so a
# page-level `t = ts.now()` stays frozen at whenever the page last fully
# loaded. The map would faithfully redraw every 10 seconds and re-plot the
# same stale instant forever -- a live-looking dashboard showing a fixed
# moment, which is worse than an obviously static one.
#
# This is NOT a walk-back of Day 11's one-clock fix. Day 11 was about not
# instantiating two Timescale OBJECTS; `ts` is still the single shared
# Timescale imported from satellite_data.py. Each fragment asks that one
# clock what time it is now. One clock, many readings.
# ------------------------------------------------------------

@st.fragment(run_every=POSITION_REFRESH_SECONDS)
def live_orbital_metrics(sat):
    """Live orbital parameters for the selected satellite."""
    t_now = ts.now()
    params = compute_orbital_params(sat, t_now)
    altitude_km = compute_subpoints([sat], t_now)["altitude_km"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Altitude", f"{altitude_km:.0f} km")
    col2.metric("Inclination", f"{params['inclination_deg']:.2f}°")
    col3.metric("Period", f"{params['period_min']:.1f} min")
    col4.metric("Speed", f"{params['speed_km_s']:.2f} km/s")

    st.caption(
        f"Live as of {t_now.utc_strftime('%H:%M:%S')} UTC, recomputed every "
        f"{POSITION_REFRESH_SECONDS} s. Inclination and period come from the "
        f"TLE's mean elements and are effectively constant; altitude and "
        f"speed genuinely vary around the orbit, so those are the two that "
        f"should visibly move."
    )


@st.fragment(run_every=POSITION_REFRESH_SECONDS)
def live_map(satellites, selected_name):
    """Live subpoint map of the whole tracked population."""
    t_now = ts.now()
    df = compute_subpoints(satellites, t_now)

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

    # uirevision preserves the user's zoom/pan across refreshes. Without
    # it, Plotly resets the view every time the figure object is replaced
    # -- which is now every POSITION_REFRESH_SECONDS, so a user mid-zoom
    # would get yanked back to the default view on every tick. The value
    # only has to stay CONSTANT across reruns; any stable string works.
    # This is a correctness fix for interactive use, not cosmetic.
    fig.update_layout(uirevision="satellite-map")

    st.plotly_chart(fig, key="live_map_chart")
    st.caption(
        f"{len(satellites)} tracked objects: CelesTrak 'visual' group + the "
        f"Canadian asset catalogue. Positions recomputed every "
        f"{POSITION_REFRESH_SECONDS} s from TLEs cached up to "
        f"{TLE_TTL_SECONDS // 3600} h -- display cadence and data cadence are "
        f"deliberately different. As of {t_now.utc_strftime('%H:%M:%S')} UTC."
    )


@st.fragment(run_every=SWPC_REFRESH_SECONDS)
def live_space_weather():
    """
    Space weather panel: raw SWPC readings, the Day 16 status banners, and
    the Day 17 operational assessment.

    Worth knowing: Streamlit renders every tab's contents whether or not
    that tab is the one on screen (tabs hide content with CSS, they don't
    lazily render it). So this fragment keeps ticking while the user is
    looking at the Map tab. That's affordable precisely because of the
    cache -- most ticks are a cache hit and touch no network at all --
    but it's a real cost worth knowing about rather than discovering.
    """
    st.caption(
        "Raw NOAA SWPC data, a rule-based status classification, and a "
        "plain-language operational impact assessment. This is a "
        "simplified educational tool, not a real space weather warning "
        "system -- see the disclaimer below the status banners."
    )

    try:
        kp_df, xray_df, wind_df = get_space_weather_frames()
    except Exception as exc:
        st.error(
            f"Could not reach NOAA SWPC: {exc}. This tab depends on live "
            f"network access to services.swpc.noaa.gov; if this is a "
            f"connectivity problem rather than a code problem, the rest of "
            f"the app (Map/Catalogue/Next Passes) is unaffected since they "
            f"talk to CelesTrak, a completely separate service."
        )
        return

    latest_kp = kp_df.iloc[-1]
    latest_xray = xray_df.iloc[-1]
    latest_wind = wind_df.iloc[-1]

    # Day 16: two independent status bands, not one fused "overall"
    # number. Geomagnetic activity and radio blackout risk are different
    # physical hazards driven by different mechanisms (magnetospheric
    # coupling vs. sunlit-ionosphere X-ray exposure), so collapsing them
    # into a single banner would hide which hazard is actually elevated.
    # Solar wind/Bz has no formal NOAA scale this project can reduce (that
    # needs >=10 MeV proton flux data this project doesn't ingest), so it
    # stays informational-only below rather than getting an invented
    # threshold.
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
    # scope names. Shown right after the status banners and before the raw
    # metrics/charts below, so an operator gets the bottom line first and
    # the supporting data second, rather than having to read numbers
    # before getting to what they mean.
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

    # Timestamps here are the OBSERVATION times from NOAA's own feeds, not
    # the time this panel last redrew -- those are different things, and
    # for space weather the observation time is the one that matters
    # operationally. Panel refresh cadence is stated separately below.
    st.caption(
        f"Kp as of {latest_kp['time_utc']:%Y-%m-%d %H:%M} UTC. "
        f"X-ray as of {latest_xray['time_utc']:%Y-%m-%d %H:%M} UTC. "
        f"Solar wind as of {latest_wind['time_utc']:%Y-%m-%d %H:%M} UTC. "
        f"Panel refreshes every {SWPC_REFRESH_SECONDS} s against feeds "
        f"cached for {SWPC_TTL_SECONDS} s."
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


# ============================================================
# Page body.
# ============================================================

# Day 12: the trackable pool is the 'visual' group plus the 8 Canadian
# catalogue satellites (none of which are members of 'visual' -- confirmed
# on Day 12). Day 20: both loads now go through the cached accessors above
# rather than hitting the network on every page load.
satellites = get_satellite_pool()
catalogue_df = get_catalogue_df()

# Day 19: classify_orbit_regime moved to satellite_data.py so globe.py can
# share the exact same LEO/MEO/GEO thresholds instead of a second copy
# silently drifting from this one.

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

# Live metrics panel (Day 20 fragment -- refreshes on its own timer).
live_orbital_metrics(selected_sat)

# Orbit regime and TLE age are computed at PAGE level, not inside a
# fragment, on purpose: both are effectively static for a given satellite
# over a viewing session (a satellite does not change orbit regime, and
# TLE age creeps by seconds), and `regime` drives which branch the Next
# Passes tab takes below -- so it needs to exist in page scope regardless.
t_page = ts.now()
altitude_km = compute_subpoints([selected_sat], t_page)["altitude_km"].iloc[0]
regime = classify_orbit_regime(altitude_km)

st.caption(f"Orbit regime: {regime}")

age_days = t_page - selected_sat.epoch
if age_days > 3:
    st.warning(f"TLE is {age_days:.1f} days old — position accuracy may be degraded.")
else:
    st.caption(f"TLE age: {age_days:.1f} days")

# ============================================================
# Day 13: four tabs instead of one long vertical scroll. Map/Catalogue/
# Passes are logically separate operator questions ("where is it," "what
# do we have," "when's it overhead") and don't need to share screen space
# -- the search/select panel and metrics above stay outside the tabs since
# they drive all of them.
# ============================================================
map_tab, catalogue_tab, passes_tab, weather_tab = st.tabs(
    ["Map", "Canadian Asset Catalogue", "Next Passes", "Space Weather"]
)

with map_tab:
    live_map(satellites, selected_name)

# ============================================================
# Day 12: Canadian asset catalogue -- filterable reference table.
# Static curated data (Day 7 CSV), not live-computed, and deliberately not
# wrapped in a refresh fragment -- a CSV in the repo cannot change while
# the app is running. Filter options are pulled from the data itself
# rather than hardcoded, so this doesn't silently break if a category is
# renamed or a satellite is added/removed from the CSV later.
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
    st.dataframe(filtered_catalogue, hide_index=True)

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
# Day 20: deliberately NOT a refresh fragment. Pass prediction searches a
# 48-hour window for elevation threshold crossings -- expensive, and the
# answer barely moves minute to minute. This recomputes when the user
# changes satellite, site, or count, which is when it can actually differ.
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
                f"{selected_city} at roughly {vis['elevation_deg']}° elevation "
                f"(azimuth {vis['azimuth_deg']}°). GEO satellites don't have "
                f"discrete passes; this is a parked, roughly-constant view, not "
                f"a live sighting."
            )
        else:
            st.warning(
                f"{selected_sat.name} is geostationary and sits at roughly "
                f"{vis['elevation_deg']}° elevation from {selected_city} -- below "
                f"the {MIN_ELEVATION_DEG}° usable threshold, and it will stay "
                f"there. This isn't 'no passes yet' -- it's not reachable from "
                f"this site at all, by geometry."
            )
    else:
        city_passes = get_next_n_passes(selected_sat, selected_city, n=n_passes)

        if not city_passes:
            st.info(
                f"No passes above {MIN_ELEVATION_DEG:.0f}° elevation for "
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
            st.dataframe(passes_df, hide_index=True)

# ============================================================
# Day 15: raw NOAA SWPC ingestion and display.
# Day 16: the two LOW/MODERATE/HIGH status banners.
# Day 17 (protected): the Operational Impact panel -- the differentiator.
# Day 20: the whole panel moved into a refresh fragment on its own cadence
# (see live_space_weather above).
# ============================================================
with weather_tab:
    live_space_weather()