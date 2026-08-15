"""
Space Operations Dashboard -- Streamlit application.

The single entry point that ties every math/data module in this project
into one interactive tool: `streamlit run app.py`. Everything Streamlit-
specific (page layout, caching, live-refresh fragments, tabs, widgets)
lives in this file on purpose, so `satellite_data.py`, `catalogue.py`,
`passes.py`, `space_weather.py`, `space_weather_status.py`,
`operational_assessment.py`, and `conjunction.py` stay plain Python that
knows nothing about Streamlit and can be imported and tested (or run
standalone, like `globe.py`) with no Streamlit runtime involved.

Five tabs, each answering a different operator question, in this order:
Map (where are satellites right now, 2D or 3D), Canadian Asset Catalogue
(what does Canada have up there), Next Passes (when is a satellite
overhead a given site), Conjunction Screening (is a Canadian asset's
neighbourhood clear, per the reduced screening heuristic -- see
`conjunction.py`'s DISCLAIMER), and Space Weather (what's the current
space environment and what does it mean operationally, per
`operational_assessment.py`'s DISCLAIMER).

Caching (`get_catalogue_df`, `get_earth_sphere`, `get_satellite_pool`,
`get_space_weather_frames`) and the four live-refresh fragments
(`live_orbital_metrics`, `live_map`, `live_globe`, `live_space_weather`)
each run on their own display cadence, separate from how often the
underlying data actually changes -- see the README's Day 20 entry for why
that distinction matters and how the page-level clock is kept from
freezing inside a fragment.

This is a public-data educational prototype, not a real military,
intelligence, or operational CAF/3 CSD system. See the README's honesty
constraints and each tab's on-screen disclaimer for specifics.
"""

import streamlit as st
import pandas as pd
from satellite_data import (
    ts,
    load_tle_group,
    compute_subpoints,
    compute_orbital_params,
    classify_orbit_regime,
    compute_ecef_positions,
    compute_orbit_arc,
)
from earth_mesh import build_earth_sphere, build_coastlines
from passes import get_next_n_passes, get_static_visibility, CITIES, MIN_ELEVATION_DEG
from catalogue import load_catalogue, get_catalogue_satellites, merge_satellite_lists
from space_weather import get_kp_index, get_xray_flux, get_solar_wind
from space_weather_status import classify_geomagnetic_status, classify_radio_blackout_status
from operational_assessment import (
    assess_geomagnetic_impact,
    assess_radio_blackout_impact,
    DISCLAIMER as SPACE_WEATHER_DISCLAIMER,
)
from conjunction import (
    bound_population,
    screen_conjunctions,
    DEFAULT_MARGIN_KM,
    DEFAULT_WINDOW_HOURS,
    DEFAULT_STEP_SECONDS,
    HIGH_RISK_MAX_KM,
    MODERATE_RISK_MAX_KM,
    DISCLAIMER as CONJUNCTION_DISCLAIMER,
)
import plotly.express as px
import plotly.graph_objects as go

# Day 24: both operational_assessment.py and conjunction.py export their own
# module-level DISCLAIMER constant (same pattern, deliberately -- see Day 17
# and Day 23) -- imported above with distinct names rather than letting the
# second import silently shadow the first, which would have been a subtle
# bug (the Space Weather tab would start showing the conjunction-screening
# disclaimer instead of its own).

st.set_page_config(
    page_title="Space Operations Dashboard",
    layout="wide",
    # Day 26: explicit rather than "auto" -- the sidebar now holds the
    # satellite search/select/metrics panel every tab depends on, so it
    # should never start collapsed on first load.
    initial_sidebar_state="expanded",
)
st.title("Space Operations Dashboard")
st.caption(
    "A public-data space domain awareness prototype -- not a real "
    "military, intelligence, or operational CAF/3 CSD system. See the "
    "sidebar and each tab's disclaimer for specifics."
)

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

# Day 16 originally, promoted to module level Day 24: a shared LOW/MODERATE/
# HIGH -> Streamlit-banner-function mapping. Lived as a local inside
# live_space_weather() until the Day 24 conjunction tab needed the exact
# same three-way mapping for its own risk_level banner -- promoted rather
# than copy-pasted, same instinct as Day 19's classify_orbit_regime and
# Day 23's build_time_array promotions for the same reason.
_STATUS_RENDER = {"LOW": st.success, "MODERATE": st.warning, "HIGH": st.error}


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


@st.cache_data
def get_earth_sphere():
    """
    Day 21: the Earth sphere mesh (Day 18's earth_mesh.py) that backs the
    3D globe view. cache_data, not cache_resource -- build_earth_sphere()
    returns plain numpy arrays (they pickle fine), and there's no
    C-extension object here the way there is for the satellite pool below.

    No ttl, same reasoning as get_catalogue_df: this geometry has no time
    dependence at all (it's a static WGS84-equatorial-radius sphere, not a
    live computation), so it can never go stale within a session.
    """
    return build_earth_sphere(resolution=50)


@st.cache_data
def get_coastlines():
    """
    Day 27: the bundled coastline outline data (earth_mesh.py's
    build_coastlines(), Natural Earth 110m, no network dependency) for the
    3D globe. Same reasoning as get_earth_sphere just above -- this is
    static geometry with no time dependence, parsed from a file that ships
    in the repo, so there's no reason to re-parse and re-project ~5,000
    points on every live_globe refresh tick when the result never changes
    within a session.
    """
    return build_coastlines()


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

    # Day 26: single column, not a 2x2 or 4-across grid. A 2x2 grid was
    # tried first (reasonable-looking guess for the sidebar's ~300px), but
    # an actual rendered screenshot (not just eyeballing the code) showed
    # "92.9 min" and "7.78 km/s" truncating with an ellipsis at that
    # column width -- st.metric's value text is large by default and two
    # columns just isn't enough room once a unit is appended. Stacked
    # single-column costs vertical space, which the sidebar has plenty of,
    # in exchange for every value actually being readable.
    st.metric("Altitude", f"{altitude_km:.0f} km")
    st.metric("Inclination", f"{params['inclination_deg']:.2f}°")
    st.metric("Period", f"{params['period_min']:.1f} min")
    st.metric("Speed", f"{params['speed_km_s']:.2f} km/s")

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
    # Day 26: explicit blue rather than Plotly's default marker color, so
    # "the whole tracked population" reads as one consistent color across
    # this map and the 3D globe's legend, instead of two different blues
    # that happen to both be called "default."
    fig.update_traces(marker=dict(color="#3987e5", size=5))

    # Day 27: back to red, by request, after Day 26 had briefly moved this
    # to white over a concern about colliding with red's HIGH/critical
    # status meaning elsewhere. Red is also simply more reliable here
    # across both the light and dark themes Day 27 brought back -- a white
    # marker on the (now possible) light theme's near-white background was
    # verified nearly invisible, where red reads clearly against light,
    # dark, and the navy globe sphere alike.
    selected_row = df[df["name"] == selected_name]
    fig.add_scattergeo(
        lat=selected_row["latitude_deg"],
        lon=selected_row["longitude_deg"],
        text=selected_row["name"],
        mode="markers",
        marker=dict(size=14, color="red"),
        name="Selected",
    )

    # Day 26: dark land/ocean colors so this map matches the app's dark
    # theme instead of sitting as a light-mode island inside it. This is
    # NOT redundant with Streamlit's automatic plotly theming (st.plotly_
    # chart re-themes general chart chrome -- background, fonts, colorway
    # -- to match .streamlit/config.toml on its own) -- geo subplot
    # properties like land/ocean color are geography-specific and aren't
    # part of that automatic theming, so they need setting explicitly.
    # Ocean reuses the exact navy from the 3D globe's Earth sphere
    # (earth_mesh.py / globe.py's "rgb(25,55,109)"), so the 2D and 3D
    # views read as the same planet rather than two different palettes.
    fig.update_geos(
        showland=True,
        showframe=False,
        bgcolor="#1a1a19",
        landcolor="#33332d",
        oceancolor="rgb(25,55,109)",
        showocean=True,
        showlakes=True,
        lakecolor="rgb(25,55,109)",
        showcountries=True,
        countrycolor="#44443d",
        coastlinecolor="#55554d",
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


@st.fragment(run_every=POSITION_REFRESH_SECONDS)
def live_globe(satellites, selected_sat, selected_name, regime_filter):
    """
    Day 21: globe.py's Day 18/19 technique (Earth sphere + ECEF satellite
    positions, split into LEO/MEO/GEO traces, plus an orbit arc for the
    highlighted satellite), wired into the app as a second view alongside
    live_map instead of staying a standalone script.

    Deliberately NOT a replacement for live_map -- the Map tab now offers
    both behind a toggle (see page body below), since the two answer
    different operator questions: "where is this over Canadian territory"
    (2D, live_map) versus "what does the orbital environment actually look
    like" (3D, this function). Same POSITION_REFRESH_SECONDS cadence as
    live_map, since it depends on the same live position computation.

    regime_filter lets the LEO/MEO/GEO scale problem from Day 19's Real
    Findings (GEO sits ~5.6x farther from Earth's center than a typical LEO
    satellite, so no single camera makes both read at equal detail) be
    stepped around interactively -- e.g. filter to LEO only to actually see
    the local shell -- rather than only described in a docstring.
    """
    t_now = ts.now()
    ecef_df = compute_ecef_positions(satellites, t_now)
    subpoint_df = compute_subpoints(satellites, t_now)
    df = ecef_df.merge(subpoint_df[["catnr", "altitude_km"]], on="catnr")
    df["regime"] = df["altitude_km"].apply(classify_orbit_regime)
    df = df[df["regime"].isin(regime_filter)]

    ex, ey, ez = get_earth_sphere()
    coast_x, coast_y, coast_z = get_coastlines()

    fig = go.Figure()

    # Solid-color sphere, same as globe.py -- a full photographic texture
    # is real extra machinery (Plotly's indexed-colorscale trick) for a
    # look that's mostly visual polish; coastline outlines just below get
    # most of the "recognizable planet" benefit for far less effort.
    fig.add_trace(go.Surface(
        x=ex, y=ey, z=ez,
        colorscale=[[0, "rgb(25,55,109)"], [1, "rgb(25,55,109)"]],
        showscale=False,
        opacity=1.0,
        hoverinfo="skip",
        name="Earth",
    ))

    # Day 27: coastline outlines -- see globe.py's matching comment and
    # earth_mesh.py's build_coastlines() for the full reasoning (bundled
    # Natural Earth data, no network dependency, deliberately using the
    # sphere's own simplified-sphere formula so the lines land exactly on
    # this sphere rather than up to ~21 km off it).
    fig.add_trace(go.Scatter3d(
        x=coast_x, y=coast_y, z=coast_z,
        mode="lines",
        line=dict(color="rgb(150,165,190)", width=1.5),
        opacity=0.9,
        hoverinfo="skip",
        name="Coastlines",
        showlegend=False,
    ))

    # One trace per regime, same reasoning as globe.py: lumping LEO/MEO/GEO
    # into one trace would hide the scale difference this view exists to
    # show honestly.
    #
    # Day 26: colors kept in sync with globe.py's own _REGIME_STYLE -- see
    # that file's comment for the full reasoning (orange/gold failed a
    # colorblind + contrast validator's plain normal-vision check; silver
    # stays for LEO as a deliberate, render-and-checked exception, not an
    # oversight). Two copies of this dict exist on purpose (globe.py stays
    # a standalone, Streamlit-free proof-of-concept per Day 21's own note
    # below) -- keep them matching by hand if either changes.
    #
    # Day 27: MEO moved from Day 26's #d95926 (a validated orange) to
    # blue. Not a validator failure this time -- it passed every real
    # check -- but a real-world one: at marker size and a glance, that
    # orange read close enough to red to raise a mix-up concern once red
    # went back to meaning "selected" (see below). Blue reuses the exact
    # hex already used for "the general population" on the 2D map, and
    # re-validated cleanly against silver/aqua in both the light and dark
    # themes Day 27 brought back (all-pairs check, both surfaces).
    _REGIME_STYLE = {
        "LEO": dict(color="silver", size=2.5),
        "MEO": dict(color="#3987e5", size=4),
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

    # Highlight + orbit arc for whichever satellite is selected up top --
    # globe.py hardcodes RADARSAT-2 since it's a standalone proof-of-concept;
    # here the app's own selection drives it, same as live_map's marker.
    # If the regime filter excludes the selected satellite's own regime, it
    # simply won't appear -- flagged below rather than silently empty.
    #
    # Day 27: back to red, by request. See live_map's matching comment --
    # red is also just more reliable across both themes than white turned
    # out to be (verified nearly invisible against the light theme's
    # near-white page).
    selected_row = df[df["name"] == selected_name]
    if not selected_row.empty:
        fig.add_trace(go.Scatter3d(
            x=selected_row["x_km"], y=selected_row["y_km"], z=selected_row["z_km"],
            text=[selected_name],
            mode="markers+text",
            textposition="top center",
            marker=dict(size=7, color="red", symbol="diamond"),
            name=f"{selected_name} (selected)",
        ))

        arc_df = compute_orbit_arc(selected_sat, t_now)
        fig.add_trace(go.Scatter3d(
            x=arc_df["x_km"], y=arc_df["y_km"], z=arc_df["z_km"],
            mode="lines",
            line=dict(color="red", width=3),
            opacity=0.6,
            name=f"{selected_name} orbit arc (1 period, ECEF)",
            hoverinfo="skip",
        ))
    else:
        st.info(
            f"{selected_name} is outside the selected regime filter, so it "
            f"isn't shown on this globe -- widen the filter to bring it back."
        )

    # aspectmode="data" and the camera eye vector are copied verbatim from
    # globe.py's Day 19 finding: no single static camera makes LEO and GEO
    # both read at equal detail (real geometry, not a framing bug), so this
    # is a deliberately LEO/MEO-favoring starting view, not the only view --
    # drag to rotate, scroll to zoom.
    #
    # uirevision is the Day 20 pattern applied to a 3D scene instead of a 2D
    # scattergeo: without it, the user's rotate/zoom would reset to this
    # default every POSITION_REFRESH_SECONDS.
    fig.update_layout(
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
        uirevision="satellite-globe",
        margin=dict(l=0, r=0, t=10, b=0),
    )

    st.plotly_chart(fig, key="live_globe_chart")
    st.caption(
        f"{len(df)} of {len(satellites)} tracked objects shown (regime filter "
        f"below). Positions recomputed every {POSITION_REFRESH_SECONDS} s from "
        f"TLEs cached up to {TLE_TTL_SECONDS // 3600} h. As of "
        f"{t_now.utc_strftime('%H:%M:%S')} UTC. Earth is a WGS84-equatorial-"
        f"radius sphere, not a true oblate spheroid -- see earth_mesh.py."
    )


def render_conjunction_tab(satellites, catalogue_df, t_page):
    """
    Day 24: reduced conjunction screening, wired into the app for the first
    time. Deliberately NOT an @st.fragment, unlike live_map/live_globe/
    live_space_weather above -- same reasoning Day 20 already recorded for
    passes_tab below: screening is comparatively expensive (propagating
    every survivor over up to thousands of timesteps) and the answer
    doesn't meaningfully change on a 10-second cadence the way a live
    position does. This recomputes when the user changes the primary
    satellite or a screening parameter -- which is when the answer can
    actually differ -- not on a timer. It reuses t_page (computed once at
    page level, same clock as the TLE-age check above) rather than calling
    ts.now() itself, since there's no fragment boundary here for a frozen
    page-level timestamp to be a problem across.

    Primary choices are restricted to the Canadian catalogue, not the full
    ~170-object tracked population: conjunction.py's own framing is "a
    PRIMARY satellite (a Canadian asset worth protecting)," and this UI
    holds to that rather than offering every tracked object as a possible
    primary, most of which aren't Canadian assets this project has any
    stake in protecting. The candidate POPULATION being screened against,
    on the other hand, is still the full tracked pool -- a conjunction
    doesn't care whether the other object is Canadian.
    """
    st.caption(CONJUNCTION_DISCLAIMER)

    primary_name = st.selectbox(
        "Primary asset (the Canadian satellite being screened)",
        catalogue_df["name"].tolist(),
        help=(
            "Screening runs Day 22's altitude-band filter, then Day 23's "
            "propagated minimum-separation search, against every other "
            "object in the tracked population."
        ),
    )
    primary_catnr = int(catalogue_df.loc[catalogue_df["name"] == primary_name, "catnr"].iloc[0])
    primary_sat = next(sat for sat in satellites if sat.model.satnum == primary_catnr)

    with st.expander("Screening parameters (advanced)"):
        st.caption(
            "These are heuristic knobs, not physical constants -- adjusting "
            "them changes what this tool flags, not what's actually true in "
            "orbit. In particular, try widening the time-grid step on a "
            "close result: a fast-crossing pair's reported minimum can "
            "visibly shift between step sizes -- that's the grid-sampling "
            "limitation in the disclaimer above showing up directly, not a "
            "bug."
        )
        margin_km = st.slider(
            "Day 22 altitude-band margin (km)",
            min_value=0, max_value=200, value=int(DEFAULT_MARGIN_KM), step=5,
            help="How far outside the primary's own altitude band a candidate can sit and still be screened.",
        )
        window_hours = st.slider(
            "Day 23 look-ahead window (hours)",
            min_value=1, max_value=72, value=int(DEFAULT_WINDOW_HOURS),
            help="How far forward in time to search for a close approach. Longer windows compound TLE staleness.",
        )
        step_seconds = st.select_slider(
            "Day 23 time-grid step (seconds)",
            options=[5, 10, 30, 60, 120, 300], value=DEFAULT_STEP_SECONDS,
            help="Finer steps catch faster/closer approaches but cost more compute -- see the grid-sampling limitation above.",
        )

    survivors_df = bound_population(primary_sat, satellites, margin_km=margin_km)

    if survivors_df.empty:
        st.info(
            f"No tracked objects fall within {margin_km:.0f} km of "
            f"{primary_name}'s altitude band. This is a legitimate result, "
            f"not an error -- {primary_name} simply has no altitude "
            f"neighbours in the current tracked population at this margin."
        )
        return

    survivor_catnrs = set(survivors_df["catnr"])
    survivor_sats = [sat for sat in satellites if sat.model.satnum in survivor_catnrs]

    results_df = screen_conjunctions(
        primary_sat, survivor_sats, t_page,
        window_hours=window_hours, step_seconds=step_seconds,
    )

    # Results are already sorted ascending by min_separation_km (see
    # screen_conjunctions' docstring) -- row 0 is the most concerning.
    worst = results_df.iloc[0]
    _STATUS_RENDER[worst["risk_level"]](
        f"Closest predicted approach: **{worst['name']}** at "
        f"{worst['min_separation_km']:.2f} km on "
        f"{worst['time_of_closest_approach_utc']:%Y-%m-%d %H:%M} UTC -- "
        f"{worst['risk_level']}"
    )

    counts = results_df["risk_level"].value_counts()
    count_col1, count_col2, count_col3 = st.columns(3)
    count_col1.metric("HIGH", int(counts.get("HIGH", 0)))
    count_col2.metric("MODERATE", int(counts.get("MODERATE", 0)))
    count_col3.metric("LOW", int(counts.get("LOW", 0)))

    display_df = results_df.rename(columns={
        "name": "Object",
        "catnr": "NORAD Catalog #",
        "min_separation_km": "Min Separation (km)",
        "time_of_closest_approach_utc": "Time of Closest Approach (UTC)",
        "risk_level": "Risk Level",
    })
    st.dataframe(display_df, hide_index=True)

    st.caption(
        f"{len(survivors_df)} of {len(satellites)} tracked objects survived "
        f"the Day 22 altitude-band filter (±{margin_km:.0f} km of "
        f"{primary_name}'s band); all {len(results_df)} were propagated over "
        f"{window_hours:.0f}h at a {step_seconds}s step and classified. "
        f"Screening evaluated as of {t_page.utc_strftime('%Y-%m-%d %H:%M:%S')} "
        f"UTC. Risk bands: HIGH < {HIGH_RISK_MAX_KM:.0f} km, MODERATE "
        f"{HIGH_RISK_MAX_KM:.0f}–{MODERATE_RISK_MAX_KM:.0f} km, LOW "
        f"≥ {MODERATE_RISK_MAX_KM:.0f} km."
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
    st.caption(SPACE_WEATHER_DISCLAIMER)
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
    # Day 21: log scale, fixing the Day 15 flag. st.line_chart can't set
    # axis scale at all, which is why this one chart is a Plotly figure
    # (px.line) while the other space-weather charts below stay
    # st.line_chart -- Kp, solar wind speed/density, and Bz don't span
    # orders of magnitude the way flux does (A-class to X-class flux
    # covers roughly 1e-8 to 1e-3 W/m^2), so linear is still the honest
    # read for those.
    xray_fig = px.line(xray_df, x="time_utc", y="flux_w_m2")
    xray_fig.update_yaxes(type="log", title="Flux (W/m^2, log scale)")
    xray_fig.update_xaxes(title="Time (UTC)")
    xray_fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        uirevision="xray-flux-chart",
    )
    st.plotly_chart(xray_fig, key="xray_flux_chart")
    st.caption(
        "Log scale, fixed Day 21. X-ray flux spans several orders of "
        "magnitude across flare classes (A/B/C/M/X) -- on the linear scale "
        "shipped Day 15, everything below an X-class flare compressed into "
        "a flat line near zero."
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

# ============================================================
# Day 26: search/select/metrics moved into the sidebar, resolving the
# question Day 13 deliberately left open ("sidebar redesign... skipped").
# It was a real trade then (a buffer day is the wrong place for a layout
# rewrite) and a real decision now, on the day meant for exactly this:
# these controls drive every tab below, so they read more like a
# persistent instrument panel than content belonging to any one tab --
# a sidebar is the conventional place for "applies everywhere" controls,
# and it frees the tabs themselves to start right at each tab's own
# content instead of repeating the same header block underneath it.
# All the underlying variables (selected_sat, regime, t_page, age_days,
# ...) stay plain page-scope Python either way -- moving the RENDERING
# calls into `st.sidebar` doesn't change what's computed or when, only
# where it's drawn.
# ============================================================
with st.sidebar:
    st.header("Satellite Tracking")

    # Day 27: the standalone "Search satellites" text input was removed.
    # It was pre-filtering the dropdown's option list, but st.selectbox's
    # own dropdown already supports typing to filter its options once it's
    # open (a plain browser combobox behavior, not something this app
    # built) -- so the text input was a second, separate control doing a
    # job the selectbox already does on its own. One control, not two.
    selected_name = st.selectbox("Select a satellite", names)
    selected_sat = next(sat for sat in satellites if sat.name == selected_name)

    st.write(f"Tracking: **{selected_sat.name}** (NORAD {selected_sat.model.satnum})")

    # Live metrics panel (Day 20 fragment -- refreshes on its own timer).
    # Day 26: 2x2 grid rather than the 4-across row this used when it sat
    # at page width -- four st.metric columns squeezed into the sidebar's
    # ~300px would each have almost no room and wrap awkwardly.
    live_orbital_metrics(selected_sat)

    # Orbit regime and TLE age are computed at PAGE level, not inside a
    # fragment, on purpose: both are effectively static for a given
    # satellite over a viewing session (a satellite does not change orbit
    # regime, and TLE age creeps by seconds), and `regime` drives which
    # branch the Next Passes tab takes below -- so it needs to exist in
    # page scope regardless of which container renders it.
    t_page = ts.now()
    altitude_km = compute_subpoints([selected_sat], t_page)["altitude_km"].iloc[0]
    regime = classify_orbit_regime(altitude_km)

    st.caption(f"Orbit regime: {regime}")

    age_days = t_page - selected_sat.epoch
    if age_days > 3:
        st.warning(f"TLE is {age_days:.1f} days old — position accuracy may be degraded.")
    else:
        st.caption(f"TLE age: {age_days:.1f} days")

    st.divider()
    st.caption(
        "Public-data educational prototype. Conjunction screening is a "
        "simplified heuristic, not a certified Pc system -- public TLEs "
        "carry no usable covariance. Space weather assessment is "
        "simplified and educational, not a warning system. Full "
        "disclaimers are on the relevant tabs and in the README."
    )

# ============================================================
# Day 13: separate tabs instead of one long vertical scroll. Map/Catalogue/
# Passes/Conjunction are logically separate operator questions ("where is
# it," "what do we have," "when's it overhead," "is anything too close to
# it") and don't need to share screen space -- the search/select panel and
# metrics moved into the sidebar (Day 26) since they drive all of them.
# Day 24: Conjunction Screening added as its own tab, placed right after
# Next Passes rather than at the end -- it continues the same asset-focused
# question sequence (where/what/when/is-it-safe) that Map through Passes
# already builds, while Space Weather is a separate environmental-awareness
# question that reads naturally as the last stop.
# ============================================================
map_tab, catalogue_tab, passes_tab, conjunction_tab, weather_tab = st.tabs(
    ["Map", "Canadian Asset Catalogue", "Next Passes", "Conjunction Screening", "Space Weather"]
)

with map_tab:
    # Day 21: 2D/3D toggle rather than replacing the flat map, per the
    # decision recorded going into Day 20 -- the two views answer different
    # operator questions ("where is this over Canadian territory" vs. "what
    # does the orbital environment actually look like") and a toggle keeps
    # the tab count at four while only building one figure per render.
    #
    # Day 27: the toggle and the regime filter moved BELOW the chart, and
    # 3D Globe is now the default view. Streamlit executes top-to-bottom,
    # so putting the widgets below the chart they control needs a
    # placeholder: `chart_slot` reserves the chart's position in the page
    # first, the controls render (and are read) further down the script as
    # normal, and the chart is drawn into that same reserved slot last --
    # so on screen it still appears above the controls, even though the
    # code that decides its content runs after them. The live_map/
    # live_globe fragments' own `run_every` auto-refresh was verified to
    # keep redrawing into that same slot correctly across multiple ticks,
    # not just on first load, before shipping this.
    chart_slot = st.empty()

    view_mode = st.radio(
        "View",
        ["3D Globe", "2D Map"],
        index=0,
        horizontal=True,
        help=(
            "3D is the ECEF globe (Day 18-19), best for seeing true "
            "altitude and the LEO/MEO/GEO scale difference -- the default "
            "view. 2D is the flat scattergeo projection (Day 4), best for "
            "reading positions against Canadian geography. Neither "
            "replaces the other."
        ),
    )

    if view_mode == "2D Map":
        with chart_slot.container():
            live_map(satellites, selected_name)
    else:
        regime_options = ["LEO", "MEO", "GEO"]
        selected_regimes = st.multiselect(
            "Orbit regime",
            regime_options,
            default=regime_options,
            help=(
                "GEO sits roughly 5.6x farther from Earth's center than a "
                "typical LEO satellite (Day 19 finding) -- no single camera "
                "view makes both read at equal detail, so filtering to one "
                "regime is often more useful than the default drag/zoom."
            ),
        )
        with chart_slot.container():
            if not selected_regimes:
                st.warning("Select at least one orbit regime to render the globe.")
            else:
                live_globe(satellites, selected_sat, selected_name, tuple(selected_regimes))

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
        """True if a compound category tag (e.g. "EO/Defence") shares at
        least one component with the selected filter categories. Splits on
        "/" rather than treating the tag as one opaque string -- see the
        Day 12 "Real Findings" entry in the README for the bug this fixed
        (an "EO"-only filter was silently hiding every RCM satellite)."""
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
# Day 24: reduced conjunction screening. The full pipeline built across
# Days 22-23 (bound_population -> screen_conjunctions -> classify) gets a
# UI for the first time here -- see render_conjunction_tab's docstring
# above for why this is deliberately NOT a refresh fragment.
# ============================================================
with conjunction_tab:
    render_conjunction_tab(satellites, catalogue_df, t_page)

# ============================================================
# Day 15: raw NOAA SWPC ingestion and display.
# Day 16: the two LOW/MODERATE/HIGH status banners.
# Day 17 (protected): the Operational Impact panel -- the differentiator.
# Day 20: the whole panel moved into a refresh fragment on its own cadence
# (see live_space_weather above).
# ============================================================
with weather_tab:
    live_space_weather()