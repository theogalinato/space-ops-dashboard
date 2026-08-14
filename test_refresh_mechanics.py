"""
test_refresh_mechanics.py

Day 20 verification harness. UNLIKE every other test_*.py in this repo,
this one is a Streamlit app, not a plain script -- run it with:

    streamlit run test_refresh_mechanics.py

It exists because Day 20's claims are about RUNTIME BEHAVIOUR, not about
math. A pure-Python test can prove compute_subpoints() returns the right
latitude; it cannot prove that "@st.cache_resource stops the TLE load from
re-running" or that "a fragment rerun doesn't re-execute the whole script."
Those are claims about how Streamlit schedules work, and the only honest
way to check them is to actually run a Streamlit app and watch counters.

It uses a hardcoded TLE and touches no network at all (same fixture
approach as test_satellite_data.py / test_globe.py, and necessary here for
the same reason -- the dev sandbox can't reach celestrak.org). That's also
what makes this harness a fair test: if the counters below stay flat, it's
because caching worked, not because a network call quietly failed.

WHAT TO LOOK FOR once it's running -- leave it open for ~10 seconds:

  SCRIPT_RUNS        should stay at 1. Fragment auto-reruns must NOT
                     re-execute the enclosing script. If this climbs, the
                     app is doing a full rerun every tick, which is the
                     exact thing fragments are supposed to avoid.
  CACHED_TLE_LOADS   should stay at 1. The "expensive fetch" body runs
                     once and every later call is a cache hit.
  FRAGMENT_TICKS     should climb by roughly one per FRAGMENT_SECONDS.
                     This is the display cadence doing its job.
  TICK_TIME          should advance in step with FRAGMENT_TICKS. If it
                     freezes while FRAGMENT_TICKS climbs, the fragment is
                     redrawing a stale timestamp -- which is precisely the
                     bug that calling ts.now() INSIDE the fragment (rather
                     than closing over a page-level `t`) prevents.

The point of the last one is worth stating plainly: a dashboard that
redraws on a timer while displaying a frozen instant looks more live than
a static page while being less honest than one. This harness makes that
failure mode visible instead of theoretical.
"""

import streamlit as st
from skyfield.api import EarthSatellite

from satellite_data import ts, compute_subpoints

st.set_page_config(page_title="Day 20 refresh mechanics", layout="wide")
st.title("Day 20 verification: caching + fragment refresh")

FRAGMENT_SECONDS = 2
CACHE_TTL_SECONDS = 3600  # long, so nothing expires during a short check

# Same ISS fixture as test_satellite_data.py / test_globe.py. Not fresh,
# and that does not matter here -- this harness measures WHEN work happens,
# not whether the resulting position is accurate.
ISS_LINE1 = "1 25544U 98067A   24079.51782528  .00016717  00000-0  30187-3 0  9994"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49560343440557"


@st.cache_resource
def call_log() -> dict:
    """
    A mutable counter dict that survives reruns.

    This works precisely BECAUSE it's @st.cache_resource and not
    cache_data: cache_resource hands back the same object every time, so
    mutations to it persist. cache_data would hand back a fresh copy on
    each call and every counter would read 0 forever -- which would make
    this harness silently useless rather than loudly broken. The same
    copy-vs-share distinction is why the real app.py caches its
    EarthSatellite pool with cache_resource and its DataFrames with
    cache_data.
    """
    return {"script_runs": 0, "tle_loads": 0, "fragment_ticks": 0}


@st.cache_resource(ttl=CACHE_TTL_SECONDS)
def load_fixture_satellites() -> list:
    """
    Stands in for app.py's get_satellite_pool(): the "expensive load" whose
    body should execute exactly once no matter how many times it's called.
    """
    call_log()["tle_loads"] += 1
    return [EarthSatellite(ISS_LINE1, ISS_LINE2, "ISS (ZARYA)", ts)]


@st.fragment(run_every=FRAGMENT_SECONDS)
def live_panel():
    log = call_log()
    log["fragment_ticks"] += 1

    # Called on every tick. Should be a cache HIT every time after the
    # first -- CACHED_TLE_LOADS staying at 1 is what proves that.
    satellites = load_fixture_satellites()

    # Fresh reading from the shared Timescale, taken INSIDE the fragment.
    # This is the line the harness exists to validate.
    t_now = ts.now()
    df = compute_subpoints(satellites, t_now)

    st.subheader("Counters")
    # Rendered as plain parseable text on purpose: the automated check
    # scrapes these strings, so they need to survive without a human
    # reading a screenshot.
    st.text(f"SCRIPT_RUNS={log['script_runs']}")
    st.text(f"CACHED_TLE_LOADS={log['tle_loads']}")
    st.text(f"FRAGMENT_TICKS={log['fragment_ticks']}")
    st.text(f"TICK_TIME={t_now.utc_strftime('%H:%M:%S')}")

    st.subheader("Live computed position (recomputed each tick)")
    st.dataframe(df, hide_index=True)

    st.caption(
        f"Fragment reruns every {FRAGMENT_SECONDS} s. The cached loader has a "
        f"{CACHE_TTL_SECONDS} s ttl, so it should not re-execute during this "
        f"check -- CACHED_TLE_LOADS staying at 1 while FRAGMENT_TICKS climbs "
        f"is the whole point."
    )


# Page-body counter. Incremented once per FULL script execution. Fragment
# auto-reruns should never touch this.
call_log()["script_runs"] += 1

st.caption(
    "If SCRIPT_RUNS climbs while you watch, fragments aren't isolating "
    "reruns and the whole Day 20 design needs revisiting."
)

live_panel()