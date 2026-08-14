"""
Day 1 -- the first working script in this project.

Loads the ISS's TLE from CelesTrak's "stations" group and prints its
current latitude, longitude, and altitude. Proved out the basic pipeline
(fetch a TLE, propagate to now, convert to a ground position) that every
later feature builds on. Superseded by `satellite_data.py`'s
`compute_subpoints()` (same math, generalized to many satellites and
returned as a DataFrame instead of printed). Kept in place, unmodified,
as a visible record of the project's build order -- see the README's
"Open Items" section for the open question of whether to trim it.
"""

from skyfield.api import load, wgs84

# --- 1. Load timescale (Skyfield's clock system) ---
ts = load.timescale()

# --- 2. Load ISS TLE from CelesTrak ---
# A TLE is a state + propagation model snapshot — similar to giving Simulink
# initial conditions plus a model (SGP4) instead of raw position/velocity.
# Skyfield caches this file locally after the first download.
stations_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle'
satellites = load.tle_file(stations_url)
by_name = {sat.name: sat for sat in satellites}
iss = by_name['ISS (ZARYA)']

# --- 3. Get current time and propagate ---
t = ts.now()
geocentric = iss.at(t)

# --- 4. Convert to subpoint (lat/lon/altitude) ---
subpoint = wgs84.subpoint(geocentric)

print(f"Time (UTC):  {t.utc_strftime()}")
print(f"Latitude:    {subpoint.latitude.degrees:.4f}°")
print(f"Longitude:   {subpoint.longitude.degrees:.4f}°")
print(f"Altitude:    {subpoint.elevation.km:.2f} km")